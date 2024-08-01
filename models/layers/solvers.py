'''
Adapted from Implicit Normalizing Flow (ICLR 2021).
Link: https://github.com/thu-ml/implicit-normalizing-flows/blob/master/lib/layers/broyden.py
'''

import models.layers.base as base_layers
import torch
from torch import nn
import torch.nn.functional as functional
from torch.autograd import Function
import numpy as np 
import math
import pickle
import sys
import os
from scipy.optimize import root
import time
# from termcolor import colored
import logging
import torch.nn.functional as F
logger = logging.getLogger()


def get_parameters(m, recurse=True):
    def model_parameters(m):
        ps = m._former_parameters.values() if hasattr(m, "_former_parameters") else m.parameters(recurse=False)
        for p in ps:
            yield p
    for m in m.modules() if recurse else [m]:
        for p in model_parameters(m):
            yield p
def find_fixed_point_noaccel1(f, x0, threshold=1000, eps=1e-3):
    B = x0.size(0)
    N = x0.size(1)
    b_shape = (B, N, 1)
    alpha = 0.5 * torch.ones(b_shape, device=x0.device)  # [B, N, 1]
    x, x_prev = (1-alpha)*x0 + (alpha)*f(x0), x0  # [B, N, C]
    i = 0
    tol = eps + eps * x0.abs()

    best_err = 1e9 * torch.ones(b_shape, device=x0.device)    # [B, N, 1]
    best_iter = torch.zeros(b_shape, dtype=torch.int64, device=x0.device)  # [B, N, 1]
    while True:
        fx = f(x)  # [B, N, C]
        err_values = torch.abs(fx - x) / tol  # [B, N, C]
        # cur_err = torch.max(err_values.view(b, -1), dim=1)[0].view(b_shape)
        cur_err = torch.max(err_values, dim=2)[0].view(b_shape)  # [B, N, 1]

        if torch.all(cur_err < 1.):
            break
        alpha = torch.where(torch.logical_and(cur_err >= best_err, i >= best_iter + 30),  # err超过30个iter未减小
            alpha * 0.9,
            alpha)
        alpha = torch.max(alpha, 0.1*torch.ones_like(alpha))  # alpha最小值0.1
        best_iter = torch.where(torch.logical_or(cur_err < best_err, i >= best_iter + 30),  # 至少30iter更新一次
            i * torch.ones(b_shape, dtype=torch.int64, device=x0.device),
            best_iter)
        best_err = torch.min(best_err, cur_err)

        x, x_prev = (1-alpha)*x + (alpha)*fx, x
        i += 1
        if i > threshold:
            dx = torch.abs(f(x) - x)
            rel_err = torch.max(dx/tol).item()
            abs_err = torch.max(dx).item()
            if rel_err > 3 or abs_err > 3 * max(eps, 1e-9):
                logger.info('Relative/Absolute error maximum: %.10f/%.10f' % (rel_err, abs_err))
                logger.info('Iterations exceeded %d for fixed point noaccel.' % (threshold))
                print('rel_err:', rel_err)
                print('abs_err:', abs_err)
            break
    return x





def find_fixed_point(Gnet, y, atol=1e-6, rtol=1e-6):
    x, x_prev = y - Gnet(y), y
    i = 0
    tol = atol + y.abs() * rtol

    while not torch.all((x - x_prev) ** 2 / tol < 1):
        # dx = x.detach() - x_prev.detach()
        x, x_prev = y - Gnet(x), x  # 求解：y - G(x) = x, 把y看成常数， d(nnet)/dx < 1 才能保证可逆
        i += 1

        if i > 10:
            logger.info('Iterations exceeded 50 for inverse.')
            # return find_fixed_point_noaccel1(lambda z: y - Gnet(z), x)
            break
    return x

class RootFind(torch.autograd.Function):
    @staticmethod
    def banach_find_root(Gnet, x):  # *args
        # eps = args[-2]
        # threshold = args[-1]    # Can also set this to be different, based on training/inference
        y_est = find_fixed_point(Gnet, x) # [B, N, C]
        return y_est.clone().detach()

    torch.cuda.empty_cache()
    @staticmethod
    def forward(ctx, Gnet, x):
        with torch.no_grad():
            y_est = RootFind.banach_find_root(Gnet, x)

            # If one would like to analyze the convergence process (e.g., failures, stability), should
            # insert here or in broyden_find_root.
            return y_est

    @staticmethod
    def backward(ctx, grad_y):
        assert 0, 'Cannot backward to this function.'
        grad_args = [None]
        return (None, grad_y)


'''
Provides backward propagation for the implicit mapping F^-1(x).
'''
class MonotoneBlockBackward(torch.autograd.Function):
    """
    A 'dummy' function that does nothing in the forward pass and perform implicit differentiation
    in the backward pass. Essentially a wrapper that provides backprop for the `MonotoneBlock` class.
    You should use this class in MonotoneBlock's forward() function by calling:

        MonotoneBlockBackward.apply(self.func, ...)

    """
    @staticmethod
    def forward(ctx, Gnet, y, x):  # backward的return和前向forward参数一一对应
        ctx.save_for_backward(y, x)
        ctx.Gnet = Gnet
        return y

    @staticmethod
    def backward(ctx, grad):
        grad = grad.clone()
        y, x = ctx.saved_tensors

        Gnet = ctx.Gnet
        y = y.clone().detach().requires_grad_()

        with torch.enable_grad():
            Gy = Gnet(y)

        def h(x_):  # dl/dw - y * (dG(w)/dw) = y, 每次算出来dG(w)/dw都要和当前y做点乘的意思
            y.grad = None
            Gy.backward(x_, retain_graph=True)  #实际上是先求出Jacobian矩阵中每一个元素的梯度值
            xJ = y.grad.clone().detach()        #然后将这个Jacobian矩阵与grad_tensors参数对应的矩阵进行对应的点乘
            y.grad = None
            return xJ

        dl_dh = RootFind.apply(h, grad)  # dl/du = (dl/dw)(Id + JG)^(-1) page17 eq(10)
        dl_dx = dl_dh  # page 22 line7

        return (None, dl_dh, dl_dx)


# o = []
# Ds = []
# G = []
# def Gn(x):
#     return y - Gnet(x)
# dx1 = x.detach() - x_prev.detach()
        # D = torch.norm(dx[0][0])
        # D1 = torch.norm(dx1[0][0])
        # Ds.append(D.abs())
        # d = torch.where(torch.logical_and(dx1[0][0] == 0, dx[0][0] == 0),
        #                 dx1[0][0].abs() - dx[0][0].abs() - 1,
        #                 dx1[0][0].abs() - dx[0][0].abs())

        # if i > 990:
        #     x1 = x.clone().detach().requires_grad_()
        #     with torch.enable_grad():
        #         Gx = Gn(x1)
        #         print('x1:', x1)
        #         print('Gx:', Gx)
        #     x1.grad = None
        #     Gx.backward(torch.ones_like(Gx), retain_graph=True)
        #     grad = x1.grad.clone().detach()
        #     ggrad = torch.where((x - x_prev).abs() > 1e-4,
        #                     grad,
        #                     torch.zeros(x.size()).to(x))
        #     print('i:', i)
        #     # print('maxd', torch.max(x - x_prev))
        #     # print('mind', torch.min(x - x_prev))
        #     d = torch.where((x - x_prev).abs() > 1e-4,
        #                     x,
        #                     torch.zeros(x.size()).to(x))
        #     GGx = Gnet(x)
        #     GGGx = torch.where((x - x_prev).abs() > 1e-4,
        #                        GGx,
        #                        torch.zeros(x.size()).to(x))
        #     print('maxd', torch.max(d))
        #     print('maxggrad', torch.max(ggrad))
        #     print('mind', torch.min(d))
        #     print('maxGGGx', torch.max(GGGx))
        #     x1 = x.clone().detach().requires_grad_()
        #     with torch.enable_grad():
        #         Gx = Gn(x1)
        #     x1.grad = None
        #     Gx.backward(torch.ones_like(Gx), retain_graph=True)
        #     grad = torch.abs(x1.grad.clone().detach())
        #     x1.grad = None
        #     print('G:', torch.max(grad).item())
            #         if torch.max(d) > 0:
            #             x1 = x.clone().detach().requires_grad_()
            #             with torch.enable_grad():
            #                 Gx = Gn(x1)
            #             x1.grad = None
            #             Gx.backward(torch.ones_like(Gx), retain_graph=True)
            #             grad = torch.abs(x1.grad.clone().detach())
            #             G.append(torch.max(grad).item())
            #             x1.grad = None
            #             print('i:', i)

            #             print('x_prev:',x_prev[0][0])
            #             print('x:',x[0][0])
            #             print('dx:',dx[0][0])
            #             print('G',G)
        # o.append(torch.max(d))

# print('max(D1 / D):', torch.max(torch.tensor(o)))
            # print('max D:', torch.max(dx))
            # print('max G:', torch.max(torch.tensor(G)))
            # print(y[0][0])
            # for m in Gnet:
            #     if isinstance(m, base_layers.SpectralNormLinear):
            #         print('m.scale:', m.scale)
            #     if isinstance(m, base_layers.InducedNormLinear):
            #         print('m.scale:', m.scale)

            # rel_err = torch.max((x - x_prev)**2 / tol).item()
            # if rel_err > 10:
            #     return find_fixed_point_noaccel(Gnet, y, atol=1e-5, rtol=1e-5)
            # else:
            #     return find_fixed_point_noaccel(Gnet, x, atol=1e-5, rtol=1e-5)
# def find_fixed_point1(Gnet, x0, threshold=1000, eps=1e-5):
#     # b = x0.size(0)
#     # def g(w):
#     #     return f(w.view(x0.shape)).view(b, -1)
#     def f(x):
#         return x0 - Gnet(x)
#     with torch.no_grad():
#         X0 = x0
#         X1 = f(X0)
#         Gnm1 = X1
#         dXnm1 = X1 - X0
#         Xn = X1
#
#         tol = eps + eps * X0.abs()
#         best_err = math.inf
#         best_iter = 0
#         n = 1
#         while n < threshold:
#             Gn = f(Xn)
#             dXn = Gn - Xn
#             cur_err = torch.max(torch.abs(dXn) / tol).item()
#             if cur_err <= 1.:
#                 break
#             if cur_err < best_err:
#                 best_err = cur_err
#                 best_iter = n
#             elif n >= best_iter + 10:
#                 break
#
#             d2Xn = dXn - dXnm1  # [B, N, C]
#             d2Xn_norm = torch.linalg.vector_norm(d2Xn, dim=2)  # [B, N]
#             mult = (d2Xn * dXn).sum(dim=2) / (d2Xn_norm**2 + 1e-8)  # [B, N]
#             mult = torch.unsqueeze(mult, 2)  # [B, N, 1]
#             Xnp1 = Gn - mult*(Gn - Gnm1)  # [B, N, C]
#
#             dXnm1 = dXn
#             Gnm1 = Gn
#             Xn = Xnp1
#             n = n + 1
#
#         rel_err = torch.max(torch.abs(dXn)/tol).item()
#         if rel_err > 1:
#             # abs_err = torch.max(torch.abs(dXn)).item()
#             # if rel_err > 10:
#             #     return find_fixed_point_noaccel(f, x0, threshold=threshold, eps=eps)
#             # else:
#             #     return find_fixed_point_noaccel(f, Xn.view(x0.shape), threshold=threshold, eps=eps)
#             print('rel_err > 1')
#             for m in f:
#                 if isinstance(m, base_layers.SpectralNormLinear):
#                     print('m.scale:', m.scale)
#                 if isinstance(m, base_layers.InducedNormLinear):
#                     print('m.scale:', m.scale)
#             return Xn
#         else:
#             return Xn  # [B, N, C]
#
# def find_fixed_point_noaccel(Gnet, x0, atol=1e-5, rtol=1e-5):
#     B = x0.size(0)
#     N = x0.size(1)
#     b_shape = (B, N, 1)
#     alpha = 0.5 * torch.ones(b_shape, device=x0.device)  # [B, N, 1]
#     x, x_prev = (1-alpha)*x0 + alpha * (x0 - Gnet(x0)), x0  # [B, N, C]
#     i = 0
#     tol = atol + rtol * x0.abs()
#
#     best_err = 1e9 * torch.ones(b_shape, device=x0.device)    # [B, N, 1]
#     best_iter = torch.zeros(b_shape, dtype=torch.int64, device=x0.device)  # [B, N, 1]
#     while True:
#         fx = x0 - Gnet(x)  # [B, N, C]
#         err_values = torch.abs(fx - x) / tol  # [B, N, C]
#         # cur_err = torch.max(err_values.view(b, -1), dim=1)[0].view(b_shape)
#         cur_err = torch.max(err_values, dim=2)[0].view(b_shape)  # [B, N, 1]
#
#         if torch.all(cur_err < 1.):
#             break
#         alpha = torch.where(torch.logical_and(cur_err >= best_err, i >= best_iter + 30),
#             alpha * 0.9,
#             alpha)
#         alpha = torch.max(alpha, 0.1*torch.ones_like(alpha))
#         best_iter = torch.where(torch.logical_or(cur_err < best_err, i >= best_iter + 30),
#             i * torch.ones(b_shape, dtype=torch.int64, device=x0.device),
#             best_iter)
#         best_err = torch.min(best_err, cur_err)
#
#         x, x_prev = (1-alpha)*x + (alpha)*fx, x
#         i += 1
#         if i > 1000:
#             dx = torch.abs(x0 - Gnet(x) - x)
#             rel_err = torch.max(dx/tol).item()
#             abs_err = torch.max(dx).item()
#             if rel_err > 3 or abs_err > 3 * atol:
#                 logger.info('Relative/Absolute error maximum: %.10f/%.10f' % (rel_err, abs_err))
#                 logger.info('Iterations exceeded 1000 for fixed point noaccel.')
#                 print('Iterations exceeded 1000 for fixed point noaccel.')
#             break
#     return x