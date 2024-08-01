import math
import numpy as np
import torch
import torch.nn as nn
import models.layers.solvers as solvers

import copy
import logging

logger = logging.getLogger()

__all__ = ['iMonotoneBlockCond']


class iMonotoneBlockCond(nn.Module):

    def __init__(
        self,
        nnet,
        geom_p=0.5,
        lamb=2.,
        n_power_series=None,
        n_dist='geometric',
        n_samples=1,
        n_exact_terms=2,
        neumann_grad=False,
        grad_in_forward=False,
        exact_trace=False,
    ):
        """
        Args:
            nnet: a nn.Module
            n_power_series: number of power series. If not None, uses a biased approximation to logdet.
            exact_trace: if False, uses a Hutchinson trace estimator. Otherwise computes the exact full Jacobian.
        """
        super(iMonotoneBlockCond, self).__init__()
        self.nnet = nnet
        self.n_dist = n_dist
        self.geom_p = nn.Parameter(torch.tensor(np.log(geom_p) - np.log(1. - geom_p)), requires_grad=False)
        self.lamb = nn.Parameter(torch.tensor(lamb), requires_grad=False)
        self.n_samples = n_samples
        self.n_power_series = n_power_series
        self.n_exact_terms = n_exact_terms
        self.neumann_grad = neumann_grad
        self.grad_in_forward = grad_in_forward
        self.exact_trace = exact_trace

    def forward(self, x, c, logpx=None):
        class Wrapper(nn.Module):
            def __init__(self, nnet, c):
                super(Wrapper, self).__init__()
                self.nnet = nnet
                self.c = c
            def forward(self, x):
                c = self.c
                for m in self.nnet:
                    x = m(x, c)
                return x
            def build_jvp_net(self, x):
                jvp_net, x, c = self.nnet.build_jvp_net(x, self.c)
                return jvp_net, x
        wrapper = Wrapper(self.nnet, c)
        wrapper_copy = Wrapper(self.nnet.build_clone(), c.detach().clone())
        x0 = x.clone().detach()
        w_value = solvers.RootFind.apply(lambda z: wrapper_copy(z), math.sqrt(2) * x0).detach()
        w_proxy = math.sqrt(2) * x0 - wrapper(w_value)  # For backwarding to parameters in func 为了创造G(w)，将逆函数转化为能计算梯度的新形势
        w = solvers.MonotoneBlockBackward.apply(lambda z: wrapper_copy(z), w_proxy, math.sqrt(2) * x)
        y = math.sqrt(2) * w - x

        if logpx is None:
            return y
        else:
            return y, logpx + self._logdetgrad_monotone_resolvent(w, wrapper) # 两条路传递的w都要计算梯度grad_w，然后求和

    def inverse(self, y, c, logpy=None):
        class Wrapper(nn.Module):
            def __init__(self, nnet, c):
                super(Wrapper, self).__init__()
                self.nnet = nnet
                self.c = c

            def forward(self, y):
                c = self.c
                for m in self.nnet:
                    y = m(y, c)
                return y

            def build_jvp_net(self, y):
                jvp_net, y, c = self.nnet.build_jvp_net(y, self.c)
                return jvp_net, y

        wrapper = Wrapper(self.nnet, c)
        wrapper_copy = Wrapper(self.nnet.build_clone(), c.detach().clone())
        y0 = y.clone().detach()
        w_value = solvers.RootFind.apply(lambda z: -wrapper_copy(z), math.sqrt(2) * y0).detach()
        w_proxy = math.sqrt(2) * y0 + wrapper(w_value)  # For backwarding to parameters in func 为了创造G(w)，将逆函数转化为能计算梯度的新形势
        w = solvers.MonotoneBlockBackward.apply(lambda z: -wrapper_copy(z), w_proxy, math.sqrt(2) * y)
        x = math.sqrt(2) * w - y

        if logpy is None:
            return y
        else:
            return x, logpy - self._logdetgrad_monotone_resolvent(w, wrapper)  # 两条路传递的w都要计算梯度grad_w，然后求和

    def _inverse_fixed_point(self, y, atol=1e-6, rtol=1e-6):
        x, x_prev = y - self.nnet(y), y
        i = 0
        tol = atol + y.abs() * rtol
        while not torch.all(torch.abs(x - x_prev) / tol < 1):
            x, x_prev = y - self.nnet(x), x
            i += 1
            if i > 1000:
                logger.info('Iterations exceeded 1000 for inverse.')
                break
        return x


    def _logdetgrad_monotone_resolvent(self, w, wrapper):
        """Returns logdet|d(sqrt(2)*(Id+g)^{-1}(sqrt(2)*x))/dx|."""
        with torch.enable_grad():

            if self.n_dist == 'geometric':
                geom_p = torch.sigmoid(self.geom_p).item()
                sample_fn = lambda m: geometric_sample(geom_p, m)
                rcdf_fn = lambda k, offset: geometric_1mcdf(geom_p, k, offset)
            elif self.n_dist == 'poisson':
                lamb = self.lamb.item()
                sample_fn = lambda m: poisson_sample(lamb, m)
                rcdf_fn = lambda k, offset: poisson_1mcdf(lamb, k, offset)

            if self.training:
                if self.n_power_series is None:
                    # Unbiased estimation.
                    lamb = self.lamb.item()
                    n_samples = sample_fn(self.n_samples)
                    n_power_series = max(n_samples) + self.n_exact_terms
                    coeff_fn = lambda k: 1 / rcdf_fn(k, self.n_exact_terms) * \
                        sum(n_samples >= k - self.n_exact_terms) / len(n_samples)
                else:
                    # Truncated estimation.
                    n_power_series = self.n_power_series
                    coeff_fn = lambda k: 1.
            else:
                # Unbiased estimation with more exact terms.
                lamb = self.lamb.item()
                n_samples = sample_fn(self.n_samples)
                n_power_series = max(n_samples) + 20
                coeff_fn = lambda k: 1 / rcdf_fn(k, 20) * \
                    sum(n_samples >= k - 20) / len(n_samples)

            if not self.exact_trace:
                ####################################
                # Power series with trace estimator.
                ####################################
                vareps = torch.randn_like(w)

                # Choose the type of estimator.
                if self.training and self.neumann_grad:
                    estimator_fn = neumann_logdet_estimator
                else:
                    estimator_fn = basic_logdet_estimator

                # Do backprop-in-forward to save memory.
                power_series_coeff_fn = lambda k: (-2)/k if k%2 == 1 else 0
                if self.training and self.grad_in_forward:
                    g, logdetgrad = mem_eff_wrapper(
                        estimator_fn, wrapper, w, n_power_series, power_series_coeff_fn, vareps, coeff_fn, self.training
                    )
                else:
                    w = w.requires_grad_(True)
                    g, logdetgrad = estimator_fn(wrapper, w, n_power_series, power_series_coeff_fn, vareps, coeff_fn, self.training)
            else:
                raise NotImplementedError()
                '''
                ############################################
                # Power series with exact trace computation.
                ############################################
                x = x.requires_grad_(True)
                g = self.nnet(x)
                jac = batch_jacobian(g, x)
                logdetgrad = batch_trace(jac)
                jac_k = jac
                for k in range(2, n_power_series + 1):
                    jac_k = torch.bmm(jac, jac_k)
                    logdetgrad = logdetgrad + coeff_fn(k) * batch_trace(jac_k)
                '''

            if self.training and self.n_power_series is None:
                # self.last_n_samples.copy_(torch.tensor(n_samples).to(self.last_n_samples))
                estimator = logdetgrad.detach()
                # self.last_firmom.copy_(torch.mean(estimator).to(self.last_firmom))
                # self.last_secmom.copy_(torch.mean(estimator**2).to(self.last_secmom))
            return logdetgrad.view(-1, 1)


    def extra_repr(self):
        return 'dist={}, n_samples={}, n_power_series={}, neumann_grad={}, exact_trace={}'.format(
            self.n_dist, self.n_samples, self.n_power_series, self.neumann_grad, self.exact_trace
        )





def batch_trace(M):
    return M.view(M.shape[0], -1)[:, ::M.shape[1] + 1].sum(1)


#####################
# Logdet Estimators
#####################
class MemoryEfficientLogDetEstimator(torch.autograd.Function):

    @staticmethod
    def forward(ctx, estimator_fn, gnet, x, n_power_series, power_series_coeff_fn, vareps, coeff_fn, training, *g_params):
        ctx.training = training
        with torch.enable_grad():
            x = x.detach().requires_grad_(True)# detach()返回一个新的tensor，是从当前计算图中分离下来的，但是仍指向原变量的存放位置，其grad_fn=None且requires_grad=False，得到的这个tensor永远不需要计算其梯度，不具有梯度grad，即使之后重新将它的requires_grad置为true,它也不会具有梯度grad
            g, logdetgrad = estimator_fn(gnet, x, n_power_series, power_series_coeff_fn, vareps, coeff_fn, training)
            ctx.g = g
            ctx.x = x

            if training:
                grad_x, *grad_params = torch.autograd.grad(  #grad_x, *grad_params和x、params的维度一一对应
                    logdetgrad.sum(), (x,) + g_params, retain_graph=True, allow_unused=True
                )
                if grad_x is None:
                    grad_x = torch.zeros_like(x)
                ctx.save_for_backward(grad_x, *g_params, *grad_params)

        return safe_detach(g), safe_detach(logdetgrad)

    @staticmethod
    def backward(ctx, grad_g, grad_logdetgrad):
        training = ctx.training
        if not training:
            raise ValueError('Provide training=True if using backward.')

        with torch.enable_grad():
            grad_x, *params_and_grad = ctx.saved_tensors
            g, x = ctx.g, ctx.x

            # Precomputed gradients.
            g_params = params_and_grad[:len(params_and_grad) // 2]
            grad_params = params_and_grad[len(params_and_grad) // 2:]

            dg_x, *dg_params = torch.autograd.grad(g, [x] + g_params, grad_g, allow_unused=True) # 这个部分不回传了

        # Update based on gradient from logdetgrad.
        dL = grad_logdetgrad[0].detach()
        with torch.no_grad():
            grad_x.mul_(dL)
            grad_params = tuple([g.mul_(dL) if g is not None else None for g in grad_params])

        # Update based on gradient from g.
        with torch.no_grad():  # 这一部分不回传了
            grad_x.add_(dg_x)
            grad_params = tuple([dg.add_(djac) if djac is not None else dg for dg, djac in zip(dg_params, grad_params)])

        return (None, None, grad_x, None, None, None, None, None) + grad_params


def basic_logdet_estimator(gnet, x, n_power_series, power_series_coeff_fn, vareps, coeff_fn, training):
    g = gnet(x)

    vjp = vareps
    logdetgrad = torch.tensor(0.).to(x)
    for k in range(1, n_power_series + 1):
        vjp = torch.autograd.grad(g, x, vjp, create_graph=training, retain_graph=True)[0]
        tr = torch.sum(vjp.view(x.shape[0], -1) * vareps.view(x.shape[0], -1), 1)
        delta = power_series_coeff_fn(k) * coeff_fn(k) * tr
        logdetgrad = logdetgrad + delta
    return g, logdetgrad

def neumann_logdet_estimator(gnet, x, n_power_series, power_series_coeff_fn, vareps, coeff_fn, training):
    g = gnet(x)
    jvp_net, _ = gnet.build_jvp_net(x)  # 没有bias 减少计算量

    jvp = vareps
    neumann_jvp = power_series_coeff_fn(1) * vareps
    with torch.no_grad():
        for k in range(1, n_power_series + 1):
            jvp = jvp_net(jvp)  # W本身就是雅可比矩阵
            neumann_jvp = neumann_jvp + (k+1) * power_series_coeff_fn(k+1) * coeff_fn(k) * jvp

    vjp_vareps = torch.autograd.grad(g, x, vareps, create_graph=training)[0]
    logdetgrad = vjp_vareps * neumann_jvp
    logdetgrad = torch.sum(logdetgrad.view(x.shape[0], -1), 1)
    return g, logdetgrad  # [B]

'''
Temporary workaround (works with i-DenseNet but still problematic with Monotone Flow i-DenseNet)
'''
def get_parameters(m, recurse=True):
    def model_parameters(m):
        ps = m._former_parameters.values() if hasattr(m, "_former_parameters") else m.parameters(recurse=False)
        for p in ps:
            yield p
    for m in m.modules() if recurse else [m]:
        for p in model_parameters(m):
            yield p

def mem_eff_wrapper(estimator_fn, gnet, x, n_power_series, power_series_coeff_fn, vareps, coeff_fn, training):

    # We need this in order to access the variables inside this module,
    # since we have no other way of getting variables along the execution path.
    if not isinstance(gnet, nn.Module):
        raise ValueError('g is required to be an instance of nn.Module.')

    return MemoryEfficientLogDetEstimator.apply(
        estimator_fn, gnet, x, n_power_series, power_series_coeff_fn, vareps, coeff_fn, training, *list(get_parameters(gnet))
    )


# -------- Helper distribution functions --------
# These take python ints or floats, not PyTorch tensors.


def geometric_sample(p, n_samples):
    return np.random.geometric(p, n_samples)


def geometric_1mcdf(p, k, offset):
    if k <= offset:
        return 1.
    else:
        k = k - offset
    """P(n >= k)"""
    return (1 - p)**max(k - 1, 0)


def poisson_sample(lamb, n_samples):
    return np.random.poisson(lamb, n_samples)


def poisson_1mcdf(lamb, k, offset):
    if k <= offset:
        return 1.
    else:
        k = k - offset
    """P(n >= k)"""
    sum = 1.
    for i in range(1, k):
        sum += lamb**i / math.factorial(i)
    return 1 - np.exp(-lamb) * sum


def sample_rademacher_like(y):
    return torch.randint(low=0, high=2, size=y.shape).to(y) * 2 - 1


# -------------- Helper functions --------------


def safe_detach(tensor):
    return tensor.detach().requires_grad_(tensor.requires_grad)


def _flatten(sequence):
    flat = [p.reshape(-1) for p in sequence]
    return torch.cat(flat) if len(flat) > 0 else torch.tensor([])


def _flatten_convert_none_to_zeros(sequence, like_sequence):
    flat = [p.reshape(-1) if p is not None else torch.zeros_like(q).view(-1) for p, q in zip(sequence, like_sequence)]
    return torch.cat(flat) if len(flat) > 0 else torch.tensor([])
