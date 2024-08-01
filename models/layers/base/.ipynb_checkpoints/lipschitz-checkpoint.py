import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

from .utils import _pair
from .mixed_lipschitz import InducedNormLinear, InducedNormLinear1

__all__ = ['SpectralNormLinear', 'get_linear', 'get_linear1']


class SpectralNormLinear(nn.Module):

    def __init__(
        self, in_features, out_features, bias=True, coeff=0.97, n_iterations=None, atol=None, rtol=None, **unused_kwargs
    ):
        del unused_kwargs
        super(SpectralNormLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.coeff = coeff
        self.n_iterations = n_iterations
        self.atol = atol
        self.rtol = rtol
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

        h, w = self.weight.shape
        self.register_buffer('scale', torch.tensor(0.))
        self.register_buffer('u', F.normalize(self.weight.new_empty(h).normal_(0, 1), dim=0))
        self.register_buffer('v', F.normalize(self.weight.new_empty(w).normal_(0, 1), dim=0))
        self.compute_weight(True, 200)

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def compute_weight(self, update=True, n_iterations=None, atol=None, rtol=None):
        n_iterations = self.n_iterations if n_iterations is None else n_iterations
        atol = self.atol if atol is None else atol
        rtol = self.rtol if rtol is None else atol

        if n_iterations is None and (atol is None or rtol is None):
            raise ValueError('Need one of n_iteration or (atol, rtol).')

        if n_iterations is None:
            n_iterations = 20000

        u = self.u
        v = self.v
        weight = self.weight
        if update:
            with torch.no_grad():
                itrs_used = 0.
                for _ in range(n_iterations):
                    old_v = v.clone()
                    old_u = u.clone()
                    # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
                    # are the first left and right singular vectors.
                    # This power iteration produces approximations of `u` and `v`.
                    v = F.normalize(torch.mv(weight.t(), u), dim=0, out=v)
                    u = F.normalize(torch.mv(weight, v), dim=0, out=u)
                    itrs_used = itrs_used + 1
                    if atol is not None and rtol is not None:
                        err_u = torch.norm(u - old_u) / (u.nelement()**0.5)
                        err_v = torch.norm(v - old_v) / (v.nelement()**0.5)
                        tol_u = atol + rtol * torch.max(u)
                        tol_v = atol + rtol * torch.max(v)
                        if err_u < tol_u and err_v < tol_v:
                            break
                if itrs_used > 0:
                    u = u.clone()
                    v = v.clone()

        sigma = torch.dot(u, torch.mv(weight, v))
        with torch.no_grad():
            self.scale.copy_(sigma)
        # soft normalization: only when sigma larger than coeff
        factor = torch.max(torch.ones(1).to(weight.device), sigma / self.coeff)
        weight = weight / factor
        return weight

    def forward(self, input):
        weight = self.compute_weight(update=self.training)
        return F.linear(input, weight, self.bias)

    def extra_repr(self):
        return 'in_features={}, out_features={}, bias={}, coeff={}, n_iters={}, atol={}, rtol={}'.format(
            self.in_features, self.out_features, self.bias is not None, self.coeff, self.n_iterations, self.atol,
            self.rtol
        )

def get_linear(in_features, out_features, bias=True, coeff=0.97, domain=None, codomain=None, **kwargs):
    _linear = InducedNormLinear
    return _linear(in_features, out_features, bias, coeff, domain, codomain, **kwargs)

def get_linear1(in_features, out_features, bias=True, coeff=0.97, **kwargs):
    _linear = InducedNormLinear1
    return _linear(in_features, out_features, bias, coeff, **kwargs)
