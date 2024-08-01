import collections.abc as container_abcs
from itertools import repeat
import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

__all__ = ['InducedNormLinear']


class InducedNormLinear(nn.Module):

    def __init__(
        self, in_features, out_features, bias=True, coeff=0.97, domain=2, codomain=2, n_iterations=None, atol=None,
        rtol=None, zero_init=False, **unused_kwargs
    ):
        del unused_kwargs
        super(InducedNormLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.coeff = coeff
        self.n_iterations = n_iterations
        self.atol = atol
        self.rtol = rtol
        self.domain = domain
        self.codomain = codomain
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:  # To be changed by injector
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters(zero_init)

        # with torch.no_grad():
        #     domain, codomain = self.compute_domain_codomain()
        domain = self.domain
        codomain = self.codomain

        h, w = self.weight.shape
        self.register_buffer('scale', torch.tensor(0.)) # new_empty返回的张量与此张量具有相同的torch.dtype 和torch.device
        self.register_buffer('u', normalize_u(self.weight.new_empty(h).normal_(0, 1), codomain))
        self.register_buffer('v', normalize_v(self.weight.new_empty(w).normal_(0, 1), domain))
        with torch.no_grad():
            self.compute_weight(True, n_iterations=200, atol=None, rtol=None)

    def reset_parameters(self, zero_init=False):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5)) # 其本质是初始参数的选择应使得objective function便于被优化
        if zero_init:
            # normalize cannot handle zero weight in some cases.
            self.weight.data.div_(1000)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def compute_domain_codomain(self):
        if torch.is_tensor(self.domain):
            domain = asym_squash(self.domain)
            codomain = asym_squash(self.codomain)
        else:
            domain, codomain = self.domain, self.codomain
        return domain, codomain

    def compute_one_iter(self):
        domain, codomain = self.compute_domain_codomain()
        u = self.u.detach()
        v = self.v.detach()
        weight = self.weight.detach()
        u = normalize_u(torch.mv(weight, v), codomain)
        v = normalize_v(torch.mv(weight.t(), u), domain)
        return torch.dot(u, torch.mv(weight, v))

    def compute_weight(self, update=True, n_iterations=None, atol=None, rtol=None):
        u = self.u
        v = self.v
        weight = self.weight

        if update:
            # print('update')
            n_iterations = self.n_iterations if n_iterations is None else n_iterations
            atol = self.atol if atol is None else atol
            rtol = self.rtol if rtol is None else atol

            if n_iterations is None and (atol is None or rtol is None):
                raise ValueError('Need one of n_iteration or (atol, rtol).')

            max_itrs = 200
            if n_iterations is not None:
                max_itrs = n_iterations

            with torch.no_grad():
                domain, codomain = self.compute_domain_codomain()
                for _ in range(max_itrs):
                    # Algorithm from http://www.qetlab.com/InducedMatrixNorm.
                    if n_iterations is None and atol is not None and rtol is not None:
                        old_v = v.clone()
                        old_u = u.clone()

                    u = normalize_u(torch.mv(weight, v), codomain, out=u)
                    v = normalize_v(torch.mv(weight.t(), u), domain, out=v)

                    if n_iterations is None and atol is not None and rtol is not None:
                        err_u = torch.norm(u - old_u) / (u.nelement()**0.5)
                        err_v = torch.norm(v - old_v) / (v.nelement()**0.5)
                        tol_u = atol + rtol * torch.max(u)
                        tol_v = atol + rtol * torch.max(v)
                        if err_u < tol_u and err_v < tol_v:
                            break
                self.v.copy_(v)
                self.u.copy_(u)
                u = u.clone()
                v = v.clone()

        sigma = torch.dot(u, torch.mv(weight, v))
        with torch.no_grad():
            self.scale.copy_(sigma)
        # soft normalization: only when sigma larger than coeff
        factor = torch.max(torch.ones(1, device=weight.device), sigma / self.coeff)
        weight = weight / factor
        return weight

    def forward(self, input):
        weight = self.compute_weight(update=False)
        return F.linear(input, weight, self.bias)

    def build_clone(self):
        with torch.no_grad():
            weight = self.compute_weight(update=False).detach().requires_grad_(False)
            if self.bias is not None:
                bias = self.bias.detach().requires_grad_(False)
            m = nn.Linear(self.in_features, self.out_features, bias=self.bias is not None, device=self.weight.device)
            m.weight.data.copy_(weight)
            if self.bias is not None:
                m.bias.data.copy_(bias)
            return m

    def build_jvp_net(self, x):
        '''
        Bias is omitted in contrast to self.build_clone().
        '''
        with torch.no_grad():
            weight = self.compute_weight(update=False).detach().requires_grad_(False)
            m = nn.Linear(self.in_features, self.out_features, bias=None, device=self.weight.device)
            m.weight.data.copy_(weight)
            return m, self.forward(x).detach().clone()

    def extra_repr(self):
        domain, codomain = self.compute_domain_codomain()
        return (
            'in_features={}, out_features={}, bias={}'
            ', coeff={}, domain={:.2f}, codomain={:.2f}, n_iters={}, atol={}, rtol={}, learnable_ord={}'.format(
                self.in_features, self.out_features, self.bias is not None, self.coeff, domain, codomain,
                self.n_iterations, self.atol, self.rtol, torch.is_tensor(self.domain)
            )
        )
class InducedNormLinear1(nn.Module):

    def __init__(
        self, in_features, out_features, bias=True, coeff=0.97, domain=2, codomain=2, n_iterations=None, atol=None,
        rtol=None, zero_init=False, **unused_kwargs
    ):
        del unused_kwargs
        super(InducedNormLinear1, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.coeff = coeff
        self.n_iterations = n_iterations
        self.atol = atol
        self.rtol = rtol
        self.domain = domain
        self.codomain = codomain
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:  # To be changed by injector
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters(zero_init)

        # with torch.no_grad():
        #     domain, codomain = self.compute_domain_codomain()
        domain = self.domain
        codomain = self.codomain

        h, w = self.weight.shape
        self.register_buffer('scale', torch.tensor(0.)) # new_empty返回的张量与此张量具有相同的torch.dtype 和torch.device
        self.register_buffer('u', normalize_u(self.weight.new_empty(h).normal_(0, 1), codomain))
        self.register_buffer('v', normalize_v(self.weight.new_empty(w).normal_(0, 1), domain))
        self.compute_weight(True, n_iterations=200, atol=None, rtol=None)

    def reset_parameters(self, zero_init=False):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5)) # 其本质是初始参数的选择应使得objective function便于被优化
        if zero_init:
            # normalize cannot handle zero weight in some cases.
            self.weight.data.div_(1000)
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            init.uniform_(self.bias, -bound, bound)

    def compute_domain_codomain(self):
        if torch.is_tensor(self.domain):
            domain = asym_squash(self.domain)
            codomain = asym_squash(self.codomain)
        else:
            domain, codomain = self.domain, self.codomain
        return domain, codomain

    def compute_one_iter(self):
        domain, codomain = self.compute_domain_codomain()
        u = self.u.detach()
        v = self.v.detach()
        weight = self.weight.detach()
        u = normalize_u(torch.mv(weight, v), codomain)
        v = normalize_v(torch.mv(weight.t(), u), domain)
        return torch.dot(u, torch.mv(weight, v))

    def compute_weight(self, update=True, n_iterations=None, atol=None, rtol=None):
        u = self.u
        v = self.v
        weight = self.weight

        if update:
            # print('update')
            n_iterations = self.n_iterations if n_iterations is None else n_iterations
            atol = self.atol if atol is None else atol
            rtol = self.rtol if rtol is None else atol

            if n_iterations is None and (atol is None or rtol is None):
                raise ValueError('Need one of n_iteration or (atol, rtol).')

            max_itrs = 200
            if n_iterations is not None:
                max_itrs = n_iterations

            with torch.no_grad():
                domain, codomain = self.compute_domain_codomain()
                for _ in range(max_itrs):
                    # Algorithm from http://www.qetlab.com/InducedMatrixNorm.
                    if n_iterations is None and atol is not None and rtol is not None:
                        old_v = v.clone()
                        old_u = u.clone()

                    u = normalize_u(torch.mv(weight, v), codomain, out=u)
                    v = normalize_v(torch.mv(weight.t(), u), domain, out=v)

                    if n_iterations is None and atol is not None and rtol is not None:
                        err_u = torch.norm(u - old_u) / (u.nelement()**0.5)
                        err_v = torch.norm(v - old_v) / (v.nelement()**0.5)
                        tol_u = atol + rtol * torch.max(u)
                        tol_v = atol + rtol * torch.max(v)
                        if err_u < tol_u and err_v < tol_v:
                            break
                self.v.copy_(v)
                self.u.copy_(u)
                u = u.clone()
                v = v.clone()

        sigma = torch.dot(u, torch.mv(weight, v))
        with torch.no_grad():
            self.scale.copy_(sigma)
        # soft normalization: only when sigma larger than coeff
        factor = torch.max(torch.ones(1, device=weight.device), sigma / self.coeff)
        weight = weight / factor
        return weight

    def forward(self, input):
        weight = self.compute_weight(update=self.training)
        return F.linear(input, weight, self.bias)

    def build_clone(self):
        with torch.no_grad():
            weight = self.compute_weight(update=False).detach().requires_grad_(False)
            if self.bias is not None:
                bias = self.bias.detach().requires_grad_(False)
            m = nn.Linear(self.in_features, self.out_features, bias=self.bias is not None, device=self.weight.device)
            m.weight.data.copy_(weight)
            if self.bias is not None:
                m.bias.data.copy_(bias)
            return m

    def build_jvp_net(self, x):
        '''
        Bias is omitted in contrast to self.build_clone().
        '''
        with torch.no_grad():
            weight = self.compute_weight(update=False).detach().requires_grad_(False)
            m = nn.Linear(self.in_features, self.out_features, bias=None, device=self.weight.device)
            m.weight.data.copy_(weight)
            return m, self.forward(x).detach().clone()

    def extra_repr(self):
        domain, codomain = self.compute_domain_codomain()
        return (
            'in_features={}, out_features={}, bias={}'
            ', coeff={}, domain={:.2f}, codomain={:.2f}, n_iters={}, atol={}, rtol={}, learnable_ord={}'.format(
                self.in_features, self.out_features, self.bias is not None, self.coeff, domain, codomain,
                self.n_iterations, self.atol, self.rtol, torch.is_tensor(self.domain)
            )
        )
def projmax_(v):
    """Inplace argmax on absolute value."""
    ind = torch.argmax(torch.abs(v))
    v.zero_()
    v[ind] = 1
    return v


def normalize_v(v, domain, out=None):
    if not torch.is_tensor(domain) and domain == 2:
        v = F.normalize(v, p=2, dim=0, out=out)
    elif domain == 1:
        v = projmax_(v)
    else:
        vabs = torch.abs(v)
        vph = v / vabs
        vph[torch.isnan(vph)] = 1
        vabs = vabs / torch.max(vabs)
        vabs = vabs**(1 / (domain - 1))
        v = vph * vabs / vector_norm(vabs, domain)
    return v


def normalize_u(u, codomain, out=None):
    if not torch.is_tensor(codomain) and codomain == 2:
        u = F.normalize(u, p=2, dim=0, out=out)
    elif codomain == float('inf'):
        u = projmax_(u)
    else:
        uabs = torch.abs(u)
        uph = u / uabs
        uph[torch.isnan(uph)] = 1
        uabs = uabs / torch.max(uabs)
        uabs = uabs**(codomain - 1)
        if codomain == 1:
            u = uph * uabs / vector_norm(uabs, float('inf'))
        else:
            u = uph * uabs / vector_norm(uabs, codomain / (codomain - 1))
    return u


def vector_norm(x, p):
    x = x.view(-1)
    return torch.sum(x**p)**(1 / p)


def leaky_elu(x, a=0.3):
    return a * x + (1 - a) * F.elu(x)


def asym_squash(x):
    return torch.tanh(-leaky_elu(-x + 0.5493061829986572)) * 2 + 3


# def asym_squash(x):
#     return torch.tanh(x) / 2. + 2.


def _ntuple(n):

    def parse(x):
        if isinstance(x, container_abcs.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


_single = _ntuple(1)
_pair = _ntuple(2)
_triple = _ntuple(3)
_quadruple = _ntuple(4)


