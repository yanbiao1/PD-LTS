
import torch
import torch.nn as nn

from torch import Tensor
from typing import Dict

from modules.linear import IdentityLayer


# -----------------------------------------------------------------------------------------
class AffineCouplingLayer(nn.Module):

    def __init__(self, coupling: str, transform_net: nn.Module, params: Dict, split_dim=1, clamp=None):
        super(AffineCouplingLayer, self).__init__()
        assert coupling in ['additive', 'affine', 'affineEx']
        assert split_dim in [-1, 1, 2, 3]
        self.coupling  = coupling
        self.dim = split_dim

        self.bias_net = transform_net(**params)

        if self.coupling == 'affine':
            self.scale_net = transform_net(**params)
        elif self.coupling == 'affineEx':
            params_t = params
            params_t['in_channel'] = params['out_channel']
            params_t['out_channel'] = params['in_channel']
            self.g1 = transform_net(**params_t)
            self.g2 = transform_net(**params)
            self.g3 = transform_net(**params)

        self.clamping = clamp or IdentityLayer()

    def forward(self, x: Tensor, c: Tensor=None, **kwargs):

        h1, h2 = self.channel_split(x)
        if self.coupling == 'affine':
            scale = self.clamping(self.scale_net(h1, c, **kwargs))
            bias  = self.bias_net(h1, c, **kwargs)

            h2 = (h2 - bias) * torch.exp(-scale)
            log_det_J = -scale.flatten(start_dim=1).sum(1)
        elif self.coupling == 'additive':
            bias = self.bias_net(h1, c, **kwargs)
            h2 = h2 - bias
            log_det_J = None
        elif self.coupling == 'affineEx':
            scale = self.clamping(self.g2(h1, c, **kwargs))
            bias  = self.g3(h1, c, **kwargs)
    
            h1 = h1 + self.g1(h2)
            h2 = torch.exp(scale) * h2 + bias
            log_det_J = scale.flatten(start_dim=1).sum(1)
        else:
            raise NotImplementedError()

        x = self.channel_cat(h1, h2)
        return x, log_det_J

    def inverse(self, z: Tensor, c: Tensor=None, **kwargs):
        
        h1, h2 = self.channel_split(z)

        if self.coupling == 'affine':
            scale = self.clamping(self.scale_net(h1, c, **kwargs))
            bias  =  self.bias_net(h1, c, **kwargs)

            h2 = h2 * torch.exp(scale) + bias

        elif self.coupling == 'additive':
            bias = self.bias_net(h1, c, **kwargs)
            h2 = h2 + bias
        elif self.coupling == 'affineEx':
            scale = self.clamping(self.g2(h1, c, **kwargs))
            bias  = self.g3(h1, c, **kwargs)

            h2 = (h2 - bias) * torch.exp(-scale)
            h1 = h1 - self.g1(h2)
        else:
            raise NotImplementedError()

        z = self.channel_cat(h1, h2)
        return z

    def channel_split(self, x: Tensor):
        return torch.chunk(x, 2, dim=self.dim)

    def channel_cat(self, h1: Tensor, h2: Tensor):
        return torch.cat([h1, h2], dim=self.dim)



# -----------------------------------------------------------------------------------------
class AffineSpatialCouplingLayer(AffineCouplingLayer):

    def __init__(self, coupling, transform_net, params, is_even, split_dim, clamp=None):
        super().__init__(coupling, transform_net, params, split_dim=split_dim, clamp=clamp)
        self.is_even = is_even

    def channel_split(self, x: Tensor):
        if self.is_even:
            return torch.split(x, [1, 2], dim=self.dim)
        else:
            return torch.split(x, [2, 1], dim=self.dim)


# -----------------------------------------------------------------------------------------
class AffineInjectorLayer(AffineCouplingLayer):

    def __init__(self, coupling, transform_net, params, clamp=None):
        super().__init__(coupling, transform_net, params, split_dim=-1, clamp=clamp)

    def forward(self, x: Tensor, c: Tensor):
        log_det_J = None

        if self.coupling == 'additive':
            x = x - self.bias_net(c)
        if self.coupling == 'affine':
            scale = self.clamping(self.scale_net(c))
            bias  = self.bias_net(c)

            x = (x - bias) * torch.exp(-scale)
            log_det_J = -torch.sum(torch.flatten(scale, start_dim=1), dim=1)

        return x, log_det_J

    def inverse(self, z: Tensor, c: Tensor):
        if self.coupling == 'additive':
            z = z + self.bias_net(c)
        if self.coupling == 'affine':
            scale = self.clamping(self.scale_net(c))
            bias  = self.bias_net(c)
            z = z * torch.exp(scale) + bias
        return z
# -----------------------------------------------------------------------------------------
