import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from torch import Tensor

class InvertibleLinear(nn.Module):

    def __init__(self, dim):
        super(InvertibleLinear, self).__init__()
        self.dim = dim
        w_init = np.random.randn(dim, dim)
        w_init = np.linalg.qr(w_init)[0].astype(np.float32)

        self.W = nn.Parameter(torch.from_numpy(w_init), requires_grad=True)

    def forward(self, x, logpx=None):
        y = F.linear(x, self.W)
        if logpx is None:
            return y
        else:
            return y, logpx + self._logdetgrad

    def inverse(self, y, logpy=None):
        x = F.linear(y, self.W.inverse())
        if math.isnan(x[0][0][0]):
            print('InvertibleLinear_inverse: nan')
        if logpy is None:
            return x
        else:
            return x, logpy - self._logdetgrad

    @property
    def _logdetgrad(self):
        return torch.log(torch.abs(torch.det(self.W)))

    def extra_repr(self):
        return 'dim={}'.format(self.dim)

class InvertibleConv1x1(nn.Module):

    def __init__(self, channel: int, dim: int):
        super(InvertibleConv1x1, self).__init__()

        w_init = np.random.randn(channel, channel)
        w_init = np.linalg.qr(w_init)[0].astype(np.float32)

        self.W = nn.Parameter(torch.from_numpy(w_init), requires_grad=True)
        # init.normal_(self.W, std=0.01)

        if dim == 1:
            self.equation = 'ij,bjn->bin'
        elif dim == 2 or dim == -1:
            self.equation = 'ij,bnj->bni'
        else:
            raise NotImplementedError(f"Unsupport dim {dim} for InvertibleConv1x1 Layer.")

    def forward(self, x: Tensor, logpx=None):
        z = torch.einsum(self.equation, self.W, x)
        logdet = torch.slogdet(self.W)[1] * x.shape[-1]   # slogdet[0]是符号，[1]是绝对值
        if logpx is None:
            return z
        else:
            return z, logpx + logdet

    def inverse(self, z: Tensor, logpx=None):
        inv_W = torch.inverse(self.W)
        x = torch.einsum(self.equation, inv_W, z)
        return x

