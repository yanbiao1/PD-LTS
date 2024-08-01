
import torch
import torch.nn as nn

from torch import Tensor


# -----------------------------------------------------------------------------------------
class ActNorm(nn.Module):
    """Yet, another ActNorm implementation for Point Cloud."""

    def __init__(self, channel: int, dim=1):
        super(ActNorm, self).__init__()

        assert dim in [-1, 1, 2]
        self.dim = 2 if dim == -1 else dim

        if self.dim == 1:
            self.logs = nn.Parameter(torch.zeros((1, channel, 1)))  # log sigma
            self.bias = nn.Parameter(torch.zeros((1, channel, 1)))
            self.Ndim = 2
        if self.dim == 2:
            self.logs = nn.Parameter(torch.zeros((1, 1, channel)))
            self.bias = nn.Parameter(torch.zeros((1, 1, channel)))
            self.Ndim = 1

        self.eps = 1e-6
        self.is_inited = False

    def forward(self, x: Tensor, _: Tensor=None):
        """
        x: [B, C, N]
        """
        if not self.is_inited:
            self.__initialize(x)
        z = x * torch.exp(self.logs) + self.bias
        # print('maxz:', torch.max(z))
        # z = (x - self.bias) * torch.exp(-self.logs)
        logdet = x.shape[self.Ndim] * torch.sum(self.logs)
        return z, logdet

    def inverse(self, z: Tensor, _:Tensor=None):
        # x = z * torch.exp(self.logs) + self.bias
        x = (z - self.bias) * torch.exp(-self.logs)
        return x

    def __initialize(self, x: Tensor):
        with torch.no_grad():
            dims = [0, 1, 2]
            dims.remove(self.dim)

            bias = -torch.mean(x.detach(), dim=dims, keepdim=True)
            logs = -torch.log(torch.std(x.detach(), dim=dims, keepdim=True) + self.eps)
            self.bias.data.copy_(bias.data)
            self.logs.data.copy_(logs.data)
            self.is_inited = True
# -----------------------------------------------------------------------------------------
