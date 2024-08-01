import torch
import torch.nn as nn
import torch.distributed as dist
from torch import Tensor
import math

# -----------------------------------------------------------------------------------------
class ActNorm1(nn.Module):
    """Yet, another ActNorm implementation for Point Cloud."""

    def __init__(self, channel: int, dim=1):
        super(ActNorm1, self).__init__()
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

    def forward(self, x: Tensor, logpx=None):
        """
        x: [B, C, N]
        """
        if not self.is_inited:
            self.__initialize(x)

        z = x * torch.exp(self.logs) + self.bias
        # z = (x - self.bias) * torch.exp(-self.logs)
        logdet = x.shape[self.Ndim] * torch.sum(self.logs)  # B维还是标量都差不多，最后都要取mean
        if logpx is None:
            return z
        else:
            return z, logpx + logdet

    def inverse(self, z: Tensor, _: Tensor = None):
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
class ActNorm(nn.Module):

    def __init__(self, num_features, eps=1e-12):
        super(ActNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.weight = nn.Parameter(torch.Tensor(num_features))
        self.bias = nn.Parameter(torch.Tensor(num_features))
        self.register_buffer('initialized', torch.tensor(0))


    def forward(self, x, logpx=None):
        c = x.size(2)

        if not self.initialized:
            with torch.no_grad():
                # compute batch statistics
                x_t = x.transpose(0, 2).contiguous().view(c, -1)

                # sync if distributed is used
                if dist.is_initialized():
                    world_size = dist.get_world_size(group=None)

                    batch_mean_intra = torch.mean(x_t, dim=1)
                    batch_var_intra = torch.var(x_t, dim=1)

                    batch_mean = batch_mean_intra.detach().clone()
                    dist.all_reduce(batch_mean, op=dist.ReduceOp.SUM)
                    batch_mean = batch_mean / world_size
                    dist.all_reduce(batch_var_intra, op=dist.ReduceOp.SUM)
                    batch_var_intra = batch_var_intra / world_size

                    batch_var_inter = torch.pow(batch_mean_intra - batch_mean, 2)
                    dist.all_reduce(batch_var_inter, op=dist.ReduceOp.SUM)
                    batch_var_inter = batch_var_inter / world_size
                    batch_var = batch_var_intra + batch_var_inter
                else:
                    batch_mean = torch.mean(x_t, dim=1)
                    batch_var = torch.var(x_t, dim=1)

                # for numerical issues
                batch_var = torch.max(batch_var, torch.tensor(0.2).to(batch_var))

                self.bias.data.copy_(-batch_mean)
                self.weight.data.copy_(-0.5 * torch.log(batch_var))
                self.initialized.fill_(1)

        bias = self.bias.view([1, 1, -1]).expand_as(x)
        weight = self.weight.view([1, 1, -1]).expand_as(x)

        y = (x + bias) * torch.exp(weight)
        if math.isnan(y[0][0][0]):
            print('Actnorm_forward: nan')
        if logpx is None:
            return y
        else:
            return y, logpx + self._logdetgrad(x)

    def inverse(self, y, logpy=None):
        assert self.initialized
        bias = self.bias.view([1, 1, -1]).expand_as(y)
        weight = self.weight.view([1, 1, -1]).expand_as(y)

        x = y * torch.exp(-weight) - bias
        if math.isnan(x[0][0][0]):
            print('Actnorm_inverse: nan')
        if logpy is None:
            return x
        else:
            return x, logpy - self._logdetgrad(x)

    def _logdetgrad(self, x):
        return self.weight.view([1, 1, -1]).expand(*x.size()).contiguous().view(x.size(0), -1).sum(1, keepdim=True)
        # return self.weight.view([1, 1, -1]).expand(*x.size()).contiguous().sum(-1)

