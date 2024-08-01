
import torch

from torch import nn, Tensor
from torch.nn import functional as F

from modules.utils.distribution import Distribution
# from modules.normalize import ActNorm
# from modules.permutate import InvertibleConv1x1_1D
from modules.linear import GatedConv1D
from modules.linear import Conv1DZeros


# -----------------------------------------------------------------------------------------
class Sigmoid(nn.Module):
    def __init__(self):
        super(Sigmoid, self).__init__()

    def forward(self, x: Tensor):
        y = torch.sigmoid(x)
        ldj = -F.softplus(x) - F.softplus(-x)
        ldj = ldj.flatten(1).sum(-1)
        return y, ldj
    
    def inverse(self, x: Tensor):
        y = -(torch.reciprocal(x) - 1.).log()
        ldj = -x.log() - (1. - x).log()
        ldj = ldj.flatten(1).sum(-1)
        return y, ldj

# -----------------------------------------------------------------------------------------
class NoAugmentLayer(nn.Module):

    def forward(self, x: Tensor, **kwargs):
        (B, N, _), device = x.shape, x.device
        empty = torch.randn((B, N, 0), device=device)
        ldj = torch.zeros(B, device=device)
        return empty, ldj 

# -----------------------------------------------------------------------------------------
class AugmentLayer(nn.Module):

    def __init__(self, dist: Distribution, aug_channel: int, shallow_net: nn.Module, argument_steps: nn.ModuleList):
        super(AugmentLayer, self).__init__()

        self.dist    = dist
        self.channel = aug_channel
        self.shallow = shallow_net
        self.steps   = argument_steps
        self.n_steps = len(argument_steps)
        self.sigmoid = Sigmoid()

    def forward(self, x: Tensor, **kwargs):

        (B, N, _), device = x.shape, x.device
        ldj = torch.zeros(B, device=device)  # 增强维度时需要与原数据同步到一个设备上

        a = self.shallow(x, **kwargs)
        a = torch.transpose(a, 1, 2)

        eps = self.dist.sample((B, self.channel, N), device)  # 同步设备
        eps_ldj = self.dist.log_prob(eps)

        y = eps
        for i in range(self.n_steps):
            y, log_det_J = self.steps[i](y, a)
            ldj += log_det_J

        y, log_det_J = self.sigmoid(y)
        ldj += log_det_J

        y = torch.transpose(y, 1, 2)
        return y, eps_ldj - ldj

# -----------------------------------------------------------------------------------------
class AugmentStep(nn.Module):

    def __init__(self, channel, hidden_channel, reverse=False):
        super(AugmentStep, self).__init__()

        # self.norm   = ActNorm(channel)
        # self.inv1x1 = InvertibleConv1x1_1D(channel, dim=1)
        self.is_reverse = reverse

        if reverse:
            in_channel = channel - channel // 2
        else:
            in_channel = channel // 2

        self.conv_in = nn.Conv1d(in_channel, hidden_channel, kernel_size=1)
        self.gated_conv = GatedConv1D(hidden_channel)
        # self.layer_norm = nn.LayerNorm([hidden_channel, N])
        self.layer_norm = nn.BatchNorm1d(hidden_channel)
        self.conv_out = Conv1DZeros(hidden_channel, (channel - in_channel) * 2)

    def forward(self, x: Tensor, a: Tensor):
        if self.is_reverse:
            x2, x1 = torch.chunk(x, chunks=2, dim=1)
        else:
            x1, x2 = torch.chunk(x, chunks=2, dim=1)

        st = self.conv_in(x2)
        st = self.gated_conv(st, a)
        st = self.layer_norm(st)
        st = self.conv_out(st)

        shift, scale = st[:, 0::2], torch.sigmoid(st[:, 1::2])

        x1 = (x1 + shift) * scale
        ldj = torch.sum(torch.log(scale).flatten(start_dim=1), dim=-1)

        if self.is_reverse:
            x = torch.cat([x2, x1], dim=1)
        else:
            x = torch.cat([x1, x2], dim=1)
        return x, ldj

    def inverse(self, x: Tensor, a: Tensor):
        x1, x2 = torch.chunk(x, chunks=2, dim=1)

        st = self.conv_in(x2)
        st = self.gated_conv(st, a)
        st = self.layer_norm(st)
        st = self.conv_out(st)

        shift, scale = st[:, 0::2], torch.sigmoid(st[:, 1::2])

        x1 = x1 / scale - shift
        ldj = -torch.sum(torch.log(scale).flatten(start_dim=1), dim=-1)
        x = torch.cat([x1, x2], dim=1)
        return x, ldj        
# -----------------------------------------------------------------------------------------
