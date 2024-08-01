
import torch
import torch.nn as nn
import math

from torch import Tensor
from torch.nn import functional as F


# -----------------------------------------------------------------------------------------
class IdentityLayer(nn.Module):
    def forward(self, x: Tensor, **kwargs):
        return x


# ---------------------------------------------------------------------
class Conv2dZeros(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, logscale_factor=3.0):
        super(Conv2dZeros, self).__init__()
        self.logscale_factor = logscale_factor
        self.bias = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        self.logs = nn.Parameter(torch.zeros(1, out_channels, 1, 1))
        self.conv2d = nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, padding=padding, bias=False)
        nn.init.zeros_(self.conv2d.weight)

    def forward(self, x):
        h = self.logs.mul(self.logscale_factor).exp()
        x = self.conv2d(x)
        x = (x + self.bias) * h
        return x

# ---------------------------------------------------------------------
class Conv1DZeros(nn.Module):

    def __init__(self, in_channel: int, out_channel: int, log_scale=3.0):
        super(Conv1DZeros, self).__init__()

        self.logscale_factor = log_scale
        self.bias = nn.Parameter(torch.zeros(1, out_channel, 1))
        self.logs = nn.Parameter(torch.zeros(1, out_channel, 1))
        self.conv1d = nn.Conv1d(in_channel, out_channel, kernel_size=1, bias=False)

        nn.init.zeros_(self.conv1d.weight)

    def forward(self, x: Tensor):
        s = self.logs.mul(self.logscale_factor).exp()
        x = self.conv1d(x)
        x = (x + self.bias) * s
        return x

# ---------------------------------------------------------------------
class LinearNN1D(nn.Module):
    """Small convolutional network used to compute scale and translate factors."""

    def __init__(self, in_channels, hidden_channels, out_channels):
        super(LinearNN1D, self).__init__()

        from modules.normalize import ActNorm

        self.in_ch = in_channels
        self.in_hi = hidden_channels
        self.out_ch = out_channels

        self.in_conv  = nn.Conv1d(in_channels, hidden_channels, kernel_size=1)
        self.norm1    = ActNorm(hidden_channels)
        self.mid_conv = nn.Conv1d(hidden_channels, hidden_channels, kernel_size=1)
        self.norm2    = ActNorm(hidden_channels)
        self.out_conv = Conv1DZeros(hidden_channels, out_channels)

    def forward(self, x: Tensor, c: Tensor=None):
        if c is not None:
            x = torch.cat([x, c], dim=1)
        x = self.in_conv(x)
        x, _ = self.norm1(x)
        x = F.relu(x)
        x = self.mid_conv(x)
        x, _ = self.norm2(x)
        x = F.relu(x)
        x = self.out_conv(x)
        return x

# -----------------------------------------------------------------------------------------
class GatedConv1D(nn.Module):

    def __init__(self, channels: int, weight_std=0.05):
        super(GatedConv1D, self).__init__()

        def weight_init(conv):
            nn.init.normal_(conv.weight, 0.0, weight_std)
            nn.init.zeros_(conv.bias)

        self.conv1 = nn.Conv1d(channels * 2, channels, kernel_size=1)
        self.conv2 = nn.Conv1d(channels * 2, channels * 2, kernel_size=1)
        self.conv3 = nn.Conv1d(channels * 2, channels * 2, kernel_size=1)

        weight_init(self.conv1)
        weight_init(self.conv2)
        weight_init(self.conv3)
    
    @staticmethod
    def nonlinearity(x):
        return F.elu(torch.cat((x, -x), dim=1))

    @staticmethod
    def gate(x):
        a, b = x.chunk(2, dim=1)
        return a * torch.sigmoid(b)

    def forward(self, x: Tensor, a: Tensor):
        y = GatedConv1D.nonlinearity(x)
        y = self.conv1(y)
        y = GatedConv1D.nonlinearity(y)
        y = self.conv2(y)
        a = GatedConv1D.nonlinearity(a)
        a = self.conv3(a)
        return x + GatedConv1D.gate(y + a)


# -----------------------------------------------------------------------------------------
class ResidualBlock(nn.Module):
    """ResNet basic block with weight norm."""
    def __init__(self, in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.in_norm = nn.BatchNorm2d(in_channels)
        self.in_conv = nn.utils.weight_norm(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False))

        self.out_norm = nn.BatchNorm2d(out_channels)
        self.out_conv = nn.utils.weight_norm(nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=True))

    def forward(self, x):
        skip = x
        x = self.in_norm(x)
        x = F.relu(x)
        x = self.in_conv(x)

        x = self.out_norm(x)
        x = F.relu(x)
        x = self.out_conv(x)

        x = x + skip
        return x


# -----------------------------------------------------------------------------------------
class ResNet(nn.Module):

    def __init__(self, d_in, d_mid, d_out, nblocks, kernel_size, padding):
        super(ResNet, self).__init__()

        self.bn1     = nn.BatchNorm2d(d_in)
        self.conv1   = nn.utils.weight_norm(nn.Conv2d(d_in * 2, d_mid, kernel_size, padding=padding))
        self.in_skip = nn.utils.weight_norm(nn.Conv2d(d_mid, d_mid, kernel_size=1, padding=0))

        self.blocks = nn.ModuleList([
            ResidualBlock(d_mid, d_mid)
            for _ in range(nblocks)
        ])

        self.skips  = nn.ModuleList([
            nn.utils.weight_norm(nn.Conv2d(d_mid, d_mid, kernel_size=1, padding=0))
            for _ in range(nblocks)
        ])

        self.bn2 = nn.BatchNorm2d(d_mid)
        self.conv2 = nn.utils.weight_norm(nn.Conv2d(d_mid, d_out, kernel_size=1, padding=0))
    
    def forward(self, x: Tensor, _c: Tensor):
        x = self.bn1(x)
        x = x * 2  # double_after_norm
        x = torch.cat([x, -x], dim=1)
        x = F.relu(x, inplace=True)
        x = self.conv1(x)
        x_skip = self.in_skip(x)

        for block, skip in zip(self.blocks, self.skips):
            x = block(x)
            x_skip += skip(x)
        
        x = self.bn2(x_skip)
        x = F.relu(x, inplace=True)
        x = self.conv2(x)
        return x


# -----------------------------------------------------------------------------------------
class SoftClampling(nn.Module):
    """
    From https://github.com/VLL-HD/FrEIA/blob/a5069018382d3bef25a6f7fa5a51c810b9f66dc5/FrEIA/modules/coupling_layers.py#L88
    """

    def __init__(self, is_enable=True, clamp=1.9):
        super(SoftClampling, self).__init__()

        self.is_enable = is_enable
        if is_enable:
            self.clamp = 2.0 * clamp / math.pi
        else:
            self.clamp = None

    def forward(self, scale: Tensor):
        if self.is_enable:
            return self.clamp * torch.atan(scale)
        else:
            return scale
# -----------------------------------------------------------------------------------------
