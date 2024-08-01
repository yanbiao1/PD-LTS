
import torch
import torch.nn as nn

from torch import Tensor
from torch.nn import functional as F

from pytorch3d.ops import knn_gather



# -----------------------------------------------------------------------------------------
def knn_group(x: Tensor, i: Tensor):
    """
    x: [B, N, C]
    i: [B, M, k]
    return: [B, M, k, C]
    """
    (B, N, C), (_, M, k) = x.shape, i.shape

    # approach 1
    # x = x.unsqueeze(1).expand(B, M, N, C)
    # i = i.unsqueeze(3).expand(B, M, k, C)
    # return torch.gather(x, dim=2, index=i)

    # approach 2 (save some gpu memory)
    idxb = torch.arange(B).view(-1, 1)
    i = i.reshape(B, M * k)
    y = x[idxb, i].view(B, M, k, C)  # [B, M * k, C]
    return y

# -----------------------------------------------------------------------------------------
def get_knn_idx(k: int, f: Tensor, q: Tensor=None, offset=None, return_features=False):
    """
    f: [B, N, C]
    q: [B, M, C]
    index of points in f: [B, M, k]
    """
    if offset is None:
        offset = 0
    if q is None:
        q = f

    (B, N, C), (_, M, _) = f.shape, q.shape

    _f = f.unsqueeze(1).expand(B, M, N, C)
    _q = q.unsqueeze(2).expand(B, M, N, C)

    dist = torch.sum((_f - _q) ** 2, dim=3, keepdim=False)  # [B, M, N]
    knn_idx = torch.argsort(dist, dim=2)[..., offset:k+offset]  # [B, M, k]

    if return_features is True:
        knn_f = knn_group(f, knn_idx)
        return knn_f, knn_idx
    else:
        return knn_idx


# -----------------------------------------------------------------------------------------
class FullyConnectedLayer(torch.nn.Module):

    def __init__(self, in_features, out_features, bias=True, activation=None):
        super(FullyConnectedLayer, self).__init__()

        self.linear = torch.nn.Linear(in_features, out_features, bias=bias)

        if activation is None:
            self.activation = torch.nn.Identity()
        elif activation == 'relu':
            self.activation = torch.nn.ReLU()
        elif activation == 'elu':
            self.activation = torch.nn.ELU(alpha=1.0)
        elif activation == 'lrelu':
            self.activation = torch.nn.LeakyReLU(0.1)
        else:
            raise ValueError()

    def forward(self, x):
        return self.activation(self.linear(x))

# -----------------------------------------------------------------------------------------

class noiseEdgeConv(nn.Module):
    def __init__(self, in_channel, hidden_channel, out_channel, bias=True):
        super(noiseEdgeConv, self).__init__()

        self.linear1 = nn.Linear(in_channel * 2, hidden_channel, bias=bias)
        self.linear2 = nn.Linear(hidden_channel, hidden_channel, bias=bias)
        self.linear3 = nn.Linear(in_channel, hidden_channel, bias=bias)
        self.linear4 = nn.Linear(hidden_channel, hidden_channel, bias=bias)
        self.linear5 = nn.Linear(hidden_channel, out_channel, bias=bias)

        nn.init.normal_(self.linear5.weight, mean=0.0, std=0.05)
        nn.init.zeros_(self.linear5.bias)

    def forward(self, f: Tensor, _c=None, knn_idx: Tensor = None):
        """
        f: [B, N, C]
        knn_idx: [B, N, k]
        return: [B, N, out]
        """
        if knn_idx is None:
            knn_idx = get_knn_idx(32, f)
        knn_feat = knn_gather(f, knn_idx)  # [B, M, k, C]

        f_tiled = f.unsqueeze(2).expand_as(knn_feat)  # [B, N, k, C]
        x = torch.cat([knn_feat, knn_feat - f_tiled], dim=-1)  # [B, N, k, C * 3]

        x = F.relu(self.linear1(x))  # [B, N, k, h]
        x = F.relu(self.linear2(x))  # [B, N, k, h]
        x, _ = torch.max(x, dim=2, keepdim=False)  # [B, N, h]
        f = F.relu(self.linear3(f))
        f = F.relu(self.linear4(f))
        x = x + f
        x = self.linear5(x)  # [B, N, out]
        return x

class PreConv(nn.Module):
    def __init__(self, in_channel, out_channel, bias=True):
        super(PreConv, self).__init__()
        in_channel = in_channel * 2
        self.conv = nn.Sequential(*[
            nn.Conv2d(in_channel, out_channel, kernel_size=[1, 1]),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.05, inplace=True),
        ])
    def forward(self, f: Tensor, knn_idx: Tensor = None):
        """
        f: [B, N, C]
        knn_idx: [B, N, k]
        return: [B, N, out]
        """
        if knn_idx is None:
            knn_idx = get_knn_idx(32, f)
        knn_feat = knn_gather(f, knn_idx)  # [B, M, k, 3]
        f_tiled = f.unsqueeze(2).expand_as(knn_feat)  # [B, N, k, 3]
        x = torch.cat([f_tiled, knn_feat - f_tiled], dim=-1)  # [B, N, k, 2 * 3]
        x = x.permute(0, 3, 1, 2)  # [B, 2 * 3, N, k]
        x = self.conv(x)
        x, _ = torch.max(x, dim=-1, keepdim=False)
        x = torch.transpose(x, 1, 2)
        return x

class EdgeConv(nn.Module):
    def __init__(self, in_channel, hidden_channel, out_channel, concat=True):
        super(EdgeConv, self).__init__()
        self.concat = concat
        if concat == False:
            hidden_channel = hidden_channel + 32
        self.convs = nn.ModuleList()
        conv_first = nn.Sequential(*[
            nn.Conv2d(in_channel * 2, hidden_channel, kernel_size=[1, 1]),
            nn.BatchNorm2d(hidden_channel),
            nn.LeakyReLU(0.05, inplace=True),
        ])
        self.convs.append(conv_first)
        conv = nn.Sequential(*[
            nn.Conv2d(hidden_channel, out_channel, kernel_size=[1, 1], bias=True),
            nn.BatchNorm2d(out_channel),
            nn.LeakyReLU(0.05, inplace=True),
        ])
        self.convs.append(conv)

    def forward(self, f: Tensor, knn_idx: Tensor = None):
        """
        f: [B, N, C]
        knn_idx: [B, N, k]
        return: [B, N, out]
        """
        # if self.convs[0][0].weight.grad != None:
        #     print(torch.max(self.convs[0][0].weight.grad))
        if knn_idx is None:
            knn_idx = get_knn_idx(32, f)
        knn_feat = knn_gather(f, knn_idx)  # [B, M, k, C]
        f_tiled = f.unsqueeze(2).expand_as(knn_feat)  # [B, N, k, C]
        x = torch.cat([f_tiled, knn_feat - f_tiled], dim=-1)  # [B, N, k, C * 2]

        x = x.permute(0, 3, 1, 2)  # [B, C * 2, N, k]
        for i in range(len(self.convs)):
            x = self.convs[i](x)
        x, _ = torch.max(x, dim=-1, keepdim=False)
        x = torch.transpose(x, 1, 2)

        if self.concat:
            x = torch.cat([x, f], dim=-1)
        return x

class FeatMergeUnit(nn.Module):
    def __init__(self, in_channel, hidden_channel, out_channel):
        super(FeatMergeUnit, self).__init__()
        self.convs = nn.ModuleList()
        conv1 = nn.Sequential(*[
            nn.Conv1d(in_channel, hidden_channel, kernel_size=1),
            nn.BatchNorm1d(hidden_channel),
            nn.ReLU(inplace=True)
        ])
        self.convs.append(conv1)
        conv2 = nn.Sequential(*[
            nn.Conv1d(hidden_channel, out_channel, kernel_size=1),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(inplace=True)
        ])
        self.convs.append(conv2)
    def forward(self, x: Tensor):
        x = torch.transpose(x, 1, 2)
        for i in range(len(self.convs)):
            x = self.convs[i](x)
        x = torch.transpose(x, 1, 2)
        return x

