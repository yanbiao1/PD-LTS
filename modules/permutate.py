
import torch
import torch.nn as nn
import numpy as np

from torch import Tensor
from torch.nn import init


# -----------------------------------------------------------------------------------------
class Permutation(nn.Module):
    
    def __init__(self, permutation: str, n_channel: int, dim: int):
        super(Permutation, self).__init__()

        assert permutation in ['reverse', 'random', 'inv1x1']
        assert dim in [-1, 1, 2, 3]
        if permutation == 'inv1x1':
            self.permutater = InvertibleConv1x1_1D(n_channel, dim)
        else:
            if dim == -1: self.permutater = _ShufflePermutationXD(permutation, n_channel)
            if dim ==  1: self.permutater = _ShufflePermutation1D(permutation, n_channel)
            if dim ==  2: self.permutater = _ShufflePermutation2D(permutation, n_channel)
            if dim ==  3: self.permutater = _ShufflePermutation3D(permutation, n_channel)
    
    def forward(self, x: Tensor, _c: Tensor):
        return self.permutater(x)
    
    def inverse(self, z: Tensor, _c: Tensor):
        return self.permutater.inverse(z)


# -----------------------------------------------------------------------------------------
class _ShufflePermutation(nn.Module):

    def __init__(self, permutation: str, n_channel: int):
        super(_ShufflePermutation, self).__init__()

        if permutation == 'reverse':
            direct_idx = np.arange(n_channel - 1, -1, -1).astype(np.long)
            inverse_idx = _ShufflePermutation.get_reverse(direct_idx, n_channel)
        if permutation == 'random':
            direct_idx = np.arange(n_channel - 1, -1, -1).astype(np.long)
            np.random.shuffle(direct_idx)
            inverse_idx = _ShufflePermutation.get_reverse(direct_idx, n_channel)

        self.register_buffer('direct_idx', torch.from_numpy(direct_idx))
        self.register_buffer('inverse_idx', torch.from_numpy(inverse_idx))

    @staticmethod
    def get_reverse(idx, n_channel: int):
        indices_inverse = np.zeros((n_channel,), dtype=np.long)
        for i in range(n_channel):
            indices_inverse[idx[i]] = i
        return indices_inverse
    
    def forward(self, _x):
        raise NotImplementedError()

    def inverse(self, _z):
        raise NotImplementedError()

class _ShufflePermutation1D(_ShufflePermutation):

    def forward(self, x: Tensor):
        return x[:, self.direct_idx], None
    def inverse(self, z: Tensor):
        return z[:, self.inverse_idx]

class _ShufflePermutation2D(_ShufflePermutation):

    def forward(self, x: Tensor):
        return x[:, :, self.direct_idx], None
    def inverse(self, z: Tensor):
        return z[:, :, self.inverse_idx]

class _ShufflePermutation3D(_ShufflePermutation):

    def forward(self, x: Tensor):
        return x[:, :, :, self.direct_idx], None
    def inverse(self, z: Tensor):
        return z[:, :, :, self.inverse_idx]

class _ShufflePermutationXD(_ShufflePermutation):

    def forward(self, x: Tensor):
        return x[..., self.direct_idx], None
    def inverse(self, z: Tensor):
        return z[..., self.inverse_idx]



# -----------------------------------------------------------------------------------------
class InvertibleConv1x1_1D(nn.Module):

    def __init__(self, channel: int, dim: int):
        super(InvertibleConv1x1_1D, self).__init__()

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

    def forward(self, x: Tensor):
        z = torch.einsum(self.equation, self.W, x)
        logdet = torch.slogdet(self.W)[1] * x.shape[-1]   # slogdet[0]是符号，[1]是绝对值
        return z, logdet

    def inverse(self, z: Tensor, logpx=None):
        inv_W = torch.inverse(self.W)
        x = torch.einsum(self.equation, inv_W, z)
        return x
# -----------------------------------------------------------------------------------------
