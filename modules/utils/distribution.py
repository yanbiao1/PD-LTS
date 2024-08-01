
import torch
import numpy as np

from torch import Tensor


# -----------------------------------------------------------------------------------------
class Distribution:
    def log_prob(self, x: Tensor):
        raise NotImplementedError()
    def sample(self, shape, device):
        raise NotImplementedError()

# -----------------------------------------------------------------------------------------
class GaussianDistribution(Distribution):

    def log_prob(self, x: Tensor, means=None, logs=None):
        if means is None:
            means = torch.zeros_like(x)
        if logs is None:
            logs = torch.zeros_like(x)
        sldj = -0.5 * ((x - means) ** 2 / (2 * logs).exp() + np.log(2 * np.pi) + 2 * logs)
        sldj = sldj.flatten(1).sum(-1)
        return sldj

    def sample(self, shape, device):
        return torch.randn(shape, device=device)
# -----------------------------------------------------------------------------------------
