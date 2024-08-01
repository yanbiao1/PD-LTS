import numpy as np
import torch
@torch.no_grad()
def Weight(pcl_noisy):
    """
    pcl_noisy: [B, N, 3]
    cent_pt  : [B, 1, 3]
    """
    B, N, C = pcl_noisy.shape
    d = pcl_noisy ** 2  # [B, N, 3]
    dist = torch.sum(d, dim=-1)          # [B, N]
    radius, _ = torch.max(dist, dim=1, keepdim=False)     # [B,]
    R = torch.div(radius, 4)     # [B,]
    _R = R.unsqueeze(1).expand_as(dist)                 # [B, N]
    w = torch.exp(- dist / _R)              # [B, N]
    divid = torch.sum(torch.exp(- dist / _R), dim=1).unsqueeze(1).expand_as(_R) # [B, N]
    weight = w / divid * N
    return weight