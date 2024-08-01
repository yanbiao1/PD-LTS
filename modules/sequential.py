
import torch
import torch.nn as nn

from torch import Tensor


# -----------------------------------------------------------------------------------------
class FlowSequential(nn.Module):

    def __init__(self, modules):
        super(FlowSequential, self).__init__()

        self.n_module = len(modules)
        self.flow_modules = nn.ModuleList(modules)
    
    def forward(self, x: Tensor, c: Tensor=None):
        log_det_J = torch.zeros((x.shape[0],), device=x.device)

        for i in range(self.n_module):
            x, _log_det_J = self.flow_modules[i](x, c)
            if _log_det_J is not None:
                log_det_J += _log_det_J
        
        return x, log_det_J
    
    def inverse(self, z: Tensor, c: Tensor=None):

        for i in reversed(range(self.n_module)):
            z = self.flow_modules[i].inverse(z, c)
        return z
# -----------------------------------------------------------------------------------------
