import torch
import torch.nn as nn


class InverseLayer(nn.Module):
    def __init__(self, nnet):
        super(InverseLayer, self).__init__()
        self.nnet = nnet

    def forward(self, x, logpx=None):
        return self.nnet.inverse(x, logpx)
    def inverse(self, y, logpy=None):
        return self.nnet.forward(y, logpy)

    def build_clone(self):
        return InverseLayer(self.nnet.build_clone())

    def build_jvp_net(self, x):
        raise NotImplementedError()