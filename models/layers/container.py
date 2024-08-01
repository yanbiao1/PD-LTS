import torch.nn as nn
import torch
import models.layers as layers
class SequentialFlow(nn.Module):
    """A generalized nn.Sequential container for normalizing flows.
    """

    def __init__(self, layersList):
        super(SequentialFlow, self).__init__()
        self.chain = nn.ModuleList(layersList)


    def forward(self, x, logpx=None):
        if logpx is None:
            for i in range(len(self.chain)):
                x = self.chain[i](x)
            return x
        else:
            for i in range(len(self.chain)):
                x, logpx = self.chain[i](x, logpx)
            return x, logpx

    def inverse(self, y, logpy=None):
        if logpy is None:
            for i in range(len(self.chain) - 1, -1, -1):
                y = self.chain[i].inverse(y)
            return y
        else:
            for i in range(len(self.chain) - 1, -1, -1):
                y, logpy = self.chain[i].inverse(y, logpy)
            return y, logpy

class Inverse(nn.Module):

    def __init__(self, flow):
        super(Inverse, self).__init__()
        self.flow = flow

    def forward(self, x, logpx=None):
        return self.flow.inverse(x, logpx)

    def inverse(self, y, logpy=None):
        return self.flow.forward(y, logpy)

class Cond_Absorb_Layer(nn.Module):
    def __init__(self, lipschitz_layer):
        super(Cond_Absorb_Layer, self).__init__()
        self.lipschitz_layer = lipschitz_layer

    def forward(self, x, c):
        x = self.lipschitz_layer.forward(x)
        x = x + c
        return x

    def build_clone(self):
        return Cond_Absorb_Layer(self.lipschitz_layer.build_clone())

    def build_jvp_net(self, x, c):
        with torch.no_grad():
            jvp_net, x = self.lipschitz_layer.build_jvp_net(x)
            x = x + c
            return jvp_net, x, c

class LipschitzWrapper(nn.Module):
    def __init__(self, nnet):
        super(LipschitzWrapper, self).__init__()

        self.nnet = nnet

    def forward(self, x, c):
        return self.nnet(x)

    def build_clone(self):
        return LipschitzWrapper(self.nnet.build_clone())

    def build_jvp_net(self, x, c):
        jvp_net, x = self.nnet.build_jvp_net(x)
        return jvp_net, x, c