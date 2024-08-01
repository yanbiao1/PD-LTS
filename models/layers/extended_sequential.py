import torch
import torch.nn as nn


class ExtendedSequential(nn.Sequential):
    def build_clone(self):
        modules = []
        for m in self:
            modules.append(m.build_clone())
        return ExtendedSequential(*modules)

    def build_jvp_net(self, *args):
        with torch.no_grad():
            modules = []
            y = args
            for m in self:
                jvp_net_and_y = m.build_jvp_net(*y)
                jvp_net = jvp_net_and_y[0]
                y = jvp_net_and_y[1:]
                modules.append(jvp_net)
            return ExtendedSequential(*modules), *y