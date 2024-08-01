import torch
import torch.nn as nn
import torch.nn.functional as F
import copy


class MyReLU(nn.Module):
    def __init__(self, inplace=False):
        super(MyReLU, self).__init__()

        self.relu = nn.ReLU(inplace=inplace)

    def forward(self, x):
        return self.relu(x)

    def build_clone(self):
        return copy.deepcopy(self)


class CReLU(nn.Module):
    def __init__(self):
        super(CReLU, self).__init__()

    def forward(self, x):
        return torch.cat([F.relu(x), F.relu(-x)], dim=1)

    def build_clone(self):
        return copy.deepcopy(self)


class Identity(nn.Module):

    def forward(self, x):
        return x


class FullSort(nn.Module):

    def forward(self, x):
        return torch.sort(x, 1)[0]


class MaxMin(nn.Module):

    def forward(self, x):
        b, d = x.shape
        max_vals = torch.max(x.view(b, d // 2, 2), 2)[0]
        min_vals = torch.min(x.view(b, d // 2, 2), 2)[0]
        return torch.cat([max_vals, min_vals], 1)


class LipschitzCube(nn.Module):

    def forward(self, x):
        return (x >= 1).to(x) * (x - 2 / 3) + (x <= -1).to(x) * (x + 2 / 3) + ((x > -1) * (x < 1)).to(x) * x**3 / 3


class SwishFn(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, beta):
        beta_sigm = torch.sigmoid(beta * x)
        output = x * beta_sigm
        ctx.save_for_backward(x, output, beta)
        return output / 1.1

    @staticmethod
    def backward(ctx, grad_output):
        x, output, beta = ctx.saved_tensors
        beta_sigm = output / x
        grad_x = grad_output * (beta * output + beta_sigm * (1 - beta * output))
        grad_beta = torch.sum(grad_output * (x * output - output * output)).expand_as(beta)
        return grad_x / 1.1, grad_beta / 1.1


class Swish(nn.Module):

    def __init__(self, device=None):
        super(Swish, self).__init__()
        self.beta = nn.Parameter(torch.tensor([0.5], device=device))

    def forward(self, x):
        return (x * torch.sigmoid_(x * F.softplus(self.beta))).div_(1.1)

    def grad(self, x):
        bx = x * F.softplus(self.beta)
        sbx = torch.sigmoid(bx)
        return (sbx + bx * sbx * (1-sbx)).div_(1.1)

    def build_clone(self):
        return copy.deepcopy(self) # 将a的地址和值一并复制过来，与a没有半点关系

    def build_jvp_net(self, x):
        class LipSwishJVP(nn.Module):
            def __init__(self, grad):
                super(LipSwishJVP, self).__init__()
                self.register_buffer('grad', grad)

            def forward(self, x):
                return x * self.grad

        with torch.no_grad():
            y = self.forward(x)

            grad = self.grad(x)
            grad.div_(1.004)
            return LipSwishJVP(grad), y

class LeakyLSwish(nn.Module):

    def __init__(self):
        super(LeakyLSwish, self).__init__()
        self.alpha = nn.Parameter(torch.tensor([-3.]))
        self.beta = nn.Parameter(torch.tensor([0.5]))

    def forward(self, x):
        alpha = torch.sigmoid(self.alpha)
        return alpha * x + (1 - alpha) * (x * torch.sigmoid_(x * F.softplus(self.beta))).div_(1.1)


class CLipSwish(nn.Module):

    def __init__(self, device=None):
        super(CLipSwish, self).__init__()
        self.swish = Swish(device=device)

    def forward(self, x):
        x = torch.cat((x, -x), 2)
        return self.swish(x) / 1.004

    def build_clone(self):
        return copy.deepcopy(self) # 将a的地址和值一并复制过来，与a没有半点关系

    def build_jvp_net(self, x):
        class CLipSwishJVP(nn.Module):
            def __init__(self, grad):
                super(CLipSwishJVP, self).__init__()
                self.register_buffer('grad', grad)

            def forward(self, x):
                return torch.cat((x,x), dim=2) * self.grad

        with torch.no_grad():
            y = self.forward(x)

            grad = torch.cat((self.swish.grad(x), -self.swish.grad(-x)), dim=2)
            grad.div_(1.004)
            return CLipSwishJVP(grad), y


class ALCLipSiLU(nn.Module):

    def __init__(self):
        super(ALCLipSiLU, self).__init__()
        self.beta = nn.Parameter(torch.tensor([0.0]))

    def forward(self, x):
        from torch import tanh
        beta = 1.0 + 0.9 * torch.tanh(self.beta)
        x = beta*x
        r = 0.997*x + 0.147020*tanh(1.971963*x) + (0.142953)*(tanh(-0.501533*(x-1.106198))+tanh(-0.501533*(x+1.106198))) + (0.152010)*(tanh(0.904555*(x-0.189486))+tanh(0.904555*(x+0.189486))) + (-0.119371)*(tanh(0.414999*(x-2.570108))+tanh(0.414999*(x+2.570108)))
        y = torch.cat((r,-r), 1)
        return F.silu(y)/beta

if __name__ == '__main__':

    m = Swish()
    xx = torch.linspace(-5, 5, 1000).requires_grad_(True)
    yy = m(xx)
    dd, dbeta = torch.autograd.grad(yy.sum() * 2, [xx, m.beta])

    import matplotlib.pyplot as plt

    plt.plot(xx.detach().numpy(), yy.detach().numpy(), label='Func')
    plt.plot(xx.detach().numpy(), dd.detach().numpy(), label='Deriv')
    plt.plot(xx.detach().numpy(), torch.max(dd.detach().abs() - 1, torch.zeros_like(dd)).numpy(), label='|Deriv| > 1')
    plt.legend()
    plt.tight_layout()
    plt.show()
