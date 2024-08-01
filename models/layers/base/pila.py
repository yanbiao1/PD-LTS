import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.cpp_extension as cpp_extension
import copy

pila_cpp = cpp_extension.load(name='pila_cpp', sources=['models/layers/base/pila.cpp', 'models/layers/base/pila.cu'],
    extra_cuda_cflags=['-allow-unsupported-compiler'], verbose=True)


'''
By default, forward() and backward() of torch.autograd.Function
   is supplied with torch.no_grad() context.
   Hence, no need to worry about that inefficiency.
'''
class PilaFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, kabcdmn):
        y = pila_cpp.forward(x, kabcdmn)
        ctx.save_for_backward(x, kabcdmn)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, kabcdmn = ctx.saved_tensors
        k, a, b, c, d, m, n = kabcdmn
        kabcdmn_new = torch.stack([k, k*a, k*b+3*a, k*c+2*b, k*d+c, 0*k, m])
        return grad_output * PilaFunction.apply(x, kabcdmn_new), None


class Pila(nn.Module):
    def __init__(self, k=5, device=None):
        super(Pila, self).__init__()
        assert k > 0
        self.k = k

    def forward(self, x):
        k = self.k
        kabcdmn = torch.tensor((k,k**2/2,-k,1,0,1,0), dtype=x.dtype, device=x.device)
        return PilaFunction.apply(x, kabcdmn)




class CPila(nn.Module):
    def __init__(self, k=5, device=None):
        super(CPila, self).__init__()
        assert k > 0
        self.k = k
        self.pila = Pila(k=k, device=device)

    def forward(self, x):
        x = torch.cat((x-0.2, -x-0.2), 2)
        return self.pila(x) / 1.06

    def build_clone(self):
        return copy.deepcopy(self)

    def build_jvp_net(self, x):
        class CPilaJVP(nn.Module):
            def __init__(self, grad):
                super(CPilaJVP, self).__init__()
                self.register_buffer('grad', grad)

            def forward(self, x):
                return torch.cat((x,x), dim=2) * self.grad

        with torch.no_grad():
            y = self.forward(x)

            k = self.k
            a, b, c, d, m, n = k**2/2,-k,1,0,1,0
            kabcdmn = torch.tensor((k, k*a, k*b+3*a, k*c+2*b, k*d+c, 0*k, m), dtype=x.dtype, device=x.device)
            grad = torch.cat((pila_cpp.forward(x-0.2, kabcdmn), -pila_cpp.forward(-x-0.2, kabcdmn)), dim=2)
            grad.div_(1.06)

            return CPilaJVP(grad), y


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    x = torch.linspace(-5, +5, 500, dtype=torch.double).requires_grad_(True).cuda()
    #x = torch.randn(500, dtype=torch.double).requires_grad_(True).cuda()
    #x = x.view(5, 4, 25)

    y = Pila()(x)

    grad_x, = torch.autograd.grad(y.sum(), x, create_graph=True)
    import torchviz
    torchviz.make_dot((grad_x, x, y), params={"grad_x": grad_x, "x": x, "y": y}).render("torchviz", format="png")

    k = 5
    kabcdmn = torch.tensor((k,k**2/2,-k,1,0,1,0), dtype=torch.double, device=x.device)
    if torch.autograd.gradcheck(PilaFunction.apply, (x, kabcdmn)):
        print('Pila passed torch.autograd.gradcheck')
    if torch.autograd.gradgradcheck(PilaFunction.apply, (x, kabcdmn)):
        print('Pila passed torch.autograd.gradgradcheck')

    x = x.view(-1)
    y = y.view(-1)
    x_np = x.detach().cpu().numpy()
    y_np = y.detach().cpu().numpy()
    plt.scatter(x_np,y_np)
    plt.title('Pila with k=5')
    plt.show()