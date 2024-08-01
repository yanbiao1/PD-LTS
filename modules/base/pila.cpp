#include <torch/extension.h>

#define CONST(c, device_tensor) (torch::tensor(c, torch::dtype(torch::kFloat32).device(device_tensor.device())))


torch::Tensor pila_cpu_forward(torch::Tensor x, torch::Tensor kabcdmn) {
	const auto k = kabcdmn[0];
	const auto a = kabcdmn[1];
	const auto b = kabcdmn[2];
	const auto c = kabcdmn[3];
	const auto d = kabcdmn[4];
	const auto m = kabcdmn[5];
	const auto n = kabcdmn[6];

    auto x2 = x*x;
    auto x3 = x2*x;

    auto p = torch::min(k*x, CONST(0.01, x));
    auto q = torch::exp(p);
    auto r = a*x3 + b*x2 + c*x + d;

	return torch::where(x > 0, m*x+n, r*q);
}

torch::Tensor pila_cuda_forward(
    torch::Tensor x,
	torch::Tensor kabcdmn);

// NOTE: AT_ASSERT has become AT_CHECK on master after 0.4.
#define CHECK_CUDA(x) AT_ASSERTM(x.type().is_cuda(), #x " must be a CUDA tensor")
#define CHECK_CONTIGUOUS(x) AT_ASSERTM(x.is_contiguous(), #x " must be contiguous")
#define CHECK_INPUT(x) CHECK_CUDA(x); CHECK_CONTIGUOUS(x)

torch::Tensor pila_forward(
	torch::Tensor x,
	torch::Tensor kabcdmn) {
	TORCH_CHECK(kabcdmn.dim() == 1 && kabcdmn.size(0) == 7,
		"kabcdmn has wrong dim/size; it must be 1-dimensional 7-element tensor, but got dim size(0)",
		kabcdmn.dim(), kabcdmn.size(0))
	switch (x.device().type()) {
	case c10::kCUDA:
		CHECK_INPUT(x);
		CHECK_INPUT(kabcdmn);
		return pila_cuda_forward(x, kabcdmn);
	case c10::kCPU:
		return pila_cpu_forward(x, kabcdmn);
    default:
		TORCH_CHECK(false, "Unsupported device type, should be CPU or CUDA but got ", x.device().type());
	}
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward", &pila_forward, "Pila forward");
}