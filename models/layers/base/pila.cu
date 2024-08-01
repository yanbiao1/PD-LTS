#include <torch/extension.h>

#include <cuda.h>
#include <cuda_runtime.h>

#include <vector>


namespace kernel {

template <typename scalar_t>
__device__ __forceinline__ scalar_t pila(scalar_t z, scalar_t k, scalar_t a, scalar_t b, scalar_t c, scalar_t d, scalar_t m, scalar_t n) {
const auto q = exp(k*z);
const auto r = a*z*z*z + b*z*z + c*z + d;

return (z>0) ? (m*z+n) : (r*q);
}

template <typename scalar_t>
__global__ void pila_cuda_forward_kernel(
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> x,
    const torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> kabcdmn,
    torch::PackedTensorAccessor32<scalar_t,1,torch::RestrictPtrTraits> output
    ) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.x;
    const int numel = x.size(0);
    for (int i = index; i < numel; i += stride) {
        output[i] = pila(x[i], kabcdmn[0], kabcdmn[1], kabcdmn[2], kabcdmn[3], kabcdmn[4], kabcdmn[5], kabcdmn[6]);
    }
}

} // namespace kernel



torch::Tensor pila_cuda_forward(
    torch::Tensor x,
    torch::Tensor kabcdmn) {

    auto x_1d = x.view(-1);
    auto kabcdmn_1d = kabcdmn.view(-1);

    const int numel = x.numel();
    const int threads = 1024;
    const int blocks = (numel + threads - 1) / threads;

    auto output_1d = torch::zeros_like(x_1d);

    AT_DISPATCH_FLOATING_TYPES(x.type(), "pila_cuda_forward", ([&] {
        kernel::pila_cuda_forward_kernel<scalar_t><<<blocks, threads>>>(
            x_1d.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            kabcdmn_1d.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>(),
            output_1d.packed_accessor32<scalar_t,1,torch::RestrictPtrTraits>()
        );
    }));

    return output_1d.view_as(x);
}
