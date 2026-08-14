// group-128 Q8 가중치를 복호화 버퍼 없이 직접 행렬-벡터 곱으로 계산합니다.
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

__global__ void q8_matvec_kernel(
    const float* input,
    const std::int8_t* codes,
    const std::uint16_t* scales_bf16,
    float* output,
    std::int64_t rows,
    std::int64_t columns) {
    const auto row = static_cast<std::int64_t>(blockIdx.x);
    if (row >= rows) {
        return;
    }
    float sum = 0.0F;
    const auto row_offset = row * columns;
    for (auto column = static_cast<std::int64_t>(threadIdx.x);
         column < columns;
         column += static_cast<std::int64_t>(blockDim.x)) {
        const auto logical = row_offset + column;
        const __nv_bfloat16_raw raw{scales_bf16[logical / 128]};
        sum = fmaf(
            static_cast<float>(codes[logical]) * __bfloat162float(raw),
            input[column],
            sum);
    }
    __shared__ float partial[256];
    partial[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride; stride >>= 1U) {
        if (threadIdx.x < stride) {
            partial[threadIdx.x] += partial[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        output[row] = partial[0];
    }
}

torch::Tensor q8_matvec(
    const torch::Tensor& input,
    const torch::Tensor& codes,
    const torch::Tensor& scales,
    std::int64_t rows,
    std::int64_t columns) {
    TORCH_CHECK(input.is_cuda() && codes.is_cuda() && scales.is_cuda());
    TORCH_CHECK(input.device() == codes.device() && input.device() == scales.device());
    TORCH_CHECK(input.scalar_type() == torch::kFloat32);
    TORCH_CHECK(codes.scalar_type() == torch::kInt8);
    TORCH_CHECK(scales.scalar_type() == torch::kBFloat16);
    TORCH_CHECK(input.is_contiguous() && codes.is_contiguous() && scales.is_contiguous());
    TORCH_CHECK(rows > 0 && columns > 0 && columns % 128 == 0);
    TORCH_CHECK(input.numel() == columns);
    TORCH_CHECK(codes.numel() == rows * columns);
    TORCH_CHECK(scales.numel() == rows * columns / 128);

    const c10::cuda::CUDAGuard guard(input.device());
    auto output = torch::empty({rows}, input.options());
    const auto stream = at::cuda::getCurrentCUDAStream(input.device().index());
    q8_matvec_kernel<<<static_cast<unsigned>(rows), 256, 0, stream>>>(
        input.data_ptr<float>(),
        codes.data_ptr<std::int8_t>(),
        reinterpret_cast<const std::uint16_t*>(scales.data_ptr()),
        output.data_ptr<float>(),
        rows,
        columns);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("q8_matvec", &q8_matvec, "K3X direct-packed Q8 matvec");
}
