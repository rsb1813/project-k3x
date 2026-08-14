// native MXFP4 바이트를 복호화 버퍼 없이 직접 행렬-벡터 곱으로 계산합니다.
#include <torch/extension.h>

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAGuard.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace {

__device__ __forceinline__ float decode_e2m1(std::uint8_t code) {
    float magnitude = 0.0F;
    switch (code & 0x07U) {
        case 1: magnitude = 0.5F; break;
        case 2: magnitude = 1.0F; break;
        case 3: magnitude = 1.5F; break;
        case 4: magnitude = 2.0F; break;
        case 5: magnitude = 3.0F; break;
        case 6: magnitude = 4.0F; break;
        case 7: magnitude = 6.0F; break;
        default: break;
    }
    return (code & 0x08U) != 0 ? -magnitude : magnitude;
}

__global__ void mxfp4_matvec_kernel(
    const float* input,
    const std::uint8_t* packed,
    const std::uint8_t* scales,
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
        const auto pair = packed[logical / 2];
        const auto code = static_cast<std::uint8_t>(
            logical % 2 == 0 ? pair & 0x0FU : pair >> 4U);
        const auto exponent = static_cast<int>(scales[logical / 32]) - 127;
        sum = fmaf(input[column], ldexpf(decode_e2m1(code), exponent), sum);
    }
    extern __shared__ float partial[];
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

torch::Tensor mxfp4_matvec(
    const torch::Tensor& input,
    const torch::Tensor& packed,
    const torch::Tensor& scales,
    std::int64_t rows,
    std::int64_t columns) {
    TORCH_CHECK(input.is_cuda() && packed.is_cuda() && scales.is_cuda());
    TORCH_CHECK(input.device() == packed.device() && input.device() == scales.device());
    TORCH_CHECK(input.scalar_type() == torch::kFloat32);
    TORCH_CHECK(packed.scalar_type() == torch::kUInt8);
    TORCH_CHECK(scales.scalar_type() == torch::kUInt8);
    TORCH_CHECK(input.is_contiguous() && packed.is_contiguous() && scales.is_contiguous());
    TORCH_CHECK(rows > 0 && columns > 0 && columns % 32 == 0);
    TORCH_CHECK(input.numel() == columns);
    TORCH_CHECK(packed.numel() == rows * columns / 2);
    TORCH_CHECK(scales.numel() == rows * columns / 32);

    const c10::cuda::CUDAGuard guard(input.device());
    auto output = torch::empty({rows}, input.options());
    const auto stream = at::cuda::getCurrentCUDAStream(input.device().index());
    constexpr unsigned threads = 256;
    mxfp4_matvec_kernel<<<static_cast<unsigned>(rows), threads,
                           threads * sizeof(float), stream>>>(
        input.data_ptr<float>(),
        packed.data_ptr<std::uint8_t>(),
        scales.data_ptr<std::uint8_t>(),
        output.data_ptr<float>(),
        rows,
        columns);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return output;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("mxfp4_matvec", &mxfp4_matvec, "K3X native MXFP4 matvec");
}
