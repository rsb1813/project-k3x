// 3-bit code와 BF16 group scale을 재패킹 없이 직접 누산합니다.
#include "quant3.cuh"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace k3x::cuda {
namespace {
__global__ void quant3_matvec_kernel(
    const float* input, const std::uint8_t* packed,
    const std::uint16_t* scales_bf16, float* output,
    std::size_t rows, std::size_t cols) {
    const auto row = static_cast<std::size_t>(blockIdx.x);
    if (row >= rows) return;
    float sum = 0.0F;
    for (std::size_t column = threadIdx.x; column < cols;
         column += blockDim.x) {
        const auto logical = row * cols + column;
        const auto group = logical / 32;
        const auto within = logical % 32;
        const auto offset = group * 12 + (within / 8) * 3;
        const auto word =
            static_cast<std::uint32_t>(packed[offset]) |
            (static_cast<std::uint32_t>(packed[offset + 1]) << 8U) |
            (static_cast<std::uint32_t>(packed[offset + 2]) << 16U);
        const auto code = (word >> ((within % 8) * 3U)) & 7U;
        __nv_bfloat16_raw raw{scales_bf16[group]};
        const auto scale = __bfloat162float(raw);
        sum += input[column] *
               static_cast<float>(static_cast<int>(code) - 3) * scale;
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
    if (threadIdx.x == 0) output[row] = partial[0];
}
}

cudaError_t launch_quant3_matvec(
    const float* input, const std::uint8_t* packed,
    const std::uint16_t* scales_bf16, float* output,
    std::size_t rows, std::size_t cols, cudaStream_t stream) {
    if (!input || !packed || !scales_bf16 || !output || !rows || !cols ||
        cols % 32 != 0) {
        return cudaErrorInvalidValue;
    }
    quant3_matvec_kernel<<<static_cast<unsigned>(rows), 256, 0, stream>>>(
        input, packed, scales_bf16, output, rows, cols);
    return cudaGetLastError();
}
}
