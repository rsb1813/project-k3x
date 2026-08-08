// K3 E2M1 값과 E8M0/32 scale을 재패킹 없이 decode하고 FP32로 누산합니다.
#include "mxfp4.cuh"

#include <cuda_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace k3x::cuda {
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
    const float* input, const std::uint8_t* packed,
    const std::uint8_t* scales, float* output,
    std::size_t rows, std::size_t cols) {
    const auto row = static_cast<std::size_t>(blockIdx.x);
    if (row >= rows) return;

    float sum = 0.0F;
    for (std::size_t column = threadIdx.x; column < cols;
         column += blockDim.x) {
        const auto index = row * cols + column;
        const auto pair = packed[index / 2];
        const auto code = static_cast<std::uint8_t>(
            index % 2 == 0 ? pair & 0x0FU : pair >> 4U);
        const auto exponent = static_cast<int>(scales[index / 32]) - 127;
        const auto weight = ldexpf(decode_e2m1(code), exponent);
        sum = fmaf(input[column], weight, sum);
    }

    extern __shared__ float partials[];
    partials[threadIdx.x] = sum;
    __syncthreads();
    for (unsigned stride = blockDim.x / 2; stride != 0; stride /= 2) {
        if (threadIdx.x < stride) {
            partials[threadIdx.x] += partials[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) output[row] = partials[0];
}

}  // namespace

cudaError_t launch_mxfp4_matvec(
    const float* input, const std::uint8_t* packed,
    const std::uint8_t* scales, float* output,
    std::size_t rows, std::size_t cols, cudaStream_t stream) {
    constexpr unsigned threads = 256;
    mxfp4_matvec_kernel<<<static_cast<unsigned>(rows), threads,
                           threads * sizeof(float), stream>>>(
        input, packed, scales, output, rows, cols);
    return cudaGetLastError();
}

}  // namespace k3x::cuda
