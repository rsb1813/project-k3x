// group-wise signed 3-bit packed matvec CUDA 실행 계약을 정의합니다.
#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace k3x::cuda {
cudaError_t launch_quant3_matvec(
    const float* input, const std::uint8_t* packed,
    const std::uint16_t* scales_bf16, float* output,
    std::size_t rows, std::size_t cols, cudaStream_t stream);
}
