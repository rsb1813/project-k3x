// native K3 MXFP4 바이트를 직접 계산하는 CUDA kernel launch 계약을 정의합니다.
#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace k3x::cuda {

cudaError_t launch_mxfp4_matvec(
    const float* input, const std::uint8_t* packed,
    const std::uint8_t* scales, float* output,
    std::size_t rows, std::size_t cols, cudaStream_t stream);

}  // namespace k3x::cuda
