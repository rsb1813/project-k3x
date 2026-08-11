// resident MoE layer의 ordered mix, strict RMSNorm, 최종 합산 CUDA launcher를 선언합니다.
#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>
#include <span>

namespace k3x::cuda {

cudaError_t launch_ordered_expert_mix(
    const float* expert_outputs, const float* device_contributions,
    std::span<const float> host_contributions, float* mixed,
    std::size_t width, cudaStream_t stream);

cudaError_t launch_strict_rms_norm(
    const float* input, const float* weight, float* output,
    std::size_t width, float epsilon, cudaStream_t stream);

cudaError_t launch_vector_add(
    const float* left, const float* right, float* output,
    std::size_t width, cudaStream_t stream);

cudaError_t launch_bf16_matvec(
    const float* input, const std::uint16_t* weight, float* output,
    std::size_t rows, std::size_t cols, cudaStream_t stream);

cudaError_t launch_bf16_rms_norm(
    const float* input, const std::uint16_t* weight, float* output,
    std::size_t width, float epsilon, cudaStream_t stream);

cudaError_t launch_bf16_vector_add(
    const float* left, const float* right, float* output,
    std::size_t width, cudaStream_t stream);

cudaError_t launch_round_bf16_inplace(
    float* values, std::size_t count, cudaStream_t stream);

cudaError_t launch_ordered_expert_mix_bf16(
    const float* expert_outputs, const float* device_contributions,
    std::span<const float> host_contributions, float* mixed,
    std::size_t width, cudaStream_t stream);

}  // namespace k3x::cuda
