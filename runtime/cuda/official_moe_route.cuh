// 공식 MoE residual 준비와 raw router logit CUDA launcher를 선언합니다.
#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace k3x::cuda {

cudaError_t launch_official_moe_prepare(
    const float* prefix, const float* block,
    const std::uint16_t* residual_norm,
    const std::uint16_t* residual_projection,
    const std::uint16_t* post_norm,
    float* prepared_prefix, float* prepared_hidden,
    std::size_t width, float epsilon, cudaStream_t stream);

cudaError_t launch_official_moe_router_logits(
    const float* prepared_hidden, const std::uint16_t* router,
    float* logits, std::size_t rows, std::size_t cols,
    cudaStream_t stream);

}  // namespace k3x::cuda
