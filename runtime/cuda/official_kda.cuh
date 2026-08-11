// 공식 KDA의 convolution, decay, recurrence와 output gate CUDA launch 계약입니다.
#pragma once

#include <cuda_runtime_api.h>

#include <cstddef>
#include <cstdint>

namespace k3x::cuda {

cudaError_t launch_official_kda_short_conv(
    const float* projected, std::uint16_t* state, const float* weight,
    float* output, std::size_t sequence, std::size_t projection,
    std::size_t width, cudaStream_t stream);

cudaError_t launch_official_kda_normalize_qk(
    const float* convolved_q, const float* convolved_k, float* q, float* k,
    std::size_t sequence, std::size_t heads, std::size_t head_dim,
    cudaStream_t stream);

cudaError_t launch_official_kda_decay_beta(
    const float* forget, const float* beta_projection, const float* a_log,
    const float* dt_bias, float* log_decay, float* beta,
    std::size_t sequence, std::size_t heads, std::size_t head_dim,
    float lower_bound, cudaStream_t stream);

cudaError_t launch_official_kda_recurrence(
    const float* q, const float* k, const float* v,
    const float* log_decay, const float* beta, float* recurrent_v_first,
    float* recurrent_output, std::size_t sequence, std::size_t heads,
    std::size_t head_dim, cudaStream_t stream);

cudaError_t launch_official_kda_gate_norm(
    const float* recurrent_output, const float* output_gate,
    const float* output_norm, float* gated, std::size_t sequence,
    std::size_t heads, std::size_t head_dim, float epsilon,
    cudaStream_t stream);

}  // namespace k3x::cuda
