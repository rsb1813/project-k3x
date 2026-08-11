// 공식 KDA의 BF16 경계와 FP32 V-first recurrence CUDA kernel을 구현합니다.
#include "official_kda.cuh"

#include <cuda_bf16.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace k3x::cuda {
namespace {

__device__ float bf16_round(float value) {
    return __bfloat162float(__float2bfloat16_rn(value));
}

__global__ void short_conv_kernel(
    const float* projected, __nv_bfloat16* state, const float* weight,
    float* output, std::size_t sequence, std::size_t projection,
    std::size_t width) {
    const auto channel = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                         threadIdx.x;
    if (channel >= projection) return;
    for (std::size_t token = 0; token < sequence; ++token) {
        float sum = 0.0F;
        for (std::size_t history = 0; history + 1 < width; ++history) {
            sum += __bfloat162float(state[history * projection + channel]) *
                   weight[channel * width + history];
        }
        sum += projected[token * projection + channel] *
               weight[channel * width + width - 1];
        output[token * projection + channel] =
            bf16_round(sum / (1.0F + expf(-sum)));
        for (std::size_t history = 0; history + 2 < width; ++history) {
            state[history * projection + channel] =
                state[(history + 1) * projection + channel];
        }
        state[(width - 2) * projection + channel] =
            __float2bfloat16_rn(projected[token * projection + channel]);
    }
}

__global__ void normalize_qk_kernel(
    const float* convolved_q, const float* convolved_k, float* q, float* k,
    std::size_t sequence, std::size_t heads, std::size_t head_dim) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    if (index >= sequence * heads) return;
    const auto base = index * head_dim;
    float q_squares = 0.0F;
    float k_squares = 0.0F;
    for (std::size_t channel = 0; channel < head_dim; ++channel) {
        q_squares += convolved_q[base + channel] * convolved_q[base + channel];
        k_squares += convolved_k[base + channel] * convolved_k[base + channel];
    }
    const auto q_inverse = rsqrtf(fmaxf(q_squares, 1.0e-24F)) /
                           sqrtf(static_cast<float>(head_dim));
    const auto k_inverse = rsqrtf(fmaxf(k_squares, 1.0e-24F));
    for (std::size_t channel = 0; channel < head_dim; ++channel) {
        q[base + channel] = bf16_round(convolved_q[base + channel] * q_inverse);
        k[base + channel] = bf16_round(convolved_k[base + channel] * k_inverse);
    }
}

__global__ void decay_beta_kernel(
    const float* forget, const float* beta_projection, const float* a_log,
    const float* dt_bias, float* log_decay, float* beta,
    std::size_t sequence, std::size_t heads, std::size_t head_dim,
    float lower_bound) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    const auto projection = heads * head_dim;
    if (index < sequence * projection) {
        const auto channel = index % head_dim;
        const auto projection_index = index % projection;
        const auto argument = expf(a_log[channel]) *
                              (forget[index] + dt_bias[projection_index]);
        log_decay[index] = lower_bound / (1.0F + expf(-argument));
    }
    if (index < sequence * heads) {
        beta[index] = 1.0F / (1.0F + expf(-beta_projection[index]));
    }
}

__global__ void recurrence_kernel(
    const float* q, const float* k, const float* v,
    const float* log_decay, const float* beta, float* state_v_first,
    float* output, std::size_t sequence, std::size_t head_dim) {
    const auto head = static_cast<std::size_t>(blockIdx.x);
    const auto value = static_cast<std::size_t>(threadIdx.x);
    if (value >= head_dim) return;
    const auto state_base = head * head_dim * head_dim;
    for (std::size_t token = 0; token < sequence; ++token) {
        const auto vector_base = (token * gridDim.x + head) * head_dim;
        float prediction = 0.0F;
        for (std::size_t key = 0; key < head_dim; ++key) {
            const auto state_index = state_base + value * head_dim + key;
            const auto decayed = expf(log_decay[vector_base + key]) *
                                 state_v_first[state_index];
            state_v_first[state_index] = decayed;
            prediction += k[vector_base + key] * decayed;
        }
        const auto delta = (v[vector_base + value] - prediction) *
                           beta[token * gridDim.x + head];
        for (std::size_t key = 0; key < head_dim; ++key) {
            state_v_first[state_base + value * head_dim + key] +=
                k[vector_base + key] * delta;
        }
        float result = 0.0F;
        for (std::size_t key = 0; key < head_dim; ++key) {
            result += q[vector_base + key] *
                      state_v_first[state_base + value * head_dim + key];
        }
        output[vector_base + value] = result;
    }
}

__global__ void gate_norm_kernel(
    const float* recurrent, const float* gate, const float* norm,
    float* output, std::size_t sequence, std::size_t heads,
    std::size_t head_dim, float epsilon) {
    const auto index = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                       threadIdx.x;
    if (index >= sequence * heads) return;
    const auto base = index * head_dim;
    float squares = 0.0F;
    for (std::size_t value = 0; value < head_dim; ++value) {
        squares += recurrent[base + value] * recurrent[base + value];
    }
    const auto inverse = rsqrtf(squares / static_cast<float>(head_dim) + epsilon);
    for (std::size_t value = 0; value < head_dim; ++value) {
        const auto sigmoid = 1.0F / (1.0F + expf(-gate[base + value]));
        output[base + value] = bf16_round(
            recurrent[base + value] * inverse * norm[value] * sigmoid);
    }
}

}  // namespace

cudaError_t launch_official_kda_short_conv(
    const float* projected, std::uint16_t* state, const float* weight,
    float* output, std::size_t sequence, std::size_t projection,
    std::size_t width, cudaStream_t stream) {
    if (!projected || !state || !weight || !output || !sequence ||
        !projection || width < 2) return cudaErrorInvalidValue;
    constexpr unsigned threads = 256;
    short_conv_kernel<<<(projection + threads - 1) / threads, threads, 0, stream>>>(
        projected, reinterpret_cast<__nv_bfloat16*>(state), weight, output,
        sequence, projection, width);
    return cudaGetLastError();
}

cudaError_t launch_official_kda_normalize_qk(
    const float* convolved_q, const float* convolved_k, float* q, float* k,
    std::size_t sequence, std::size_t heads, std::size_t head_dim,
    cudaStream_t stream) {
    if (!convolved_q || !convolved_k || !q || !k || !sequence || !heads ||
        !head_dim) return cudaErrorInvalidValue;
    constexpr unsigned threads = 256;
    const auto count = sequence * heads;
    normalize_qk_kernel<<<(count + threads - 1) / threads, threads, 0, stream>>>(
        convolved_q, convolved_k, q, k, sequence, heads, head_dim);
    return cudaGetLastError();
}

cudaError_t launch_official_kda_decay_beta(
    const float* forget, const float* beta_projection, const float* a_log,
    const float* dt_bias, float* log_decay, float* beta,
    std::size_t sequence, std::size_t heads, std::size_t head_dim,
    float lower_bound, cudaStream_t stream) {
    if (!forget || !beta_projection || !a_log || !dt_bias || !log_decay ||
        !beta || !sequence || !heads || !head_dim || !std::isfinite(lower_bound) ||
        lower_bound >= 0.0F) return cudaErrorInvalidValue;
    constexpr unsigned threads = 256;
    const auto count = sequence * heads * head_dim;
    decay_beta_kernel<<<(count + threads - 1) / threads, threads, 0, stream>>>(
        forget, beta_projection, a_log, dt_bias, log_decay, beta,
        sequence, heads, head_dim, lower_bound);
    return cudaGetLastError();
}

cudaError_t launch_official_kda_recurrence(
    const float* q, const float* k, const float* v,
    const float* log_decay, const float* beta, float* recurrent_v_first,
    float* recurrent_output, std::size_t sequence, std::size_t heads,
    std::size_t head_dim, cudaStream_t stream) {
    if (!q || !k || !v || !log_decay || !beta || !recurrent_v_first ||
        !recurrent_output || !sequence || !heads || !head_dim ||
        head_dim > 1024) return cudaErrorInvalidValue;
    recurrence_kernel<<<heads, static_cast<unsigned>(head_dim), 0, stream>>>(
        q, k, v, log_decay, beta, recurrent_v_first, recurrent_output,
        sequence, head_dim);
    return cudaGetLastError();
}

cudaError_t launch_official_kda_gate_norm(
    const float* recurrent_output, const float* output_gate,
    const float* output_norm, float* gated, std::size_t sequence,
    std::size_t heads, std::size_t head_dim, float epsilon,
    cudaStream_t stream) {
    if (!recurrent_output || !output_gate || !output_norm || !gated ||
        !sequence || !heads || !head_dim || !std::isfinite(epsilon) ||
        epsilon <= 0.0F) return cudaErrorInvalidValue;
    constexpr unsigned threads = 256;
    const auto count = sequence * heads;
    gate_norm_kernel<<<(count + threads - 1) / threads, threads, 0, stream>>>(
        recurrent_output, output_gate, output_norm, gated,
        sequence, heads, head_dim, epsilon);
    return cudaGetLastError();
}

}  // namespace k3x::cuda
