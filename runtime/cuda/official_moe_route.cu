// 공식 MoE residual 준비와 raw router logit을 결정적으로 계산합니다.
#include "official_moe_route.cuh"

#include <cuda_bf16.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace k3x::cuda {
namespace {

__device__ float bf16_round(float value) {
    return __bfloat162float(__float2bfloat16_rn(value));
}

__global__ void official_moe_prepare_kernel(
    const float* prefix, const float* block,
    const __nv_bfloat16* residual_norm,
    const __nv_bfloat16* residual_projection,
    const __nv_bfloat16* post_norm,
    float* prepared_prefix, float* prepared_hidden,
    std::size_t width, float epsilon) {
    if (blockIdx.x || threadIdx.x) return;
    double prefix_squares = 0.0;
    double block_squares = 0.0;
    for (std::size_t index = 0; index < width; ++index) {
        prepared_prefix[index] = bf16_round(prefix[index]);
        prepared_hidden[index] = bf16_round(block[index]);
        prefix_squares += static_cast<double>(prepared_prefix[index]) *
                          prepared_prefix[index];
        block_squares += static_cast<double>(prepared_hidden[index]) *
                         prepared_hidden[index];
    }
    const auto prefix_inverse = 1.0F / sqrtf(
        static_cast<float>(prefix_squares / width) + epsilon);
    const auto block_inverse = 1.0F / sqrtf(
        static_cast<float>(block_squares / width) + epsilon);
    double prefix_score = 0.0;
    double block_score = 0.0;
    for (std::size_t index = 0; index < width; ++index) {
        const auto scale = __bfloat162float(residual_norm[index]) *
                           __bfloat162float(residual_projection[index]);
        prefix_score += static_cast<double>(
            prepared_prefix[index] * prefix_inverse) * scale;
        block_score += static_cast<double>(
            prepared_hidden[index] * block_inverse) * scale;
    }
    const auto prefix_value = static_cast<float>(prefix_score);
    const auto block_value = static_cast<float>(block_score);
    const auto maximum = fmaxf(prefix_value, block_value);
    const auto prefix_exp = expf(prefix_value - maximum);
    const auto block_exp = expf(block_value - maximum);
    const auto denominator = prefix_exp + block_exp;
    const auto prefix_probability = prefix_exp / denominator;
    const auto block_probability = block_exp / denominator;
    double hidden_squares = 0.0;
    for (std::size_t index = 0; index < width; ++index) {
        prepared_hidden[index] = bf16_round(
            block_probability * prepared_hidden[index] +
            prefix_probability * prepared_prefix[index]);
        hidden_squares += static_cast<double>(prepared_hidden[index]) *
                          prepared_hidden[index];
    }
    const auto hidden_inverse = 1.0F / sqrtf(
        static_cast<float>(hidden_squares / width) + epsilon);
    for (std::size_t index = 0; index < width; ++index) {
        prepared_hidden[index] = bf16_round(
            prepared_hidden[index] * hidden_inverse *
            __bfloat162float(post_norm[index]));
    }
}

__global__ void official_moe_router_logits_kernel(
    const float* hidden, const __nv_bfloat16* router,
    float* logits, std::size_t rows, std::size_t cols) {
    const auto row = static_cast<std::size_t>(blockIdx.x) * blockDim.x +
                     threadIdx.x;
    if (row >= rows) return;
    double sum = 0.0;
    for (std::size_t column = 0; column < cols; ++column) {
        sum += static_cast<double>(
                   __bfloat162float(router[row * cols + column])) *
               hidden[column];
    }
    logits[row] = static_cast<float>(sum);
}

}  // namespace

cudaError_t launch_official_moe_prepare(
    const float* prefix, const float* block,
    const std::uint16_t* residual_norm,
    const std::uint16_t* residual_projection,
    const std::uint16_t* post_norm,
    float* prepared_prefix, float* prepared_hidden,
    std::size_t width, float epsilon, cudaStream_t stream) {
    if (!prefix || !block || !residual_norm || !residual_projection ||
        !post_norm || !prepared_prefix || !prepared_hidden || !width ||
        !std::isfinite(epsilon) || epsilon <= 0.0F) {
        return cudaErrorInvalidValue;
    }
    official_moe_prepare_kernel<<<1, 1, 0, stream>>>(
        prefix, block,
        reinterpret_cast<const __nv_bfloat16*>(residual_norm),
        reinterpret_cast<const __nv_bfloat16*>(residual_projection),
        reinterpret_cast<const __nv_bfloat16*>(post_norm),
        prepared_prefix, prepared_hidden, width, epsilon);
    return cudaGetLastError();
}

cudaError_t launch_official_moe_router_logits(
    const float* prepared_hidden, const std::uint16_t* router,
    float* logits, std::size_t rows, std::size_t cols,
    cudaStream_t stream) {
    if (!prepared_hidden || !router || !logits || !rows || !cols) {
        return cudaErrorInvalidValue;
    }
    constexpr unsigned threads = 256;
    const auto blocks = static_cast<unsigned>((rows + threads - 1) / threads);
    official_moe_router_logits_kernel<<<blocks, threads, 0, stream>>>(
        prepared_hidden,
        reinterpret_cast<const __nv_bfloat16*>(router),
        logits, rows, cols);
    return cudaGetLastError();
}

}  // namespace k3x::cuda
