// 공식 layer-1 KDA와 MoE를 합성하는 portable complete-layer 경계를 정의합니다.
#pragma once

#include "k3x/official_kda.hpp"
#include "k3x/official_moe.hpp"

#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace k3x {

struct OfficialLayerWeights {
    Bf16VectorView self_residual_norm;
    Bf16WeightView self_residual_proj;
    Bf16VectorView input_norm;
    OfficialKdaWeightsView kda;
    OfficialMoeWeights moe;
};

struct OfficialLayerInput {
    std::vector<float> hidden_input;
    std::vector<float> block_source;
};

struct OfficialLayerStepResult {
    std::vector<float> self_attention_residual;
    std::vector<float> input_normalized;
    std::vector<float> post_kda_prefix;
    std::vector<float> mlp_attention_residual;
    std::vector<float> normalized_moe_input;
    OfficialRoute route;
    OfficialMoeResult moe;
};

struct OfficialLayerResult {
    OfficialKdaResult kda;
    std::vector<OfficialLayerStepResult> steps;
};

struct OfficialLayerCudaWeights {
    Bf16VectorView self_residual_norm;
    Bf16WeightView self_residual_proj;
    Bf16VectorView input_norm;
    OfficialKdaCudaView kda;
    OfficialMoeWeights moe;
    OfficialMoeFfnView moe_ffn;
};

struct OfficialLayerCudaStepResult {
    std::vector<float> self_attention_residual;
    std::vector<float> input_normalized;
    std::vector<float> post_kda_prefix;
    std::vector<float> mlp_attention_residual;
    std::vector<float> normalized_moe_input;
    OfficialRoute route;
    std::vector<float> output;
};

struct OfficialLayerCudaResult {
    bool executed{};
    bool kda_state_published{};
    OfficialKdaDeviceStateToken kda_device_state;
    OfficialKdaState kda_state;
    std::vector<OfficialLayerCudaStepResult> steps;
};

enum class OfficialMoeRoutePreparationMode { host, device };

Result<OfficialLayerResult> official_layer_cpu(
    std::span<const OfficialLayerInput> inputs,
    const OfficialLayerWeights& weights,
    const OfficialKdaState& initial_state,
    const OfficialKdaConfig& config,
    std::size_t top_k,
    float situ_beta,
    std::optional<float> situ_linear_beta);

Result<OfficialLayerCudaResult> official_layer_cuda(
    ComputeBackend& backend,
    std::span<const OfficialLayerInput> inputs,
    const OfficialLayerCudaWeights& weights,
    const OfficialKdaState& initial_state,
    const OfficialKdaConfig& config,
    std::size_t top_k,
    float situ_beta,
    std::optional<float> situ_linear_beta,
    std::uint32_t layer,
    ProfilePhase phase,
    OfficialKdaStateControl state_control = {},
    OfficialMoeRoutePreparationMode route_preparation =
        OfficialMoeRoutePreparationMode::host);

}  // namespace k3x
