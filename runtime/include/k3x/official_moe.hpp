// 공식 Kimi K3 MoE FFN의 BF16/MXFP4 portable oracle 경계를 정의합니다.
#pragma once

#include "k3x/backend.hpp"
#include "k3x/status.hpp"

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace k3x {

struct OfficialExpertView {
    std::uint32_t expert_id{};
    Mxfp4MlpView weights;
};

struct OfficialMoeWeights {
    Bf16VectorView residual_norm;
    Bf16WeightView residual_proj;
    Bf16VectorView post_norm;
    Bf16WeightView router;
    std::span<const float> correction;
    Bf16WeightView routed_down;
    Bf16VectorView routed_norm;
    Bf16WeightView routed_up;
    Bf16MlpView shared;
    std::span<const OfficialExpertView> experts;
};

struct OfficialMoeInput {
    std::vector<float> prefix_sum;
    std::vector<float> block_residual;
};

struct OfficialRoute {
    std::vector<std::uint32_t> expert_ids;
    std::vector<float> contributions;
    std::vector<float> scores;
};

struct OfficialMoeResult {
    std::vector<float> hidden;
    std::vector<float> latent;
    std::vector<std::vector<float>> expert_outputs;
    std::vector<float> mixed_latent;
    std::vector<float> routed;
    std::vector<float> shared;
    std::vector<float> combined;
    std::vector<float> output;
};

float decode_bf16_word(std::uint16_t word) noexcept;

std::array<OfficialMoeInput, 2> official_moe_inputs();

Result<std::vector<float>> prepare_official_moe_input(
    const OfficialMoeInput& input,
    const OfficialMoeWeights& weights,
    float rms_norm_epsilon);

Result<OfficialRoute> route_official_moe(
    std::span<const float> hidden,
    Bf16WeightView router,
    std::span<const float> correction,
    std::size_t top_k);

Result<OfficialMoeResult> official_moe_cpu(
    const OfficialMoeInput& input,
    const OfficialMoeWeights& weights,
    const OfficialRoute& route,
    float rms_norm_epsilon,
    float situ_beta,
    std::optional<float> situ_linear_beta);

}  // namespace k3x
