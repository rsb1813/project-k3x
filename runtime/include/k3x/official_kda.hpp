// 공식 KDA의 native BF16 가중치와 V-first portable state 경계를 정의합니다.
#pragma once

#include "k3x/backend.hpp"
#include "k3x/status.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace k3x {

struct OfficialKdaConfig {
    std::size_t hidden_size{};
    std::size_t heads{};
    std::size_t head_dim{};
    std::size_t conv_width{};
    float rms_norm_epsilon{};
    float gate_lower_bound{};
};

struct OfficialKdaWeightsView {
    Bf16WeightView q_proj;
    Bf16WeightView k_proj;
    Bf16WeightView v_proj;
    std::span<const float> q_conv;
    std::span<const float> k_conv;
    std::span<const float> v_conv;
    Bf16WeightView f_a_proj;
    Bf16WeightView f_b_proj;
    std::span<const float> a_log;
    std::span<const float> dt_bias;
    Bf16WeightView b_proj;
    Bf16WeightView g_proj;
    std::span<const float> o_norm;
    Bf16WeightView o_proj;
};

struct OfficialKdaState {
    std::vector<std::uint16_t> conv_q;
    std::vector<std::uint16_t> conv_k;
    std::vector<std::uint16_t> conv_v;
    std::vector<float> recurrent_v_first;
};

struct OfficialKdaBoundaries {
    std::vector<float> projected_q;
    std::vector<float> projected_k;
    std::vector<float> projected_v;
    std::vector<float> convolved_q;
    std::vector<float> convolved_k;
    std::vector<float> convolved_v;
    std::vector<float> q;
    std::vector<float> k;
    std::vector<float> v;
    std::vector<float> log_decay;
    std::vector<float> beta;
    std::vector<float> recurrent_output;
    std::vector<float> gated;
};

struct OfficialKdaResult {
    std::vector<float> output;
    OfficialKdaState state;
    OfficialKdaBoundaries boundaries;
};

OfficialKdaState zero_official_kda_state(const OfficialKdaConfig& config);

Result<OfficialKdaResult> official_kda_cpu(
    std::span<const float> hidden,
    const OfficialKdaWeightsView& weights,
    const OfficialKdaState& state,
    const OfficialKdaConfig& config);

}  // namespace k3x
