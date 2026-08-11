// 공식 두 레이어 portable 실행의 상태와 결과 계약을 정의합니다.
#pragma once

#include "k3x/official_layer.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace k3x {

struct OfficialTwoLayerWeights {
    std::uint32_t layer_id{};
    OfficialLayerWeights weights;
};

struct OfficialTwoLayerStepResult {
    std::size_t position{};
    std::uint32_t layer_id{};
    OfficialLayerStepResult result;
};

struct OfficialTwoLayerResult {
    std::array<OfficialKdaState, 2> final_states;
    std::vector<OfficialTwoLayerStepResult> steps;
    std::array<std::vector<float>, 2> outputs;
};

Result<OfficialTwoLayerResult> official_two_layer_cpu(
    std::span<const OfficialLayerInput> inputs,
    std::span<const OfficialTwoLayerWeights> layers,
    std::span<const OfficialKdaState> initial_states,
    const OfficialKdaConfig& config,
    std::size_t top_k,
    float situ_beta,
    std::optional<float> situ_linear_beta);

}  // namespace k3x
