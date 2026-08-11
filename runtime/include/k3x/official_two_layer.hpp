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

struct OfficialTwoLayerCudaWeights {
    std::uint32_t layer_id{};
    OfficialLayerCudaWeights weights;
};

enum class OfficialTwoLayerCudaMode { host_round_trip, device_closure };

struct OfficialTwoLayerCudaStepResult {
    std::size_t position{};
    std::uint32_t layer_id{};
    OfficialRoute route;
    std::vector<float> output;
};

struct OfficialTwoLayerCudaTelemetry {
    std::uint64_t weight_h2d_bytes{};
    std::uint64_t activation_h2d_bytes{};
    std::uint64_t device_to_host_bytes{};
    std::uint64_t state_h2d_bytes{};
    std::uint64_t state_d2h_bytes{};
    std::uint64_t kda_output_d2h_bytes{};
    std::uint64_t router_logit_d2h_bytes{};
    std::uint64_t inter_layer_hidden_h2d_bytes{};
    std::uint64_t inter_layer_hidden_d2h_bytes{};
    std::uint64_t final_hidden_d2h_bytes{};
    std::uint64_t layer_front_calls{};
    std::uint64_t layer_tail_calls{};
};

struct OfficialTwoLayerCudaResult {
    bool executed{};
    std::array<OfficialKdaState, 2> final_states;
    std::vector<OfficialTwoLayerCudaStepResult> steps;
    std::array<std::vector<float>, 2> outputs;
    OfficialTwoLayerCudaTelemetry telemetry;
};

Result<OfficialTwoLayerResult> official_two_layer_cpu(
    std::span<const OfficialLayerInput> inputs,
    std::span<const OfficialTwoLayerWeights> layers,
    std::span<const OfficialKdaState> initial_states,
    const OfficialKdaConfig& config,
    std::size_t top_k,
    float situ_beta,
    std::optional<float> situ_linear_beta);

Result<OfficialTwoLayerCudaResult> official_two_layer_cuda(
    ComputeBackend& backend,
    std::span<const OfficialLayerInput> inputs,
    std::span<const OfficialTwoLayerCudaWeights> layers,
    std::span<const OfficialKdaState> initial_states,
    const OfficialKdaConfig& config,
    std::size_t top_k,
    float situ_beta,
    std::optional<float> situ_linear_beta,
    ProfilePhase phase,
    OfficialTwoLayerCudaMode mode);

}  // namespace k3x
