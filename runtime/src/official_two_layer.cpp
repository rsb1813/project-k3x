// 공식 두 레이어를 모델 순서대로 실행하고 독립 KDA 상태를 유지합니다.
#include "k3x/official_two_layer.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <limits>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace k3x {

Result<OfficialTwoLayerResult> official_two_layer_cpu(
    std::span<const OfficialLayerInput> inputs,
    std::span<const OfficialTwoLayerWeights> layers,
    std::span<const OfficialKdaState> initial_states,
    const OfficialKdaConfig& config,
    std::size_t top_k,
    float situ_beta,
    std::optional<float> situ_linear_beta) {
    if (inputs.size() != 2 || layers.size() != 2 ||
        initial_states.size() != 2 || layers[0].layer_id != 1 ||
        layers[1].layer_id != 2) {
        return Result<OfficialTwoLayerResult>::failure(
            ErrorCode::invalid_state);
    }

    std::array<OfficialKdaState, 2> states{
        initial_states[0], initial_states[1]};
    std::vector<OfficialTwoLayerStepResult> steps;
    steps.reserve(4);
    std::array<std::vector<float>, 2> outputs;
    for (std::size_t position = 0; position < inputs.size(); ++position) {
        OfficialLayerInput current = inputs[position];
        for (std::size_t layer_index = 0; layer_index < layers.size();
             ++layer_index) {
            auto result = official_layer_cpu(
                std::span<const OfficialLayerInput>(&current, 1),
                layers[layer_index].weights,
                states[layer_index],
                config,
                top_k,
                situ_beta,
                situ_linear_beta);
            if (!result) {
                return Result<OfficialTwoLayerResult>::failure(
                    result.error(), result.message());
            }
            if (result.value().steps.size() != 1) {
                return Result<OfficialTwoLayerResult>::failure(
                    ErrorCode::invalid_state);
            }
            states[layer_index] = std::move(result.value().kda.state);
            current.hidden_input = result.value().steps[0].moe.output;
            steps.push_back({
                position,
                layers[layer_index].layer_id,
                std::move(result.value().steps[0]),
                states[layer_index],
                std::move(result.value().kda.output),
            });
        }
        outputs[position] = std::move(current.hidden_input);
    }
    return Result<OfficialTwoLayerResult>::success({
        std::move(states), std::move(steps), std::move(outputs)});
}

namespace {

Result<std::vector<Mxfp4MlpView>> selected_experts(
    const OfficialMoeWeights& weights,
    std::span<const std::uint32_t> expert_ids) {
    std::vector<Mxfp4MlpView> selected;
    selected.reserve(expert_ids.size());
    for (const auto expert_id : expert_ids) {
        const auto found = std::find_if(
            weights.experts.begin(), weights.experts.end(),
            [expert_id](const OfficialExpertView& expert) {
                return expert.expert_id == expert_id;
            });
        if (found == weights.experts.end()) {
            return Result<std::vector<Mxfp4MlpView>>::failure(
                ErrorCode::invalid_extent);
        }
        selected.push_back(found->weights);
    }
    return Result<std::vector<Mxfp4MlpView>>::success(std::move(selected));
}

OfficialKdaState published_state(OfficialKdaCudaResult& result) {
    return {std::move(result.conv_q), std::move(result.conv_k),
            std::move(result.conv_v),
            std::move(result.recurrent_v_first)};
}

}  // namespace

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
    OfficialTwoLayerCudaMode mode,
    Profiler* attribution_profiler,
    OfficialTwoLayerAttribution* attribution) {
    if (inputs.size() != 2 || layers.size() != 2 ||
        initial_states.size() != 2 || layers[0].layer_id != 1 ||
        layers[1].layer_id != 2 || !config.hidden_size || !top_k ||
        (mode != OfficialTwoLayerCudaMode::host_round_trip &&
         mode != OfficialTwoLayerCudaMode::device_closure) ||
        ((attribution_profiler == nullptr) != (attribution == nullptr))) {
        return Result<OfficialTwoLayerCudaResult>::failure(
            ErrorCode::invalid_state);
    }

    OfficialTwoLayerAttribution measured_attribution;
    const auto wrapper_start = std::chrono::steady_clock::now();
    const auto elapsed = [](std::chrono::steady_clock::time_point start) {
        return static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - start)
                .count());
    };
    const auto device_delta = [&](std::size_t first)
        -> std::optional<std::uint64_t> {
        if (!attribution_profiler) return std::uint64_t{};
        const auto& events = attribution_profiler->events();
        if (events.size() < first) return std::nullopt;
        std::uint64_t total{};
        for (std::size_t index = first; index < events.size(); ++index) {
            if (!events[index].success) continue;
            if (events[index].device_nanoseconds >
                std::numeric_limits<std::uint64_t>::max() - total) {
                return std::nullopt;
            }
            total += events[index].device_nanoseconds;
        }
        return total;
    };

    std::array<OfficialKdaDeviceStateToken, 2> state_tokens{};
    OfficialLayerHiddenToken hidden_token{};
    const auto stats_before = backend.runtime_stats();
    const auto cleanup = [&]() {
        bool clean = true;
        if (hidden_token.owner) {
            clean = backend.discard_official_layer_hidden(hidden_token) && clean;
            hidden_token = {};
        }
        for (auto& token : state_tokens) {
            if (token.owner) {
                clean = backend.discard_official_kda_device_state(token) && clean;
                token = {};
            }
        }
        return clean;
    };
    const auto fail = [&](ErrorCode code, const std::string& message = {}) {
        if (!cleanup()) {
            return Result<OfficialTwoLayerCudaResult>::failure(
                ErrorCode::invalid_state);
        }
        return Result<OfficialTwoLayerCudaResult>::failure(code, message);
    };

    OfficialTwoLayerCudaResult result;
    const OfficialKdaState empty_state;
    result.steps.reserve(4);
    for (std::size_t position = 0; position < inputs.size(); ++position) {
        OfficialLayerInput current = inputs[position];
        for (std::size_t layer_index = 0; layer_index < layers.size();
             ++layer_index) {
            const auto layer = layers[layer_index].layer_id;
            const auto state_mode = position == 0
                ? OfficialKdaStateMode::device_seed
                : OfficialKdaStateMode::device_publish;
            const OfficialKdaStateControl state_control{
                state_mode, position == 0
                                ? OfficialKdaDeviceStateToken{}
                                : state_tokens[layer_index]};

            if (mode == OfficialTwoLayerCudaMode::host_round_trip) {
                const std::array<OfficialLayerInput, 1> one{{current}};
                auto layer_result = official_layer_cuda(
                    backend, one, layers[layer_index].weights,
                    position == 0 ? initial_states[layer_index] : empty_state,
                    config, top_k, situ_beta,
                    situ_linear_beta, layer, phase, state_control,
                    OfficialMoeRoutePreparationMode::device);
                if (!layer_result) {
                    return fail(
                        layer_result.error(),
                        "host layer " + std::to_string(layer) +
                            " position " + std::to_string(position) +
                            ": " + layer_result.message());
                }
                if (!layer_result.value().executed ||
                    layer_result.value().steps.size() != 1) {
                    return fail(ErrorCode::invalid_state);
                }
                if (position == 0) {
                    state_tokens[layer_index] =
                        layer_result.value().kda_device_state;
                    if (!state_tokens[layer_index].owner) {
                        return fail(ErrorCode::invalid_state);
                    }
                } else {
                    state_tokens[layer_index] = {};
                    if (!layer_result.value().kda_state_published) {
                        return fail(ErrorCode::invalid_state);
                    }
                    result.final_states[layer_index] =
                        std::move(layer_result.value().kda_state);
                }
                auto& step = layer_result.value().steps[0];
                current.hidden_input = step.output;
                result.steps.push_back({position, layer,
                                        std::move(step.route),
                                        std::move(step.output)});
                continue;
            }

            const auto& cuda_weights = layers[layer_index].weights;
            const OfficialLayerFrontWeights front_weights{
                cuda_weights.self_residual_norm,
                cuda_weights.self_residual_proj,
                cuda_weights.input_norm,
                cuda_weights.kda,
                {cuda_weights.moe.residual_norm,
                 cuda_weights.moe.residual_proj,
                 cuda_weights.moe.post_norm,
                 cuda_weights.moe.router},
            };
            const OfficialKdaCudaStateView state_view{
                position == 0 ? initial_states[layer_index].conv_q
                              : empty_state.conv_q,
                position == 0 ? initial_states[layer_index].conv_k
                              : empty_state.conv_k,
                position == 0 ? initial_states[layer_index].conv_v
                              : empty_state.conv_v,
                position == 0
                    ? initial_states[layer_index].recurrent_v_first
                    : empty_state.recurrent_v_first};
            const OfficialKdaCudaConfig cuda_config{
                config.hidden_size, config.heads, config.head_dim,
                config.conv_width, config.rms_norm_epsilon,
                config.gate_lower_bound};
            const auto front_event = attribution_profiler
                ? attribution_profiler->events().size() : 0;
            const auto front_start = std::chrono::steady_clock::now();
            auto front = backend.official_layer_front(
                layer_index == 0 ? std::span<const float>(current.hidden_input)
                                 : std::span<const float>{},
                layer_index == 0 ? std::span<const float>(current.block_source)
                                 : std::span<const float>{},
                layer_index == 0 ? OfficialLayerHiddenToken{} : hidden_token,
                front_weights, state_view, cuda_config, layer, phase,
                state_control);
            measured_attribution.front_wall_nanoseconds +=
                elapsed(front_start);
            const auto front_device = device_delta(front_event);
            if (!front_device ||
                *front_device > std::numeric_limits<std::uint64_t>::max() -
                    measured_attribution.front_device_nanoseconds) {
                return fail(ErrorCode::invalid_state);
            }
            measured_attribution.front_device_nanoseconds += *front_device;
            if (!front) {
                return fail(front.error(),
                            "device front layer " + std::to_string(layer) +
                                " position " + std::to_string(position) +
                                ": " + front.message());
            }
            if (!front.value().executed ||
                !front.value().route.prepared.owner) {
                return fail(ErrorCode::invalid_state);
            }
            if (layer_index == 1) hidden_token = {};
            if (position == 0) {
                state_tokens[layer_index] = front.value().kda.device_state;
                if (!state_tokens[layer_index].owner) {
                    const auto discarded = backend.discard_official_moe_prepared(
                        front.value().route.prepared);
                    if (!discarded) return fail(ErrorCode::invalid_state);
                    return fail(ErrorCode::invalid_state);
                }
            } else {
                state_tokens[layer_index] = {};
                if (!front.value().kda.state_published) {
                    const auto discarded = backend.discard_official_moe_prepared(
                        front.value().route.prepared);
                    if (!discarded) return fail(ErrorCode::invalid_state);
                    return fail(ErrorCode::invalid_state);
                }
                result.final_states[layer_index] =
                    published_state(front.value().kda);
            }

            const auto route_start = std::chrono::steady_clock::now();
            auto route = route_official_moe_logits(
                front.value().route.router_logits,
                cuda_weights.moe.correction, top_k);
            if (!route) {
                const auto discarded = backend.discard_official_moe_prepared(
                    front.value().route.prepared);
                if (!discarded) return fail(ErrorCode::invalid_state);
                return fail(route.error(), route.message());
            }
            auto selected = selected_experts(
                cuda_weights.moe, route.value().expert_ids);
            measured_attribution.route_wall_nanoseconds +=
                elapsed(route_start);
            if (!selected) {
                const auto discarded = backend.discard_official_moe_prepared(
                    front.value().route.prepared);
                if (!discarded) return fail(ErrorCode::invalid_state);
                return fail(selected.error(), selected.message());
            }
            const bool retain = layer_index == 0;
            const auto tail_event = attribution_profiler
                ? attribution_profiler->events().size() : 0;
            const auto tail_start = std::chrono::steady_clock::now();
            auto tail = backend.official_layer_tail(
                front.value().route.prepared, cuda_weights.moe_ffn,
                selected.value(), route.value().expert_ids,
                route.value().contributions, config.rms_norm_epsilon,
                situ_beta, situ_linear_beta, layer, phase, retain);
            measured_attribution.tail_wall_nanoseconds += elapsed(tail_start);
            const auto tail_device = device_delta(tail_event);
            if (!tail_device ||
                *tail_device > std::numeric_limits<std::uint64_t>::max() -
                    measured_attribution.tail_device_nanoseconds) {
                return fail(ErrorCode::invalid_state);
            }
            measured_attribution.tail_device_nanoseconds += *tail_device;
            if (!tail) {
                return fail(tail.error(),
                            "device tail layer " + std::to_string(layer) +
                                " position " + std::to_string(position) +
                                ": " + tail.message());
            }
            if (!tail.value().executed ||
                tail.value().selected_expert_ids !=
                    route.value().expert_ids ||
                (retain ? (!tail.value().hidden.owner ||
                           !tail.value().output.empty())
                        : (tail.value().hidden.owner ||
                           tail.value().output.size() != config.hidden_size))) {
                if (tail.value().hidden.owner) hidden_token = tail.value().hidden;
                return fail(ErrorCode::invalid_state);
            }
            hidden_token = tail.value().hidden;
            if (!retain) current.hidden_input = tail.value().output;
            result.steps.push_back({position, layer,
                                    std::move(route.value()),
                                    std::move(tail.value().output)});
        }
        if (hidden_token.owner) return fail(ErrorCode::invalid_state);
        result.outputs[position] = std::move(current.hidden_input);
    }
    if (state_tokens[0].owner || state_tokens[1].owner) {
        return fail(ErrorCode::invalid_state);
    }
    const auto stats_after = backend.runtime_stats();
    const auto delta = [](std::uint64_t after, std::uint64_t before) {
        return after >= before ? after - before : std::uint64_t{};
    };
    result.telemetry = {
        delta(stats_after.weight_h2d_bytes, stats_before.weight_h2d_bytes),
        delta(stats_after.activation_h2d_bytes,
              stats_before.activation_h2d_bytes),
        delta(stats_after.device_to_host_bytes,
              stats_before.device_to_host_bytes),
        delta(stats_after.official_kda_state_h2d_bytes,
              stats_before.official_kda_state_h2d_bytes),
        delta(stats_after.official_kda_state_d2h_bytes,
              stats_before.official_kda_state_d2h_bytes),
        delta(stats_after.official_kda_output_d2h_bytes,
              stats_before.official_kda_output_d2h_bytes),
        delta(stats_after.official_moe_router_logit_d2h_bytes,
              stats_before.official_moe_router_logit_d2h_bytes),
        mode == OfficialTwoLayerCudaMode::host_round_trip
            ? inputs.size() * config.hidden_size * sizeof(float)
            : 0,
        mode == OfficialTwoLayerCudaMode::host_round_trip
            ? inputs.size() * config.hidden_size * sizeof(float)
            : 0,
        inputs.size() * config.hidden_size * sizeof(float),
        mode == OfficialTwoLayerCudaMode::device_closure ? 4U : 0U,
        mode == OfficialTwoLayerCudaMode::device_closure ? 4U : 0U,
    };
    measured_attribution.total_wall_nanoseconds = elapsed(wrapper_start);
    const auto attributed_wall =
        measured_attribution.front_wall_nanoseconds +
        measured_attribution.route_wall_nanoseconds +
        measured_attribution.tail_wall_nanoseconds;
    if (attributed_wall > measured_attribution.total_wall_nanoseconds) {
        return Result<OfficialTwoLayerCudaResult>::failure(
            ErrorCode::invalid_state);
    }
    measured_attribution.unattributed_wall_nanoseconds =
        measured_attribution.total_wall_nanoseconds - attributed_wall;
    if (attribution) *attribution = measured_attribution;
    result.executed = true;
    return Result<OfficialTwoLayerCudaResult>::success(std::move(result));
}

}  // namespace k3x
