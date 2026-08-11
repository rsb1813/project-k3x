// 공식 self Attention Residual, KDA, MoE를 하나의 portable layer로 합성합니다.
#include "k3x/official_layer.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace k3x {
namespace {

std::uint16_t encode_bf16(float value) noexcept {
    auto bits = std::bit_cast<std::uint32_t>(value);
    if ((bits & 0x7f800000U) == 0x7f800000U) {
        if (bits & 0x007fffffU) bits |= 0x00400000U;
        return static_cast<std::uint16_t>(bits >> 16U);
    }
    bits += 0x7fffU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>(bits >> 16U);
}

float rounded_bf16(float value) noexcept {
    return decode_bf16_word(encode_bf16(value));
}

bool finite(std::span<const float> values) {
    return std::all_of(values.begin(), values.end(),
                       [](float value) { return std::isfinite(value); });
}

bool valid_words(std::span<const std::uint16_t> values) {
    return std::all_of(values.begin(), values.end(), [](std::uint16_t value) {
        return std::isfinite(decode_bf16_word(value));
    });
}

Result<std::vector<float>> attention_residual(
    std::span<const float> prefix,
    std::span<const float> block,
    Bf16VectorView norm,
    Bf16WeightView projection,
    float epsilon) {
    const auto width = prefix.size();
    if (!width || block.size() != width || norm.values.size() != width ||
        projection.rows != 1 || projection.cols != width ||
        projection.values.size() != width || !finite(prefix) || !finite(block) ||
        !valid_words(norm.values) || !valid_words(projection.values) ||
        !std::isfinite(epsilon) || epsilon <= 0.0F) {
        return Result<std::vector<float>>::failure(ErrorCode::invalid_extent);
    }
    std::vector<float> rounded_prefix(width);
    std::vector<float> rounded_block(width);
    for (std::size_t index = 0; index < width; ++index) {
        rounded_prefix[index] = rounded_bf16(prefix[index]);
        rounded_block[index] = rounded_bf16(block[index]);
        if (!std::isfinite(rounded_prefix[index]) ||
            !std::isfinite(rounded_block[index])) {
            return Result<std::vector<float>>::failure(ErrorCode::invalid_extent);
        }
    }
    const auto score = [&](std::span<const float> values) {
        double squares = 0.0;
        for (const auto value : values) {
            squares += static_cast<double>(value) * value;
        }
        const auto inverse = 1.0F / std::sqrt(
            static_cast<float>(squares / width) + epsilon);
        double result = 0.0;
        for (std::size_t index = 0; index < width; ++index) {
            result += static_cast<double>(values[index] * inverse) *
                      decode_bf16_word(norm.values[index]) *
                      decode_bf16_word(projection.values[index]);
        }
        return static_cast<float>(result);
    };
    const auto block_score = score(rounded_block);
    const auto prefix_score = score(rounded_prefix);
    const auto maximum = std::max(block_score, prefix_score);
    const auto block_exp = std::exp(block_score - maximum);
    const auto prefix_exp = std::exp(prefix_score - maximum);
    const auto denominator = block_exp + prefix_exp;
    if (!std::isfinite(denominator) || denominator <= 0.0F) {
        return Result<std::vector<float>>::failure(ErrorCode::invalid_extent);
    }
    const auto block_probability = block_exp / denominator;
    const auto prefix_probability = prefix_exp / denominator;
    std::vector<float> output(width);
    for (std::size_t index = 0; index < width; ++index) {
        output[index] = rounded_bf16(
            block_probability * rounded_block[index] +
            prefix_probability * rounded_prefix[index]);
        if (!std::isfinite(output[index])) {
            return Result<std::vector<float>>::failure(ErrorCode::invalid_extent);
        }
    }
    return Result<std::vector<float>>::success(std::move(output));
}

Result<std::vector<float>> rms_norm(
    std::span<const float> input,
    Bf16VectorView weight,
    float epsilon) {
    if (input.empty() || weight.values.size() != input.size() || !finite(input) ||
        !valid_words(weight.values) || !std::isfinite(epsilon) || epsilon <= 0.0F) {
        return Result<std::vector<float>>::failure(ErrorCode::invalid_extent);
    }
    double squares = 0.0;
    for (const auto value : input) squares += static_cast<double>(value) * value;
    const auto inverse = 1.0F / std::sqrt(
        static_cast<float>(squares / input.size()) + epsilon);
    std::vector<float> output(input.size());
    for (std::size_t index = 0; index < input.size(); ++index) {
        output[index] = rounded_bf16(
            input[index] * inverse * decode_bf16_word(weight.values[index]));
        if (!std::isfinite(output[index])) {
            return Result<std::vector<float>>::failure(ErrorCode::invalid_extent);
        }
    }
    return Result<std::vector<float>>::success(std::move(output));
}

bool equal(std::span<const float> left, std::span<const float> right) {
    return left.size() == right.size() &&
           std::equal(left.begin(), left.end(), right.begin());
}

}  // namespace

Result<OfficialLayerResult> official_layer_cpu(
    std::span<const OfficialLayerInput> inputs,
    const OfficialLayerWeights& weights,
    const OfficialKdaState& initial_state,
    const OfficialKdaConfig& config,
    std::size_t top_k,
    float situ_beta,
    std::optional<float> situ_linear_beta) {
    if (inputs.empty() || !config.hidden_size || !top_k ||
        !std::isfinite(situ_beta) || situ_beta <= 0.0F ||
        (situ_linear_beta &&
         (!std::isfinite(*situ_linear_beta) || *situ_linear_beta <= 0.0F))) {
        return Result<OfficialLayerResult>::failure(ErrorCode::invalid_extent);
    }
    if (inputs.size() >
        std::numeric_limits<std::size_t>::max() / config.hidden_size) {
        return Result<OfficialLayerResult>::failure(ErrorCode::invalid_extent);
    }
    std::vector<std::vector<float>> self_residuals;
    std::vector<std::vector<float>> normalized_inputs;
    std::vector<float> flattened;
    self_residuals.reserve(inputs.size());
    normalized_inputs.reserve(inputs.size());
    flattened.reserve(inputs.size() * config.hidden_size);
    for (const auto& input : inputs) {
        if (input.hidden_input.size() != config.hidden_size ||
            input.block_source.size() != config.hidden_size) {
            return Result<OfficialLayerResult>::failure(ErrorCode::invalid_extent);
        }
        auto residual = attention_residual(
            input.hidden_input,
            input.block_source,
            weights.self_residual_norm,
            weights.self_residual_proj,
            config.rms_norm_epsilon);
        if (!residual) {
            return Result<OfficialLayerResult>::failure(
                residual.error(), residual.message());
        }
        auto normalized = rms_norm(
            residual.value(), weights.input_norm, config.rms_norm_epsilon);
        if (!normalized) {
            return Result<OfficialLayerResult>::failure(
                normalized.error(), normalized.message());
        }
        flattened.insert(flattened.end(), normalized.value().begin(),
                         normalized.value().end());
        self_residuals.push_back(std::move(residual.value()));
        normalized_inputs.push_back(std::move(normalized.value()));
    }
    auto kda = official_kda_cpu(flattened, weights.kda, initial_state, config);
    if (!kda || kda.value().output.size() != flattened.size()) {
        return Result<OfficialLayerResult>::failure(
            kda ? ErrorCode::invalid_state : kda.error(),
            kda ? std::string{} : kda.message());
    }

    std::vector<OfficialLayerStepResult> steps;
    steps.reserve(inputs.size());
    for (std::size_t index = 0; index < inputs.size(); ++index) {
        std::vector<float> prefix(config.hidden_size);
        for (std::size_t channel = 0; channel < config.hidden_size; ++channel) {
            prefix[channel] = rounded_bf16(
                rounded_bf16(inputs[index].hidden_input[channel]) +
                kda.value().output[index * config.hidden_size + channel]);
        }
        auto mlp_residual = attention_residual(
            prefix,
            inputs[index].block_source,
            weights.moe.residual_norm,
            weights.moe.residual_proj,
            config.rms_norm_epsilon);
        auto normalized = rms_norm(
            mlp_residual ? std::span<const float>(mlp_residual.value())
                         : std::span<const float>{},
            weights.moe.post_norm,
            config.rms_norm_epsilon);
        const OfficialMoeInput moe_input{prefix, inputs[index].block_source};
        auto prepared = prepare_official_moe_input(
            moe_input, weights.moe, config.rms_norm_epsilon);
        if (!mlp_residual || !normalized || !prepared ||
            !equal(normalized.value(), prepared.value())) {
            return Result<OfficialLayerResult>::failure(ErrorCode::invalid_state);
        }
        auto route = route_official_moe(
            prepared.value(), weights.moe.router, weights.moe.correction, top_k);
        if (!route) {
            return Result<OfficialLayerResult>::failure(
                route.error(), route.message());
        }
        auto moe = official_moe_cpu(
            moe_input, weights.moe, route.value(), config.rms_norm_epsilon,
            situ_beta, situ_linear_beta);
        if (!moe || !equal(moe.value().hidden, prepared.value())) {
            return Result<OfficialLayerResult>::failure(
                moe ? ErrorCode::invalid_state : moe.error(),
                moe ? std::string{} : moe.message());
        }
        steps.push_back({
            std::move(self_residuals[index]),
            std::move(normalized_inputs[index]),
            std::move(prefix),
            std::move(mlp_residual.value()),
            std::move(prepared.value()),
            std::move(route.value()),
            std::move(moe.value()),
        });
    }
    return Result<OfficialLayerResult>::success(
        {std::move(kda.value()), std::move(steps)});
}

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
    OfficialKdaStateControl state_control,
    OfficialMoeRoutePreparationMode route_preparation) {
    if (inputs.empty() || !config.hidden_size || !top_k ||
        !std::isfinite(situ_beta) || situ_beta <= 0.0F ||
        (situ_linear_beta &&
         (!std::isfinite(*situ_linear_beta) || *situ_linear_beta <= 0.0F)) ||
        inputs.size() >
            std::numeric_limits<std::size_t>::max() / config.hidden_size) {
        return Result<OfficialLayerCudaResult>::failure(
            ErrorCode::invalid_extent);
    }
    if (route_preparation != OfficialMoeRoutePreparationMode::host &&
        route_preparation != OfficialMoeRoutePreparationMode::device) {
        return Result<OfficialLayerCudaResult>::failure(
            ErrorCode::invalid_state);
    }

    std::vector<std::vector<float>> self_residuals;
    std::vector<std::vector<float>> normalized_inputs;
    std::vector<float> flattened;
    self_residuals.reserve(inputs.size());
    normalized_inputs.reserve(inputs.size());
    flattened.reserve(inputs.size() * config.hidden_size);
    for (const auto& input : inputs) {
        if (input.hidden_input.size() != config.hidden_size ||
            input.block_source.size() != config.hidden_size) {
            return Result<OfficialLayerCudaResult>::failure(
                ErrorCode::invalid_extent);
        }
        auto residual = attention_residual(
            input.hidden_input, input.block_source,
            weights.self_residual_norm, weights.self_residual_proj,
            config.rms_norm_epsilon);
        if (!residual) {
            return Result<OfficialLayerCudaResult>::failure(
                residual.error(), residual.message());
        }
        auto normalized = rms_norm(
            residual.value(), weights.input_norm, config.rms_norm_epsilon);
        if (!normalized) {
            return Result<OfficialLayerCudaResult>::failure(
                normalized.error(), normalized.message());
        }
        flattened.insert(flattened.end(), normalized.value().begin(),
                         normalized.value().end());
        self_residuals.push_back(std::move(residual.value()));
        normalized_inputs.push_back(std::move(normalized.value()));
    }

    const OfficialKdaCudaStateView state_view{
        initial_state.conv_q, initial_state.conv_k, initial_state.conv_v,
        initial_state.recurrent_v_first};
    const OfficialKdaCudaConfig cuda_config{
        config.hidden_size, config.heads, config.head_dim, config.conv_width,
        config.rms_norm_epsilon, config.gate_lower_bound};
    auto kda = backend.official_kda(
        flattened, weights.kda, state_view, cuda_config, layer, phase,
        state_control);
    if (!kda) {
        return Result<OfficialLayerCudaResult>::failure(
            kda.error(), kda.message());
    }
    if (!kda.value().executed || kda.value().output.size() != flattened.size()) {
        return Result<OfficialLayerCudaResult>::failure(
            ErrorCode::invalid_state);
    }

    const auto kda_token = kda.value().device_state;
    const auto fail_after_kda = [&](ErrorCode code,
                                    const std::string& message = {}) {
        if (kda_token.owner || kda_token.generation) {
            const auto discarded =
                backend.discard_official_kda_device_state(kda_token);
            if (!discarded) {
                return Result<OfficialLayerCudaResult>::failure(
                    ErrorCode::invalid_state);
            }
        }
        return Result<OfficialLayerCudaResult>::failure(code, message);
    };
    const bool expected_publication =
        state_control.mode == OfficialKdaStateMode::host_roundtrip ||
        state_control.mode == OfficialKdaStateMode::device_publish;
    const bool expected_token =
        state_control.mode == OfficialKdaStateMode::device_seed ||
        state_control.mode == OfficialKdaStateMode::device_continue;
    if (kda.value().state_published != expected_publication ||
        ((kda_token.owner != 0 && kda_token.generation != 0) !=
         expected_token)) {
        return fail_after_kda(ErrorCode::invalid_state);
    }

    OfficialKdaState final_state{
        std::move(kda.value().conv_q), std::move(kda.value().conv_k),
        std::move(kda.value().conv_v),
        std::move(kda.value().recurrent_v_first)};
    std::vector<OfficialLayerCudaStepResult> steps;
    steps.reserve(inputs.size());
    for (std::size_t index = 0; index < inputs.size(); ++index) {
        std::vector<float> prefix(config.hidden_size);
        for (std::size_t channel = 0; channel < config.hidden_size; ++channel) {
            prefix[channel] = rounded_bf16(
                rounded_bf16(inputs[index].hidden_input[channel]) +
                kda.value().output[index * config.hidden_size + channel]);
        }
        if (route_preparation == OfficialMoeRoutePreparationMode::device) {
            auto prepared = backend.prepare_official_moe_route(
                prefix, inputs[index].block_source,
                {weights.moe.residual_norm, weights.moe.residual_proj,
                 weights.moe.post_norm, weights.moe.router},
                config.rms_norm_epsilon, layer, phase);
            if (!prepared) {
                return fail_after_kda(prepared.error(), prepared.message());
            }
            const auto prepared_token = prepared.value().prepared;
            const auto fail_after_prepare = [&](ErrorCode code,
                                                const std::string& message = {},
                                                bool consumed = false) {
                const auto discarded =
                    backend.discard_official_moe_prepared(prepared_token);
                if (!discarded &&
                    !(consumed && discarded.error() == ErrorCode::invalid_state)) {
                    return fail_after_kda(ErrorCode::invalid_state);
                }
                return fail_after_kda(code, message);
            };
            if (!prepared.value().executed || !prepared_token.owner ||
                !prepared_token.generation) {
                return fail_after_prepare(ErrorCode::invalid_state);
            }
            auto route = route_official_moe_logits(
                prepared.value().router_logits, weights.moe.correction, top_k);
            if (!route) {
                return fail_after_prepare(route.error(), route.message());
            }
            std::vector<Mxfp4MlpView> selected;
            selected.reserve(route.value().expert_ids.size());
            for (const auto expert_id : route.value().expert_ids) {
                const auto match = std::find_if(
                    weights.moe.experts.begin(), weights.moe.experts.end(),
                    [expert_id](const OfficialExpertView& expert) {
                        return expert.expert_id == expert_id;
                    });
                if (match == weights.moe.experts.end()) {
                    return fail_after_prepare(ErrorCode::invalid_extent);
                }
                selected.push_back(match->weights);
            }
            auto moe = backend.official_mxfp4_moe_ffn_prepared(
                prepared_token, weights.moe_ffn, selected,
                route.value().expert_ids, route.value().contributions,
                config.rms_norm_epsilon, situ_beta, situ_linear_beta,
                layer, phase);
            if (!moe) {
                return fail_after_prepare(
                    moe.error(), moe.message(), true);
            }
            if (!moe.value().executed ||
                moe.value().selected_expert_ids != route.value().expert_ids ||
                moe.value().output.size() != config.hidden_size) {
                return fail_after_prepare(ErrorCode::invalid_state, {}, true);
            }
            steps.push_back({
                std::move(self_residuals[index]),
                std::move(normalized_inputs[index]),
                std::move(prefix),
                {},
                {},
                std::move(route.value()),
                std::move(moe.value().output),
            });
            continue;
        }
        auto mlp_residual = attention_residual(
            prefix, inputs[index].block_source, weights.moe.residual_norm,
            weights.moe.residual_proj, config.rms_norm_epsilon);
        if (!mlp_residual) {
            return fail_after_kda(
                mlp_residual.error(), mlp_residual.message());
        }
        const OfficialMoeInput moe_input{prefix, inputs[index].block_source};
        auto prepared = prepare_official_moe_input(
            moe_input, weights.moe, config.rms_norm_epsilon);
        if (!prepared) {
            return fail_after_kda(prepared.error(), prepared.message());
        }
        auto route = route_official_moe(
            prepared.value(), weights.moe.router, weights.moe.correction, top_k);
        if (!route) {
            return fail_after_kda(route.error(), route.message());
        }
        std::vector<Mxfp4MlpView> selected;
        selected.reserve(route.value().expert_ids.size());
        for (const auto expert_id : route.value().expert_ids) {
            const auto match = std::find_if(
                weights.moe.experts.begin(), weights.moe.experts.end(),
                [expert_id](const OfficialExpertView& expert) {
                    return expert.expert_id == expert_id;
                });
            if (match == weights.moe.experts.end()) {
                return fail_after_kda(ErrorCode::invalid_extent);
            }
            selected.push_back(match->weights);
        }
        auto moe = backend.official_mxfp4_moe_ffn(
            prepared.value(), prefix, weights.moe_ffn, selected,
            route.value().expert_ids, route.value().contributions,
            config.rms_norm_epsilon, situ_beta, situ_linear_beta,
            layer, phase);
        if (!moe) {
            return fail_after_kda(moe.error(), moe.message());
        }
        if (!moe.value().executed ||
            moe.value().selected_expert_ids != route.value().expert_ids ||
            moe.value().output.size() != config.hidden_size) {
            return fail_after_kda(ErrorCode::invalid_state);
        }
        steps.push_back({
            std::move(self_residuals[index]),
            std::move(normalized_inputs[index]),
            std::move(prefix),
            std::move(mlp_residual.value()),
            std::move(prepared.value()),
            std::move(route.value()),
            std::move(moe.value().output),
        });
    }
    OfficialLayerCudaResult result;
    result.executed = true;
    result.kda_state_published = kda.value().state_published;
    result.kda_device_state = kda_token;
    result.kda_state = std::move(final_state);
    result.steps = std::move(steps);
    return Result<OfficialLayerCudaResult>::success(std::move(result));
}

}  // namespace k3x
