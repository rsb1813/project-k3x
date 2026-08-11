// 공식 KDA의 BF16 convolution과 FP32 V-first recurrence를 portable하게 실행합니다.
#include "k3x/official_kda.hpp"

#include "k3x/official_moe.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>
#include <vector>

namespace k3x {
namespace {

bool product(std::size_t left, std::size_t right, std::size_t& output) {
    if (right && left > std::numeric_limits<std::size_t>::max() / right) {
        return false;
    }
    output = left * right;
    return true;
}

bool finite(std::span<const float> values) {
    return std::all_of(values.begin(), values.end(),
                       [](float value) { return std::isfinite(value); });
}

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

bool valid_words(std::span<const std::uint16_t> values) {
    return std::all_of(values.begin(), values.end(), [](std::uint16_t value) {
        return std::isfinite(decode_bf16_word(value));
    });
}

bool valid_matrix(Bf16WeightView view, std::size_t rows, std::size_t cols) {
    std::size_t elements{};
    return view.rows == rows && view.cols == cols &&
           product(rows, cols, elements) && view.values.size() == elements &&
           valid_words(view.values);
}

Result<std::vector<float>> matmul_sequence(
    std::span<const float> input, std::size_t sequence,
    std::size_t input_width, Bf16WeightView weight) {
    std::size_t input_count{};
    if (!product(sequence, input_width, input_count) ||
        !valid_matrix(weight, weight.rows, input_width) ||
        input.size() != input_count || !finite(input)) {
        return Result<std::vector<float>>::failure(ErrorCode::invalid_extent);
    }
    std::size_t count{};
    if (!product(sequence, weight.rows, count)) {
        return Result<std::vector<float>>::failure(ErrorCode::invalid_extent);
    }
    std::vector<float> output(count);
    for (std::size_t token = 0; token < sequence; ++token) {
        for (std::size_t row = 0; row < weight.rows; ++row) {
            double sum = 0.0;
            for (std::size_t column = 0; column < input_width; ++column) {
                sum += static_cast<double>(input[token * input_width + column]) *
                       decode_bf16_word(weight.values[row * input_width + column]);
            }
            const auto value = rounded_bf16(static_cast<float>(sum));
            if (!std::isfinite(value)) {
                return Result<std::vector<float>>::failure(
                    ErrorCode::invalid_extent);
            }
            output[token * weight.rows + row] = value;
        }
    }
    return Result<std::vector<float>>::success(std::move(output));
}

struct ConvResult {
    std::vector<float> output;
    std::vector<std::uint16_t> state;
};

Result<ConvResult> short_conv(
    std::span<const float> projected,
    std::span<const std::uint16_t> initial_state,
    std::span<const float> weight,
    std::size_t sequence,
    std::size_t projection,
    std::size_t width) {
    std::size_t history_count{};
    std::size_t weight_count{};
    std::size_t projected_count{};
    if (width < 2 || !product(width - 1, projection, history_count) ||
        !product(width, projection, weight_count) ||
        !product(sequence, projection, projected_count) ||
        initial_state.size() != history_count || weight.size() != weight_count ||
        projected.size() != projected_count || !valid_words(initial_state) ||
        !finite(weight)) {
        return Result<ConvResult>::failure(ErrorCode::invalid_extent);
    }
    std::vector<std::uint16_t> state(initial_state.begin(), initial_state.end());
    std::vector<float> output(sequence * projection);
    for (std::size_t token = 0; token < sequence; ++token) {
        for (std::size_t channel = 0; channel < projection; ++channel) {
            float sum = 0.0F;
            for (std::size_t history = 0; history + 1 < width; ++history) {
                sum += decode_bf16_word(state[history * projection + channel]) *
                       weight[channel * width + history];
            }
            sum += projected[token * projection + channel] *
                   weight[channel * width + width - 1];
            const auto activated = sum / (1.0F + std::exp(-sum));
            const auto value = rounded_bf16(activated);
            if (!std::isfinite(value)) {
                return Result<ConvResult>::failure(ErrorCode::invalid_extent);
            }
            output[token * projection + channel] = value;
        }
        for (std::size_t history = 0; history + 2 < width; ++history) {
            for (std::size_t channel = 0; channel < projection; ++channel) {
                state[history * projection + channel] =
                    state[(history + 1) * projection + channel];
            }
        }
        for (std::size_t channel = 0; channel < projection; ++channel) {
            state[(width - 2) * projection + channel] =
                encode_bf16(projected[token * projection + channel]);
        }
    }
    return Result<ConvResult>::success({std::move(output), std::move(state)});
}

bool valid_config(const OfficialKdaConfig& config,
                  std::size_t& projection,
                  std::size_t& history,
                  std::size_t& recurrent) {
    std::size_t head_square{};
    return config.hidden_size && config.heads && config.head_dim &&
           config.conv_width >= 2 &&
           std::isfinite(config.rms_norm_epsilon) &&
           config.rms_norm_epsilon > 0.0F &&
           std::isfinite(config.gate_lower_bound) &&
           config.gate_lower_bound < 0.0F &&
           product(config.heads, config.head_dim, projection) &&
           product(config.conv_width - 1, projection, history) &&
           product(config.head_dim, config.head_dim, head_square) &&
           product(config.heads, head_square, recurrent);
}

bool valid_weights(const OfficialKdaWeightsView& weights,
                   const OfficialKdaConfig& config,
                   std::size_t projection) {
    std::size_t conv_count{};
    return product(projection, config.conv_width, conv_count) &&
           valid_matrix(weights.q_proj, projection, config.hidden_size) &&
           valid_matrix(weights.k_proj, projection, config.hidden_size) &&
           valid_matrix(weights.v_proj, projection, config.hidden_size) &&
           weights.q_conv.size() == conv_count && finite(weights.q_conv) &&
           weights.k_conv.size() == conv_count && finite(weights.k_conv) &&
           weights.v_conv.size() == conv_count && finite(weights.v_conv) &&
           valid_matrix(weights.f_a_proj, config.head_dim, config.hidden_size) &&
           valid_matrix(weights.f_b_proj, projection, config.head_dim) &&
           weights.a_log.size() == config.head_dim && finite(weights.a_log) &&
           weights.dt_bias.size() == projection && finite(weights.dt_bias) &&
           valid_matrix(weights.b_proj, config.heads, config.hidden_size) &&
           valid_matrix(weights.g_proj, projection, config.hidden_size) &&
           weights.o_norm.size() == config.head_dim && finite(weights.o_norm) &&
           valid_matrix(weights.o_proj, config.hidden_size, projection);
}

}  // namespace

OfficialKdaState zero_official_kda_state(const OfficialKdaConfig& config) {
    std::size_t projection{};
    std::size_t history{};
    std::size_t recurrent{};
    if (!valid_config(config, projection, history, recurrent)) return {};
    return {
        std::vector<std::uint16_t>(history),
        std::vector<std::uint16_t>(history),
        std::vector<std::uint16_t>(history),
        std::vector<float>(recurrent),
    };
}

Result<OfficialKdaResult> official_kda_cpu(
    std::span<const float> hidden,
    const OfficialKdaWeightsView& weights,
    const OfficialKdaState& state,
    const OfficialKdaConfig& config) {
    std::size_t projection{};
    std::size_t history_count{};
    std::size_t recurrent_count{};
    if (!valid_config(config, projection, history_count, recurrent_count) ||
        hidden.empty() || hidden.size() % config.hidden_size != 0 ||
        !finite(hidden) || !valid_weights(weights, config, projection) ||
        state.conv_q.size() != history_count ||
        state.conv_k.size() != history_count ||
        state.conv_v.size() != history_count ||
        state.recurrent_v_first.size() != recurrent_count ||
        !valid_words(state.conv_q) || !valid_words(state.conv_k) ||
        !valid_words(state.conv_v) || !finite(state.recurrent_v_first)) {
        return Result<OfficialKdaResult>::failure(ErrorCode::invalid_extent);
    }
    const auto sequence = hidden.size() / config.hidden_size;
    std::size_t sequence_projection{};
    if (!product(sequence, projection, sequence_projection)) {
        return Result<OfficialKdaResult>::failure(ErrorCode::invalid_extent);
    }
    std::vector<float> rounded_hidden(hidden.size());
    std::transform(hidden.begin(), hidden.end(), rounded_hidden.begin(),
                   [](float value) { return rounded_bf16(value); });
    if (!finite(rounded_hidden)) {
        return Result<OfficialKdaResult>::failure(ErrorCode::invalid_extent);
    }
    auto projected_q = matmul_sequence(
        rounded_hidden, sequence, config.hidden_size, weights.q_proj);
    auto projected_k = matmul_sequence(
        rounded_hidden, sequence, config.hidden_size, weights.k_proj);
    auto projected_v = matmul_sequence(
        rounded_hidden, sequence, config.hidden_size, weights.v_proj);
    auto forget_low = matmul_sequence(
        rounded_hidden, sequence, config.hidden_size, weights.f_a_proj);
    auto beta_projection = matmul_sequence(
        rounded_hidden, sequence, config.hidden_size, weights.b_proj);
    auto output_gate = matmul_sequence(
        rounded_hidden, sequence, config.hidden_size, weights.g_proj);
    if (!projected_q || !projected_k || !projected_v || !forget_low ||
        !beta_projection || !output_gate) {
        return Result<OfficialKdaResult>::failure(ErrorCode::invalid_extent);
    }
    auto forget = matmul_sequence(
        forget_low.value(), sequence, config.head_dim, weights.f_b_proj);
    auto convolved_q = short_conv(
        projected_q.value(), state.conv_q, weights.q_conv,
        sequence, projection, config.conv_width);
    auto convolved_k = short_conv(
        projected_k.value(), state.conv_k, weights.k_conv,
        sequence, projection, config.conv_width);
    auto convolved_v = short_conv(
        projected_v.value(), state.conv_v, weights.v_conv,
        sequence, projection, config.conv_width);
    if (!forget || !convolved_q || !convolved_k || !convolved_v) {
        return Result<OfficialKdaResult>::failure(ErrorCode::invalid_extent);
    }

    std::vector<float> q(sequence_projection);
    std::vector<float> k(sequence_projection);
    const auto q_scale = 1.0F / std::sqrt(static_cast<float>(config.head_dim));
    for (std::size_t token = 0; token < sequence; ++token) {
        for (std::size_t head = 0; head < config.heads; ++head) {
            const auto base = token * projection + head * config.head_dim;
            float q_squares = 0.0F;
            float k_squares = 0.0F;
            for (std::size_t channel = 0; channel < config.head_dim; ++channel) {
                const auto q_value = convolved_q.value().output[base + channel];
                const auto k_value = convolved_k.value().output[base + channel];
                q_squares += q_value * q_value;
                k_squares += k_value * k_value;
            }
            const auto q_inverse = 1.0F /
                std::max(std::sqrt(q_squares), 1.0e-12F);
            const auto k_inverse = 1.0F /
                std::max(std::sqrt(k_squares), 1.0e-12F);
            for (std::size_t channel = 0; channel < config.head_dim; ++channel) {
                q[base + channel] = rounded_bf16(
                    convolved_q.value().output[base + channel] *
                    q_inverse * q_scale);
                k[base + channel] = rounded_bf16(
                    convolved_k.value().output[base + channel] * k_inverse);
                if (!std::isfinite(q[base + channel]) ||
                    !std::isfinite(k[base + channel])) {
                    return Result<OfficialKdaResult>::failure(
                        ErrorCode::invalid_extent);
                }
            }
        }
    }

    std::vector<float> log_decay(sequence_projection);
    for (std::size_t token = 0; token < sequence; ++token) {
        for (std::size_t head = 0; head < config.heads; ++head) {
            for (std::size_t channel = 0; channel < config.head_dim; ++channel) {
                const auto index = token * projection +
                                   head * config.head_dim + channel;
                const auto argument = std::exp(weights.a_log[channel]) *
                    (forget.value()[index] + weights.dt_bias[
                        head * config.head_dim + channel]);
                log_decay[index] = config.gate_lower_bound /
                    (1.0F + std::exp(-argument));
                if (!std::isfinite(argument) || !std::isfinite(log_decay[index])) {
                    return Result<OfficialKdaResult>::failure(
                        ErrorCode::invalid_extent);
                }
            }
        }
    }
    std::vector<float> beta(beta_projection.value().size());
    std::transform(beta_projection.value().begin(), beta_projection.value().end(),
                   beta.begin(), [](float value) {
                       return 1.0F / (1.0F + std::exp(-value));
                   });
    if (!finite(beta)) {
        return Result<OfficialKdaResult>::failure(ErrorCode::invalid_extent);
    }

    auto recurrent = state.recurrent_v_first;
    std::vector<float> recurrent_output(sequence_projection);
    std::vector<float> decayed(config.head_dim * config.head_dim);
    std::vector<float> prediction(config.head_dim);
    std::vector<float> delta(config.head_dim);
    for (std::size_t token = 0; token < sequence; ++token) {
        for (std::size_t head = 0; head < config.heads; ++head) {
            const auto vector_base = token * projection + head * config.head_dim;
            const auto state_base = head * config.head_dim * config.head_dim;
            std::fill(prediction.begin(), prediction.end(), 0.0F);
            for (std::size_t key = 0; key < config.head_dim; ++key) {
                const auto alpha = std::exp(log_decay[vector_base + key]);
                for (std::size_t value = 0; value < config.head_dim; ++value) {
                    const auto state_index = state_base +
                        value * config.head_dim + key;
                    const auto kv_index = key * config.head_dim + value;
                    decayed[kv_index] = alpha * recurrent[state_index];
                    prediction[value] +=
                        k[vector_base + key] * decayed[kv_index];
                }
            }
            for (std::size_t value = 0; value < config.head_dim; ++value) {
                delta[value] =
                    (convolved_v.value().output[vector_base + value] -
                     prediction[value]) * beta[token * config.heads + head];
            }
            for (std::size_t key = 0; key < config.head_dim; ++key) {
                for (std::size_t value = 0; value < config.head_dim; ++value) {
                    const auto updated = decayed[key * config.head_dim + value] +
                        k[vector_base + key] * delta[value];
                    if (!std::isfinite(updated)) {
                        return Result<OfficialKdaResult>::failure(
                            ErrorCode::invalid_extent);
                    }
                    recurrent[state_base + value * config.head_dim + key] = updated;
                }
            }
            for (std::size_t value = 0; value < config.head_dim; ++value) {
                float output = 0.0F;
                for (std::size_t key = 0; key < config.head_dim; ++key) {
                    output += q[vector_base + key] *
                        recurrent[state_base + value * config.head_dim + key];
                }
                recurrent_output[vector_base + value] = output;
                if (!std::isfinite(output)) {
                    return Result<OfficialKdaResult>::failure(
                        ErrorCode::invalid_extent);
                }
            }
        }
    }

    std::vector<float> gated(sequence_projection);
    for (std::size_t token = 0; token < sequence; ++token) {
        for (std::size_t head = 0; head < config.heads; ++head) {
            const auto base = token * projection + head * config.head_dim;
            float squares = 0.0F;
            for (std::size_t value = 0; value < config.head_dim; ++value) {
                const auto item = recurrent_output[base + value];
                squares += item * item;
            }
            const auto inverse = 1.0F / std::sqrt(
                squares / static_cast<float>(config.head_dim) +
                config.rms_norm_epsilon);
            for (std::size_t value = 0; value < config.head_dim; ++value) {
                const auto gate = 1.0F /
                    (1.0F + std::exp(-output_gate.value()[base + value]));
                gated[base + value] = rounded_bf16(
                    recurrent_output[base + value] * inverse *
                    weights.o_norm[value] * gate);
                if (!std::isfinite(gated[base + value])) {
                    return Result<OfficialKdaResult>::failure(
                        ErrorCode::invalid_extent);
                }
            }
        }
    }
    auto output = matmul_sequence(
        gated, sequence, projection, weights.o_proj);
    if (!output) {
        return Result<OfficialKdaResult>::failure(
            output.error(), output.message());
    }
    OfficialKdaBoundaries boundaries{
        std::move(projected_q.value()),
        std::move(projected_k.value()),
        std::move(projected_v.value()),
        std::move(convolved_q.value().output),
        std::move(convolved_k.value().output),
        std::move(convolved_v.value().output),
        std::move(q),
        std::move(k),
        {},
        std::move(log_decay),
        std::move(beta),
        std::move(recurrent_output),
        std::move(gated),
    };
    boundaries.v = boundaries.convolved_v;
    OfficialKdaState final_state{
        std::move(convolved_q.value().state),
        std::move(convolved_k.value().state),
        std::move(convolved_v.value().state),
        std::move(recurrent),
    };
    return Result<OfficialKdaResult>::success({
        std::move(output.value()), std::move(final_state), std::move(boundaries)});
}

}  // namespace k3x
