// 공식 Kimi K3 MoE FFN의 BF16/MXFP4 portable oracle를 구현합니다.
#include "k3x/official_moe.hpp"

#include "k3x/ops.hpp"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <unordered_map>
#include <unordered_set>

namespace k3x {
namespace {

float round_bf16(float value) noexcept {
    auto bits = std::bit_cast<std::uint32_t>(value);
    if ((bits & 0x7f800000U) == 0x7f800000U) {
        if (bits & 0x007fffffU) bits |= 0x00400000U;
        return std::bit_cast<float>(bits & 0xffff0000U);
    }
    bits += 0x7fffU + ((bits >> 16U) & 1U);
    return std::bit_cast<float>(bits & 0xffff0000U);
}

bool finite(std::span<const float> values) {
    return std::all_of(values.begin(), values.end(),
                       [](float value) { return std::isfinite(value); });
}

bool valid(Bf16WeightView weight) {
    return weight.rows && weight.cols &&
           weight.rows <= std::numeric_limits<std::size_t>::max() / weight.cols &&
           weight.values.size() == weight.rows * weight.cols;
}

Result<std::vector<float>> bf16_matvec(
    std::span<const float> input, Bf16WeightView weight) {
    if (!valid(weight) || input.size() != weight.cols || !finite(input)) {
        return Result<std::vector<float>>::failure(ErrorCode::invalid_extent);
    }
    std::vector<float> output(weight.rows);
    for (std::size_t row = 0; row < weight.rows; ++row) {
        double sum = 0.0;
        for (std::size_t column = 0; column < weight.cols; ++column) {
            const auto value = decode_bf16_word(
                weight.values[row * weight.cols + column]);
            if (!std::isfinite(value)) {
                return Result<std::vector<float>>::failure(
                    ErrorCode::invalid_extent);
            }
            sum += static_cast<double>(value) * input[column];
        }
        output[row] = round_bf16(static_cast<float>(sum));
    }
    return Result<std::vector<float>>::success(std::move(output));
}

Result<std::vector<float>> normalized(
    std::span<const float> input, Bf16VectorView weight, float epsilon) {
    if (input.empty() || input.size() != weight.values.size() ||
        !finite(input) || !std::isfinite(epsilon) || epsilon <= 0.0F) {
        return Result<std::vector<float>>::failure(ErrorCode::invalid_extent);
    }
    double squares = 0.0;
    for (const auto value : input) squares += static_cast<double>(value) * value;
    const auto inverse = 1.0F / std::sqrt(
        static_cast<float>(squares / input.size()) + epsilon);
    std::vector<float> output(input.size());
    for (std::size_t index = 0; index < input.size(); ++index) {
        const auto scale = decode_bf16_word(weight.values[index]);
        if (!std::isfinite(scale)) {
            return Result<std::vector<float>>::failure(ErrorCode::invalid_extent);
        }
        output[index] = round_bf16(input[index] * inverse * scale);
    }
    return Result<std::vector<float>>::success(std::move(output));
}

Result<std::vector<float>> expert_ffn(
    std::span<const float> input, Mxfp4MlpView expert,
    float situ_beta, std::optional<float> situ_linear_beta) {
    auto gate = mxfp4_matmul(input, expert.gate.packed, expert.gate.scales,
                             expert.gate.rows, expert.gate.cols,
                             expert.gate.group_size);
    auto up = mxfp4_matmul(input, expert.up.packed, expert.up.scales,
                           expert.up.rows, expert.up.cols,
                           expert.up.group_size);
    if (!gate || !up || gate.value().size() != up.value().size() ||
        expert.down.cols != gate.value().size()) {
        return Result<std::vector<float>>::failure(ErrorCode::invalid_mxfp4);
    }
    std::vector<float> activated(gate.value().size());
    situ_glu(activated, gate.value(), up.value(), situ_beta, situ_linear_beta);
    auto output = mxfp4_matmul(
        activated, expert.down.packed, expert.down.scales,
        expert.down.rows, expert.down.cols, expert.down.group_size);
    if (!output) return output;
    for (auto& value : output.value()) value = round_bf16(value);
    return output;
}

Result<std::vector<float>> shared_ffn(
    std::span<const float> input, Bf16MlpView weights,
    float situ_beta, std::optional<float> situ_linear_beta) {
    auto gate = bf16_matvec(input, weights.gate);
    auto up = bf16_matvec(input, weights.up);
    if (!gate || !up || gate.value().size() != up.value().size() ||
        weights.down.cols != gate.value().size()) {
        return Result<std::vector<float>>::failure(ErrorCode::invalid_extent);
    }
    std::vector<float> activated(gate.value().size());
    situ_glu(activated, gate.value(), up.value(), situ_beta, situ_linear_beta);
    return bf16_matvec(activated, weights.down);
}

bool valid_route(const OfficialRoute& route) {
    if (route.expert_ids.empty() ||
        route.expert_ids.size() != route.contributions.size()) return false;
    std::unordered_set<std::uint32_t> ids;
    double sum = 0.0;
    for (std::size_t index = 0; index < route.expert_ids.size(); ++index) {
        const auto contribution = route.contributions[index];
        if (!ids.insert(route.expert_ids[index]).second ||
            !std::isfinite(contribution) || contribution <= 0.0F) return false;
        sum += contribution;
    }
    return std::abs(sum - 1.0) <= 1.0e-5;
}

}  // namespace

float decode_bf16_word(std::uint16_t word) noexcept {
    return std::bit_cast<float>(static_cast<std::uint32_t>(word) << 16U);
}

std::array<OfficialMoeInput, 2> official_moe_inputs() {
    const auto values = [](int multiplier, int increment, int modulus,
                           int offset) {
        std::vector<float> output(7'168);
        for (std::size_t index = 0; index < output.size(); ++index) {
            output[index] = static_cast<float>(
                (multiplier * static_cast<int>(index) + increment) % modulus -
                offset) / 1024.0F;
        }
        return output;
    };
    return {{
        {values(17, 3, 257, 128), values(29, 11, 251, 125)},
        {values(31, 7, 263, 131), values(43, 19, 269, 134)},
    }};
}

Result<std::vector<float>> prepare_official_moe_input(
    const OfficialMoeInput& input, const OfficialMoeWeights& weights,
    float epsilon) {
    const auto width = input.prefix_sum.size();
    if (!width || input.block_residual.size() != width ||
        weights.residual_norm.values.size() != width ||
        !valid(weights.residual_proj) || weights.residual_proj.rows != 1 ||
        weights.residual_proj.cols != width ||
        weights.post_norm.values.size() != width ||
        !finite(input.prefix_sum) || !finite(input.block_residual) ||
        !std::isfinite(epsilon) || epsilon <= 0.0F) {
        return Result<std::vector<float>>::failure(ErrorCode::invalid_extent);
    }
    std::array<std::vector<float>, 2> values{
        input.block_residual, input.prefix_sum};
    for (auto& row : values) {
        for (auto& value : row) value = round_bf16(value);
    }
    std::array<float, 2> attention_scores{};
    for (std::size_t row = 0; row < values.size(); ++row) {
        double squares = 0.0;
        for (const auto value : values[row])
            squares += static_cast<double>(value) * value;
        const auto inverse = 1.0F / std::sqrt(
            static_cast<float>(squares / width) + epsilon);
        double score = 0.0;
        for (std::size_t index = 0; index < width; ++index) {
            const auto norm = decode_bf16_word(weights.residual_norm.values[index]);
            const auto projection = decode_bf16_word(
                weights.residual_proj.values[index]);
            if (!std::isfinite(norm) || !std::isfinite(projection)) {
                return Result<std::vector<float>>::failure(
                    ErrorCode::invalid_extent);
            }
            score += static_cast<double>(values[row][index] * inverse) *
                     norm * projection;
        }
        attention_scores[row] = static_cast<float>(score);
    }
    const auto maximum = std::max(attention_scores[0], attention_scores[1]);
    std::array<float, 2> probabilities{
        std::exp(attention_scores[0] - maximum),
        std::exp(attention_scores[1] - maximum)};
    const auto denominator = probabilities[0] + probabilities[1];
    probabilities[0] /= denominator;
    probabilities[1] /= denominator;
    std::vector<float> hidden(width);
    for (std::size_t index = 0; index < width; ++index) {
        hidden[index] = round_bf16(
            probabilities[0] * values[0][index] +
            probabilities[1] * values[1][index]);
    }
    return normalized(hidden, weights.post_norm, epsilon);
}

Result<OfficialRoute> route_official_moe(
    std::span<const float> hidden, Bf16WeightView router,
    std::span<const float> correction, std::size_t top_k) {
    if (!valid(router) || router.cols != hidden.size() ||
        correction.size() != router.rows || !top_k || top_k > router.rows ||
        !finite(hidden) || !finite(correction)) {
        return Result<OfficialRoute>::failure(ErrorCode::invalid_extent);
    }
    std::vector<float> scores(router.rows);
    std::vector<float> adjusted(router.rows);
    for (std::size_t row = 0; row < router.rows; ++row) {
        double sum = 0.0;
        for (std::size_t column = 0; column < router.cols; ++column) {
            const auto weight = decode_bf16_word(
                router.values[row * router.cols + column]);
            if (!std::isfinite(weight)) {
                return Result<OfficialRoute>::failure(ErrorCode::invalid_extent);
            }
            sum += static_cast<double>(weight) * hidden[column];
        }
        scores[row] = 1.0F / (1.0F + std::exp(-static_cast<float>(sum)));
        adjusted[row] = scores[row] + correction[row];
    }
    std::vector<std::uint32_t> order(router.rows);
    std::iota(order.begin(), order.end(), 0U);
    std::partial_sort(
        order.begin(), order.begin() + static_cast<std::ptrdiff_t>(top_k),
        order.end(), [&adjusted](std::uint32_t left, std::uint32_t right) {
            return adjusted[left] != adjusted[right]
                ? adjusted[left] > adjusted[right]
                : left < right;
        });
    order.resize(top_k);
    std::vector<float> contributions(top_k);
    double total = 0.0;
    for (std::size_t index = 0; index < top_k; ++index)
        total += scores[order[index]];
    for (std::size_t index = 0; index < top_k; ++index)
        contributions[index] = static_cast<float>(scores[order[index]] / total);
    return Result<OfficialRoute>::success(
        {std::move(order), std::move(contributions), std::move(scores)});
}

Result<OfficialMoeResult> official_moe_cpu(
    const OfficialMoeInput& input, const OfficialMoeWeights& weights,
    const OfficialRoute& route, float epsilon, float situ_beta,
    std::optional<float> situ_linear_beta) {
    if (!valid_route(route) || !std::isfinite(situ_beta) || situ_beta <= 0.0F ||
        (situ_linear_beta &&
         (!std::isfinite(*situ_linear_beta) || *situ_linear_beta <= 0.0F))) {
        return Result<OfficialMoeResult>::failure(ErrorCode::invalid_state);
    }
    std::unordered_map<std::uint32_t, const OfficialExpertView*> experts;
    for (const auto& expert : weights.experts) {
        if (!experts.emplace(expert.expert_id, &expert).second) {
            return Result<OfficialMoeResult>::failure(ErrorCode::invalid_state);
        }
    }
    for (const auto id : route.expert_ids) {
        if (!experts.contains(id)) {
            return Result<OfficialMoeResult>::failure(ErrorCode::invalid_state);
        }
    }

    auto hidden = prepare_official_moe_input(input, weights, epsilon);
    if (!hidden) {
        return Result<OfficialMoeResult>::failure(hidden.error(), hidden.message());
    }
    auto latent = bf16_matvec(hidden.value(), weights.routed_down);
    auto shared = shared_ffn(hidden.value(), weights.shared, situ_beta,
                             situ_linear_beta);
    if (!latent || !shared) {
        return Result<OfficialMoeResult>::failure(ErrorCode::invalid_extent);
    }
    std::vector<std::vector<float>> expert_outputs;
    expert_outputs.reserve(route.expert_ids.size());
    for (const auto id : route.expert_ids) {
        auto output = expert_ffn(latent.value(), experts[id]->weights, situ_beta,
                                 situ_linear_beta);
        if (!output) {
            return Result<OfficialMoeResult>::failure(
                output.error(), output.message());
        }
        expert_outputs.push_back(std::move(output.value()));
    }
    std::vector<float> mixed(latent.value().size(), 0.0F);
    for (std::size_t slot = 0; slot < expert_outputs.size(); ++slot) {
        if (expert_outputs[slot].size() != mixed.size()) {
            return Result<OfficialMoeResult>::failure(ErrorCode::invalid_mxfp4);
        }
        for (std::size_t index = 0; index < mixed.size(); ++index) {
            mixed[index] += route.contributions[slot] * expert_outputs[slot][index];
        }
    }
    for (auto& value : mixed) value = round_bf16(value);
    auto routed_norm = normalized(mixed, weights.routed_norm, epsilon);
    if (!routed_norm) {
        return Result<OfficialMoeResult>::failure(
            routed_norm.error(), routed_norm.message());
    }
    auto routed = bf16_matvec(routed_norm.value(), weights.routed_up);
    if (!routed || routed.value().size() != shared.value().size() ||
        routed.value().size() != input.prefix_sum.size()) {
        return Result<OfficialMoeResult>::failure(ErrorCode::invalid_extent);
    }
    std::vector<float> combined(routed.value().size());
    std::vector<float> output(routed.value().size());
    for (std::size_t index = 0; index < combined.size(); ++index) {
        combined[index] = round_bf16(routed.value()[index] + shared.value()[index]);
        output[index] = round_bf16(
            round_bf16(input.prefix_sum[index]) + combined[index]);
    }
    return Result<OfficialMoeResult>::success({
        std::move(hidden.value()), std::move(latent.value()),
        std::move(expert_outputs), std::move(mixed), std::move(routed.value()),
        std::move(shared.value()), std::move(combined), std::move(output)});
}

}  // namespace k3x
