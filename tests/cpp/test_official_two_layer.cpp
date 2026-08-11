// 공식 두 레이어 portable 실행의 모델 순서와 독립 상태를 검증합니다.
#include "k3x/official_two_layer.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

namespace {

std::uint16_t bf16(float value) {
    auto bits = std::bit_cast<std::uint32_t>(value);
    bits += 0x7fffU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>(bits >> 16U);
}

template <std::size_t Size>
std::array<std::uint16_t, Size> words(const std::array<float, Size>& values) {
    std::array<std::uint16_t, Size> result{};
    for (std::size_t index = 0; index < Size; ++index) {
        result[index] = bf16(values[index]);
    }
    return result;
}

bool close(std::span<const float> left, std::span<const float> right) {
    if (left.size() != right.size()) return false;
    for (std::size_t index = 0; index < left.size(); ++index) {
        if (std::abs(left[index] - right[index]) > 1.0e-6F) return false;
    }
    return true;
}

bool same_state(const k3x::OfficialKdaState& left,
                const k3x::OfficialKdaState& right) {
    return left.conv_q == right.conv_q && left.conv_k == right.conv_k &&
           left.conv_v == right.conv_v &&
           close(left.recurrent_v_first, right.recurrent_v_first);
}

struct Fixture {
    k3x::OfficialKdaConfig config{2, 1, 2, 2, 1.0e-5F, -5.0F};
    std::array<std::uint16_t, 4> identity = words<4>({1, 0, 0, 1});
    std::array<float, 4> conv{0.5F, 0.25F, -0.25F, 0.5F};
    std::array<float, 2> a_log{0, 0.25F};
    std::array<float, 2> dt_bias{0.1F, -0.2F};
    std::array<float, 2> o_norm{1, 1};
    std::array<std::uint16_t, 2> beta = words<2>({0.25F, -0.25F});
    std::array<std::uint16_t, 2> norm = words<2>({1, 1});
    std::array<std::uint16_t, 2> residual = words<2>({0.25F, -0.125F});
    std::array<std::uint16_t, 4> router_one = words<4>({1, -0.5F, -0.5F, 1});
    std::array<std::uint16_t, 4> router_two = words<4>({-0.5F, 1, 1, -0.5F});
    std::array<float, 2> correction{0, 0};
    std::array<std::uint16_t, 4> dense = words<4>({0.25F, 0, 0, 0.25F});
    std::array<std::byte, 2> packed_one{std::byte{0x12}, std::byte{0x21}};
    std::array<std::byte, 2> packed_two{std::byte{0x21}, std::byte{0x12}};
    std::array<std::byte, 2> scales{std::byte{127}, std::byte{127}};
    std::array<k3x::OfficialExpertView, 2> experts_one{};
    std::array<k3x::OfficialExpertView, 2> experts_two{};

    k3x::OfficialKdaWeightsView kda() const {
        return {
            {identity, 2, 2}, {identity, 2, 2}, {identity, 2, 2},
            conv, conv, conv, {identity, 2, 2}, {identity, 2, 2},
            a_log, dt_bias, {beta, 1, 2}, {identity, 2, 2}, o_norm,
            {identity, 2, 2},
        };
    }

    k3x::Mxfp4WeightView matrix(
        std::uint64_t id, std::span<const std::byte> packed) const {
        return {id, packed, scales, 2, 2, 2};
    }

    k3x::OfficialLayerWeights layer(
        bool second, bool include_experts = true) {
        auto& experts = second ? experts_two : experts_one;
        experts = {{
                {0, {matrix(second ? 110 : 10, packed_one),
                     matrix(second ? 111 : 11, packed_two),
                     matrix(second ? 112 : 12, packed_one)}},
                {1, {matrix(second ? 120 : 20, packed_two),
                     matrix(second ? 121 : 21, packed_one),
                     matrix(second ? 122 : 22, packed_two)}},
            }};
        const std::span<const k3x::OfficialExpertView> expert_view =
            include_experts ? std::span<const k3x::OfficialExpertView>(experts)
                            : std::span<const k3x::OfficialExpertView>{};
        const k3x::OfficialMoeWeights moe{
            {norm}, {residual, 1, 2}, {norm},
            {second ? router_two : router_one, 2, 2}, correction,
            {dense, 2, 2}, {norm}, {dense, 2, 2},
            {{dense, 2, 2}, {dense, 2, 2}, {dense, 2, 2}},
            expert_view,
        };
        return {{norm}, {residual, 1, 2}, {norm}, kda(), std::move(moe)};
    }
};

}  // namespace

int main() {
    Fixture fixture;
    const std::array<k3x::OfficialTwoLayerWeights, 2> layers{{
        {1, fixture.layer(false)},
        {2, fixture.layer(true)},
    }};
    const std::array<k3x::OfficialLayerInput, 2> inputs{{
        {{0.5F, -0.25F}, {0.125F, 0.375F}},
        {{-0.375F, 0.625F}, {0.25F, -0.125F}},
    }};
    const auto zero = k3x::zero_official_kda_state(fixture.config);
    const std::array initial{zero, zero};
    const auto actual = k3x::official_two_layer_cpu(
        inputs, layers, initial, fixture.config, 1, 4, 25);
    if (!actual || actual.value().steps.size() != 4) return 1;

    std::array expected_states{zero, zero};
    std::array<std::vector<float>, 2> expected_outputs;
    std::size_t step = 0;
    for (std::size_t position = 0; position < inputs.size(); ++position) {
        auto current = inputs[position];
        for (std::size_t layer = 0; layer < layers.size(); ++layer) {
            const auto expected = k3x::official_layer_cpu(
                std::span(&current, 1), layers[layer].weights,
                expected_states[layer], fixture.config, 1, 4, 25);
            if (!expected) return 2;
            const auto& observed = actual.value().steps[step++];
            if (observed.position != position ||
                observed.layer_id != layer + 1 ||
                !close(observed.result.moe.output,
                       expected.value().steps[0].moe.output)) return 3;
            expected_states[layer] = expected.value().kda.state;
            current.hidden_input = expected.value().steps[0].moe.output;
        }
        expected_outputs[position] = current.hidden_input;
    }
    if (!same_state(actual.value().final_states[0], expected_states[0]) ||
        !same_state(actual.value().final_states[1], expected_states[1]) ||
        !close(actual.value().outputs[0], expected_outputs[0]) ||
        !close(actual.value().outputs[1], expected_outputs[1])) return 4;

    auto swapped = layers;
    std::swap(swapped[0], swapped[1]);
    if (k3x::official_two_layer_cpu(
            inputs, swapped, initial, fixture.config, 1, 4, 25)) return 5;

    auto missing = layers;
    missing[1] = {2, fixture.layer(true, false)};
    if (k3x::official_two_layer_cpu(
            inputs, missing, initial, fixture.config, 1, 4, 25)) return 6;

    auto bad_inputs = inputs;
    bad_inputs[1].block_source.pop_back();
    if (k3x::official_two_layer_cpu(
            bad_inputs, layers, initial, fixture.config, 1, 4, 25)) return 7;
    return 0;
}
