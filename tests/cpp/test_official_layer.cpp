// 공식 portable complete-layer의 graph 순서와 full/incremental parity를 검증합니다.
#include "k3x/official_layer.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <span>
#include <string_view>
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
    for (std::size_t index = 0; index < Size; ++index) result[index] = bf16(values[index]);
    return result;
}

bool close(std::span<const float> left, std::span<const float> right,
           float tolerance = 1.0e-6F) {
    if (left.size() != right.size()) return false;
    for (std::size_t index = 0; index < left.size(); ++index) {
        if (std::abs(left[index] - right[index]) > tolerance) return false;
    }
    return true;
}

bool same_step(const k3x::OfficialLayerStepResult& left,
               const k3x::OfficialLayerStepResult& right) {
    if (!close(left.self_attention_residual, right.self_attention_residual) ||
        !close(left.input_normalized, right.input_normalized) ||
        !close(left.post_kda_prefix, right.post_kda_prefix) ||
        !close(left.mlp_attention_residual, right.mlp_attention_residual) ||
        !close(left.normalized_moe_input, right.normalized_moe_input) ||
        left.route.expert_ids != right.route.expert_ids ||
        !close(left.route.contributions, right.route.contributions) ||
        !close(left.moe.hidden, right.moe.hidden) ||
        !close(left.moe.latent, right.moe.latent) ||
        !close(left.moe.mixed_latent, right.moe.mixed_latent) ||
        !close(left.moe.routed, right.moe.routed) ||
        !close(left.moe.shared, right.moe.shared) ||
        !close(left.moe.combined, right.moe.combined) ||
        !close(left.moe.output, right.moe.output) ||
        left.moe.expert_outputs.size() != right.moe.expert_outputs.size()) {
        return false;
    }
    for (std::size_t index = 0; index < left.moe.expert_outputs.size(); ++index) {
        if (!close(left.moe.expert_outputs[index], right.moe.expert_outputs[index])) {
            return false;
        }
    }
    return true;
}

void print_vector(std::span<const float> values) {
    std::cout << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index) std::cout << ',';
        std::cout << values[index];
    }
    std::cout << ']';
}

}  // namespace

int main(int argc, char** argv) {
    const k3x::OfficialKdaConfig config{4, 2, 2, 3, 1.0e-5F, -5.0F};
    const auto identity4 = words<16>({
        1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1});
    const auto k_proj = words<16>({
        0.5F, 0, 0.25F, 0, 0, 0.75F, 0, -0.25F,
        0.25F, 0, 1, 0, 0, -0.5F, 0, 0.5F});
    const std::array<float, 12> q_conv{
        0.25F, -0.5F, 1, -0.25F, 0.5F, 0.75F,
        0.5F, 0.25F, -0.5F, -0.5F, 0.25F, 1.25F};
    std::array<float, 12> k_conv{};
    std::array<float, 12> v_conv{};
    for (std::size_t index = 0; index < q_conv.size(); ++index) {
        k_conv[index] = q_conv[index] * 0.75F;
        v_conv[index] = q_conv[index] * -0.5F;
    }
    const auto f_a = words<8>({
        0.5F, -0.25F, 0, 0.25F, 0, 0.5F, -0.5F, 0.25F});
    const auto f_b = words<8>({1, 0, 0.5F, -0.5F, 0, 1, -0.25F, 0.75F});
    const std::array<float, 2> a_log{0, 0.5F};
    const std::array<float, 4> dt_bias{0.25F, -0.5F, 0.75F, -0.25F};
    const auto beta = words<8>({
        0.5F, -0.25F, 0.25F, 0, 0, 0.5F, -0.5F, 0.25F});
    const auto gate = words<16>({
        0.5F, 0, 0, -0.25F, 0, 0.5F, 0.25F, 0,
        -0.25F, 0, 0.5F, 0, 0, 0.25F, 0, 0.75F});
    const std::array<float, 2> o_norm{1, 1.5F};
    const k3x::OfficialKdaWeightsView kda{
        {identity4, 4, 4}, {k_proj, 4, 4}, {identity4, 4, 4},
        q_conv, k_conv, v_conv, {f_a, 2, 4}, {f_b, 4, 2},
        a_log, dt_bias, {beta, 2, 4}, {gate, 4, 4}, o_norm,
        {identity4, 4, 4}};

    const auto ones4 = words<4>({1, 1, 1, 1});
    const auto residual_proj = words<4>({0.5F, -0.25F, 0.25F, 0});
    const auto router = words<12>({
        1, 0, 0, 0, 0, 1, 0, 0, -1, 0.5F, 0.25F, 0});
    const std::array<float, 3> correction{0, 0.1F, -0.05F};
    const auto down2x4 = words<8>({1, 0, 0.5F, 0, 0, 1, 0, 0.5F});
    const auto routed_norm = words<2>({1, 1});
    const auto up4x2 = words<8>({1, 0, 0, 1, 0.5F, 0, 0, 0.5F});
    const auto shared2x4 = words<8>({1, 0, 0, 0.5F, 0, 1, 0.5F, 0});
    const auto shared_down = words<8>({1, 0, 0, 1, 0.5F, 0, 0, 0.5F});
    const std::array<std::byte, 2> e0_gate{std::byte{0x12}, std::byte{0x21}};
    const std::array<std::byte, 2> e0_up{std::byte{0x22}, std::byte{0x31}};
    const std::array<std::byte, 2> e0_down{std::byte{0x12}, std::byte{0x21}};
    const std::array<std::byte, 2> e1_gate{std::byte{0x21}, std::byte{0x12}};
    const std::array<std::byte, 2> e1_up{std::byte{0x12}, std::byte{0x23}};
    const std::array<std::byte, 2> e1_down{std::byte{0x12}, std::byte{0x12}};
    const std::array<std::byte, 2> scales{std::byte{127}, std::byte{127}};
    const auto matrix = [](std::uint64_t id, const auto& packed, const auto& scale) {
        return k3x::Mxfp4WeightView{id, packed, scale, 2, 2, 2};
    };
    const std::vector<k3x::OfficialExpertView> experts{
        {0, {matrix(10, e0_gate, scales), matrix(11, e0_up, scales),
             matrix(12, e0_down, scales)}},
        {1, {matrix(20, e1_gate, scales), matrix(21, e1_up, scales),
             matrix(22, e1_down, scales)}},
    };
    const k3x::OfficialMoeWeights moe{
        {ones4}, {residual_proj, 1, 4}, {ones4}, {router, 3, 4}, correction,
        {down2x4, 2, 4}, {routed_norm}, {up4x2, 4, 2},
        {{shared2x4, 2, 4}, {shared2x4, 2, 4}, {shared_down, 4, 2}},
        experts};
    const k3x::OfficialLayerWeights weights{
        {ones4}, {residual_proj, 1, 4}, {ones4}, kda, moe};
    const std::array<k3x::OfficialLayerInput, 2> inputs{{
        {{0.5F, -1, 0.25F, 0.75F}, {0.75F, -0.25F, 0.5F, 0.25F}},
        {{-0.25F, 0.5F, 1, -0.5F}, {-0.5F, 0.75F, -0.25F, 0.5F}},
    }};
    const auto zero = k3x::zero_official_kda_state(config);
    const auto original_zero = zero;
    const auto full = k3x::official_layer_cpu(inputs, weights, zero, config, 2, 4, 25);
    const auto first = k3x::official_layer_cpu(
        std::span(inputs).first(1), weights, zero, config, 2, 4, 25);
    if (!full || !first) return 1;
    const auto second = k3x::official_layer_cpu(
        std::span(inputs).last(1), weights, first.value().kda.state,
        config, 2, 4, 25);
    if (!second || full.value().steps.size() != 2) return 2;
    if (!same_step(full.value().steps[0], first.value().steps[0]) ||
        !same_step(full.value().steps[1], second.value().steps[0]) ||
        full.value().kda.state.conv_q != second.value().kda.state.conv_q ||
        full.value().kda.state.conv_k != second.value().kda.state.conv_k ||
        full.value().kda.state.conv_v != second.value().kda.state.conv_v ||
        !close(full.value().kda.state.recurrent_v_first,
               second.value().kda.state.recurrent_v_first) ||
        zero.conv_q != original_zero.conv_q ||
        zero.conv_k != original_zero.conv_k ||
        zero.conv_v != original_zero.conv_v ||
        zero.recurrent_v_first != original_zero.recurrent_v_first) return 3;

    if (argc == 2 && std::string_view(argv[1]) == "--dump") {
        std::cout.precision(9);
        std::cout << "{\"steps\":[";
        for (std::size_t index = 0; index < full.value().steps.size(); ++index) {
            if (index) std::cout << ',';
            const auto& step = full.value().steps[index];
            std::cout << "{\"self_attention_residual\":";
            print_vector(step.self_attention_residual);
            std::cout << ",\"input_normalized\":"; print_vector(step.input_normalized);
            std::cout << ",\"post_kda_prefix\":"; print_vector(step.post_kda_prefix);
            std::cout << ",\"mlp_attention_residual\":";
            print_vector(step.mlp_attention_residual);
            std::cout << ",\"normalized_moe_input\":";
            print_vector(step.normalized_moe_input);
            std::cout << ",\"expert_ids\":[";
            for (std::size_t slot = 0; slot < step.route.expert_ids.size(); ++slot) {
                if (slot) std::cout << ',';
                std::cout << step.route.expert_ids[slot];
            }
            std::cout << "],\"contributions\":"; print_vector(step.route.contributions);
            std::cout << ",\"output\":"; print_vector(step.moe.output);
            std::cout << '}';
        }
        std::cout << "],\"recurrent_v_first\":";
        print_vector(full.value().kda.state.recurrent_v_first);
        std::cout << "}\n";
        return 0;
    }

    auto bad = inputs;
    bad[0].block_source.pop_back();
    if (k3x::official_layer_cpu(bad, weights, zero, config, 2, 4, 25)) return 4;
    return 0;
}
