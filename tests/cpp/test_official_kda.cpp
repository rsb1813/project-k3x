// 공식 KDA portable oracle의 full/incremental state와 모든 경계를 검증합니다.
#include "k3x/official_kda.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
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
    for (std::size_t index = 0; index < Size; ++index) {
        result[index] = bf16(values[index]);
    }
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

bool joined_close(std::span<const float> full,
                  std::span<const float> first,
                  std::span<const float> second,
                  float tolerance = 1.0e-6F) {
    std::vector<float> joined(first.begin(), first.end());
    joined.insert(joined.end(), second.begin(), second.end());
    return close(full, joined, tolerance);
}

void print_vector(std::span<const float> values) {
    std::cout << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index) std::cout << ',';
        std::cout << values[index];
    }
    std::cout << ']';
}

void print_words(std::span<const std::uint16_t> values) {
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
    const auto identity = words<16>({
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
    const auto f_b = words<8>({
        1, 0, 0.5F, -0.5F, 0, 1, -0.25F, 0.75F});
    const std::array<float, 2> a_log{0, 0.5F};
    const std::array<float, 4> dt_bias{0.25F, -0.5F, 0.75F, -0.25F};
    const auto b_proj = words<8>({
        0.5F, -0.25F, 0.25F, 0, 0, 0.5F, -0.5F, 0.25F});
    const auto g_proj = words<16>({
        0.5F, 0, 0, -0.25F, 0, 0.5F, 0.25F, 0,
        -0.25F, 0, 0.5F, 0, 0, 0.25F, 0, 0.75F});
    const std::array<float, 2> o_norm{1, 1.5F};
    const k3x::OfficialKdaWeightsView weights{
        {identity, 4, 4}, {k_proj, 4, 4}, {identity, 4, 4},
        q_conv, k_conv, v_conv,
        {f_a, 2, 4}, {f_b, 4, 2}, a_log, dt_bias,
        {b_proj, 2, 4}, {g_proj, 4, 4}, o_norm, {identity, 4, 4}};
    const std::array<float, 8> hidden{
        0.5F, -1, 0.25F, 0.75F, -0.25F, 0.5F, 1, -0.5F};
    const auto zero = k3x::zero_official_kda_state(config);
    const auto original_zero = zero;
    const auto full = k3x::official_kda_cpu(hidden, weights, zero, config);
    const auto first = k3x::official_kda_cpu(
        std::span(hidden).first(4), weights, zero, config);
    if (!full || !first) return 1;
    const auto second = k3x::official_kda_cpu(
        std::span(hidden).last(4), weights, first.value().state, config);
    if (!second) return 2;
    std::vector<float> incremental = first.value().output;
    incremental.insert(incremental.end(), second.value().output.begin(),
                       second.value().output.end());
    if (!close(full.value().output, incremental, 0.0F) ||
        full.value().state.conv_q != second.value().state.conv_q ||
        full.value().state.conv_k != second.value().state.conv_k ||
        full.value().state.conv_v != second.value().state.conv_v ||
        !close(full.value().state.recurrent_v_first,
               second.value().state.recurrent_v_first) ||
        zero.conv_q != original_zero.conv_q ||
        zero.conv_k != original_zero.conv_k ||
        zero.conv_v != original_zero.conv_v ||
        zero.recurrent_v_first != original_zero.recurrent_v_first) return 3;
    const auto& all = full.value().boundaries;
    const auto& one = first.value().boundaries;
    const auto& two = second.value().boundaries;
    if (!joined_close(all.projected_q, one.projected_q, two.projected_q, 0.0F) ||
        !joined_close(all.projected_k, one.projected_k, two.projected_k, 0.0F) ||
        !joined_close(all.projected_v, one.projected_v, two.projected_v, 0.0F) ||
        !joined_close(all.convolved_q, one.convolved_q, two.convolved_q, 0.0F) ||
        !joined_close(all.convolved_k, one.convolved_k, two.convolved_k, 0.0F) ||
        !joined_close(all.convolved_v, one.convolved_v, two.convolved_v, 0.0F) ||
        !joined_close(all.q, one.q, two.q, 0.0F) ||
        !joined_close(all.k, one.k, two.k, 0.0F) ||
        !joined_close(all.v, one.v, two.v, 0.0F) ||
        !joined_close(all.log_decay, one.log_decay, two.log_decay) ||
        !joined_close(all.beta, one.beta, two.beta) ||
        !joined_close(all.recurrent_output, one.recurrent_output,
                      two.recurrent_output) ||
        !joined_close(all.gated, one.gated, two.gated, 0.0F)) return 7;

    if (argc == 2 && std::string_view(argv[1]) == "--dump") {
        const auto& result = full.value();
        std::cout.precision(9);
        std::cout << "{\"projected_q\":"; print_vector(result.boundaries.projected_q);
        std::cout << ",\"projected_k\":"; print_vector(result.boundaries.projected_k);
        std::cout << ",\"projected_v\":"; print_vector(result.boundaries.projected_v);
        std::cout << ",\"convolved_q\":"; print_vector(result.boundaries.convolved_q);
        std::cout << ",\"convolved_k\":"; print_vector(result.boundaries.convolved_k);
        std::cout << ",\"convolved_v\":"; print_vector(result.boundaries.convolved_v);
        std::cout << ",\"q\":"; print_vector(result.boundaries.q);
        std::cout << ",\"k\":"; print_vector(result.boundaries.k);
        std::cout << ",\"v\":"; print_vector(result.boundaries.v);
        std::cout << ",\"log_decay\":"; print_vector(result.boundaries.log_decay);
        std::cout << ",\"beta\":"; print_vector(result.boundaries.beta);
        std::cout << ",\"recurrent_output\":"; print_vector(result.boundaries.recurrent_output);
        std::cout << ",\"gated\":"; print_vector(result.boundaries.gated);
        std::cout << ",\"output\":"; print_vector(result.output);
        std::cout << ",\"conv_q\":"; print_words(result.state.conv_q);
        std::cout << ",\"conv_k\":"; print_words(result.state.conv_k);
        std::cout << ",\"conv_v\":"; print_words(result.state.conv_v);
        std::cout << ",\"recurrent_v_first\":";
        print_vector(result.state.recurrent_v_first);
        std::cout << "}\n";
        return 0;
    }

    auto bad_weights = weights;
    bad_weights.a_log = std::span(a_log).first(1);
    if (k3x::official_kda_cpu(hidden, bad_weights, zero, config)) return 8;
    bad_weights = weights;
    const std::array<float, 2> bad_a_log{0, std::numeric_limits<float>::infinity()};
    bad_weights.a_log = bad_a_log;
    if (k3x::official_kda_cpu(hidden, bad_weights, zero, config)) return 9;
    auto bad_state = zero;
    bad_state.recurrent_v_first.push_back(0);
    if (k3x::official_kda_cpu(hidden, weights, bad_state, config)) return 10;
    return 0;
}
