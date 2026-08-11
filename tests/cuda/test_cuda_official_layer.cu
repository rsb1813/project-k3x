// 공식 complete-layer CUDA 경계의 portable parity와 증분 상태를 검증합니다.
#include "k3x/official_layer.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>
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

bool close(std::span<const float> actual, std::span<const float> expected,
           float tolerance = 2.0e-2F) {
    if (actual.size() != expected.size()) return false;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (!std::isfinite(actual[index]) ||
            std::abs(actual[index] - expected[index]) > tolerance) return false;
    }
    return true;
}

struct Fixture {
    static constexpr std::size_t hidden_width = 4;
    static constexpr std::size_t latent_width = 32;
    static constexpr std::size_t intermediate_width = 32;

    k3x::OfficialKdaConfig config{4, 2, 2, 3, 1.0e-5F, -5.0F};
    std::array<k3x::OfficialLayerInput, 2> inputs{{
        {{0.5F, -1, 0.25F, 0.75F}, {0.75F, -0.25F, 0.5F, 0.25F}},
        {{-0.25F, 0.5F, 1, -0.5F}, {-0.5F, 0.75F, -0.25F, 0.5F}},
    }};
    std::array<std::uint16_t, 16> identity = words<16>({
        1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1});
    std::array<std::uint16_t, 16> k_proj = words<16>({
        0.5F, 0, 0.25F, 0, 0, 0.75F, 0, -0.25F,
        0.25F, 0, 1, 0, 0, -0.5F, 0, 0.5F});
    std::array<float, 12> q_conv{
        0.25F, -0.5F, 1, -0.25F, 0.5F, 0.75F,
        0.5F, 0.25F, -0.5F, -0.5F, 0.25F, 1.25F};
    std::array<float, 12> k_conv{};
    std::array<float, 12> v_conv{};
    std::array<std::uint16_t, 8> f_a = words<8>({
        0.5F, -0.25F, 0, 0.25F, 0, 0.5F, -0.5F, 0.25F});
    std::array<std::uint16_t, 8> f_b = words<8>({
        1, 0, 0.5F, -0.5F, 0, 1, -0.25F, 0.75F});
    std::array<float, 2> a_log{0, 0.5F};
    std::array<float, 4> dt_bias{0.25F, -0.5F, 0.75F, -0.25F};
    std::array<std::uint16_t, 8> beta = words<8>({
        0.5F, -0.25F, 0.25F, 0, 0, 0.5F, -0.5F, 0.25F});
    std::array<std::uint16_t, 16> gate = words<16>({
        0.5F, 0, 0, -0.25F, 0, 0.5F, 0.25F, 0,
        -0.25F, 0, 0.5F, 0, 0, 0.25F, 0, 0.75F});
    std::array<float, 2> o_norm{1, 1.5F};
    std::array<std::uint16_t, 4> ones4 = words<4>({1, 1, 1, 1});
    std::array<std::uint16_t, 4> residual_proj =
        words<4>({0.5F, -0.25F, 0.25F, 0});
    std::array<std::uint16_t, 8> router = words<8>({
        1, 0, 0, 0, 0, 1, 0, 0});
    std::array<float, 2> correction{0, 0.1F};
    std::array<std::uint16_t, latent_width * hidden_width> routed_down{};
    std::array<std::uint16_t, latent_width> routed_norm{};
    std::array<std::uint16_t, hidden_width * latent_width> routed_up{};
    std::array<std::uint16_t, 2 * hidden_width> shared_gate = words<8>({
        1, 0, 0, 0.5F, 0, 1, 0.5F, 0});
    std::array<std::uint16_t, 2 * hidden_width> shared_up = shared_gate;
    std::array<std::uint16_t, hidden_width * 2> shared_down = words<8>({
        1, 0, 0, 1, 0.5F, 0, 0, 0.5F});
    std::array<std::array<std::byte, intermediate_width * latent_width / 2>, 2>
        expert_gate{};
    std::array<std::array<std::byte, intermediate_width * latent_width / 2>, 2>
        expert_up{};
    std::array<std::array<std::byte, latent_width * intermediate_width / 2>, 2>
        expert_down{};
    std::array<std::array<std::byte, intermediate_width * latent_width / 32>, 2>
        expert_scales{};
    std::array<k3x::OfficialExpertView, 2> experts{};

    Fixture() {
        for (std::size_t index = 0; index < q_conv.size(); ++index) {
            k_conv[index] = q_conv[index] * 0.75F;
            v_conv[index] = q_conv[index] * -0.5F;
        }
        routed_norm.fill(bf16(1));
        for (std::size_t index = 0; index < hidden_width; ++index) {
            routed_down[index * hidden_width + index] = bf16(1);
            routed_up[index * latent_width + index] = bf16(1);
        }
        for (std::size_t expert = 0; expert < experts.size(); ++expert) {
            expert_gate[expert][0] = std::byte{0x12};
            expert_up[expert][0] = std::byte{0x21};
            expert_down[expert][0] = static_cast<std::byte>(0x12U + expert);
            expert_scales[expert].fill(std::byte{127});
            const auto matrix = [&](std::uint64_t id, const auto& packed) {
                return k3x::Mxfp4WeightView{
                    id, packed, expert_scales[expert], intermediate_width,
                    latent_width, 32};
            };
            experts[expert] = {
                static_cast<std::uint32_t>(expert),
                {matrix(100 + expert * 3, expert_gate[expert]),
                 matrix(101 + expert * 3, expert_up[expert]),
                 matrix(102 + expert * 3, expert_down[expert])}};
        }
    }

    k3x::OfficialKdaWeightsView cpu_kda() const {
        return {{identity, 4, 4}, {k_proj, 4, 4}, {identity, 4, 4},
                q_conv, k_conv, v_conv, {f_a, 2, 4}, {f_b, 4, 2},
                a_log, dt_bias, {beta, 2, 4}, {gate, 4, 4}, o_norm,
                {identity, 4, 4}};
    }

    k3x::OfficialKdaCudaView cuda_kda() const {
        return {{identity, 4, 4, 201}, {k_proj, 4, 4, 202},
                {identity, 4, 4, 203}, {301, q_conv, 4, 3},
                {302, k_conv, 4, 3}, {303, v_conv, 4, 3},
                {f_a, 2, 4, 204}, {f_b, 4, 2, 205},
                {304, a_log}, {305, dt_bias}, {beta, 2, 4, 206},
                {gate, 4, 4, 207}, {306, o_norm}, {identity, 4, 4, 208}};
    }

    k3x::OfficialMoeWeights cpu_moe() const {
        return {{ones4}, {residual_proj, 1, 4}, {ones4}, {router, 2, 4},
                correction, {routed_down, latent_width, hidden_width},
                {routed_norm}, {routed_up, hidden_width, latent_width},
                {{shared_gate, 2, hidden_width}, {shared_up, 2, hidden_width},
                 {shared_down, hidden_width, 2}}, experts};
    }

    k3x::OfficialMoeFfnView cuda_moe() const {
        return {{routed_down, latent_width, hidden_width, 401},
                {routed_norm, 402}, {routed_up, hidden_width, latent_width, 403},
                {{shared_gate, 2, hidden_width, 404},
                 {shared_up, 2, hidden_width, 405},
                 {shared_down, hidden_width, 2, 406}}};
    }

    k3x::OfficialLayerWeights cpu_weights() const {
        return {{ones4}, {residual_proj, 1, 4}, {ones4}, cpu_kda(), cpu_moe()};
    }

    k3x::OfficialLayerCudaWeights cuda_weights() const {
        return {{ones4}, {residual_proj, 1, 4}, {ones4}, cuda_kda(),
                cpu_moe(), cuda_moe()};
    }
};

k3x::BackendOptions options(k3x::CudaWeightMode mode) {
    k3x::BackendOptions value;
    value.kind = k3x::BackendKind::cuda_custom;
    value.cuda_allocation = k3x::CudaAllocationMode::reused;
    value.cuda_weights = mode;
    value.cuda_batching = k3x::CudaBatchingMode::resident_grid;
    value.cuda_boundary = k3x::CudaBoundaryMode::moe_layer;
    value.cuda_resident_bytes = 8 * 1024 * 1024;
    return value;
}

int run(k3x::CudaWeightMode mode) {
    const Fixture fixture;
    const auto zero = k3x::zero_official_kda_state(fixture.config);
    const auto oracle = k3x::official_layer_cpu(
        fixture.inputs, fixture.cpu_weights(), zero, fixture.config, 2, 4, 25);
    if (!oracle) return 1;
    auto backend = k3x::make_cuda_backend(options(mode));
    if (!backend) return 2;
    const auto full = k3x::official_layer_cuda(
        *backend.value(), fixture.inputs, fixture.cuda_weights(), zero,
        fixture.config, 2, 4, 25, 1, k3x::ProfilePhase::decode);
    if (!full || !full.value().executed || full.value().steps.size() != 2)
        return 3;
    for (std::size_t index = 0; index < 2; ++index) {
        if (full.value().steps[index].route.expert_ids !=
                oracle.value().steps[index].route.expert_ids ||
            !close(full.value().steps[index].route.contributions,
                   oracle.value().steps[index].route.contributions) ||
            !close(full.value().steps[index].output,
                   oracle.value().steps[index].moe.output)) return 4;
    }
    const auto stats_after_full = backend.value()->runtime_stats();
    const auto first = k3x::official_layer_cuda(
        *backend.value(), std::span(fixture.inputs).first(1),
        fixture.cuda_weights(), zero, fixture.config, 2, 4, 25, 1,
        k3x::ProfilePhase::decode);
    if (!first) return 5;
    const auto second = k3x::official_layer_cuda(
        *backend.value(), std::span(fixture.inputs).last(1),
        fixture.cuda_weights(), first.value().kda_state, fixture.config,
        2, 4, 25, 1, k3x::ProfilePhase::decode);
    if (!second || !close(first.value().steps[0].output,
                          full.value().steps[0].output) ||
        !close(second.value().steps[0].output, full.value().steps[1].output) ||
        second.value().kda_state.conv_q != full.value().kda_state.conv_q ||
        second.value().kda_state.conv_k != full.value().kda_state.conv_k ||
        second.value().kda_state.conv_v != full.value().kda_state.conv_v ||
        !close(second.value().kda_state.recurrent_v_first,
               full.value().kda_state.recurrent_v_first)) return 6;
    const auto stats_after_incremental = backend.value()->runtime_stats();
    if (mode == k3x::CudaWeightMode::resident &&
        (stats_after_incremental.weight_h2d_bytes !=
             stats_after_full.weight_h2d_bytes ||
         stats_after_incremental.weight_cache_hits <=
             stats_after_full.weight_cache_hits)) return 7;
    return 0;
}

}  // namespace

int main() {
    if (const auto result = run(k3x::CudaWeightMode::transient)) return result;
    if (const auto result = run(k3x::CudaWeightMode::resident)) return 10 + result;
    return 0;
}
