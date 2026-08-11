// 공식 두 레이어 CUDA wrapper의 정확도와 폐쇄형 전송 계약을 확인합니다.
#include "k3x/backend.hpp"
#include "k3x/official_two_layer.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace {

std::uint16_t bf16(float value) {
    auto bits = std::bit_cast<std::uint32_t>(value);
    const auto rounding = 0x7fffU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>((bits + rounding) >> 16U);
}

template <std::size_t N>
std::array<std::uint16_t, N> words(std::array<float, N> values) {
    std::array<std::uint16_t, N> result{};
    for (std::size_t index = 0; index < N; ++index) {
        result[index] = bf16(values[index]);
    }
    return result;
}

bool close(std::span<const float> lhs, std::span<const float> rhs,
           float tolerance = 2.0e-2F) {
    if (lhs.size() != rhs.size()) return false;
    for (std::size_t index = 0; index < lhs.size(); ++index) {
        if (std::abs(lhs[index] - rhs[index]) > tolerance) return false;
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
        return {{ones4, 601}, {residual_proj, 1, 4, 602}, {ones4, 603},
                {router, 2, 4, 604}, correction,
                {routed_down, latent_width, hidden_width}, {routed_norm},
                {routed_up, hidden_width, latent_width},
                {{shared_gate, 2, hidden_width},
                 {shared_up, 2, hidden_width},
                 {shared_down, hidden_width, 2}},
                experts};
    }

    k3x::OfficialMoeFfnView cuda_moe() const {
        return {{routed_down, latent_width, hidden_width, 401},
                {routed_norm, 402},
                {routed_up, hidden_width, latent_width, 403},
                {{shared_gate, 2, hidden_width, 404},
                 {shared_up, 2, hidden_width, 405},
                 {shared_down, hidden_width, 2, 406}}};
    }

    k3x::OfficialLayerWeights cpu_weights() const {
        return {{ones4}, {residual_proj, 1, 4}, {ones4}, cpu_kda(), cpu_moe()};
    }

    k3x::OfficialLayerCudaWeights cuda_weights() const {
        return {{ones4, 501}, {residual_proj, 1, 4, 502}, {ones4, 503},
                cuda_kda(), cpu_moe(), cuda_moe()};
    }
};

k3x::BackendOptions options() {
    k3x::BackendOptions value;
    value.kind = k3x::BackendKind::cuda_custom;
    value.cuda_allocation = k3x::CudaAllocationMode::reused;
    value.cuda_weights = k3x::CudaWeightMode::resident;
    value.cuda_batching = k3x::CudaBatchingMode::resident_grid;
    value.cuda_boundary = k3x::CudaBoundaryMode::moe_layer;
    value.cuda_resident_bytes = 8 * 1024 * 1024;
    value.cuda_weight_validation = k3x::CudaWeightValidationMode::admission;
    return value;
}

int run_mode(k3x::OfficialTwoLayerCudaMode mode) {
    const Fixture fixture;
    const auto zero = k3x::zero_official_kda_state(fixture.config);
    const std::array<k3x::OfficialTwoLayerWeights, 2> cpu_layers{{
        {1, fixture.cpu_weights()}, {2, fixture.cpu_weights()}}};
    const std::array<k3x::OfficialTwoLayerCudaWeights, 2> cuda_layers{{
        {1, fixture.cuda_weights()}, {2, fixture.cuda_weights()}}};
    const std::array<k3x::OfficialKdaState, 2> states{{zero, zero}};
    const auto oracle = k3x::official_two_layer_cpu(
        fixture.inputs, cpu_layers, states, fixture.config, 2, 4, 25);
    if (!oracle) return 1;
    auto backend = k3x::make_cuda_backend(options());
    if (!backend) return 2;
    const auto actual = k3x::official_two_layer_cuda(
        *backend.value(), fixture.inputs, cuda_layers, states, fixture.config,
        2, 4, 25, k3x::ProfilePhase::decode, mode);
    if (!actual || !actual.value().executed ||
        actual.value().steps.size() != 4) return 3;
    for (std::size_t index = 0; index < 4; ++index) {
        if (actual.value().steps[index].position != index / 2 ||
            actual.value().steps[index].layer_id != index % 2 + 1 ||
            actual.value().steps[index].route.expert_ids !=
                oracle.value().steps[index].result.route.expert_ids ||
            !close(actual.value().steps[index].route.contributions,
                   oracle.value().steps[index].result.route.contributions)) {
            return 4;
        }
    }
    for (std::size_t position = 0; position < 2; ++position) {
        if (!close(actual.value().outputs[position],
                   oracle.value().outputs[position])) return 5;
    }
    for (std::size_t layer = 0; layer < 2; ++layer) {
        const auto& lhs = actual.value().final_states[layer];
        const auto& rhs = oracle.value().final_states[layer];
        if (lhs.conv_q != rhs.conv_q || lhs.conv_k != rhs.conv_k ||
            lhs.conv_v != rhs.conv_v ||
            !close(lhs.recurrent_v_first, rhs.recurrent_v_first)) return 6;
    }
    const auto stats = backend.value()->runtime_stats();
    if (stats.official_kda_device_state_seeds != 2 ||
        stats.official_kda_device_state_publications != 2 ||
        stats.official_moe_route_prepare_calls != 4 ||
        stats.official_moe_router_logit_d2h_bytes != 4 * 2 * sizeof(float)) {
        return 7;
    }
    if (mode == k3x::OfficialTwoLayerCudaMode::device_closure) {
        if (stats.official_kda_output_d2h_bytes != 0 ||
            !actual.value().steps[0].output.empty() ||
            !actual.value().steps[2].output.empty() ||
            actual.value().telemetry.inter_layer_hidden_h2d_bytes != 0 ||
            actual.value().telemetry.inter_layer_hidden_d2h_bytes != 0 ||
            actual.value().telemetry.layer_front_calls != 4 ||
            actual.value().telemetry.layer_tail_calls != 4) return 8;
    } else if (stats.official_kda_output_d2h_bytes == 0 ||
               actual.value().steps[0].output.empty() ||
               actual.value().steps[2].output.empty() ||
               actual.value().telemetry.inter_layer_hidden_h2d_bytes !=
                   2 * Fixture::hidden_width * sizeof(float) ||
               actual.value().telemetry.inter_layer_hidden_d2h_bytes !=
                   2 * Fixture::hidden_width * sizeof(float) ||
               actual.value().telemetry.layer_front_calls != 0 ||
               actual.value().telemetry.layer_tail_calls != 0) {
        return 9;
    }
    if (actual.value().telemetry.state_h2d_bytes == 0 ||
        actual.value().telemetry.state_d2h_bytes == 0 ||
        actual.value().telemetry.router_logit_d2h_bytes !=
            4 * 2 * sizeof(float) ||
        actual.value().telemetry.final_hidden_d2h_bytes !=
            2 * Fixture::hidden_width * sizeof(float)) return 10;
    return 0;
}

int invalid_contracts() {
    const Fixture fixture;
    const auto zero = k3x::zero_official_kda_state(fixture.config);
    const std::array<k3x::OfficialKdaState, 2> states{{zero, zero}};
    std::array<k3x::OfficialTwoLayerCudaWeights, 2> layers{{
        {1, fixture.cuda_weights()}, {2, fixture.cuda_weights()}}};
    auto backend = k3x::make_cuda_backend(options());
    if (!backend) return 1;

    std::swap(layers[0], layers[1]);
    if (k3x::official_two_layer_cuda(
            *backend.value(), fixture.inputs, layers, states, fixture.config,
            2, 4, 25, k3x::ProfilePhase::decode,
            k3x::OfficialTwoLayerCudaMode::device_closure)) return 2;
    std::swap(layers[0], layers[1]);
    if (k3x::official_two_layer_cuda(
            *backend.value(), fixture.inputs, layers, states, fixture.config,
            2, 4, 25, k3x::ProfilePhase::decode,
            static_cast<k3x::OfficialTwoLayerCudaMode>(99))) return 3;

    auto missing = layers;
    missing[1].weights.moe.experts = {};
    if (k3x::official_two_layer_cuda(
            *backend.value(), fixture.inputs, missing, states, fixture.config,
            2, 4, 25, k3x::ProfilePhase::decode,
            k3x::OfficialTwoLayerCudaMode::device_closure)) return 4;
    const auto recovered = k3x::official_two_layer_cuda(
        *backend.value(), fixture.inputs, layers, states, fixture.config,
        2, 4, 25, k3x::ProfilePhase::decode,
        k3x::OfficialTwoLayerCudaMode::device_closure);
    if (!recovered) return 5;

    auto tiny_options = options();
    tiny_options.cuda_resident_bytes = 1;
    auto tiny = k3x::make_cuda_backend(tiny_options);
    if (!tiny) return 6;
    if (k3x::official_two_layer_cuda(
            *tiny.value(), fixture.inputs, layers, states, fixture.config,
            2, 4, 25, k3x::ProfilePhase::decode,
            k3x::OfficialTwoLayerCudaMode::device_closure)) return 7;
    return 0;
}

}  // namespace

int main() {
    if (const auto result = run_mode(
            k3x::OfficialTwoLayerCudaMode::host_round_trip)) return result;
    if (const auto result = run_mode(
            k3x::OfficialTwoLayerCudaMode::device_closure)) return result + 20;
    if (const auto result = invalid_contracts()) return result + 40;
    return 0;
}
