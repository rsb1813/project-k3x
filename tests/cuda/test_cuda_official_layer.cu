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
        return {{ones4, 601}, {residual_proj, 1, 4, 602}, {ones4, 603},
                {router, 2, 4, 604},
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

k3x::BackendOptions options(
    k3x::CudaWeightMode mode,
    k3x::CudaWeightValidationMode validation =
        k3x::CudaWeightValidationMode::per_call) {
    k3x::BackendOptions value;
    value.kind = k3x::BackendKind::cuda_custom;
    value.cuda_allocation = k3x::CudaAllocationMode::reused;
    value.cuda_weights = mode;
    value.cuda_batching = k3x::CudaBatchingMode::resident_grid;
    value.cuda_boundary = k3x::CudaBoundaryMode::moe_layer;
    value.cuda_resident_bytes = 8 * 1024 * 1024;
    value.cuda_weight_validation = validation;
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

int device_state() {
    const Fixture fixture;
    const auto zero = k3x::zero_official_kda_state(fixture.config);
    const auto oracle = k3x::official_layer_cpu(
        fixture.inputs, fixture.cpu_weights(), zero, fixture.config, 2, 4, 25);
    if (!oracle) return 20;
    auto backend = k3x::make_cuda_backend(options(
        k3x::CudaWeightMode::resident,
        k3x::CudaWeightValidationMode::admission));
    if (!backend) return 21;
    const auto first = k3x::official_layer_cuda(
        *backend.value(), std::span(fixture.inputs).first(1),
        fixture.cuda_weights(), zero, fixture.config, 2, 4, 25, 1,
        k3x::ProfilePhase::decode,
        {k3x::OfficialKdaStateMode::device_seed, {}});
    if (!first || !first.value().executed ||
        first.value().kda_state_published ||
        !first.value().kda_device_state.owner ||
        !first.value().kda_device_state.generation ||
        !first.value().kda_state.conv_q.empty() ||
        !first.value().kda_state.conv_k.empty() ||
        !first.value().kda_state.conv_v.empty() ||
        !first.value().kda_state.recurrent_v_first.empty() ||
        first.value().steps.size() != 1 ||
        !close(first.value().steps[0].output,
               oracle.value().steps[0].moe.output)) return 22;
    const k3x::OfficialKdaState no_host_state;
    const auto second = k3x::official_layer_cuda(
        *backend.value(), std::span(fixture.inputs).last(1),
        fixture.cuda_weights(), no_host_state, fixture.config, 2, 4, 25, 1,
        k3x::ProfilePhase::decode,
        {k3x::OfficialKdaStateMode::device_publish,
         first.value().kda_device_state});
    if (!second || !second.value().executed ||
        !second.value().kda_state_published ||
        second.value().kda_device_state.owner ||
        second.value().kda_device_state.generation ||
        second.value().steps.size() != 1 ||
        second.value().steps[0].route.expert_ids !=
            oracle.value().steps[1].route.expert_ids ||
        !close(second.value().steps[0].route.contributions,
               oracle.value().steps[1].route.contributions) ||
        !close(second.value().steps[0].output,
               oracle.value().steps[1].moe.output) ||
        second.value().kda_state.conv_q != oracle.value().kda.state.conv_q ||
        second.value().kda_state.conv_k != oracle.value().kda.state.conv_k ||
        second.value().kda_state.conv_v != oracle.value().kda.state.conv_v ||
        !close(second.value().kda_state.recurrent_v_first,
               oracle.value().kda.state.recurrent_v_first)) return 23;
    const auto stats = backend.value()->runtime_stats();
    const std::uint64_t state_bytes = 3 * 2 * 4 * sizeof(std::uint16_t) +
                                      2 * 2 * 2 * sizeof(float);
    if (stats.official_kda_state_h2d_bytes != state_bytes ||
        stats.official_kda_state_d2h_bytes != state_bytes ||
        stats.official_kda_device_state_seeds != 1 ||
        stats.official_kda_device_state_continuations != 1 ||
        stats.official_kda_device_state_publications != 1) return 24;
    return 0;
}

int device_state_failure_cleanup() {
    const Fixture fixture;
    const auto zero = k3x::zero_official_kda_state(fixture.config);
    auto backend = k3x::make_cuda_backend(options(
        k3x::CudaWeightMode::resident,
        k3x::CudaWeightValidationMode::admission));
    if (!backend) return 30;
    auto broken = fixture.cuda_weights();
    broken.moe.experts = {};
    const auto failed = k3x::official_layer_cuda(
        *backend.value(), std::span(fixture.inputs).first(1), broken, zero,
        fixture.config, 2, 4, 25, 1, k3x::ProfilePhase::decode,
        {k3x::OfficialKdaStateMode::device_seed, {}});
    const auto failed_stats = backend.value()->runtime_stats();
    if (failed || failed.error() != k3x::ErrorCode::invalid_extent ||
        failed_stats.official_kda_calls != 1 ||
        failed_stats.official_kda_device_state_seeds != 1 ||
        failed_stats.official_kda_device_state_invalidations != 1) return 31;

    const auto recovered = k3x::official_layer_cuda(
        *backend.value(), std::span(fixture.inputs).first(1),
        fixture.cuda_weights(), zero, fixture.config, 2, 4, 25, 1,
        k3x::ProfilePhase::decode,
        {k3x::OfficialKdaStateMode::device_seed, {}});
    if (!recovered || !recovered.value().kda_device_state.owner ||
        backend.value()->runtime_stats()
                .official_kda_device_state_invalidations != 1) return 32;
    const auto discarded = backend.value()->discard_official_kda_device_state(
        recovered.value().kda_device_state);
    const auto final_stats = backend.value()->runtime_stats();
    if (!discarded ||
        final_stats.official_kda_device_state_invalidations != 2) return 33;
    return 0;
}

int device_route_preparation() {
    const Fixture fixture;
    const auto zero = k3x::zero_official_kda_state(fixture.config);
    const auto oracle = k3x::official_layer_cpu(
        fixture.inputs, fixture.cpu_weights(), zero, fixture.config, 2, 4, 25);
    if (!oracle) return 40;
    auto backend = k3x::make_cuda_backend(options(
        k3x::CudaWeightMode::resident,
        k3x::CudaWeightValidationMode::admission));
    if (!backend) return 41;
    const auto first = k3x::official_layer_cuda(
        *backend.value(), std::span(fixture.inputs).first(1),
        fixture.cuda_weights(), zero, fixture.config, 2, 4, 25, 1,
        k3x::ProfilePhase::decode,
        {k3x::OfficialKdaStateMode::device_seed, {}},
        k3x::OfficialMoeRoutePreparationMode::device);
    if (!first || first.value().steps.size() != 1 ||
        first.value().steps[0].route.expert_ids !=
            oracle.value().steps[0].route.expert_ids ||
        !close(first.value().steps[0].route.contributions,
               oracle.value().steps[0].route.contributions) ||
        !close(first.value().steps[0].output,
               oracle.value().steps[0].moe.output))
        return 42;
    const k3x::OfficialKdaState no_host_state;
    const auto second = k3x::official_layer_cuda(
        *backend.value(), std::span(fixture.inputs).last(1),
        fixture.cuda_weights(), no_host_state, fixture.config, 2, 4, 25, 1,
        k3x::ProfilePhase::decode,
        {k3x::OfficialKdaStateMode::device_publish,
         first.value().kda_device_state},
        k3x::OfficialMoeRoutePreparationMode::device);
    if (!second || second.value().steps.size() != 1 ||
        second.value().steps[0].route.expert_ids !=
            oracle.value().steps[1].route.expert_ids ||
        !close(second.value().steps[0].route.contributions,
               oracle.value().steps[1].route.contributions) ||
        !close(second.value().steps[0].output,
               oracle.value().steps[1].moe.output) ||
        second.value().kda_state.conv_q != oracle.value().kda.state.conv_q ||
        second.value().kda_state.conv_k != oracle.value().kda.state.conv_k ||
        second.value().kda_state.conv_v != oracle.value().kda.state.conv_v ||
        !close(second.value().kda_state.recurrent_v_first,
               oracle.value().kda.state.recurrent_v_first))
        return 43;
    const auto stats = backend.value()->runtime_stats();
    if (stats.official_moe_route_prepare_calls != 2 ||
        stats.official_moe_route_prepare_kernel_launches != 4 ||
        stats.official_moe_router_logit_d2h_bytes !=
            2 * 2 * sizeof(float) ||
        stats.official_moe_prepared_seeds != 2 ||
        stats.official_moe_prepared_consumes != 2 ||
        stats.official_moe_prepared_discards != 0 ||
        stats.official_moe_prepared_invalidations != 0)
        return 44;
    return 0;
}

int device_route_failure_cleanup() {
    const Fixture fixture;
    const auto zero = k3x::zero_official_kda_state(fixture.config);
    auto backend = k3x::make_cuda_backend(options(
        k3x::CudaWeightMode::resident,
        k3x::CudaWeightValidationMode::admission));
    if (!backend) return 50;
    auto broken = fixture.cuda_weights();
    broken.moe.experts = {};
    const auto failed = k3x::official_layer_cuda(
        *backend.value(), std::span(fixture.inputs).first(1), broken, zero,
        fixture.config, 2, 4, 25, 1, k3x::ProfilePhase::decode,
        {k3x::OfficialKdaStateMode::device_seed, {}},
        k3x::OfficialMoeRoutePreparationMode::device);
    const auto stats = backend.value()->runtime_stats();
    if (failed || failed.error() != k3x::ErrorCode::invalid_extent ||
        stats.official_moe_prepared_seeds != 1 ||
        stats.official_moe_prepared_consumes != 0 ||
        stats.official_moe_prepared_discards != 1 ||
        stats.official_kda_device_state_invalidations != 1)
        return 51;
    broken = fixture.cuda_weights();
    broken.moe.correction = {};
    const auto route_failed = k3x::official_layer_cuda(
        *backend.value(), std::span(fixture.inputs).first(1), broken, zero,
        fixture.config, 2, 4, 25, 1, k3x::ProfilePhase::decode,
        {k3x::OfficialKdaStateMode::device_seed, {}},
        k3x::OfficialMoeRoutePreparationMode::device);
    const auto route_stats = backend.value()->runtime_stats();
    if (route_failed || route_failed.error() != k3x::ErrorCode::invalid_extent ||
        route_stats.official_moe_prepared_seeds != 2 ||
        route_stats.official_moe_prepared_consumes != 0 ||
        route_stats.official_moe_prepared_discards != 2 ||
        route_stats.official_kda_device_state_invalidations != 2)
        return 52;
    broken = fixture.cuda_weights();
    broken.moe_ffn.routed_down.tensor_id = 0;
    const auto ffn_failed = k3x::official_layer_cuda(
        *backend.value(), std::span(fixture.inputs).first(1), broken, zero,
        fixture.config, 2, 4, 25, 1, k3x::ProfilePhase::decode,
        {k3x::OfficialKdaStateMode::device_seed, {}},
        k3x::OfficialMoeRoutePreparationMode::device);
    const auto ffn_stats = backend.value()->runtime_stats();
    if (ffn_failed || ffn_failed.error() != k3x::ErrorCode::invalid_mxfp4 ||
        ffn_stats.official_moe_prepared_seeds != 3 ||
        ffn_stats.official_moe_prepared_consumes != 0 ||
        ffn_stats.official_moe_prepared_discards != 3 ||
        ffn_stats.official_kda_device_state_invalidations != 3)
        return 53;
    return 0;
}

}  // namespace

int main() {
    if (const auto result = run(k3x::CudaWeightMode::transient)) return result;
    if (const auto result = run(k3x::CudaWeightMode::resident)) return 10 + result;
    if (const auto result = device_state()) return result;
    if (const auto result = device_state_failure_cleanup()) return result;
    if (const auto result = device_route_preparation()) return result;
    return device_route_failure_cleanup();
}
