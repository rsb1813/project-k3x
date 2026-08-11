// 공식 KDA CUDA 경계의 portable parity, incremental state와 residency를 검증합니다.
#include "k3x/backend.hpp"
#include "k3x/official_kda.hpp"

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
           float tolerance = 2.0e-3F) {
    if (actual.size() != expected.size()) return false;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (!std::isfinite(actual[index]) ||
            std::abs(actual[index] - expected[index]) > tolerance) return false;
    }
    return true;
}

struct Fixture {
    k3x::OfficialKdaConfig cpu_config{4, 2, 2, 3, 1.0e-5F, -5.0F};
    k3x::OfficialKdaCudaConfig cuda_config{4, 2, 2, 3, 1.0e-5F, -5.0F};
    std::array<float, 8> hidden{0.5F, -1, 0.25F, 0.75F,
                                -0.25F, 0.5F, 1, -0.5F};
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

    Fixture() {
        for (std::size_t index = 0; index < q_conv.size(); ++index) {
            k_conv[index] = q_conv[index] * 0.75F;
            v_conv[index] = q_conv[index] * -0.5F;
        }
    }

    k3x::OfficialKdaWeightsView cpu_weights() const {
        return {{identity, 4, 4}, {k_proj, 4, 4}, {identity, 4, 4},
                q_conv, k_conv, v_conv, {f_a, 2, 4}, {f_b, 4, 2},
                a_log, dt_bias, {beta, 2, 4}, {gate, 4, 4}, o_norm,
                {identity, 4, 4}};
    }

    k3x::OfficialKdaCudaView cuda_weights() const {
        return {{identity, 4, 4, 101}, {k_proj, 4, 4, 102},
                {identity, 4, 4, 103}, {201, q_conv, 4, 3},
                {202, k_conv, 4, 3}, {203, v_conv, 4, 3},
                {f_a, 2, 4, 104}, {f_b, 4, 2, 105},
                {301, a_log}, {302, dt_bias}, {beta, 2, 4, 106},
                {gate, 4, 4, 107}, {303, o_norm}, {identity, 4, 4, 108}};
    }
};

k3x::BackendOptions options(k3x::CudaWeightMode mode, std::uint64_t capacity) {
    k3x::BackendOptions value;
    value.kind = k3x::BackendKind::cuda_custom;
    value.cuda_allocation = k3x::CudaAllocationMode::reused;
    value.cuda_weights = mode;
    value.cuda_batching = k3x::CudaBatchingMode::resident_grid;
    value.cuda_boundary = k3x::CudaBoundaryMode::moe_layer;
    value.cuda_resident_bytes = capacity;
    return value;
}

int run(k3x::CudaWeightMode mode) {
    const Fixture fixture;
    const auto zero = k3x::zero_official_kda_state(fixture.cpu_config);
    const auto oracle = k3x::official_kda_cpu(
        fixture.hidden, fixture.cpu_weights(), zero, fixture.cpu_config);
    if (!oracle) return 1;
    auto backend = k3x::make_cuda_backend(options(mode, 1 << 20));
    if (!backend) return 2;
    const k3x::OfficialKdaCudaStateView state{
        zero.conv_q, zero.conv_k, zero.conv_v, zero.recurrent_v_first};
    const auto first = backend.value()->official_kda(
        fixture.hidden, fixture.cuda_weights(), state, fixture.cuda_config,
        1, k3x::ProfilePhase::decode);
    if (!first || !first.value().executed ||
        !close(first.value().output, oracle.value().output) ||
        first.value().conv_q != oracle.value().state.conv_q ||
        first.value().conv_k != oracle.value().state.conv_k ||
        first.value().conv_v != oracle.value().state.conv_v ||
        !close(first.value().recurrent_v_first,
               oracle.value().state.recurrent_v_first)) return 3;
    const auto first_stats = backend.value()->runtime_stats();
    const auto incremental_a = backend.value()->official_kda(
        std::span(fixture.hidden).first(4), fixture.cuda_weights(), state,
        fixture.cuda_config, 1, k3x::ProfilePhase::decode);
    if (!incremental_a) return 4;
    const k3x::OfficialKdaCudaStateView incremental_state{
        incremental_a.value().conv_q, incremental_a.value().conv_k,
        incremental_a.value().conv_v, incremental_a.value().recurrent_v_first};
    const auto incremental_b = backend.value()->official_kda(
        std::span(fixture.hidden).last(4), fixture.cuda_weights(),
        incremental_state, fixture.cuda_config,
        1, k3x::ProfilePhase::decode);
    if (!incremental_b) return 5;
    std::vector<float> incremental = incremental_a.value().output;
    incremental.insert(incremental.end(), incremental_b.value().output.begin(),
                       incremental_b.value().output.end());
    if (!close(incremental, first.value().output) ||
        incremental_b.value().conv_q != first.value().conv_q ||
        incremental_b.value().conv_k != first.value().conv_k ||
        incremental_b.value().conv_v != first.value().conv_v ||
        !close(incremental_b.value().recurrent_v_first,
               first.value().recurrent_v_first)) return 6;
    const auto second_stats = backend.value()->runtime_stats();
    const std::uint64_t one_state_bytes = 3 * 2 * 4 * sizeof(std::uint16_t) +
                                          2 * 2 * 2 * sizeof(float);
    if (first_stats.official_kda_calls != 1 ||
        second_stats.official_kda_calls != 3 ||
        first_stats.official_kda_state_h2d_bytes != one_state_bytes ||
        first_stats.official_kda_state_d2h_bytes != one_state_bytes ||
        first_stats.official_kda_output_d2h_bytes !=
            std::span(fixture.hidden).size_bytes())
        return 7;
    if (mode == k3x::CudaWeightMode::resident &&
        (second_stats.weight_h2d_bytes != first_stats.weight_h2d_bytes ||
         second_stats.weight_cache_hits <= first_stats.weight_cache_hits)) return 8;
    return 0;
}

int invalid() {
    Fixture fixture;
    const auto zero = k3x::zero_official_kda_state(fixture.cpu_config);
    const k3x::OfficialKdaCudaStateView state{
        zero.conv_q, zero.conv_k, zero.conv_v, zero.recurrent_v_first};
    auto backend = k3x::make_cuda_backend(
        options(k3x::CudaWeightMode::resident, 1 << 20));
    if (!backend) return 20;
    auto weights = fixture.cuda_weights();
    weights.k_proj.tensor_id = weights.q_proj.tensor_id;
    if (backend.value()->official_kda(fixture.hidden, weights, state,
                                      fixture.cuda_config, 1,
                                      k3x::ProfilePhase::decode)) return 21;
    auto config = fixture.cuda_config;
    config.head_dim = 3;
    if (backend.value()->official_kda(fixture.hidden, fixture.cuda_weights(),
                                      state, config, 1,
                                      k3x::ProfilePhase::decode)) return 22;
    auto tiny = k3x::make_cuda_backend(
        options(k3x::CudaWeightMode::resident, 1));
    if (!tiny) return 23;
    if (tiny.value()->official_kda(fixture.hidden, fixture.cuda_weights(),
                                   state, fixture.cuda_config, 1,
                                   k3x::ProfilePhase::decode)) return 24;
    auto invalid_conv = zero.conv_q;
    invalid_conv[0] = 0x7f80U;
    const k3x::OfficialKdaCudaStateView invalid_state{
        invalid_conv, zero.conv_k, zero.conv_v, zero.recurrent_v_first};
    const auto before = backend.value()->runtime_stats();
    if (backend.value()->official_kda(fixture.hidden, fixture.cuda_weights(),
                                      invalid_state, fixture.cuda_config, 1,
                                      k3x::ProfilePhase::decode)) return 25;
    const auto after = backend.value()->runtime_stats();
    if (after.official_kda_calls != before.official_kda_calls ||
        after.official_kda_kernel_launches !=
            before.official_kda_kernel_launches) return 26;
    auto short_a_log = fixture.cuda_weights();
    short_a_log.a_log.values = std::span(fixture.a_log).first(1);
    if (backend.value()->official_kda(fixture.hidden, short_a_log, state,
                                      fixture.cuda_config, 1,
                                      k3x::ProfilePhase::decode)) return 27;
    return 0;
}

}  // namespace

int main() {
    if (const auto result = run(k3x::CudaWeightMode::transient)) return result;
    if (const auto result = run(k3x::CudaWeightMode::resident)) return 10 + result;
    return invalid();
}
