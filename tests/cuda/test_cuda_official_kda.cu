// 공식 KDA CUDA 경계의 portable parity, incremental state와 residency를 검증합니다.
#include "k3x/backend.hpp"
#include "k3x/official_kda.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
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

k3x::BackendOptions options(
    k3x::CudaWeightMode mode, std::uint64_t capacity,
    k3x::CudaWeightValidationMode validation =
        k3x::CudaWeightValidationMode::per_call) {
    k3x::BackendOptions value;
    value.kind = k3x::BackendKind::cuda_custom;
    value.cuda_allocation = k3x::CudaAllocationMode::reused;
    value.cuda_weights = mode;
    value.cuda_batching = k3x::CudaBatchingMode::resident_grid;
    value.cuda_boundary = k3x::CudaBoundaryMode::moe_layer;
    value.cuda_resident_bytes = capacity;
    value.cuda_weight_validation = validation;
    return value;
}

constexpr std::uint64_t kImmutableViews = 14;
constexpr std::uint64_t kImmutableBytes = 384;

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
        second_stats.immutable_validation_scans != 3 * kImmutableViews ||
        second_stats.immutable_validation_hits != 0 ||
        second_stats.immutable_validation_bytes != 3 * kImmutableBytes ||
        second_stats.immutable_validation_nanoseconds == 0 ||
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

int admission_validation() {
    const Fixture fixture;
    const auto zero = k3x::zero_official_kda_state(fixture.cpu_config);
    const k3x::OfficialKdaCudaStateView state{
        zero.conv_q, zero.conv_k, zero.conv_v, zero.recurrent_v_first};
    auto transient = k3x::make_cuda_backend(options(
        k3x::CudaWeightMode::transient, 0,
        k3x::CudaWeightValidationMode::admission));
    if (!transient) return 29;
    const auto unsupported = transient.value()->official_kda(
        fixture.hidden, fixture.cuda_weights(), state, fixture.cuda_config,
        1, k3x::ProfilePhase::decode);
    const auto unsupported_stats = transient.value()->runtime_stats();
    if (unsupported || unsupported.error() != k3x::ErrorCode::invalid_extent ||
        unsupported_stats.immutable_validation_scans != 0 ||
        unsupported_stats.official_kda_calls != 0 ||
        unsupported_stats.official_kda_kernel_launches != 0) return 29;
    auto backend = k3x::make_cuda_backend(options(
        k3x::CudaWeightMode::resident, 1 << 20,
        k3x::CudaWeightValidationMode::admission));
    if (!backend) return 30;
    const auto first = backend.value()->official_kda(
        fixture.hidden, fixture.cuda_weights(), state, fixture.cuda_config,
        1, k3x::ProfilePhase::decode);
    if (!first) return 31;
    const auto first_stats = backend.value()->runtime_stats();
    if (first_stats.immutable_validation_scans != kImmutableViews ||
        first_stats.immutable_validation_hits != 0 ||
        first_stats.immutable_validation_bytes != kImmutableBytes ||
        first_stats.immutable_validation_nanoseconds == 0) return 32;
    const auto second = backend.value()->official_kda(
        fixture.hidden, fixture.cuda_weights(), state, fixture.cuda_config,
        1, k3x::ProfilePhase::decode);
    if (!second || second.value().output != first.value().output ||
        second.value().conv_q != first.value().conv_q ||
        second.value().conv_k != first.value().conv_k ||
        second.value().conv_v != first.value().conv_v ||
        second.value().recurrent_v_first != first.value().recurrent_v_first) {
        return 33;
    }
    const auto second_stats = backend.value()->runtime_stats();
    if (second_stats.immutable_validation_scans != kImmutableViews ||
        second_stats.immutable_validation_hits != kImmutableViews ||
        second_stats.immutable_validation_bytes != kImmutableBytes) return 34;

    const Fixture different_allocation;
    const auto rejected = backend.value()->official_kda(
        different_allocation.hidden, different_allocation.cuda_weights(), state,
        different_allocation.cuda_config, 1, k3x::ProfilePhase::decode);
    const auto rejected_stats = backend.value()->runtime_stats();
    if (rejected || rejected.error() != k3x::ErrorCode::invalid_extent ||
        rejected_stats.immutable_validation_scans !=
            second_stats.immutable_validation_scans ||
        rejected_stats.immutable_validation_hits !=
            second_stats.immutable_validation_hits ||
        rejected_stats.immutable_validation_bytes !=
            second_stats.immutable_validation_bytes ||
        rejected_stats.resident_weight_bytes !=
            second_stats.resident_weight_bytes ||
        rejected_stats.official_kda_calls != second_stats.official_kda_calls ||
        rejected_stats.official_kda_kernel_launches !=
            second_stats.official_kda_kernel_launches) return 35;

    Fixture nonfinite;
    nonfinite.q_conv[0] = std::numeric_limits<float>::quiet_NaN();
    auto atomic = k3x::make_cuda_backend(options(
        k3x::CudaWeightMode::resident, 1 << 20,
        k3x::CudaWeightValidationMode::admission));
    if (!atomic) return 36;
    const auto failed = atomic.value()->official_kda(
        nonfinite.hidden, nonfinite.cuda_weights(), state,
        nonfinite.cuda_config, 1, k3x::ProfilePhase::decode);
    const auto failed_stats = atomic.value()->runtime_stats();
    if (failed || failed.error() != k3x::ErrorCode::invalid_extent ||
        failed_stats.immutable_validation_scans != kImmutableViews ||
        failed_stats.immutable_validation_hits != 0 ||
        failed_stats.immutable_validation_bytes != kImmutableBytes ||
        failed_stats.resident_weight_bytes != 0 ||
        failed_stats.official_kda_calls != 0 ||
        failed_stats.official_kda_kernel_launches != 0) return 37;
    nonfinite.q_conv[0] = 0.25F;
    const auto recovered = atomic.value()->official_kda(
        nonfinite.hidden, nonfinite.cuda_weights(), state,
        nonfinite.cuda_config, 1, k3x::ProfilePhase::decode);
    const auto recovered_stats = atomic.value()->runtime_stats();
    if (!recovered ||
        recovered_stats.immutable_validation_scans != 2 * kImmutableViews ||
        recovered_stats.immutable_validation_hits != 0 ||
        recovered_stats.immutable_validation_bytes != 2 * kImmutableBytes) {
        return 38;
    }

    Fixture nonfinite_bf16;
    nonfinite_bf16.identity[0] = 0x7f80U;
    auto bf16_backend = k3x::make_cuda_backend(options(
        k3x::CudaWeightMode::resident, 1 << 20,
        k3x::CudaWeightValidationMode::admission));
    if (!bf16_backend) return 39;
    const auto bf16_failed = bf16_backend.value()->official_kda(
        nonfinite_bf16.hidden, nonfinite_bf16.cuda_weights(), state,
        nonfinite_bf16.cuda_config, 1, k3x::ProfilePhase::decode);
    const auto bf16_stats = bf16_backend.value()->runtime_stats();
    if (bf16_failed ||
        bf16_failed.error() != k3x::ErrorCode::invalid_extent ||
        bf16_stats.immutable_validation_scans != kImmutableViews ||
        bf16_stats.immutable_validation_hits != 0 ||
        bf16_stats.immutable_validation_bytes != kImmutableBytes ||
        bf16_stats.resident_weight_bytes != 0 ||
        bf16_stats.official_kda_kernel_launches != 0) return 40;
    return 0;
}

int device_state_handoff() {
    const Fixture fixture;
    const auto zero = k3x::zero_official_kda_state(fixture.cpu_config);
    const auto oracle_a = k3x::official_kda_cpu(
        std::span(fixture.hidden).first(4), fixture.cpu_weights(), zero,
        fixture.cpu_config);
    if (!oracle_a) return 50;
    const auto oracle_b = k3x::official_kda_cpu(
        std::span(fixture.hidden).last(4), fixture.cpu_weights(),
        oracle_a.value().state, fixture.cpu_config);
    if (!oracle_b) return 51;

    auto backend = k3x::make_cuda_backend(options(
        k3x::CudaWeightMode::resident, 1 << 20,
        k3x::CudaWeightValidationMode::admission));
    if (!backend) return 52;
    const k3x::OfficialKdaCudaStateView initial_state{
        zero.conv_q, zero.conv_k, zero.conv_v, zero.recurrent_v_first};
    const auto first = backend.value()->official_kda(
        std::span(fixture.hidden).first(4), fixture.cuda_weights(),
        initial_state, fixture.cuda_config, 1, k3x::ProfilePhase::decode,
        {k3x::OfficialKdaStateMode::device_seed, {}});
    if (!first || !first.value().executed || first.value().state_published ||
        !first.value().device_state.owner ||
        !first.value().device_state.generation ||
        !first.value().conv_q.empty() || !first.value().conv_k.empty() ||
        !first.value().conv_v.empty() ||
        !first.value().recurrent_v_first.empty() ||
        !close(first.value().output, oracle_a.value().output)) return 53;

    const k3x::OfficialKdaCudaStateView no_host_state{};
    const auto second = backend.value()->official_kda(
        std::span(fixture.hidden).last(4), fixture.cuda_weights(),
        no_host_state, fixture.cuda_config, 1, k3x::ProfilePhase::decode,
        {k3x::OfficialKdaStateMode::device_publish,
         first.value().device_state});
    if (!second || !second.value().executed ||
        !second.value().state_published || second.value().device_state.owner ||
        second.value().device_state.generation ||
        second.value().conv_q != oracle_b.value().state.conv_q ||
        second.value().conv_k != oracle_b.value().state.conv_k ||
        second.value().conv_v != oracle_b.value().state.conv_v ||
        !close(second.value().recurrent_v_first,
               oracle_b.value().state.recurrent_v_first) ||
        !close(second.value().output, oracle_b.value().output)) return 54;

    const auto stats = backend.value()->runtime_stats();
    const std::uint64_t state_bytes = 3 * 2 * 4 * sizeof(std::uint16_t) +
                                      2 * 2 * 2 * sizeof(float);
    if (stats.official_kda_calls != 2 ||
        stats.official_kda_state_h2d_bytes != state_bytes ||
        stats.official_kda_state_d2h_bytes != state_bytes ||
        stats.official_kda_device_state_seeds != 1 ||
        stats.official_kda_device_state_continuations != 1 ||
        stats.official_kda_device_state_publications != 1) return 55;

    const auto stale = backend.value()->official_kda(
        std::span(fixture.hidden).last(4), fixture.cuda_weights(),
        no_host_state, fixture.cuda_config, 1, k3x::ProfilePhase::decode,
        {k3x::OfficialKdaStateMode::device_publish,
         first.value().device_state});
    const auto stale_stats = backend.value()->runtime_stats();
    if (stale || stale.error() != k3x::ErrorCode::invalid_state ||
        stale_stats.official_kda_calls != stats.official_kda_calls ||
        stale_stats.official_kda_kernel_launches !=
            stats.official_kda_kernel_launches ||
        stale_stats.official_kda_state_h2d_bytes !=
            stats.official_kda_state_h2d_bytes ||
        stale_stats.official_kda_state_d2h_bytes !=
            stats.official_kda_state_d2h_bytes) return 56;
    return 0;
}

int device_state_fail_closed() {
    const Fixture fixture;
    const auto zero = k3x::zero_official_kda_state(fixture.cpu_config);
    const auto oracle_a = k3x::official_kda_cpu(
        std::span(fixture.hidden).first(4), fixture.cpu_weights(), zero,
        fixture.cpu_config);
    if (!oracle_a) return 60;
    const auto oracle_b = k3x::official_kda_cpu(
        std::span(fixture.hidden).last(4), fixture.cpu_weights(),
        oracle_a.value().state, fixture.cpu_config);
    if (!oracle_b) return 61;

    auto backend = k3x::make_cuda_backend(options(
        k3x::CudaWeightMode::resident, 1 << 20,
        k3x::CudaWeightValidationMode::admission));
    auto other = k3x::make_cuda_backend(options(
        k3x::CudaWeightMode::resident, 1 << 20,
        k3x::CudaWeightValidationMode::admission));
    if (!backend || !other) return 62;
    const k3x::OfficialKdaCudaStateView initial_state{
        zero.conv_q, zero.conv_k, zero.conv_v, zero.recurrent_v_first};
    const k3x::OfficialKdaCudaStateView no_host_state{};

    auto invalid_mode_backend = k3x::make_cuda_backend(options(
        k3x::CudaWeightMode::resident, 1 << 20,
        k3x::CudaWeightValidationMode::admission));
    if (!invalid_mode_backend) return 69;
    const auto invalid_mode_before = invalid_mode_backend.value()->runtime_stats();
    const auto invalid_mode = invalid_mode_backend.value()->official_kda(
        std::span(fixture.hidden).first(4), fixture.cuda_weights(),
        no_host_state, fixture.cuda_config, 7, k3x::ProfilePhase::decode,
        {static_cast<k3x::OfficialKdaStateMode>(255), {}});
    const auto invalid_mode_after = invalid_mode_backend.value()->runtime_stats();
    if (invalid_mode || invalid_mode.error() != k3x::ErrorCode::invalid_extent ||
        invalid_mode_after.official_kda_calls !=
            invalid_mode_before.official_kda_calls ||
        invalid_mode_after.official_kda_kernel_launches !=
            invalid_mode_before.official_kda_kernel_launches ||
        invalid_mode_after.activation_h2d_bytes !=
            invalid_mode_before.activation_h2d_bytes ||
        invalid_mode_after.device_to_host_bytes !=
            invalid_mode_before.device_to_host_bytes) return 70;

    const auto seed = backend.value()->official_kda(
        std::span(fixture.hidden).first(4), fixture.cuda_weights(),
        initial_state, fixture.cuda_config, 7, k3x::ProfilePhase::decode,
        {k3x::OfficialKdaStateMode::device_seed, {}});
    if (!seed || seed.value().state_published ||
        !close(seed.value().output, oracle_a.value().output)) return 63;

    const auto wrong_owner = other.value()->official_kda(
        std::span(fixture.hidden).last(4), fixture.cuda_weights(),
        no_host_state, fixture.cuda_config, 7, k3x::ProfilePhase::decode,
        {k3x::OfficialKdaStateMode::device_publish,
         seed.value().device_state});
    const auto wrong_layer = backend.value()->official_kda(
        std::span(fixture.hidden).last(4), fixture.cuda_weights(),
        no_host_state, fixture.cuda_config, 8, k3x::ProfilePhase::decode,
        {k3x::OfficialKdaStateMode::device_publish,
         seed.value().device_state});
    auto wrong_config = fixture.cuda_config;
    wrong_config.rms_norm_epsilon = 2.0e-5F;
    const auto config_mismatch = backend.value()->official_kda(
        std::span(fixture.hidden).last(4), fixture.cuda_weights(),
        no_host_state, wrong_config, 7, k3x::ProfilePhase::decode,
        {k3x::OfficialKdaStateMode::device_publish,
         seed.value().device_state});
    const auto unexpected_host = backend.value()->official_kda(
        std::span(fixture.hidden).last(4), fixture.cuda_weights(),
        initial_state, fixture.cuda_config, 7, k3x::ProfilePhase::decode,
        {k3x::OfficialKdaStateMode::device_continue,
         seed.value().device_state});
    if (wrong_owner || wrong_owner.error() != k3x::ErrorCode::invalid_state ||
        wrong_layer || wrong_layer.error() != k3x::ErrorCode::invalid_state ||
        config_mismatch ||
        config_mismatch.error() != k3x::ErrorCode::invalid_state ||
        unexpected_host ||
        unexpected_host.error() != k3x::ErrorCode::invalid_extent) return 64;

    const auto continued = backend.value()->official_kda(
        std::span(fixture.hidden).last(4), fixture.cuda_weights(),
        no_host_state, fixture.cuda_config, 7, k3x::ProfilePhase::decode,
        {k3x::OfficialKdaStateMode::device_continue,
         seed.value().device_state});
    if (!continued || continued.value().state_published ||
        continued.value().device_state == seed.value().device_state ||
        !close(continued.value().output, oracle_b.value().output)) return 65;
    const auto before_host = backend.value()->runtime_stats();
    if (before_host.official_kda_calls != 2 ||
        before_host.official_kda_device_state_seeds != 1 ||
        before_host.official_kda_device_state_continuations != 1 ||
        before_host.official_kda_device_state_publications != 0 ||
        before_host.official_kda_device_state_invalidations != 0) return 66;

    const auto host = backend.value()->official_kda(
        std::span(fixture.hidden).first(4), fixture.cuda_weights(),
        initial_state, fixture.cuda_config, 7, k3x::ProfilePhase::decode);
    if (!host || !host.value().state_published) return 67;
    const auto invalidated = backend.value()->official_kda(
        std::span(fixture.hidden).last(4), fixture.cuda_weights(),
        no_host_state, fixture.cuda_config, 7, k3x::ProfilePhase::decode,
        {k3x::OfficialKdaStateMode::device_publish,
         continued.value().device_state});
    const auto after_host = backend.value()->runtime_stats();
    if (invalidated ||
        invalidated.error() != k3x::ErrorCode::invalid_state ||
        after_host.official_kda_calls != 3 ||
        after_host.official_kda_device_state_invalidations != 1) return 68;
    return 0;
}

}  // namespace

int main() {
    if (const auto result = run(k3x::CudaWeightMode::transient)) return result;
    if (const auto result = run(k3x::CudaWeightMode::resident)) return 10 + result;
    if (const auto result = invalid()) return result;
    if (const auto result = admission_validation()) return result;
    if (const auto result = device_state_handoff()) return result;
    return device_state_fail_closed();
}
