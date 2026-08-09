// cuBLASLt와 strict SiTU를 연결한 dense FFN block의 전송과 출력을 검증합니다.
#include "k3x/backend.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstdint>
#include <vector>

namespace {

float round_to_bf16(float value) {
    auto bits = std::bit_cast<std::uint32_t>(value);
    bits += 0x7FFFU + ((bits >> 16U) & 1U);
    return std::bit_cast<float>(bits & 0xFFFF0000U);
}

bool nearly_equal(const std::vector<float>& actual,
                  const std::vector<float>& expected, float tolerance) {
    if (actual.size() != expected.size()) return false;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (std::abs(actual[index] - expected[index]) > tolerance) return false;
    }
    return true;
}

struct Fixture {
    std::array<float, 3> input{1.25F, -0.75F, 0.5F};
    std::array<float, 12> gate{
        0.2F, -0.3F, 0.4F,
        -0.5F, 0.6F, 0.7F,
        0.8F, 0.1F, -0.2F,
        -0.4F, -0.9F, 0.3F,
    };
    std::array<float, 12> up{
        0.7F, -0.2F, 0.1F,
        0.3F, 0.8F, -0.6F,
        -0.5F, 0.4F, 0.9F,
        0.2F, -0.7F, 0.6F,
    };
    std::array<float, 8> down{
        0.4F, -0.2F, 0.7F, 0.1F,
        -0.6F, 0.5F, 0.3F, -0.8F,
    };

    k3x::DenseMlpView view(
        std::uint64_t gate_id = 501, std::uint64_t up_id = 502,
        std::uint64_t down_id = 503) const {
        return {
            {gate_id, gate, 4, 3},
            {up_id, up, 4, 3},
            {down_id, down, 2, 4},
        };
    }
};

std::vector<float> cpu_oracle(
    std::span<const float> input, k3x::DenseMlpView view) {
    auto cpu = k3x::make_cpu_backend();
    const auto result = cpu->dense_situ_mlp(
        input, view, 2.0F, 1.5F, 7, k3x::ProfilePhase::decode);
    return result ? result.value() : std::vector<float>{};
}

int test_fp32_and_residency() {
    const Fixture fixture;
    k3x::BackendOptions options;
    options.kind = k3x::BackendKind::cuda_custom;
    options.cuda_boundary = k3x::CudaBoundaryMode::ffn_block;
    options.cuda_allocation = k3x::CudaAllocationMode::reused;
    options.cuda_weights = k3x::CudaWeightMode::resident;
    options.cuda_resident_bytes = 128;
    k3x::Profiler profiler;
    auto backend = k3x::make_cuda_backend(options, &profiler);
    if (!backend) return 1;

    const auto weights = fixture.view();
    if (!backend.value()->dense_matvec(fixture.input, weights.gate, 7,
                                      k3x::ProfilePhase::decode)) return 2;
    if (!backend.value()->dense_matvec(fixture.input, weights.up, 7,
                                      k3x::ProfilePhase::decode)) return 3;
    const std::array<float, 4> down_input{};
    if (!backend.value()->dense_matvec(down_input, weights.down, 7,
                                      k3x::ProfilePhase::decode)) return 4;
    const auto before = backend.value()->runtime_stats();
    const auto before_profile = profiler.summary();
    const auto result = backend.value()->dense_situ_mlp(
        fixture.input, weights, 2.0F, 1.5F, 7,
        k3x::ProfilePhase::decode);
    if (!result || !nearly_equal(result.value(), cpu_oracle(fixture.input, weights),
                                 1.0e-5F)) return 5;
    const auto after = backend.value()->runtime_stats();
    const auto after_profile = profiler.summary();
    if (after.weight_cache_hits - before.weight_cache_hits != 3) return 6;
    if (after.weight_h2d_bytes != before.weight_h2d_bytes) return 7;
    if (after.activation_h2d_bytes - before.activation_h2d_bytes != 12) return 8;
    if (after.stream_synchronization_count -
            before.stream_synchronization_count != 1) return 9;
    if (after.ffn_block_calls - before.ffn_block_calls != 1) return 10;
    if (after.ffn_block_experts != before.ffn_block_experts) return 11;
    if (after_profile.device_to_host_bytes -
            before_profile.device_to_host_bytes != 8) return 12;
    return 0;
}

int test_bf16() {
    Fixture fixture;
    for (auto& value : fixture.input) value = round_to_bf16(value);
    for (auto& value : fixture.gate) value = round_to_bf16(value);
    for (auto& value : fixture.up) value = round_to_bf16(value);
    for (auto& value : fixture.down) value = round_to_bf16(value);
    const auto weights = fixture.view(601, 602, 603);
    const auto expected = cpu_oracle(fixture.input, weights);

    k3x::BackendOptions options;
    options.kind = k3x::BackendKind::cuda_custom;
    options.dense_precision = k3x::DensePrecision::bf16_rounded;
    options.cuda_boundary = k3x::CudaBoundaryMode::ffn_block;
    options.cuda_allocation = k3x::CudaAllocationMode::reused;
    auto backend = k3x::make_cuda_backend(options);
    if (!backend) return 20;
    const auto result = backend.value()->dense_situ_mlp(
        fixture.input, weights, 2.0F, 1.5F, 8,
        k3x::ProfilePhase::decode);
    if (!result || !nearly_equal(result.value(), expected, 2.0e-2F)) return 21;
    const auto stats = backend.value()->runtime_stats();
    if (stats.activation_h2d_bytes != 6 || stats.weight_h2d_bytes != 64 ||
        stats.stream_synchronization_count != 1 ||
        stats.ffn_block_calls != 1) return 22;
    return 0;
}

int test_validation() {
    const Fixture fixture;
    k3x::BackendOptions options;
    options.kind = k3x::BackendKind::cuda_custom;
    options.cuda_boundary = k3x::CudaBoundaryMode::ffn_block;
    options.cuda_allocation = k3x::CudaAllocationMode::reused;
    options.cuda_weights = k3x::CudaWeightMode::resident;
    options.cuda_resident_bytes = 256;
    auto backend = k3x::make_cuda_backend(options);
    if (!backend) return 30;

    auto invalid = fixture.view(701, 702, 703);
    invalid.up.cols = 2;
    const auto result = backend.value()->dense_situ_mlp(
        fixture.input, invalid, 2.0F, 1.5F, 9,
        k3x::ProfilePhase::decode);
    if (result || result.error() != k3x::ErrorCode::invalid_extent) return 31;
    if (backend.value()->runtime_stats().ffn_block_calls != 0) return 32;

    const std::array<float, 3> collision_weight{1.0F, 2.0F, 3.0F};
    if (!backend.value()->dense_matvec(
            fixture.input, {704, collision_weight, 1, 3}, 9,
            k3x::ProfilePhase::decode)) return 33;
    const auto collision = backend.value()->dense_situ_mlp(
        fixture.input, fixture.view(704, 705, 706), 2.0F, 1.5F, 9,
        k3x::ProfilePhase::decode);
    if (collision || collision.error() != k3x::ErrorCode::invalid_extent) return 34;
    if (backend.value()->runtime_stats().ffn_block_calls != 0) return 35;
    return 0;
}

}  // namespace

int main() {
    if (const auto result = test_fp32_and_residency()) return result;
    if (const auto result = test_bf16()) return result;
    return test_validation();
}
