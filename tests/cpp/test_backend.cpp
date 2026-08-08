// exact CPU backend의 dense와 native MXFP4 행렬 연산 계약을 검증합니다.
#include "k3x/backend.hpp"

#include <array>
#include <cstddef>

int main() {
    k3x::Profiler profiler;
    auto backend = k3x::make_cpu_backend(&profiler);

    if (backend->kind() != k3x::BackendKind::cpu) return 1;
    if (backend->device_name() != "CPU") return 2;
    if (backend->memory_stats().current_device_bytes != 0) return 3;
    if (backend->memory_stats().peak_device_bytes != 0) return 4;

    const k3x::BackendOptions defaults;
    if (defaults.cuda_allocation != k3x::CudaAllocationMode::per_operation) return 21;
    if (defaults.cuda_weights != k3x::CudaWeightMode::transient) return 22;
    if (defaults.cuda_batching != k3x::CudaBatchingMode::scalar) return 23;
    if (defaults.cuda_resident_bytes != 0) return 24;

    const auto& options = backend->options();
    if (options.kind != k3x::BackendKind::cpu) return 25;
    if (options.dense_precision != k3x::DensePrecision::fp32) return 26;
    if (options.cuda_allocation != k3x::CudaAllocationMode::per_operation) return 27;
    if (options.cuda_weights != k3x::CudaWeightMode::transient) return 28;
    if (options.cuda_batching != k3x::CudaBatchingMode::scalar) return 29;
    if (options.cuda_resident_bytes != 0) return 30;

    const auto runtime_stats = backend->runtime_stats();
    if (runtime_stats.device_allocation_count != 0) return 31;
    if (runtime_stats.device_free_count != 0) return 32;
    if (runtime_stats.stream_synchronization_count != 0) return 33;
    if (runtime_stats.weight_cache_hits != 0) return 34;
    if (runtime_stats.weight_cache_misses != 0) return 35;
    if (runtime_stats.weight_cache_bypasses != 0) return 36;
    if (runtime_stats.resident_weight_bytes != 0) return 37;
    if (runtime_stats.peak_resident_weight_bytes != 0) return 38;
    if (runtime_stats.scratch_bytes != 0) return 39;
    if (runtime_stats.peak_scratch_bytes != 0) return 40;
    if (runtime_stats.weight_h2d_bytes != 0) return 41;
    if (runtime_stats.activation_h2d_bytes != 0) return 42;
    if (runtime_stats.grouped_projection_calls != 0) return 43;
    if (runtime_stats.grouped_projection_members != 0) return 44;

    const std::array<float, 3> dense_input{2.0F, -1.0F, 0.5F};
    const std::array<float, 6> dense_weight{1.0F, 2.0F, 3.0F,
                                             -2.0F, 0.25F, 4.0F};
    const auto dense = backend->dense_matvec(
        dense_input, dense_weight, 2, 3, 7, k3x::ProfilePhase::prefill);
    if (!dense) return 5;
    if (dense.value()[0] != 1.5F) return 6;
    if (dense.value()[1] != -2.25F) return 7;

    std::array<float, 32> mxfp4_input{};
    mxfp4_input[1] = 2.0F;
    std::array<std::byte, 16> packed{};
    packed[0] = std::byte{0x10};
    const std::array<std::byte, 1> scales{std::byte{127}};
    const auto mxfp4 = backend->mxfp4_matvec(
        mxfp4_input, packed, scales, 1, 32, 32, 8,
        k3x::ProfilePhase::decode);
    if (!mxfp4) return 8;
    if (mxfp4.value()[0] != 1.0F) return 9;

    const auto invalid = backend->dense_matvec(
        dense_input, dense_weight, 2, 4, 9, k3x::ProfilePhase::decode);
    if (invalid) return 10;
    if (invalid.error() != k3x::ErrorCode::invalid_extent) return 11;

    const auto& events = profiler.events();
    if (events.size() != 3) return 12;
    if (events[0].operation != k3x::ProfileOperation::dense_matvec) return 13;
    if (events[0].precision != k3x::NumericPrecision::fp32) return 14;
    if (events[0].layer != 7) return 15;
    if (events[1].operation != k3x::ProfileOperation::mxfp4_matvec) return 16;
    if (events[1].precision != k3x::NumericPrecision::mxfp4_e2m1_e8m0) return 17;
    if (events[1].layer != 8) return 18;
    if (events[2].success) return 19;
    if (events[2].layer != 9) return 20;
    return 0;
}
