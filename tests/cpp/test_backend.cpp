// exact CPU backend의 dense와 native MXFP4 행렬 연산 계약을 검증합니다.
#include "k3x/backend.hpp"

#include <array>
#include <cassert>
#include <cstddef>

int main() {
    k3x::Profiler profiler;
    auto backend = k3x::make_cpu_backend(&profiler);

    assert(backend->kind() == k3x::BackendKind::cpu);
    assert(backend->device_name() == "CPU");
    assert(backend->memory_stats().current_device_bytes == 0);
    assert(backend->memory_stats().peak_device_bytes == 0);

    const std::array<float, 3> dense_input{2.0F, -1.0F, 0.5F};
    const std::array<float, 6> dense_weight{1.0F, 2.0F, 3.0F,
                                             -2.0F, 0.25F, 4.0F};
    const auto dense = backend->dense_matvec(
        dense_input, dense_weight, 2, 3, 7, k3x::ProfilePhase::prefill);
    assert(dense);
    assert(dense.value()[0] == 1.5F);
    assert(dense.value()[1] == -2.25F);

    std::array<float, 32> mxfp4_input{};
    mxfp4_input[1] = 2.0F;
    std::array<std::byte, 16> packed{};
    packed[0] = std::byte{0x10};
    const std::array<std::byte, 1> scales{std::byte{127}};
    const auto mxfp4 = backend->mxfp4_matvec(
        mxfp4_input, packed, scales, 1, 32, 32, 8,
        k3x::ProfilePhase::decode);
    assert(mxfp4);
    assert(mxfp4.value()[0] == 1.0F);

    const auto invalid = backend->dense_matvec(
        dense_input, dense_weight, 2, 4, 9, k3x::ProfilePhase::decode);
    assert(!invalid);
    assert(invalid.error() == k3x::ErrorCode::invalid_extent);

    const auto& events = profiler.events();
    assert(events.size() == 3);
    assert(events[0].operation == k3x::ProfileOperation::dense_matvec);
    assert(events[0].precision == k3x::NumericPrecision::fp32);
    assert(events[0].layer == 7);
    assert(events[1].operation == k3x::ProfileOperation::mxfp4_matvec);
    assert(events[1].precision == k3x::NumericPrecision::mxfp4_e2m1_e8m0);
    assert(events[1].layer == 8);
    assert(!events[2].success);
    assert(events[2].layer == 9);
}
