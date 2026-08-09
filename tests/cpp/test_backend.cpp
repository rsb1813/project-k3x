// exact CPU backend의 dense와 native MXFP4 행렬 연산 계약을 검증합니다.
#include "k3x/backend.hpp"
#include "k3x/ops.hpp"

#include <array>
#include <cmath>
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
    if (defaults.cuda_boundary != k3x::CudaBoundaryMode::operation) return 58;

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
    if (runtime_stats.ffn_block_calls != 0) return 59;
    if (runtime_stats.ffn_block_experts != 0) return 60;

    const std::array<float, 3> dense_input{2.0F, -1.0F, 0.5F};
    const std::array<float, 6> dense_weight{1.0F, 2.0F, 3.0F,
                                             -2.0F, 0.25F, 4.0F};
    const auto dense = backend->dense_matvec(
        dense_input, k3x::DenseWeightView{100, dense_weight, 2, 3}, 7,
        k3x::ProfilePhase::prefill);
    if (!dense) return 5;
    if (dense.value()[0] != 1.5F) return 6;
    if (dense.value()[1] != -2.25F) return 7;

    std::array<float, 32> mxfp4_input{};
    mxfp4_input[1] = 2.0F;
    std::array<std::byte, 16> packed{};
    packed[0] = std::byte{0x10};
    const std::array<std::byte, 1> scales{std::byte{127}};
    const auto mxfp4 = backend->mxfp4_matvec(
        mxfp4_input, k3x::Mxfp4WeightView{200, packed, scales, 1, 32, 32}, 8,
        k3x::ProfilePhase::decode);
    if (!mxfp4) return 8;
    if (mxfp4.value()[0] != 1.0F) return 9;

    const auto invalid = backend->dense_matvec(
        dense_input, k3x::DenseWeightView{300, dense_weight, 2, 4}, 9,
        k3x::ProfilePhase::decode);
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

    const std::array<float, 2> group_input{2.0F, -1.0F};
    const std::array<float, 4> first_dense{1.0F, 0.0F, 0.0F, 1.0F};
    const std::array<float, 2> second_dense{3.0F, -2.0F};
    const std::array<k3x::DenseWeightView, 2> dense_group{{
        {101, first_dense, 2, 2},
        {102, second_dense, 1, 2},
    }};
    const auto dense_outputs = backend->dense_matvec_group(
        group_input, dense_group, 10, k3x::ProfilePhase::decode);
    if (!dense_outputs || dense_outputs.value().size() != 2) return 50;
    if (dense_outputs.value()[0] != std::vector<float>{2.0F, -1.0F}) return 51;
    if (dense_outputs.value()[1] != std::vector<float>{8.0F}) return 52;

    std::array<std::byte, 32> second_packed{};
    second_packed[0] = std::byte{0x10};
    second_packed[16] = std::byte{0x10};
    const std::array<std::byte, 2> second_scales{
        std::byte{127}, std::byte{127}};
    const std::array<k3x::Mxfp4WeightView, 2> mxfp4_group{{
        {201, packed, scales, 1, 32, 32},
        {202, second_packed, second_scales, 2, 32, 32},
    }};
    const auto mxfp4_outputs = backend->mxfp4_matvec_group(
        mxfp4_input, mxfp4_group, 11, k3x::ProfilePhase::decode);
    if (!mxfp4_outputs || mxfp4_outputs.value().size() != 2) return 53;
    if (mxfp4_outputs.value()[0] != std::vector<float>{1.0F}) return 54;
    if (mxfp4_outputs.value()[1] != std::vector<float>{1.0F, 1.0F}) return 55;

    const std::array<k3x::Mxfp4WeightView, 2> invalid_group{{
        {203, packed, scales, 1, 32, 32},
        {204, packed, {}, 1, 32, 32},
    }};
    const auto event_count = profiler.events().size();
    const auto rejected = backend->mxfp4_matvec_group(
        mxfp4_input, invalid_group, 12, k3x::ProfilePhase::decode);
    if (rejected || rejected.error() != k3x::ErrorCode::invalid_mxfp4) return 56;
    if (profiler.events().size() != event_count) return 57;

    const std::array<float, 6> dense_gate{
        1.0F, 0.0F, 0.0F, 0.0F, 1.0F, 0.0F};
    const std::array<float, 6> dense_up{
        0.0F, 0.0F, 2.0F, 1.0F, 0.0F, 0.0F};
    const std::array<float, 4> dense_down{1.0F, 2.0F, -1.0F, 0.5F};
    const k3x::DenseMlpView dense_mlp{
        {301, dense_gate, 2, 3},
        {302, dense_up, 2, 3},
        {303, dense_down, 2, 2},
    };
    const auto dense_block = backend->dense_situ_mlp(
        dense_input, dense_mlp, 2.0F, 1.5F, 13,
        k3x::ProfilePhase::decode);
    if (!dense_block || dense_block.value().size() != 2) return 61;
    std::array<float, 2> dense_activation{};
    const std::array<float, 2> dense_gate_output{2.0F, -1.0F};
    const std::array<float, 2> dense_up_output{1.0F, 2.0F};
    k3x::situ_glu(dense_activation, dense_gate_output, dense_up_output, 2.0F,
                  1.5F);
    const std::array<float, 2> dense_expected{
        dense_activation[0] + 2.0F * dense_activation[1],
        -dense_activation[0] + 0.5F * dense_activation[1],
    };
    if (std::abs(dense_block.value()[0] - dense_expected[0]) > 1.0e-6F ||
        std::abs(dense_block.value()[1] - dense_expected[1]) > 1.0e-6F) {
        return 62;
    }

    std::array<std::byte, 512> expert_gate{};
    std::array<std::byte, 512> expert_up{};
    std::array<std::byte, 16> expert_down_one{};
    std::array<std::byte, 16> expert_down_two{};
    expert_gate[0] = std::byte{0x10};
    expert_up[0] = std::byte{0x20};
    expert_down_one[0] = std::byte{0x02};
    expert_down_two[0] = std::byte{0x04};
    std::array<std::byte, 32> expert_intermediate_scales{};
    expert_intermediate_scales.fill(std::byte{127});
    const std::array<std::byte, 1> expert_down_scales{std::byte{127}};
    const std::array<k3x::Mxfp4MlpView, 2> expert_mlps{{
        {
            {401, expert_gate, expert_intermediate_scales, 32, 32, 32},
            {402, expert_up, expert_intermediate_scales, 32, 32, 32},
            {403, expert_down_one, expert_down_scales, 1, 32, 32},
        },
        {
            {404, expert_gate, expert_intermediate_scales, 32, 32, 32},
            {405, expert_up, expert_intermediate_scales, 32, 32, 32},
            {406, expert_down_two, expert_down_scales, 1, 32, 32},
        },
    }};
    const auto expert_blocks = backend->mxfp4_situ_mlp_group(
        mxfp4_input, expert_mlps, 2.0F, 1.5F, 14,
        k3x::ProfilePhase::decode);
    if (!expert_blocks || expert_blocks.value().size() != 2) return 63;
    std::array<float, 1> expert_activation{};
    const std::array<float, 1> expert_gate_output{1.0F};
    const std::array<float, 1> expert_up_output{2.0F};
    k3x::situ_glu(expert_activation, expert_gate_output, expert_up_output,
                  2.0F, 1.5F);
    if (std::abs(expert_blocks.value()[0][0] - expert_activation[0]) >
            1.0e-6F ||
        std::abs(expert_blocks.value()[1][0] - 2.0F * expert_activation[0]) >
            1.0e-6F) {
        return 64;
    }

    auto invalid_expert_mlps = expert_mlps;
    invalid_expert_mlps[1].down.scales = {};
    const auto block_event_count = profiler.events().size();
    const auto rejected_block = backend->mxfp4_situ_mlp_group(
        mxfp4_input, invalid_expert_mlps, 2.0F, 1.5F, 15,
        k3x::ProfilePhase::decode);
    if (rejected_block ||
        rejected_block.error() != k3x::ErrorCode::invalid_mxfp4) {
        return 65;
    }
    if (profiler.events().size() != block_event_count) return 66;
    return 0;
}
