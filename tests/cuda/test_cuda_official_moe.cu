// 공식 BF16/MXFP4 MoE CUDA 경계의 portable parity, residency와 입력 검증을 확인합니다.
#include "k3x/backend.hpp"
#include "k3x/official_moe.hpp"

#include <array>
#include <bit>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <span>

namespace {

constexpr std::size_t hidden_width = 3;
constexpr std::size_t latent_width = 32;
constexpr std::size_t intermediate_width = 32;

std::uint16_t bf16(float value) {
    auto bits = std::bit_cast<std::uint32_t>(value);
    bits += 0x7fffU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>(bits >> 16U);
}

struct Fixture {
    k3x::OfficialMoeInput input{{0.5F, -1.0F, 0.25F},
                                {1000.1F, 0.3F, -0.7F}};
    std::array<std::uint16_t, hidden_width> residual_norm{};
    std::array<std::uint16_t, hidden_width> residual_proj{};
    std::array<std::uint16_t, hidden_width> post_norm{};
    std::array<std::uint16_t, 2 * hidden_width> router{};
    std::array<float, 2> correction{0.0F, 0.0F};
    std::array<std::uint16_t, latent_width * hidden_width> routed_down{};
    std::array<std::uint16_t, latent_width> routed_norm{};
    std::array<std::uint16_t, hidden_width * latent_width> routed_up{};
    std::array<std::uint16_t, 2 * hidden_width> shared_gate{};
    std::array<std::uint16_t, 2 * hidden_width> shared_up{};
    std::array<std::uint16_t, hidden_width * 2> shared_down{};
    std::array<std::array<std::byte, intermediate_width * latent_width / 2>, 2> gate_packed{};
    std::array<std::array<std::byte, intermediate_width * latent_width / 2>, 2> up_packed{};
    std::array<std::array<std::byte, latent_width * intermediate_width / 2>, 2> down_packed{};
    std::array<std::array<std::byte, intermediate_width * latent_width / 32>, 2> gate_scales{};
    std::array<std::array<std::byte, intermediate_width * latent_width / 32>, 2> up_scales{};
    std::array<std::array<std::byte, latent_width * intermediate_width / 32>, 2> down_scales{};
    std::array<k3x::Mxfp4MlpView, 2> experts{};
    std::array<k3x::OfficialExpertView, 2> cpu_experts{};
    std::array<std::uint32_t, 2> expert_ids{0, 1};
    std::array<float, 2> contributions{0.65F, 0.35F};

    Fixture() {
        residual_norm.fill(bf16(1.0F));
        residual_proj = {bf16(0.5F), bf16(-0.25F), bf16(0.125F)};
        post_norm.fill(bf16(1.0F));
        router = {bf16(1.0F), bf16(0.0F), bf16(0.0F),
                  bf16(0.0F), bf16(1.0F), bf16(0.0F)};
        routed_norm.fill(bf16(1.0F));
        for (std::size_t index = 0; index < hidden_width; ++index) {
            routed_down[index * hidden_width + index] = bf16(1.0F);
            routed_up[index * latent_width + index] = bf16(1.0F);
        }
        shared_gate = {bf16(1.0F), bf16(0.0F), bf16(0.0F),
                       bf16(0.0F), bf16(1.0F), bf16(0.0F)};
        shared_up = shared_gate;
        shared_down = {bf16(1.0F), bf16(0.5F), bf16(-0.5F),
                       bf16(1.0F), bf16(0.25F), bf16(-0.75F)};
        for (std::size_t expert = 0; expert < experts.size(); ++expert) {
            gate_packed[expert][0] = std::byte{0x12};
            up_packed[expert][0] = std::byte{0x21};
            down_packed[expert][0] = static_cast<std::byte>(0x12U + expert);
            gate_scales[expert].fill(std::byte{127});
            up_scales[expert].fill(std::byte{127});
            down_scales[expert].fill(std::byte{127});
            experts[expert] = {
                {100 + expert * 3, gate_packed[expert], gate_scales[expert],
                 intermediate_width, latent_width, 32},
                {101 + expert * 3, up_packed[expert], up_scales[expert],
                 intermediate_width, latent_width, 32},
                {102 + expert * 3, down_packed[expert], down_scales[expert],
                 latent_width, intermediate_width, 32},
            };
            cpu_experts[expert] = {static_cast<std::uint32_t>(expert), experts[expert]};
        }
    }

    k3x::OfficialMoeWeights cpu_weights() const {
        return {
            {residual_norm}, {residual_proj, 1, hidden_width}, {post_norm},
            {router, 2, hidden_width}, correction,
            {routed_down, latent_width, hidden_width}, {routed_norm},
            {routed_up, hidden_width, latent_width},
            {{shared_gate, 2, hidden_width}, {shared_up, 2, hidden_width},
             {shared_down, hidden_width, 2}},
            cpu_experts,
        };
    }

    k3x::OfficialMoeFfnView cuda_weights() const {
        return {
            {routed_down, latent_width, hidden_width, 501},
            {routed_norm, 502},
            {routed_up, hidden_width, latent_width, 503},
            {{shared_gate, 2, hidden_width, 504},
             {shared_up, 2, hidden_width, 505},
             {shared_down, hidden_width, 2, 506}},
        };
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

bool close(std::span<const float> actual, std::span<const float> expected) {
    if (actual.size() != expected.size()) return false;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (!std::isfinite(actual[index]) ||
            std::abs(actual[index] - expected[index]) > 2.0e-2F) return false;
    }
    return true;
}

int run(k3x::CudaWeightMode mode) {
    const Fixture fixture;
    const k3x::OfficialRoute route{
        {fixture.expert_ids.begin(), fixture.expert_ids.end()},
        {fixture.contributions.begin(), fixture.contributions.end()}, {}};
    const auto oracle = k3x::official_moe_cpu(
        fixture.input, fixture.cpu_weights(), route, 1.0e-5F, 4.0F, 25.0F);
    if (!oracle) return 1;
    k3x::Profiler profiler;
    auto backend = k3x::make_cuda_backend(options(mode, 8 * 1024 * 1024), &profiler);
    if (!backend) return 2;
    const auto before_prefix = fixture.input.prefix_sum;
    const auto before_hidden = oracle.value().hidden;
    const auto first = backend.value()->official_mxfp4_moe_ffn(
        oracle.value().hidden, fixture.input.prefix_sum, fixture.cuda_weights(),
        fixture.experts, fixture.expert_ids, fixture.contributions,
        1.0e-5F, 4.0F, 25.0F, 1, k3x::ProfilePhase::decode);
    if (!first || !first.value().executed ||
        first.value().selected_expert_ids !=
            std::vector<std::uint32_t>(fixture.expert_ids.begin(), fixture.expert_ids.end()) ||
        !close(first.value().output, oracle.value().output) ||
        fixture.input.prefix_sum != before_prefix || oracle.value().hidden != before_hidden)
        return 3;
    const auto first_stats = backend.value()->runtime_stats();
    const auto second = backend.value()->official_mxfp4_moe_ffn(
        oracle.value().hidden, fixture.input.prefix_sum, fixture.cuda_weights(),
        fixture.experts, fixture.expert_ids, fixture.contributions,
        1.0e-5F, 4.0F, 25.0F, 1, k3x::ProfilePhase::decode);
    if (!second || !close(second.value().output, oracle.value().output)) return 4;
    const auto second_stats = backend.value()->runtime_stats();
    if (second_stats.device_to_host_bytes - first_stats.device_to_host_bytes !=
        hidden_width * sizeof(float)) return 5;
    if (mode == k3x::CudaWeightMode::resident &&
        (second_stats.weight_h2d_bytes != first_stats.weight_h2d_bytes ||
         second_stats.weight_cache_hits <= first_stats.weight_cache_hits ||
         second_stats.resident_weight_bytes == 0)) return 6;
    return 0;
}

int invalid() {
    Fixture fixture;
    auto backend = k3x::make_cuda_backend(
        options(k3x::CudaWeightMode::resident, 8 * 1024 * 1024));
    if (!backend) return 20;
    const auto hidden = k3x::prepare_official_moe_input(
        fixture.input, fixture.cpu_weights(), 1.0e-5F);
    if (!hidden) return 21;
    auto view = fixture.cuda_weights();
    view.shared.gate.tensor_id = view.routed_down.tensor_id;
    if (backend.value()->official_mxfp4_moe_ffn(
            hidden.value(), fixture.input.prefix_sum, view, fixture.experts,
            fixture.expert_ids, fixture.contributions, 1.0e-5F, 4.0F, 25.0F,
            1, k3x::ProfilePhase::decode)) return 22;
    auto duplicate_ids = fixture.expert_ids;
    duplicate_ids[1] = duplicate_ids[0];
    if (backend.value()->official_mxfp4_moe_ffn(
            hidden.value(), fixture.input.prefix_sum, fixture.cuda_weights(),
            fixture.experts, duplicate_ids, fixture.contributions, 1.0e-5F,
            4.0F, 25.0F, 1, k3x::ProfilePhase::decode)) return 23;
    if (backend.value()->official_mxfp4_moe_ffn(
            hidden.value(), fixture.input.prefix_sum, fixture.cuda_weights(),
            fixture.experts, fixture.expert_ids,
            std::span(fixture.contributions).first(1), 1.0e-5F, 4.0F, 25.0F,
            1, k3x::ProfilePhase::decode)) return 24;
    auto tiny = k3x::make_cuda_backend(
        options(k3x::CudaWeightMode::resident, 1));
    if (!tiny) return 25;
    if (tiny.value()->official_mxfp4_moe_ffn(
            hidden.value(), fixture.input.prefix_sum, fixture.cuda_weights(),
            fixture.experts, fixture.expert_ids, fixture.contributions,
            1.0e-5F, 4.0F, 25.0F, 1, k3x::ProfilePhase::decode)) return 26;
    return 0;
}

}  // namespace

int main() {
    if (const auto result = run(k3x::CudaWeightMode::transient)) return result;
    if (const auto result = run(k3x::CudaWeightMode::resident)) return 10 + result;
    return invalid();
}
