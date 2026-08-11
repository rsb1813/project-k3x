// 공식 MoE portable oracle의 BF16 변환, 자연 routing, 전체 경계와 실패 원자성을 검증합니다.
#include "k3x/official_moe.hpp"

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <iostream>
#include <string_view>
#include <vector>

namespace {

bool close(float left, float right, float tolerance = 1.0e-6F) {
    return std::abs(left - right) <= tolerance;
}

bool vector_close(const std::vector<float>& actual,
                  std::initializer_list<float> expected) {
    if (actual.size() != expected.size()) return false;
    std::size_t index = 0;
    for (const auto value : expected) {
        if (!close(actual[index++], value)) return false;
    }
    return true;
}

void print_vector(const std::vector<float>& values) {
    std::cout << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index) std::cout << ',';
        std::cout << values[index];
    }
    std::cout << ']';
}

}  // namespace

int main(int argc, char** argv) {
    if (k3x::decode_bf16_word(0x0000U) != 0.0F ||
        !std::signbit(k3x::decode_bf16_word(0x8000U)) ||
        k3x::decode_bf16_word(0x3f80U) != 1.0F ||
        !std::isinf(k3x::decode_bf16_word(0x7f80U)) ||
        !std::isnan(k3x::decode_bf16_word(0x7fc1U))) return 1;

    const std::array<std::uint16_t, 2> residual_norm{0x3f80, 0x3f80};
    const std::array<std::uint16_t, 2> residual_proj{0x3f00, 0xbe80};
    const std::array<std::uint16_t, 2> post_norm{0x3f80, 0x3f00};
    const std::array<std::uint16_t, 6> router{
        0x3f80, 0x0000, 0x0000, 0x3f80, 0xbf80, 0x3f80};
    const std::array<float, 3> correction{0.0F, 0.1F, -0.05F};
    const std::array<std::uint16_t, 4> identity{
        0x3f80, 0x0000, 0x0000, 0x3f80};
    const std::array<std::uint16_t, 2> routed_norm{0x3f80, 0x3f80};

    const std::array<std::byte, 2> e0_gate{std::byte{0x12}, std::byte{0x21}};
    const std::array<std::byte, 2> e0_up{std::byte{0x22}, std::byte{0x31}};
    const std::array<std::byte, 2> e0_down{std::byte{0x12}, std::byte{0x21}};
    const std::array<std::byte, 2> e1_gate{std::byte{0x21}, std::byte{0x12}};
    const std::array<std::byte, 2> e1_up{std::byte{0x12}, std::byte{0x23}};
    const std::array<std::byte, 2> e1_down{std::byte{0x12}, std::byte{0x12}};
    const std::array<std::byte, 2> scales{std::byte{127}, std::byte{127}};
    const auto matrix = [](std::uint64_t id, const auto& packed,
                           const auto& scale) {
        return k3x::Mxfp4WeightView{id, packed, scale, 2, 2, 2};
    };
    std::vector<k3x::OfficialExpertView> experts{
        {0, {matrix(10, e0_gate, scales), matrix(11, e0_up, scales),
             matrix(12, e0_down, scales)}},
        {1, {matrix(20, e1_gate, scales), matrix(21, e1_up, scales),
             matrix(22, e1_down, scales)}},
    };
    const k3x::OfficialMoeWeights weights{
        {residual_norm}, {residual_proj, 1, 2}, {post_norm},
        {router, 3, 2}, correction,
        {identity, 2, 2}, {routed_norm}, {identity, 2, 2},
        {{identity, 2, 2}, {identity, 2, 2}, {identity, 2, 2}},
        experts,
    };
    const k3x::OfficialMoeInput input{{0.5F, -1.0F}, {1.0F, 0.25F}};

    const auto hidden = k3x::prepare_official_moe_input(input, weights, 1.0e-5F);
    if (!hidden || !vector_close(hidden.value(), {1.2578125F, -0.32421875F}))
        return 2;
    const auto route = k3x::route_official_moe(
        hidden.value(), weights.router, weights.correction, 2);
    if (!route || route.value().expert_ids != std::vector<std::uint32_t>({0, 1}) ||
        !close(route.value().contributions[0], 0.6497964859F, 1.0e-6F) ||
        !close(route.value().contributions[1], 0.3502035439F, 1.0e-6F))
        return 3;

    const auto result = k3x::official_moe_cpu(
        input, weights, route.value(), 1.0e-5F, 4.0F, 25.0F);
    if (!result) return 4;
    if (!vector_close(result.value().hidden, {1.2578125F, -0.32421875F}) ||
        !vector_close(result.value().latent, {1.2578125F, -0.32421875F}) ||
        !vector_close(result.value().expert_outputs[0], {0.76171875F, 0.3984375F}) ||
        !vector_close(result.value().expert_outputs[1], {0.81640625F, 0.81640625F}) ||
        !vector_close(result.value().mixed_latent, {0.78125F, 0.54296875F}) ||
        !vector_close(result.value().routed, {1.1640625F, 0.80859375F}) ||
        !vector_close(result.value().shared, {1.1953125F, 0.0439453125F}) ||
        !vector_close(result.value().combined, {2.359375F, 0.8515625F}) ||
        !vector_close(result.value().output, {2.859375F, -0.1484375F}))
        return 5;

    if (argc == 2 && std::string_view(argv[1]) == "--dump") {
        std::cout.precision(9);
        std::cout << "{\"expert_ids\":[";
        for (std::size_t index = 0; index < route.value().expert_ids.size(); ++index) {
            if (index) std::cout << ',';
            std::cout << route.value().expert_ids[index];
        }
        std::cout << "],\"contributions\":";
        print_vector(route.value().contributions);
        std::cout << ",\"hidden\":"; print_vector(result.value().hidden);
        std::cout << ",\"latent\":"; print_vector(result.value().latent);
        std::cout << ",\"expert_outputs\":[";
        for (std::size_t index = 0; index < result.value().expert_outputs.size(); ++index) {
            if (index) std::cout << ',';
            print_vector(result.value().expert_outputs[index]);
        }
        std::cout << "],\"mixed_latent\":";
        print_vector(result.value().mixed_latent);
        std::cout << ",\"routed\":"; print_vector(result.value().routed);
        std::cout << ",\"shared\":"; print_vector(result.value().shared);
        std::cout << ",\"combined\":"; print_vector(result.value().combined);
        std::cout << ",\"output\":"; print_vector(result.value().output);
        std::cout << "}\n";
        return 0;
    }

    auto bad_route = route.value();
    bad_route.expert_ids.pop_back();
    if (k3x::official_moe_cpu(input, weights, bad_route, 1.0e-5F, 4.0F, 25.0F))
        return 6;
    bad_route = route.value();
    bad_route.expert_ids[1] = bad_route.expert_ids[0];
    if (k3x::official_moe_cpu(input, weights, bad_route, 1.0e-5F, 4.0F, 25.0F))
        return 7;
    bad_route = route.value();
    bad_route.contributions[0] = std::numeric_limits<float>::infinity();
    if (k3x::official_moe_cpu(input, weights, bad_route, 1.0e-5F, 4.0F, 25.0F))
        return 8;
    bad_route = route.value();
    bad_route.contributions[0] = 0.5F;
    bad_route.contributions[1] = 0.4F;
    if (k3x::official_moe_cpu(input, weights, bad_route, 1.0e-5F, 4.0F, 25.0F))
        return 9;

    auto duplicate_experts = experts;
    duplicate_experts[1].expert_id = 0;
    auto bad_weights = weights;
    bad_weights.experts = duplicate_experts;
    if (k3x::official_moe_cpu(input, bad_weights, route.value(), 1.0e-5F, 4.0F, 25.0F))
        return 10;
    bad_weights = weights;
    bad_weights.experts = std::span<const k3x::OfficialExpertView>(experts).first(1);
    if (k3x::official_moe_cpu(input, bad_weights, route.value(), 1.0e-5F, 4.0F, 25.0F))
        return 11;
    bad_weights = weights;
    bad_weights.router.values = bad_weights.router.values.first(5);
    if (k3x::route_official_moe(hidden.value(), bad_weights.router,
                                bad_weights.correction, 2))
        return 12;
    return 0;
}
