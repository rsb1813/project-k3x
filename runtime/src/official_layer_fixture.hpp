// 공식 Kimi K3 레이어 fixture의 소유 데이터와 portable/CUDA view를 정의합니다.
#pragma once

#include "k3x/official_layer.hpp"
#include "k3x/reader.hpp"

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <span>
#include <vector>

namespace k3x::bench {

struct OwnedBf16 {
    std::uint64_t id{};
    std::vector<std::uint16_t> values;
    std::size_t rows{};
    std::size_t cols{};

    Bf16WeightView matrix() const;
    Bf16VectorView vector() const;
};

struct OwnedExpert {
    std::uint32_t layer_id{};
    std::uint32_t expert_id{};
    std::array<std::vector<std::byte>, 6> extents;

    Mxfp4MlpView view() const;
};

struct LoadedOfficialLayer {
    std::uint32_t layer_id{};
    OwnedBf16 self_norm, self_proj, input_norm;
    OwnedBf16 q_proj, k_proj, v_proj, f_a, f_b, beta, gate, o_proj;
    std::vector<float> q_conv, k_conv, v_conv, a_log, dt_bias, o_norm;
    OwnedBf16 mlp_norm, mlp_proj, post_norm, router;
    std::vector<float> correction;
    OwnedBf16 routed_down, routed_norm, routed_up;
    OwnedBf16 shared_gate, shared_up, shared_down;
    std::vector<OwnedExpert> experts;
    std::vector<Mxfp4MlpView> expert_mlp_views;
    std::vector<OfficialExpertView> expert_views;

    OfficialLayerWeights portable_views() const;
    OfficialLayerCudaWeights cuda_views() const;
};

std::optional<LoadedOfficialLayer> load_official_layer_fixture(
    Reader& reader, std::uint32_t layer_id,
    std::span<const std::uint32_t> selected_experts);

}  // namespace k3x::bench
