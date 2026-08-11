// 공식 Kimi K3 레이어 fixture를 엄격한 dtype/shape 계약으로 적재합니다.
#include "official_layer_fixture.hpp"

#include "k3x/storage_slice.hpp"

#include <algorithm>
#include <cstring>
#include <string>
#include <utility>

namespace k3x::bench {

Bf16WeightView OwnedBf16::matrix() const {
    return {values, rows, cols, id};
}

Bf16VectorView OwnedBf16::vector() const {
    return {values, id};
}

Mxfp4MlpView OwnedExpert::view() const {
    const auto base = "model.layers." + std::to_string(layer_id) +
        ".feed_forward.experts." + std::to_string(expert_id) + ".";
    return {
        {fnv1a64((base + "gate").c_str()), extents[0], extents[1],
         3'072, 3'584, 32},
        {fnv1a64((base + "up").c_str()), extents[2], extents[3],
         3'072, 3'584, 32},
        {fnv1a64((base + "down").c_str()), extents[4], extents[5],
         3'584, 3'072, 32}};
}

OfficialLayerWeights LoadedOfficialLayer::portable_views() const {
    const OfficialKdaWeightsView kda{
        q_proj.matrix(), k_proj.matrix(), v_proj.matrix(),
        q_conv, k_conv, v_conv, f_a.matrix(), f_b.matrix(), a_log,
        dt_bias, beta.matrix(), gate.matrix(), o_norm, o_proj.matrix()};
    const OfficialMoeWeights moe{
        mlp_norm.vector(), mlp_proj.matrix(), post_norm.vector(),
        router.matrix(), correction, routed_down.matrix(),
        routed_norm.vector(), routed_up.matrix(),
        {shared_gate.matrix(), shared_up.matrix(), shared_down.matrix()},
        expert_views};
    return {self_norm.vector(), self_proj.matrix(), input_norm.vector(),
            kda, moe};
}

OfficialLayerCudaWeights LoadedOfficialLayer::cuda_views() const {
    const auto base = "model.layers." + std::to_string(layer_id) + ".";
    const auto id = [&](std::string_view suffix) {
        return fnv1a64((base + std::string(suffix)).c_str());
    };
    const OfficialKdaCudaView kda{
        q_proj.matrix(), k_proj.matrix(), v_proj.matrix(),
        {id("self_attn.q_conv1d.weight"), q_conv, 12'288, 4},
        {id("self_attn.k_conv1d.weight"), k_conv, 12'288, 4},
        {id("self_attn.v_conv1d.weight"), v_conv, 12'288, 4},
        f_a.matrix(), f_b.matrix(),
        {id("self_attn.A_log"), a_log},
        {id("self_attn.dt_bias"), dt_bias}, beta.matrix(), gate.matrix(),
        {id("self_attn.o_norm.weight"), o_norm}, o_proj.matrix()};
    const OfficialMoeWeights moe{
        mlp_norm.vector(), mlp_proj.matrix(), post_norm.vector(),
        router.matrix(), correction, routed_down.matrix(),
        routed_norm.vector(), routed_up.matrix(),
        {shared_gate.matrix(), shared_up.matrix(), shared_down.matrix()},
        expert_views};
    const OfficialMoeFfnView moe_ffn{
        routed_down.matrix(), routed_norm.vector(), routed_up.matrix(),
        {shared_gate.matrix(), shared_up.matrix(), shared_down.matrix()}};
    return {self_norm.vector(), self_proj.matrix(), input_norm.vector(),
            kda, moe, moe_ffn};
}

namespace {

const TensorRecord* find_record(const Reader& reader, const std::string& name) {
    const auto id = fnv1a64(name.c_str());
    const auto found = std::find_if(
        reader.tensors().begin(), reader.tensors().end(),
        [id](const TensorRecord& record) { return record.tensor_id == id; });
    return found == reader.tensors().end() ? nullptr : &*found;
}

std::optional<OwnedBf16> load_bf16(
    Reader& reader, const std::string& name,
    std::size_t rows, std::size_t cols) {
    const auto* record = find_record(reader, name);
    if (!record || record->dtype != 3 || record->quantization != 0 ||
        record->data_length != rows * cols * 2 || record->auxiliary_length) {
        return std::nullopt;
    }
    auto bytes = reader.read_tensor(record->tensor_id);
    if (!bytes || bytes.value().size() != record->data_length) {
        return std::nullopt;
    }
    OwnedBf16 result{record->tensor_id,
                     std::vector<std::uint16_t>(rows * cols), rows, cols};
    std::memcpy(result.values.data(), bytes.value().data(), bytes.value().size());
    return result;
}

std::optional<std::vector<float>> load_f32(
    Reader& reader, const std::string& name, std::size_t count) {
    const auto* record = find_record(reader, name);
    if (!record || record->dtype != 1 || record->quantization != 0 ||
        record->data_length != count * sizeof(float) ||
        record->auxiliary_length) {
        return std::nullopt;
    }
    auto bytes = reader.read_tensor(record->tensor_id);
    if (!bytes || bytes.value().size() != record->data_length) {
        return std::nullopt;
    }
    std::vector<float> result(count);
    std::memcpy(result.data(), bytes.value().data(), bytes.value().size());
    return result;
}

}  // namespace

std::optional<LoadedOfficialLayer> load_official_layer_fixture(
    Reader& reader, std::uint32_t layer_id,
    std::span<const std::uint32_t> selected_experts) {
    if (layer_id < 1 || layer_id > 2 || selected_experts.empty()) {
        return std::nullopt;
    }
    const auto base = "model.layers." + std::to_string(layer_id) + ".";
    auto self_norm = load_bf16(reader, base + "self_attention_res_norm.weight", 1, 7'168);
    auto self_proj = load_bf16(reader, base + "self_attention_res_proj.weight", 1, 7'168);
    auto input_norm = load_bf16(reader, base + "input_layernorm.weight", 1, 7'168);
    auto q_proj = load_bf16(reader, base + "self_attn.q_proj.weight", 12'288, 7'168);
    auto k_proj = load_bf16(reader, base + "self_attn.k_proj.weight", 12'288, 7'168);
    auto v_proj = load_bf16(reader, base + "self_attn.v_proj.weight", 12'288, 7'168);
    auto f_a = load_bf16(reader, base + "self_attn.f_a_proj.weight", 128, 7'168);
    auto f_b = load_bf16(reader, base + "self_attn.f_b_proj.weight", 12'288, 128);
    auto beta = load_bf16(reader, base + "self_attn.b_proj.weight", 96, 7'168);
    auto gate = load_bf16(reader, base + "self_attn.g_proj.weight", 12'288, 7'168);
    auto o_proj = load_bf16(reader, base + "self_attn.o_proj.weight", 7'168, 12'288);
    auto q_conv = load_f32(reader, base + "self_attn.q_conv1d.weight", 12'288 * 4);
    auto k_conv = load_f32(reader, base + "self_attn.k_conv1d.weight", 12'288 * 4);
    auto v_conv = load_f32(reader, base + "self_attn.v_conv1d.weight", 12'288 * 4);
    auto a_log = load_f32(reader, base + "self_attn.A_log", 128);
    auto dt_bias = load_f32(reader, base + "self_attn.dt_bias", 12'288);
    auto o_norm = load_f32(reader, base + "self_attn.o_norm.weight", 128);
    auto mlp_norm = load_bf16(reader, base + "mlp_res_norm.weight", 1, 7'168);
    auto mlp_proj = load_bf16(reader, base + "mlp_res_proj.weight", 1, 7'168);
    auto post_norm = load_bf16(reader, base + "post_attention_layernorm.weight", 1, 7'168);
    auto router = load_bf16(reader, base + "block_sparse_moe.gate.weight", 896, 7'168);
    auto correction = load_f32(
        reader, base + "block_sparse_moe.gate.e_score_correction_bias", 896);
    auto routed_down = load_bf16(
        reader, base + "block_sparse_moe.routed_expert_down_proj.weight", 3'584, 7'168);
    auto routed_norm = load_bf16(
        reader, base + "block_sparse_moe.routed_expert_norm.weight", 1, 3'584);
    auto routed_up = load_bf16(
        reader, base + "block_sparse_moe.routed_expert_up_proj.weight", 7'168, 3'584);
    auto shared_gate = load_bf16(
        reader, base + "block_sparse_moe.shared_experts.gate_proj.weight", 6'144, 7'168);
    auto shared_up = load_bf16(
        reader, base + "block_sparse_moe.shared_experts.up_proj.weight", 6'144, 7'168);
    auto shared_down = load_bf16(
        reader, base + "block_sparse_moe.shared_experts.down_proj.weight", 7'168, 6'144);
    if (!self_norm || !self_proj || !input_norm || !q_proj || !k_proj ||
        !v_proj || !f_a || !f_b || !beta || !gate || !o_proj || !q_conv ||
        !k_conv || !v_conv || !a_log || !dt_bias || !o_norm || !mlp_norm ||
        !mlp_proj || !post_norm || !router || !correction || !routed_down ||
        !routed_norm || !routed_up || !shared_gate || !shared_up ||
        !shared_down) {
        return std::nullopt;
    }
    LoadedOfficialLayer result{
        layer_id, std::move(*self_norm), std::move(*self_proj),
        std::move(*input_norm), std::move(*q_proj), std::move(*k_proj),
        std::move(*v_proj), std::move(*f_a), std::move(*f_b),
        std::move(*beta), std::move(*gate), std::move(*o_proj),
        std::move(*q_conv), std::move(*k_conv), std::move(*v_conv),
        std::move(*a_log), std::move(*dt_bias), std::move(*o_norm),
        std::move(*mlp_norm), std::move(*mlp_proj), std::move(*post_norm),
        std::move(*router), std::move(*correction), std::move(*routed_down),
        std::move(*routed_norm), std::move(*routed_up),
        std::move(*shared_gate), std::move(*shared_up),
        std::move(*shared_down), {}, {}, {}};
    result.experts.reserve(selected_experts.size());
    for (const auto expert_id : selected_experts) {
        auto expert = load_storage_expert(reader, layer_id, expert_id);
        if (!expert) return std::nullopt;
        result.experts.push_back(
            {layer_id, expert_id, std::move(expert.value().extents)});
    }
    result.expert_mlp_views.reserve(result.experts.size());
    result.expert_views.reserve(result.experts.size());
    for (const auto& expert : result.experts) {
        result.expert_mlp_views.push_back(expert.view());
        result.expert_views.push_back(
            {expert.expert_id, result.expert_mlp_views.back()});
    }
    return result;
}

}  // namespace k3x::bench
