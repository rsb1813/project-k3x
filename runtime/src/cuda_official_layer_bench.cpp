// 공식 Kimi K3 complete-layer artifact를 CUDA backend 생성 전에 엄격히 검증합니다.
#include "k3x/checksums.hpp"
#include "k3x/format.hpp"
#include "k3x/reader.hpp"
#include "k3x/status.hpp"
#include "k3x/strict_json.hpp"

#include <algorithm>
#include <array>
#include <charconv>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

namespace {

namespace json = k3x::strict_json;

constexpr std::size_t kTopK = 16;
constexpr std::uint64_t kShardBytes = 16'990'911'504ULL;

enum class CaseMode { a, ab_full, ab_incremental };
enum class WeightMode { transient, resident };

struct Arguments {
    std::filesystem::path artifact;
    std::filesystem::path manifest;
    CaseMode case_mode{CaseMode::a};
    WeightMode weight_mode{WeightMode::transient};
    std::uint64_t warmups{};
    std::uint64_t iterations{1};
};

std::optional<std::uint64_t> parse_u64(std::string_view text) {
    std::uint64_t value{};
    const auto parsed = std::from_chars(text.data(), text.data() + text.size(), value);
    return !text.empty() && parsed.ec == std::errc{} &&
                   parsed.ptr == text.data() + text.size()
        ? std::optional(value)
        : std::nullopt;
}

std::optional<Arguments> parse_arguments(int argc, char** argv) {
    Arguments result;
    std::string case_name{"a"};
    std::string weight_name{"transient"};
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc) {
            std::cerr << "missing option value\n";
            return std::nullopt;
        }
        const std::string key = argv[index];
        const std::string value = argv[index + 1];
        const auto number = parse_u64(value);
        if (key == "--artifact") result.artifact = value;
        else if (key == "--manifest") result.manifest = value;
        else if (key == "--case") case_name = value;
        else if (key == "--weight-mode") weight_name = value;
        else if (key == "--warmups" && number) result.warmups = *number;
        else if (key == "--iterations" && number) result.iterations = *number;
        else {
            std::cerr << "invalid option: " << key << '\n';
            return std::nullopt;
        }
    }
    if (case_name == "a") result.case_mode = CaseMode::a;
    else if (case_name == "ab-full") result.case_mode = CaseMode::ab_full;
    else if (case_name == "ab-incremental") result.case_mode = CaseMode::ab_incremental;
    else {
        std::cerr << "unknown case: " << case_name << '\n';
        return std::nullopt;
    }
    if (weight_name == "transient") result.weight_mode = WeightMode::transient;
    else if (weight_name == "resident") result.weight_mode = WeightMode::resident;
    else {
        std::cerr << "unknown weight mode: " << weight_name << '\n';
        return std::nullopt;
    }
    if (!result.iterations) {
        std::cerr << "iterations must be positive\n";
        return std::nullopt;
    }
    if (result.artifact.empty()) {
        std::cerr << "artifact path is required\n";
        return std::nullopt;
    }
    if (result.manifest.empty()) {
        std::cerr << "manifest path is required\n";
        return std::nullopt;
    }
    return result;
}

bool hex_string(std::string_view value, std::size_t size) {
    return value.size() == size &&
           std::all_of(value.begin(), value.end(), [](char character) {
               return (character >= '0' && character <= '9') ||
                      (character >= 'a' && character <= 'f');
           });
}

bool text_is(const json::Value& root, std::string_view key,
             std::string_view expected) {
    const auto* value = json::string(json::member(root, key));
    return value && *value == expected;
}

bool integer_is(const json::Value& root, std::string_view key,
                std::uint64_t expected) {
    const auto* value = json::number(json::member(root, key));
    return value && *value >= 0 && std::floor(*value) == *value &&
           *value == static_cast<double>(expected);
}

std::optional<std::vector<std::uint32_t>> expert_ids(const json::Value* value) {
    const auto* array = json::array(value);
    if (!array) return std::nullopt;
    std::vector<std::uint32_t> result;
    std::set<std::uint32_t> unique;
    for (const auto& item : *array) {
        const auto* number = std::get_if<double>(&item.value);
        if (!number || *number < 0 || *number > 895 ||
            std::floor(*number) != *number ||
            !unique.insert(static_cast<std::uint32_t>(*number)).second) {
            return std::nullopt;
        }
        result.push_back(static_cast<std::uint32_t>(*number));
    }
    return result;
}

struct Route {
    std::vector<std::uint32_t> ids;
    std::vector<float> contributions;
};

struct Manifest {
    std::string root_sha256;
    std::string source_sha256;
    std::string k3x_source_fingerprint_sha256;
    std::map<std::string, std::string> tensor_sha256;
    std::array<Route, 2> routes;
    std::vector<std::uint32_t> selected;
};

constexpr std::array<std::string_view, 17> kKdaSuffixes{
    "self_attention_res_norm.weight", "self_attention_res_proj.weight",
    "input_layernorm.weight", "self_attn.q_proj.weight",
    "self_attn.q_conv1d.weight", "self_attn.k_proj.weight",
    "self_attn.k_conv1d.weight", "self_attn.v_proj.weight",
    "self_attn.v_conv1d.weight", "self_attn.f_a_proj.weight",
    "self_attn.f_b_proj.weight", "self_attn.A_log", "self_attn.dt_bias",
    "self_attn.b_proj.weight", "self_attn.g_proj.weight",
    "self_attn.o_norm.weight", "self_attn.o_proj.weight"};

constexpr std::array<std::pair<std::uint64_t, std::uint64_t>, 17> kKdaRanges{{
    {381'373'456, 14'336}, {381'387'792, 14'336},
    {381'316'112, 14'336}, {916'241'424, 176'160'768},
    {1'069'072, 196'608}, {563'919'888, 176'160'768},
    {871'952, 196'608}, {1'092'402'192, 176'160'768},
    {1'265'680, 196'608}, {382'778'384, 1'835'008},
    {384'613'392, 3'145'728}, {822'288, 512}, {822'800, 49'152},
    {381'402'128, 1'376'256}, {387'759'120, 176'160'768},
    {1'068'560, 512}, {740'080'656, 176'160'768}}};

constexpr std::array<std::string_view, 11> kMoeSuffixes{
    "mlp_res_norm.weight", "mlp_res_proj.weight",
    "post_attention_layernorm.weight", "block_sparse_moe.gate.weight",
    "block_sparse_moe.gate.e_score_correction_bias",
    "block_sparse_moe.routed_expert_down_proj.weight",
    "block_sparse_moe.routed_expert_norm.weight",
    "block_sparse_moe.routed_expert_up_proj.weight",
    "block_sparse_moe.shared_experts.gate_proj.weight",
    "block_sparse_moe.shared_experts.up_proj.weight",
    "block_sparse_moe.shared_experts.down_proj.weight"};

constexpr std::array<std::pair<std::uint64_t, std::uint64_t>, 11> kMoeRanges{{
    {381'330'448, 14'336}, {381'344'784, 14'336},
    {381'359'120, 14'336}, {1'462'288, 12'845'056},
    {818'704, 3'584}, {14'307'344, 51'380'224},
    {65'687'568, 7'168}, {65'694'736, 51'380'224},
    {205'155'344, 88'080'384}, {293'235'728, 88'080'384},
    {117'074'960, 88'080'384}}};

bool valid_object_list(const json::Value& root, std::string_view key,
                       std::span<const std::string_view> suffixes,
                       std::span<const std::pair<std::uint64_t, std::uint64_t>> ranges) {
    const auto* values = json::array(json::member(root, key));
    if (!values || values->size() != suffixes.size() ||
        ranges.size() != suffixes.size()) return false;
    const std::string prefix = "language_model.model.layers.1.";
    std::set<std::pair<std::uint64_t, std::uint64_t>> unique_ranges;
    for (std::size_t index = 0; index < values->size(); ++index) {
        if (!text_is((*values)[index], "name", prefix + std::string(suffixes[index]))) {
            return false;
        }
        const auto* range = json::array(json::member((*values)[index], "range"));
        const auto* digest = json::string(json::member((*values)[index], "sha256"));
        if (!range || range->size() != 2 || !digest || !hex_string(*digest, 64)) {
            return false;
        }
        const auto* begin = std::get_if<double>(&(*range)[0].value);
        const auto* end = std::get_if<double>(&(*range)[1].value);
        if (!begin || !end || *begin < 818'704 || *end <= *begin ||
            std::floor(*begin) != *begin || std::floor(*end) != *end ||
            *end > static_cast<double>(kShardBytes) ||
            static_cast<std::uint64_t>(*begin) != ranges[index].first ||
            static_cast<std::uint64_t>(*end) !=
                ranges[index].first + ranges[index].second ||
            !unique_ranges.emplace(static_cast<std::uint64_t>(*begin),
                                   static_cast<std::uint64_t>(*end)).second) {
            return false;
        }
    }
    return true;
}

std::set<std::string> expected_tensor_digests(
    std::span<const std::uint32_t> selected) {
    const std::string prefix = "model.layers.1.";
    std::set<std::string> result;
    for (const auto suffix : kKdaSuffixes) result.insert(prefix + std::string(suffix));
    for (const auto suffix : kMoeSuffixes) result.insert(prefix + std::string(suffix));
    for (const auto id : selected) {
        const auto expert = prefix + "feed_forward.experts." + std::to_string(id) + ".";
        for (const auto role : {"gate", "up", "down"}) {
            result.insert(expert + role + ".weight_packed");
            result.insert(expert + role + ".weight_scale");
        }
    }
    return result;
}

std::optional<Manifest> load_manifest(const std::filesystem::path& path,
                                     bool& syntax_valid) {
    std::error_code error;
    const auto size = std::filesystem::file_size(path, error);
    if (error || size > 16 * 1024 * 1024) return std::nullopt;
    std::ifstream stream(path, std::ios::binary);
    const std::string text((std::istreambuf_iterator<char>(stream)), {});
    auto root = json::Parser(text).parse();
    if (!root) return std::nullopt;
    syntax_valid = true;
    if (!text_is(*root, "format", "k3x-official-kda-layer-routes-v1") ||
        !text_is(*root, "converter_version", "k3x-converter-0.1.0") ||
        !text_is(*root, "repository", "moonshotai/Kimi-K3") ||
        !text_is(*root, "requested_revision", "main") ||
        !text_is(*root, "resolved_revision", "9f62e4e9fffbd0a83ddd60e1c209d828994b3569") ||
        !text_is(*root, "snapshot_sha256", "deaa6394b80afe12976ce8efbbf2463f6808c291d83b029e6b0cfb98de90a4e5") ||
        !text_is(*root, "index_sha256", "a1c5210650ce71d2d3ae9ec5a101ac4afd3cf4b10091be589853437eb967febd") ||
        !text_is(*root, "config_sha256", "9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213") ||
        !text_is(*root, "config_git_blob_id", "d7f26ead420b1d967f2759679dbebc65edfcff93") ||
        !text_is(*root, "source_blob_id", "b8c41e8bfce768d74d8da3a37e693f5ee43876a0") ||
        !text_is(*root, "shard_path", "model-00002-of-000096.safetensors") ||
        !text_is(*root, "shard_lfs_sha256", "26a3284e1d2cb567934ebef002e6a1813551d646739e8bcb1e9e3fe7f878e0f5") ||
        !text_is(*root, "state_layout", "v-first-fp32") ||
        !text_is(*root, "provenance", "transport-pinned-ranges")) {
        return std::nullopt;
    }
    const auto* header = json::member(*root, "header");
    if (!header || !integer_is(*header, "file_size", kShardBytes) ||
        !integer_is(*header, "header_length", 818'696) ||
        !integer_is(*header, "data_start", 818'704)) {
        return std::nullopt;
    }
    const auto* initial = json::string(json::member(*root, "initial_state_sha256"));
    const auto* final = json::string(json::member(*root, "final_state_sha256"));
    if (!initial || !final || !hex_string(*initial, 64) || !hex_string(*final, 64)) {
        return std::nullopt;
    }
    const auto* inputs = json::array(json::member(*root, "inputs"));
    if (!inputs || inputs->size() != 2 ||
        !text_is((*inputs)[0], "name", "a") ||
        !text_is((*inputs)[0], "hidden_sha256", "acc7746e19fcb6bb17d09ce08d387ca91d3a742c4f671046aaa0184a290d2cc3") ||
        !text_is((*inputs)[0], "block_sha256", "c7d98135ee7f46f4d82822d2e267d368dcdee51411575e578e63385a12e9bc3e") ||
        !text_is((*inputs)[1], "name", "b") ||
        !text_is((*inputs)[1], "hidden_sha256", "9b8f886591586999d0fb6a9661c938e24f2ade01cfdfbe352ea57961a642d566") ||
        !text_is((*inputs)[1], "block_sha256", "323b027923f323953dc12c6bc16618672e84d264891c6ed0a9aa3383b0045046")) {
        return std::nullopt;
    }
    const auto* steps = json::array(json::member(*root, "steps"));
    if (!steps || steps->size() != 2) return std::nullopt;
    Manifest result;
    std::vector<std::uint32_t> expected_union;
    std::set<std::uint32_t> union_set;
    std::string expected_consumed = *initial;
    for (std::size_t index = 0; index < 2; ++index) {
        const auto& step = (*steps)[index];
        const auto* consumed = json::string(json::member(step, "consumes_state_sha256"));
        const auto* state = json::string(json::member(step, "state_sha256"));
        const auto* output = json::string(json::member(step, "kda_output_sha256"));
        auto ids = expert_ids(json::member(step, "expert_ids"));
        const auto* contributions = json::array(json::member(step, "contributions"));
        if (!text_is(step, "name", index ? "b" : "a") || !consumed ||
            *consumed != expected_consumed || !state || !hex_string(*state, 64) ||
            !output || !hex_string(*output, 64) || !ids || ids->size() != kTopK ||
            !contributions || contributions->size() != kTopK) {
            return std::nullopt;
        }
        double total{};
        for (const auto& item : *contributions) {
            const auto* contribution = std::get_if<double>(&item.value);
            if (!contribution || *contribution <= 0 || !std::isfinite(*contribution)) {
                return std::nullopt;
            }
            total += *contribution;
            result.routes[index].contributions.push_back(
                static_cast<float>(*contribution));
        }
        if (std::abs(total - 1.0) > 1.0e-5) return std::nullopt;
        for (const auto id : *ids) {
            if (union_set.insert(id).second) expected_union.push_back(id);
        }
        result.routes[index].ids = std::move(*ids);
        expected_consumed = *state;
    }
    if (expected_consumed != *final) return std::nullopt;
    auto selected = expert_ids(json::member(*root, "selected_experts"));
    if (!selected || selected->empty() || selected->size() > 32 ||
        *selected != expected_union) {
        return std::nullopt;
    }
    result.selected = std::move(*selected);
    if (!valid_object_list(*root, "kda_objects", kKdaSuffixes, kKdaRanges) ||
        !valid_object_list(*root, "always_active_objects", kMoeSuffixes,
                           kMoeRanges)) {
        return std::nullopt;
    }
    const auto* expert_objects = json::array(json::member(*root, "expert_objects"));
    if (!expert_objects || expert_objects->size() != result.selected.size()) {
        return std::nullopt;
    }
    for (std::size_t index = 0; index < expert_objects->size(); ++index) {
        if (!integer_is((*expert_objects)[index], "expert_id", result.selected[index])) {
            return std::nullopt;
        }
        const auto* range = json::array(json::member((*expert_objects)[index], "range"));
        const auto* digest = json::string(json::member((*expert_objects)[index], "sha256"));
        if (!range || range->size() != 2 || !digest || !hex_string(*digest, 64)) {
            return std::nullopt;
        }
        const auto* begin = std::get_if<double>(&(*range)[0].value);
        const auto* end = std::get_if<double>(&(*range)[1].value);
        if (!begin || !end || *begin < 818'704 || *end <= *begin ||
            std::floor(*begin) != *begin || std::floor(*end) != *end ||
            *end > static_cast<double>(kShardBytes) ||
            static_cast<std::uint64_t>(*end - *begin) != 17'547'264) {
            return std::nullopt;
        }
    }
    const auto* artifact = json::member(*root, "artifact");
    const auto* root_digest = artifact
        ? json::string(json::member(*artifact, "k3x_root_sha256")) : nullptr;
    const auto* source_digest = artifact
        ? json::string(json::member(*artifact, "source_sha256")) : nullptr;
    const auto* source_fingerprint = artifact
        ? json::string(json::member(*artifact, "k3x_source_fingerprint_sha256"))
        : nullptr;
    const auto* tensor_digests = artifact
        ? json::object(json::member(*artifact, "tensor_sha256")) : nullptr;
    if (!artifact || !text_is(*artifact, "filename", "official-kda-layer-l1.k3x") ||
        !root_digest || !source_digest || !source_fingerprint ||
        !hex_string(*root_digest, 64) || !hex_string(*source_digest, 64) ||
        !hex_string(*source_fingerprint, 64) || !tensor_digests) {
        return std::nullopt;
    }
    const auto expected = expected_tensor_digests(result.selected);
    if (tensor_digests->size() != expected.size()) return std::nullopt;
    for (const auto& name : expected) {
        const auto found = tensor_digests->find(name);
        const auto* digest = found == tensor_digests->end()
            ? nullptr : std::get_if<std::string>(&found->second.value);
        if (!digest || !hex_string(*digest, 64)) return std::nullopt;
        result.tensor_sha256.emplace(name, *digest);
    }
    result.root_sha256 = *root_digest;
    result.source_sha256 = *source_digest;
    result.k3x_source_fingerprint_sha256 = *source_fingerprint;
    return result;
}

std::string digest_hex(std::span<const std::byte> digest) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string result(digest.size() * 2, '0');
    for (std::size_t index = 0; index < digest.size(); ++index) {
        const auto value = std::to_integer<unsigned>(digest[index]);
        result[index * 2] = digits[value >> 4];
        result[index * 2 + 1] = digits[value & 15];
    }
    return result;
}

struct TensorSpec {
    std::string name;
    std::uint16_t dtype{};
    std::array<std::uint64_t, 4> dimensions{};
    std::uint8_t rank{};
    std::int32_t expert_id{-1};
};

void add_dense(std::vector<TensorSpec>& specs, std::string name,
               std::uint16_t dtype, std::uint64_t first,
               std::uint64_t second = 0) {
    specs.push_back({std::move(name), dtype, {first, second},
                     static_cast<std::uint8_t>(second ? 2 : 1), -1});
}

void add_dense3(std::vector<TensorSpec>& specs, std::string name,
                std::uint16_t dtype, std::uint64_t first,
                std::uint64_t second, std::uint64_t third) {
    specs.push_back({std::move(name), dtype, {first, second, third, 0}, 3, -1});
}

std::vector<TensorSpec> expected_records(std::span<const std::uint32_t> selected) {
    const std::string base = "model.layers.1.";
    std::vector<TensorSpec> result;
    add_dense(result, base + "self_attention_res_norm.weight", 3, 7'168);
    add_dense(result, base + "self_attention_res_proj.weight", 3, 1, 7'168);
    add_dense(result, base + "input_layernorm.weight", 3, 7'168);
    add_dense(result, base + "self_attn.q_proj.weight", 3, 12'288, 7'168);
    add_dense3(result, base + "self_attn.q_conv1d.weight", 1, 12'288, 1, 4);
    add_dense(result, base + "self_attn.k_proj.weight", 3, 12'288, 7'168);
    add_dense3(result, base + "self_attn.k_conv1d.weight", 1, 12'288, 1, 4);
    add_dense(result, base + "self_attn.v_proj.weight", 3, 12'288, 7'168);
    add_dense3(result, base + "self_attn.v_conv1d.weight", 1, 12'288, 1, 4);
    add_dense(result, base + "self_attn.f_a_proj.weight", 3, 128, 7'168);
    add_dense(result, base + "self_attn.f_b_proj.weight", 3, 12'288, 128);
    add_dense(result, base + "self_attn.A_log", 1, 128);
    add_dense(result, base + "self_attn.dt_bias", 1, 12'288);
    add_dense(result, base + "self_attn.b_proj.weight", 3, 96, 7'168);
    add_dense(result, base + "self_attn.g_proj.weight", 3, 12'288, 7'168);
    add_dense(result, base + "self_attn.o_norm.weight", 1, 128);
    add_dense(result, base + "self_attn.o_proj.weight", 3, 7'168, 12'288);
    add_dense(result, base + "mlp_res_norm.weight", 3, 7'168);
    add_dense(result, base + "mlp_res_proj.weight", 3, 1, 7'168);
    add_dense(result, base + "post_attention_layernorm.weight", 3, 7'168);
    add_dense(result, base + "block_sparse_moe.gate.weight", 3, 896, 7'168);
    add_dense(result, base + "block_sparse_moe.gate.e_score_correction_bias", 1, 896);
    add_dense(result, base + "block_sparse_moe.routed_expert_down_proj.weight", 3, 3'584, 7'168);
    for (const auto id : selected) {
        const auto expert = base + "feed_forward.experts." + std::to_string(id) + ".";
        result.push_back({expert + "gate", 2, {3'072, 3'584}, 2,
                          static_cast<std::int32_t>(id)});
        result.push_back({expert + "up", 2, {3'072, 3'584}, 2,
                          static_cast<std::int32_t>(id)});
        result.push_back({expert + "down", 2, {3'584, 3'072}, 2,
                          static_cast<std::int32_t>(id)});
    }
    add_dense(result, base + "block_sparse_moe.routed_expert_norm.weight", 3, 3'584);
    add_dense(result, base + "block_sparse_moe.routed_expert_up_proj.weight", 3, 7'168, 3'584);
    add_dense(result, base + "block_sparse_moe.shared_experts.gate_proj.weight", 3, 6'144, 7'168);
    add_dense(result, base + "block_sparse_moe.shared_experts.up_proj.weight", 3, 6'144, 7'168);
    add_dense(result, base + "block_sparse_moe.shared_experts.down_proj.weight", 3, 7'168, 6'144);
    return result;
}

struct MicroTensorSpec {
    std::string name;
    std::string_view dtype;
    std::array<std::uint64_t, 4> dimensions{};
    std::uint8_t rank{};
    std::uint64_t length{};
};

std::vector<MicroTensorSpec> microshard_tensors(
    std::span<const TensorSpec> records) {
    std::vector<MicroTensorSpec> result;
    for (const auto& record : records) {
        std::uint64_t values = 1;
        for (std::size_t index = 0; index < record.rank; ++index) {
            values *= record.dimensions[index];
        }
        if (record.dtype == 2) {
            result.push_back({record.name + ".weight_packed", "U8",
                              {values / 2, 0}, 1, values / 2});
            result.push_back({record.name + ".weight_scale", "U8",
                              {values / 32, 0}, 1, values / 32});
        } else {
            result.push_back({record.name, record.dtype == 1 ? "F32" : "BF16",
                              record.dimensions, record.rank,
                              values * (record.dtype == 1 ? 4ULL : 2ULL)});
        }
    }
    return result;
}

std::string microshard_header(std::span<const MicroTensorSpec> tensors) {
    std::string result{"{"};
    std::uint64_t offset{};
    for (std::size_t index = 0; index < tensors.size(); ++index) {
        const auto& tensor = tensors[index];
        if (index) result += ',';
        result += '"' + tensor.name + "\":{\"dtype\":\"";
        result += tensor.dtype;
        result += "\",\"shape\":[" + std::to_string(tensor.dimensions[0]);
        for (std::size_t dimension = 1; dimension < tensor.rank; ++dimension) {
            result += ',' + std::to_string(tensor.dimensions[dimension]);
        }
        result += "],\"data_offsets\":[" + std::to_string(offset) + ',' +
                  std::to_string(offset + tensor.length) + "]}";
        offset += tensor.length;
    }
    result += '}';
    return result;
}

bool validate_artifact(k3x::Reader& reader, const Manifest& manifest) {
    if (digest_hex(reader.superblock().root_sha256) != manifest.root_sha256 ||
        digest_hex(reader.superblock().source_sha256) !=
            manifest.k3x_source_fingerprint_sha256) {
        return false;
    }
    const auto specs = expected_records(manifest.selected);
    if (reader.tensors().size() != specs.size()) return false;
    const auto micro_tensors = microshard_tensors(specs);
    const auto header = microshard_header(micro_tensors);
    std::array<std::byte, 8> header_length{};
    auto length = static_cast<std::uint64_t>(header.size());
    for (std::size_t index = 0; index < header_length.size(); ++index) {
        header_length[index] = std::byte((length >> (index * 8)) & 0xffU);
    }
    k3x::Sha256Hasher microshard;
    microshard.update(header_length);
    microshard.update(std::as_bytes(std::span(header)));
    for (std::size_t index = 0; index < specs.size(); ++index) {
        const auto& record = reader.tensors()[index];
        const auto& spec = specs[index];
        if (record.tensor_id != k3x::fnv1a64(spec.name.c_str()) ||
            record.dtype != spec.dtype || record.rank != spec.rank ||
            record.layer_id != 1 || record.expert_id != spec.expert_id ||
            record.dimensions != spec.dimensions) {
            return false;
        }
        if (spec.dtype == 2) {
            const auto values = spec.dimensions[0] * spec.dimensions[1];
            if (record.quantization != 1 || record.data_length != values / 2 ||
                record.auxiliary_length != values / 32) {
                return false;
            }
            auto packed = reader.read_tensor(record.tensor_id);
            auto scale = reader.read_auxiliary(record.tensor_id);
            if (!packed || !scale ||
                digest_hex(k3x::sha256(packed.value())) !=
                    manifest.tensor_sha256.at(spec.name + ".weight_packed") ||
                digest_hex(k3x::sha256(scale.value())) !=
                    manifest.tensor_sha256.at(spec.name + ".weight_scale")) {
                return false;
            }
            microshard.update(packed.value());
            microshard.update(scale.value());
        } else {
            const auto element_bytes = spec.dtype == 1 ? 4ULL : 2ULL;
            std::uint64_t values = 1;
            for (std::size_t dimension = 0; dimension < spec.rank; ++dimension) {
                values *= spec.dimensions[dimension];
            }
            if (record.quantization != 0 || record.data_length != values * element_bytes ||
                record.auxiliary_length != 0) {
                return false;
            }
            auto data = reader.read_tensor(record.tensor_id);
            if (!data || digest_hex(k3x::sha256(data.value())) !=
                             manifest.tensor_sha256.at(spec.name)) {
                return false;
            }
            microshard.update(data.value());
        }
    }
    return digest_hex(microshard.finish()) == manifest.source_sha256;
}

void write_error(k3x::ErrorCode code, std::string_view message) {
    std::cerr << k3x::error_code_name(code) << ": " << message << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    const auto arguments = parse_arguments(argc, argv);
    if (!arguments) return 2;
    bool syntax_valid = false;
    const auto manifest = load_manifest(arguments->manifest, syntax_valid);
    if (!manifest) {
        write_error(k3x::ErrorCode::invalid_extent,
                    syntax_valid ? "official layer manifest identity mismatch"
                                 : "invalid official layer manifest");
        return 4;
    }
    auto reader = k3x::Reader::open(arguments->artifact, k3x::VerifyMode::checksums);
    if (!reader) {
        write_error(reader.error(), reader.message());
        return 4;
    }
    if (reader.value().superblock().optional_features !=
        (k3x::optional_storage_fixture | k3x::optional_official_moe_fixture)) {
        write_error(k3x::ErrorCode::invalid_extent,
                    "artifact is not official layer fixture");
        return 4;
    }
    if (!validate_artifact(reader.value(), *manifest)) {
        write_error(k3x::ErrorCode::invalid_extent,
                    "official layer artifact identity mismatch");
        return 4;
    }
    write_error(k3x::ErrorCode::backend_unavailable,
                "official layer CUDA execution is not implemented");
    return 4;
}
