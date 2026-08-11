// 공식 Kimi K3 complete-layer artifact를 CUDA backend 생성 전에 엄격히 검증합니다.
#include "k3x/checksums.hpp"
#include "k3x/format.hpp"
#include "k3x/official_layer.hpp"
#include "k3x/reader.hpp"
#include "k3x/status.hpp"
#include "k3x/storage_slice.hpp"
#include "k3x/strict_json.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <optional>
#include <numeric>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <utility>
#include <vector>

#include <sys/resource.h>

namespace {

namespace json = k3x::strict_json;

constexpr std::size_t kTopK = 16;
constexpr std::uint64_t kShardBytes = 16'990'911'504ULL;
constexpr std::uint64_t kOracleBytes = 6'541'344;
constexpr double kOutputAbsoluteTolerance = 2.0e-5;
constexpr double kConvolutionAbsoluteTolerance = 8.0e-3;
constexpr double kRecurrentAbsoluteTolerance = 5.0e-5;
constexpr double kContributionAbsoluteTolerance = 2.0e-6;

enum class CaseMode { a, ab_full, ab_incremental };
enum class WeightMode { transient, resident };
enum class StateTransfer { host, device };
enum class RoutePreparation { host, device };

struct Arguments {
    std::filesystem::path artifact;
    std::filesystem::path manifest;
    CaseMode case_mode{CaseMode::a};
    WeightMode weight_mode{WeightMode::transient};
    k3x::CudaWeightValidationMode validation{
        k3x::CudaWeightValidationMode::per_call};
    bool validation_explicit{};
    StateTransfer state_transfer{StateTransfer::host};
    bool state_transfer_explicit{};
    RoutePreparation route_preparation{RoutePreparation::host};
    bool route_preparation_explicit{};
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
    std::string validation_name{"per-call"};
    std::string state_transfer_name{"host"};
    std::string route_preparation_name{"host"};
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
        else if (key == "--validation") {
            validation_name = value;
            result.validation_explicit = true;
        }
        else if (key == "--state-transfer") {
            state_transfer_name = value;
            result.state_transfer_explicit = true;
        }
        else if (key == "--route-preparation") {
            route_preparation_name = value;
            result.route_preparation_explicit = true;
        }
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
    if (validation_name == "per-call") {
        result.validation = k3x::CudaWeightValidationMode::per_call;
    } else if (validation_name == "admission") {
        result.validation = k3x::CudaWeightValidationMode::admission;
    } else {
        std::cerr << "unknown validation mode: " << validation_name << '\n';
        return std::nullopt;
    }
    if (state_transfer_name == "host") {
        result.state_transfer = StateTransfer::host;
    } else if (state_transfer_name == "device") {
        result.state_transfer = StateTransfer::device;
    } else {
        std::cerr << "unknown state transfer: " << state_transfer_name << '\n';
        return std::nullopt;
    }
    if (route_preparation_name == "host") {
        result.route_preparation = RoutePreparation::host;
    } else if (route_preparation_name == "device") {
        result.route_preparation = RoutePreparation::device;
    } else {
        std::cerr << "unknown route preparation: "
                  << route_preparation_name << '\n';
        return std::nullopt;
    }
    if (result.validation == k3x::CudaWeightValidationMode::admission &&
        result.weight_mode != WeightMode::resident) {
        std::cerr << "admission validation requires resident weights\n";
        return std::nullopt;
    }
    if (result.state_transfer == StateTransfer::device &&
        (result.case_mode != CaseMode::ab_incremental ||
         result.weight_mode != WeightMode::resident ||
         result.validation != k3x::CudaWeightValidationMode::admission)) {
        std::cerr
            << "device state requires ab-incremental resident admission\n";
        return std::nullopt;
    }
    if (result.route_preparation == RoutePreparation::device &&
        (result.case_mode != CaseMode::ab_incremental ||
         result.weight_mode != WeightMode::resident ||
         result.validation != k3x::CudaWeightValidationMode::admission ||
         result.state_transfer != StateTransfer::device)) {
        std::cerr << "device route preparation requires ab-incremental "
                     "device-state resident admission\n";
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
    std::array<std::string, 2> state_sha256;
    std::array<std::string, 2> kda_output_sha256;
    std::string initial_state_sha256;
    std::string final_state_sha256;
    std::string oracle_sha256;
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
    const auto* oracle = json::member(*root, "oracle");
    const auto* oracle_digest = oracle
        ? json::string(json::member(*oracle, "sha256")) : nullptr;
    if (!oracle ||
        !text_is(*oracle, "format", "k3x-official-layer-oracle-v1") ||
        !text_is(*oracle, "filename", "official-layer-oracle-v1.bin") ||
        !integer_is(*oracle, "bytes", kOracleBytes) || !oracle_digest ||
        !hex_string(*oracle_digest, 64)) {
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
        result.state_sha256[index] = *state;
        result.kda_output_sha256[index] = *output;
        expected_consumed = *state;
    }
    if (expected_consumed != *final) return std::nullopt;
    auto selected = expert_ids(json::member(*root, "selected_experts"));
    if (!selected || selected->empty() || selected->size() > 32 ||
        *selected != expected_union) {
        return std::nullopt;
    }
    result.selected = std::move(*selected);
    result.initial_state_sha256 = *initial;
    result.final_state_sha256 = *final;
    result.oracle_sha256 = *oracle_digest;
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

std::uint64_t read_u64(std::span<const std::byte> bytes, std::size_t offset) {
    std::uint64_t result{};
    for (std::size_t index = 0; index < 8; ++index) {
        result |= static_cast<std::uint64_t>(
            std::to_integer<unsigned>(bytes[offset + index])) << (index * 8);
    }
    return result;
}

struct OfficialOracle {
    std::vector<std::uint16_t> output;
    std::array<std::vector<std::uint16_t>, 3> convolution;
    std::vector<float> recurrent;
};

std::optional<OfficialOracle> load_oracle(const std::filesystem::path& path,
                                          std::string_view expected_digest) {
    std::error_code error;
    if (std::filesystem::file_size(path, error) != kOracleBytes || error) {
        return std::nullopt;
    }
    std::ifstream stream(path, std::ios::binary);
    std::vector<std::byte> bytes(kOracleBytes);
    stream.read(reinterpret_cast<char*>(bytes.data()),
                static_cast<std::streamsize>(bytes.size()));
    if (!stream || digest_hex(k3x::sha256(bytes)) != expected_digest ||
        std::memcmp(bytes.data(), "K3XORC1\0", 8) != 0 ||
        read_u64(bytes, 8) != 2 * 7'168 ||
        read_u64(bytes, 16) != 3 * 12'288 ||
        read_u64(bytes, 24) != 96 * 128 * 128) {
        return std::nullopt;
    }
    OfficialOracle result;
    std::size_t offset = 32;
    auto read_words = [&](std::size_t count) {
        std::vector<std::uint16_t> values(count);
        std::memcpy(values.data(), bytes.data() + offset, count * 2);
        offset += count * 2;
        return values;
    };
    result.output = read_words(2 * 7'168);
    for (auto& values : result.convolution) values = read_words(3 * 12'288);
    result.recurrent.resize(96 * 128 * 128);
    std::memcpy(result.recurrent.data(), bytes.data() + offset,
                result.recurrent.size() * sizeof(float));
    offset += result.recurrent.size() * sizeof(float);
    if (offset != bytes.size() ||
        !std::all_of(result.recurrent.begin(), result.recurrent.end(),
                     [](float value) { return std::isfinite(value); })) {
        return std::nullopt;
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
                              {record.dimensions[0], record.dimensions[1] / 2,
                               0, 0}, 2, values / 2});
            result.push_back({record.name + ".weight_scale", "U8",
                              {record.dimensions[0], record.dimensions[1] / 32,
                               0, 0}, 2, values / 32});
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

const k3x::TensorRecord* find_record(const k3x::Reader& reader,
                                     const std::string& name) {
    const auto id = k3x::fnv1a64(name.c_str());
    const auto found = std::find_if(
        reader.tensors().begin(), reader.tensors().end(),
        [id](const auto& record) { return record.tensor_id == id; });
    return found == reader.tensors().end() ? nullptr : &*found;
}

struct OwnedBf16 {
    std::uint64_t id{};
    std::vector<std::uint16_t> values;
    std::size_t rows{};
    std::size_t cols{};
    k3x::Bf16WeightView matrix() const { return {values, rows, cols, id}; }
    k3x::Bf16VectorView vector() const { return {values, id}; }
};

std::optional<OwnedBf16> load_bf16(k3x::Reader& reader,
                                   const std::string& name,
                                   std::size_t rows, std::size_t cols) {
    const auto* record = find_record(reader, name);
    if (!record || record->dtype != 3 || record->quantization != 0 ||
        record->data_length != rows * cols * 2 || record->auxiliary_length) {
        return std::nullopt;
    }
    auto bytes = reader.read_tensor(record->tensor_id);
    if (!bytes || bytes.value().size() != record->data_length) return std::nullopt;
    OwnedBf16 result{record->tensor_id,
                     std::vector<std::uint16_t>(rows * cols), rows, cols};
    std::memcpy(result.values.data(), bytes.value().data(), bytes.value().size());
    return result;
}

std::optional<std::vector<float>> load_f32(k3x::Reader& reader,
                                           const std::string& name,
                                           std::size_t count) {
    const auto* record = find_record(reader, name);
    if (!record || record->dtype != 1 || record->quantization != 0 ||
        record->data_length != count * sizeof(float) || record->auxiliary_length) {
        return std::nullopt;
    }
    auto bytes = reader.read_tensor(record->tensor_id);
    if (!bytes || bytes.value().size() != record->data_length) return std::nullopt;
    std::vector<float> result(count);
    std::memcpy(result.data(), bytes.value().data(), bytes.value().size());
    return result;
}

struct OwnedExpert {
    std::uint32_t id{};
    std::array<std::vector<std::byte>, 6> extents;
    k3x::Mxfp4MlpView view() const {
        const auto base = "model.layers.1.feed_forward.experts." +
            std::to_string(id) + ".";
        return {
            {k3x::fnv1a64((base + "gate").c_str()), extents[0], extents[1],
             3'072, 3'584, 32},
            {k3x::fnv1a64((base + "up").c_str()), extents[2], extents[3],
             3'072, 3'584, 32},
            {k3x::fnv1a64((base + "down").c_str()), extents[4], extents[5],
             3'584, 3'072, 32}};
    }
};

struct LoadedLayer {
    OwnedBf16 self_norm, self_proj, input_norm;
    OwnedBf16 q_proj, k_proj, v_proj, f_a, f_b, beta, gate, o_proj;
    std::vector<float> q_conv, k_conv, v_conv, a_log, dt_bias, o_norm;
    OwnedBf16 mlp_norm, mlp_proj, post_norm, router;
    std::vector<float> correction;
    OwnedBf16 routed_down, routed_norm, routed_up;
    OwnedBf16 shared_gate, shared_up, shared_down;
    std::vector<OwnedExpert> experts;
    std::vector<k3x::Mxfp4MlpView> expert_mlp_views;
    std::vector<k3x::OfficialExpertView> expert_views;

    k3x::OfficialLayerWeights views() const {
        const k3x::OfficialKdaWeightsView kda{
            q_proj.matrix(), k_proj.matrix(), v_proj.matrix(),
            q_conv, k_conv, v_conv, f_a.matrix(), f_b.matrix(), a_log,
            dt_bias, beta.matrix(), gate.matrix(), o_norm, o_proj.matrix()};
        const k3x::OfficialMoeWeights moe{
            mlp_norm.vector(), mlp_proj.matrix(), post_norm.vector(),
            router.matrix(), correction, routed_down.matrix(),
            routed_norm.vector(), routed_up.matrix(),
            {shared_gate.matrix(), shared_up.matrix(), shared_down.matrix()},
            expert_views};
        return {self_norm.vector(), self_proj.matrix(), input_norm.vector(),
                kda, moe};
    }

    k3x::OfficialLayerCudaWeights cuda_views() const {
        const std::string base = "model.layers.1.";
        const auto id = [&](std::string_view suffix) {
            return k3x::fnv1a64((base + std::string(suffix)).c_str());
        };
        const k3x::OfficialKdaCudaView kda{
            q_proj.matrix(), k_proj.matrix(), v_proj.matrix(),
            {id("self_attn.q_conv1d.weight"), q_conv, 12'288, 4},
            {id("self_attn.k_conv1d.weight"), k_conv, 12'288, 4},
            {id("self_attn.v_conv1d.weight"), v_conv, 12'288, 4},
            f_a.matrix(), f_b.matrix(),
            {id("self_attn.A_log"), a_log},
            {id("self_attn.dt_bias"), dt_bias}, beta.matrix(), gate.matrix(),
            {id("self_attn.o_norm.weight"), o_norm}, o_proj.matrix()};
        const k3x::OfficialMoeWeights moe{
            mlp_norm.vector(), mlp_proj.matrix(), post_norm.vector(),
            router.matrix(), correction, routed_down.matrix(),
            routed_norm.vector(), routed_up.matrix(),
            {shared_gate.matrix(), shared_up.matrix(), shared_down.matrix()},
            expert_views};
        const k3x::OfficialMoeFfnView moe_ffn{
            routed_down.matrix(), routed_norm.vector(), routed_up.matrix(),
            {shared_gate.matrix(), shared_up.matrix(), shared_down.matrix()}};
        return {self_norm.vector(), self_proj.matrix(), input_norm.vector(),
                kda, moe, moe_ffn};
    }
};

std::optional<LoadedLayer> load_layer(k3x::Reader& reader,
                                      const Manifest& manifest) {
    const std::string base = "model.layers.1.";
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
    auto correction = load_f32(reader,
        base + "block_sparse_moe.gate.e_score_correction_bias", 896);
    auto routed_down = load_bf16(reader,
        base + "block_sparse_moe.routed_expert_down_proj.weight", 3'584, 7'168);
    auto routed_norm = load_bf16(reader,
        base + "block_sparse_moe.routed_expert_norm.weight", 1, 3'584);
    auto routed_up = load_bf16(reader,
        base + "block_sparse_moe.routed_expert_up_proj.weight", 7'168, 3'584);
    auto shared_gate = load_bf16(reader,
        base + "block_sparse_moe.shared_experts.gate_proj.weight", 6'144, 7'168);
    auto shared_up = load_bf16(reader,
        base + "block_sparse_moe.shared_experts.up_proj.weight", 6'144, 7'168);
    auto shared_down = load_bf16(reader,
        base + "block_sparse_moe.shared_experts.down_proj.weight", 7'168, 6'144);
    if (!self_norm || !self_proj || !input_norm || !q_proj || !k_proj || !v_proj ||
        !f_a || !f_b || !beta || !gate || !o_proj || !q_conv || !k_conv ||
        !v_conv || !a_log || !dt_bias || !o_norm || !mlp_norm || !mlp_proj ||
        !post_norm || !router || !correction || !routed_down || !routed_norm ||
        !routed_up || !shared_gate || !shared_up || !shared_down) return std::nullopt;
    LoadedLayer result{
        std::move(*self_norm), std::move(*self_proj), std::move(*input_norm),
        std::move(*q_proj), std::move(*k_proj), std::move(*v_proj),
        std::move(*f_a), std::move(*f_b), std::move(*beta), std::move(*gate),
        std::move(*o_proj), std::move(*q_conv), std::move(*k_conv),
        std::move(*v_conv), std::move(*a_log), std::move(*dt_bias),
        std::move(*o_norm), std::move(*mlp_norm), std::move(*mlp_proj),
        std::move(*post_norm), std::move(*router), std::move(*correction),
        std::move(*routed_down), std::move(*routed_norm), std::move(*routed_up),
        std::move(*shared_gate), std::move(*shared_up), std::move(*shared_down),
        {}, {}, {}};
    result.experts.reserve(manifest.selected.size());
    for (const auto id : manifest.selected) {
        auto expert = k3x::load_storage_expert(reader, 1, id);
        if (!expert) return std::nullopt;
        result.experts.push_back({id, std::move(expert.value().extents)});
    }
    result.expert_mlp_views.reserve(result.experts.size());
    result.expert_views.reserve(result.experts.size());
    for (const auto& expert : result.experts) {
        result.expert_mlp_views.push_back(expert.view());
        result.expert_views.push_back({expert.id, result.expert_mlp_views.back()});
    }
    return result;
}

std::vector<float> input_values(int multiplier, int increment,
                                int modulus, int offset) {
    std::vector<float> result(7'168);
    for (std::size_t index = 0; index < result.size(); ++index) {
        result[index] = static_cast<float>(
            ((multiplier * static_cast<int>(index) + increment) % modulus) - offset) /
            1024.0F;
    }
    return result;
}

std::array<k3x::OfficialLayerInput, 2> layer_inputs() {
    return {{
        {input_values(17, 3, 257, 128), input_values(29, 11, 251, 125)},
        {input_values(31, 7, 263, 131), input_values(43, 19, 269, 134)}}};
}

bool close(std::span<const float> left, std::span<const float> right,
           float tolerance = 1.0e-6F) {
    if (left.size() != right.size()) return false;
    for (std::size_t index = 0; index < left.size(); ++index) {
        if (!std::isfinite(left[index]) || !std::isfinite(right[index]) ||
            std::abs(left[index] - right[index]) > tolerance) return false;
    }
    return true;
}

bool same_step(const k3x::OfficialLayerStepResult& left,
               const k3x::OfficialLayerStepResult& right) {
    if (!close(left.self_attention_residual, right.self_attention_residual) ||
        !close(left.input_normalized, right.input_normalized) ||
        !close(left.post_kda_prefix, right.post_kda_prefix) ||
        !close(left.mlp_attention_residual, right.mlp_attention_residual) ||
        !close(left.normalized_moe_input, right.normalized_moe_input) ||
        left.route.expert_ids != right.route.expert_ids ||
        !close(left.route.contributions, right.route.contributions) ||
        !close(left.moe.hidden, right.moe.hidden) ||
        !close(left.moe.latent, right.moe.latent) ||
        !close(left.moe.mixed_latent, right.moe.mixed_latent) ||
        !close(left.moe.routed, right.moe.routed) ||
        !close(left.moe.shared, right.moe.shared) ||
        !close(left.moe.combined, right.moe.combined) ||
        !close(left.moe.output, right.moe.output) ||
        left.moe.expert_outputs.size() != right.moe.expert_outputs.size()) return false;
    for (std::size_t index = 0; index < left.moe.expert_outputs.size(); ++index) {
        if (!close(left.moe.expert_outputs[index], right.moe.expert_outputs[index])) {
            return false;
        }
    }
    return true;
}

std::uint16_t encode_bf16(float value) {
    auto bits = std::bit_cast<std::uint32_t>(value);
    bits += 0x7fffU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>(bits >> 16U);
}

void update_u64(k3x::Sha256Hasher& digest, std::uint64_t value) {
    std::array<std::byte, 8> bytes{};
    for (std::size_t index = 0; index < bytes.size(); ++index) {
        bytes[index] = std::byte((value >> (index * 8)) & 0xffU);
    }
    digest.update(bytes);
}

template <typename T>
void update_state_tensor(k3x::Sha256Hasher& digest, std::string_view name,
                         std::span<const std::uint64_t> shape,
                         std::span<const T> values) {
    digest.update(std::as_bytes(std::span(name.data(), name.size())));
    const std::byte zero{};
    digest.update(std::span(&zero, 1));
    for (const auto dimension : shape) update_u64(digest, dimension);
    digest.update(std::as_bytes(values));
}

std::string state_digest(const k3x::OfficialKdaState& state) {
    static constexpr char identity[] = "k3x-official-kda-state-v1\0v-first-fp32\0";
    k3x::Sha256Hasher digest;
    digest.update(std::as_bytes(std::span(identity).first(sizeof(identity) - 1)));
    const std::array<std::uint64_t, 3> conv_shape{1, 3, 12'288};
    const std::array<std::uint64_t, 4> recurrent_shape{1, 96, 128, 128};
    update_state_tensor(digest, "conv_q", conv_shape,
                        std::span<const std::uint16_t>(state.conv_q));
    update_state_tensor(digest, "conv_k", conv_shape,
                        std::span<const std::uint16_t>(state.conv_k));
    update_state_tensor(digest, "conv_v", conv_shape,
                        std::span<const std::uint16_t>(state.conv_v));
    update_state_tensor(digest, "recurrent_v_first", recurrent_shape,
                        std::span<const float>(state.recurrent_v_first));
    return digest_hex(digest.finish());
}

std::string kda_output_digest(std::span<const std::uint16_t> output) {
    static constexpr char identity[] = "kda-output-bf16\0";
    k3x::Sha256Hasher digest;
    digest.update(std::as_bytes(std::span(identity).first(sizeof(identity) - 1)));
    digest.update(std::as_bytes(output));
    return digest_hex(digest.finish());
}

struct ErrorStats {
    double maximum_absolute{};
    double maximum_relative{};
};

void observe(ErrorStats& stats, double actual, double expected) {
    const auto absolute = std::abs(actual - expected);
    const auto relative = absolute / std::max(std::abs(expected), 1.0e-12);
    stats.maximum_absolute = std::max(stats.maximum_absolute, absolute);
    stats.maximum_relative = std::max(stats.maximum_relative, relative);
}

ErrorStats bf16_error(std::span<const std::uint16_t> actual,
                      std::span<const std::uint16_t> expected) {
    ErrorStats result;
    if (actual.size() != expected.size()) {
        return {std::numeric_limits<double>::infinity(),
                std::numeric_limits<double>::infinity()};
    }
    for (std::size_t index = 0; index < actual.size(); ++index) {
        observe(result, k3x::decode_bf16_word(actual[index]),
                k3x::decode_bf16_word(expected[index]));
    }
    return result;
}

ErrorStats f32_error(std::span<const float> actual,
                     std::span<const float> expected) {
    ErrorStats result;
    if (actual.size() != expected.size()) {
        return {std::numeric_limits<double>::infinity(),
                std::numeric_limits<double>::infinity()};
    }
    for (std::size_t index = 0; index < actual.size(); ++index) {
        observe(result, actual[index], expected[index]);
    }
    return result;
}

struct PortableReference {
    k3x::OfficialLayerResult full;
    k3x::OfficialLayerResult first;
};

bool validate_portable_oracle(const LoadedLayer& loaded,
                              const Manifest& manifest,
                              const OfficialOracle& oracle,
                              PortableReference& reference) {
    bool valid = true;
    const k3x::OfficialKdaConfig config{7'168, 96, 128, 4, 1.0e-5F, -5.0F};
    const auto zero = k3x::zero_official_kda_state(config);
    if (state_digest(zero) != manifest.initial_state_sha256) {
        std::cerr << "oracle diagnostic: initial-state-hash\n";
        return false;
    }
    const auto inputs = layer_inputs();
    const k3x::OfficialKdaState reference_state{
        oracle.convolution[0], oracle.convolution[1], oracle.convolution[2],
        oracle.recurrent};
    if (state_digest(reference_state) != manifest.final_state_sha256 ||
        kda_output_digest(std::span(oracle.output).first(7'168)) !=
            manifest.kda_output_sha256[0] ||
        kda_output_digest(std::span(oracle.output).last(7'168)) !=
            manifest.kda_output_sha256[1]) {
        std::cerr << "oracle diagnostic: sidecar-manifest-binding\n";
        return false;
    }
    const auto weights = loaded.views();
    const auto full = k3x::official_layer_cpu(inputs, weights, zero, config, 16, 4, 25);
    const auto first = k3x::official_layer_cpu(
        std::span(inputs).first(1), weights, zero, config, 16, 4, 25);
    if (!full || !first) {
        std::cerr << "oracle diagnostic: full-or-first-call\n";
        return false;
    }
    const auto second = k3x::official_layer_cpu(
        std::span(inputs).last(1), weights, first.value().kda.state,
        config, 16, 4, 25);
    if (!second || full.value().steps.size() != 2) {
        std::cerr << "oracle diagnostic: second-call-or-step-count\n";
        return false;
    }
    if (!same_step(full.value().steps[0], first.value().steps[0])) {
        std::cerr << "oracle diagnostic: incremental-step-a\n";
        return false;
    }
    if (!same_step(full.value().steps[1], second.value().steps[0])) {
        std::cerr << "oracle diagnostic: incremental-step-b\n";
        return false;
    }
    const auto output_words = [&] {
        std::vector<std::uint16_t> result(full.value().kda.output.size());
        std::transform(full.value().kda.output.begin(), full.value().kda.output.end(),
                       result.begin(), encode_bf16);
        return result;
    }();
    const auto output_error = bf16_error(output_words, oracle.output);
    const auto conv_q_error = bf16_error(full.value().kda.state.conv_q,
                                         oracle.convolution[0]);
    const auto conv_k_error = bf16_error(full.value().kda.state.conv_k,
                                         oracle.convolution[1]);
    const auto conv_v_error = bf16_error(full.value().kda.state.conv_v,
                                         oracle.convolution[2]);
    const auto recurrent_error = f32_error(
        full.value().kda.state.recurrent_v_first, oracle.recurrent);
    if (output_error.maximum_absolute > kOutputAbsoluteTolerance ||
        conv_q_error.maximum_absolute > kConvolutionAbsoluteTolerance ||
        conv_k_error.maximum_absolute > kConvolutionAbsoluteTolerance ||
        conv_v_error.maximum_absolute > kConvolutionAbsoluteTolerance ||
        recurrent_error.maximum_absolute > kRecurrentAbsoluteTolerance) {
        std::cerr << "oracle diagnostic: max-error output="
                  << output_error.maximum_absolute << ','
                  << output_error.maximum_relative << " conv-q="
                  << conv_q_error.maximum_absolute << ','
                  << conv_q_error.maximum_relative << " conv-k="
                  << conv_k_error.maximum_absolute << ','
                  << conv_k_error.maximum_relative << " conv-v="
                  << conv_v_error.maximum_absolute << ','
                  << conv_v_error.maximum_relative << " recurrent="
                  << recurrent_error.maximum_absolute << ','
                  << recurrent_error.maximum_relative << '\n';
        valid = false;
    }
    for (std::size_t index = 0; index < 2; ++index) {
        const auto output = std::span(full.value().kda.output).subspan(index * 7'168, 7'168);
        static_cast<void>(output);
        if (full.value().steps[index].route.expert_ids != manifest.routes[index].ids) {
            std::cerr << "oracle diagnostic: route-" << index << '\n';
            valid = false;
        }
        const auto contribution_error = f32_error(
            full.value().steps[index].route.contributions,
            manifest.routes[index].contributions);
        if (contribution_error.maximum_absolute >
            kContributionAbsoluteTolerance) {
            std::cerr << "oracle diagnostic: contributions-" << index << '='
                      << contribution_error.maximum_absolute << ','
                      << contribution_error.maximum_relative << '\n';
            valid = false;
        }
    }
    if (valid) {
        reference = {std::move(full.value()), std::move(first.value())};
    }
    return valid;
}

k3x::BackendOptions backend_options(
    WeightMode mode, std::uint64_t capacity,
    k3x::CudaWeightValidationMode validation) {
    k3x::BackendOptions result;
    result.kind = k3x::BackendKind::cuda_custom;
    result.cuda_allocation = k3x::CudaAllocationMode::reused;
    result.cuda_weights = mode == WeightMode::resident
        ? k3x::CudaWeightMode::resident : k3x::CudaWeightMode::transient;
    result.cuda_batching = k3x::CudaBatchingMode::resident_grid;
    result.cuda_boundary = k3x::CudaBoundaryMode::moe_layer;
    result.cuda_resident_bytes = capacity;
    result.cuda_weight_validation = validation;
    return result;
}

std::uint64_t elapsed(std::chrono::steady_clock::time_point start) {
    return static_cast<std::uint64_t>(std::chrono::duration_cast<
        std::chrono::nanoseconds>(std::chrono::steady_clock::now() - start).count());
}

void write_u32_array(std::ostream& stream,
                     std::span<const std::uint32_t> values) {
    stream << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index) stream << ',';
        stream << values[index];
    }
    stream << ']';
}

void write_float_array(std::ostream& stream, std::span<const float> values) {
    stream << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index) stream << ',';
        stream << values[index];
    }
    stream << ']';
}

std::uint64_t percentile(std::vector<std::uint64_t> values,
                         std::size_t percent) {
    std::sort(values.begin(), values.end());
    const auto index = (values.size() - 1) * percent / 100;
    return values[index];
}

std::uint64_t median(std::vector<std::uint64_t> values) {
    std::sort(values.begin(), values.end());
    return values[values.size() / 2];
}

std::string output_digest(
    std::span<const k3x::OfficialLayerCudaStepResult> first,
    std::span<const k3x::OfficialLayerCudaStepResult> second = {}) {
    static constexpr char identity[] = "k3x-official-layer-output-bf16-v1\0";
    k3x::Sha256Hasher digest;
    digest.update(std::as_bytes(std::span(identity).first(sizeof(identity) - 1)));
    for (const auto steps : {first, second}) {
        for (const auto& step : steps) {
            std::vector<std::uint16_t> words(step.output.size());
            std::transform(step.output.begin(), step.output.end(), words.begin(),
                           encode_bf16);
            digest.update(std::as_bytes(std::span(words)));
        }
    }
    return digest_hex(digest.finish());
}

bool observe_cuda_result(const k3x::OfficialLayerCudaResult& actual,
                         const k3x::OfficialLayerResult& expected,
                         float& maximum_error,
                         bool require_state = true) {
    if (!actual.executed || actual.steps.size() != expected.steps.size()) return false;
    for (std::size_t step = 0; step < actual.steps.size(); ++step) {
        if (actual.steps[step].route.expert_ids !=
                expected.steps[step].route.expert_ids ||
            actual.steps[step].route.contributions.size() !=
                expected.steps[step].route.contributions.size() ||
            actual.steps[step].output.size() !=
                expected.steps[step].moe.output.size()) return false;
        for (std::size_t index = 0;
             index < actual.steps[step].route.contributions.size(); ++index) {
            if (std::abs(actual.steps[step].route.contributions[index] -
                         expected.steps[step].route.contributions[index]) >
                kContributionAbsoluteTolerance) return false;
        }
        for (std::size_t index = 0; index < actual.steps[step].output.size(); ++index) {
            const auto error = std::abs(actual.steps[step].output[index] -
                                        expected.steps[step].moe.output[index]);
            if (!std::isfinite(error)) return false;
            maximum_error = std::max(maximum_error, error);
        }
    }
    return !require_state ||
           (actual.kda_state.conv_q == expected.kda.state.conv_q &&
           actual.kda_state.conv_k == expected.kda.state.conv_k &&
           actual.kda_state.conv_v == expected.kda.state.conv_v &&
           f32_error(actual.kda_state.recurrent_v_first,
                     expected.kda.state.recurrent_v_first).maximum_absolute <=
               kRecurrentAbsoluteTolerance);
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
    auto loaded = load_layer(reader.value(), *manifest);
    if (!loaded) {
        write_error(k3x::ErrorCode::invalid_extent,
                    "official layer weight load mismatch");
        return 4;
    }
    const auto oracle_path = arguments->manifest.parent_path() /
        "official-layer-oracle-v1.bin";
    auto oracle = load_oracle(oracle_path, manifest->oracle_sha256);
    if (!oracle) {
        write_error(k3x::ErrorCode::invalid_extent,
                    "official layer oracle identity mismatch");
        return 4;
    }
    PortableReference reference;
    if (!validate_portable_oracle(*loaded, *manifest, *oracle, reference)) {
        write_error(k3x::ErrorCode::invalid_state,
                    "official layer portable oracle mismatch");
        return 4;
    }
    const k3x::OfficialKdaConfig config{7'168, 96, 128, 4, 1.0e-5F, -5.0F};
    const auto zero = k3x::zero_official_kda_state(config);
    const auto inputs = layer_inputs();
    k3x::Profiler profiler;
    auto backend = k3x::make_cuda_backend(
        backend_options(arguments->weight_mode,
                        reader.value().superblock().file_length,
                        arguments->validation), &profiler);
    if (!backend) {
        write_error(backend.error(), backend.message());
        return 4;
    }
    const auto weights = loaded->cuda_views();
    float maximum_error{};
    std::string published_output_digest;
    std::string published_state_digest;
    const auto publish = [&](std::string output, std::string state) {
        if ((!published_output_digest.empty() &&
             published_output_digest != output) ||
            (!published_state_digest.empty() && published_state_digest != state)) {
            return false;
        }
        published_output_digest = std::move(output);
        published_state_digest = std::move(state);
        return true;
    };
    const auto route_preparation =
        arguments->route_preparation == RoutePreparation::device
            ? k3x::OfficialMoeRoutePreparationMode::device
            : k3x::OfficialMoeRoutePreparationMode::host;
    const auto execute = [&]() {
        if (arguments->case_mode == CaseMode::a) {
            auto result = k3x::official_layer_cuda(
                *backend.value(), std::span(inputs).first(1), weights, zero,
                config, 16, 4, 25, 1, k3x::ProfilePhase::decode, {},
                route_preparation);
            return result && observe_cuda_result(
                result.value(), reference.first, maximum_error) &&
                publish(output_digest(result.value().steps),
                        state_digest(result.value().kda_state));
        }
        if (arguments->case_mode == CaseMode::ab_full) {
            auto result = k3x::official_layer_cuda(
                *backend.value(), inputs, weights, zero, config, 16, 4, 25,
                1, k3x::ProfilePhase::decode, {}, route_preparation);
            return result && observe_cuda_result(
                result.value(), reference.full, maximum_error) &&
                publish(output_digest(result.value().steps),
                        state_digest(result.value().kda_state));
        }
        const bool device_state =
            arguments->state_transfer == StateTransfer::device;
        const k3x::OfficialKdaStateControl first_control =
            device_state
                ? k3x::OfficialKdaStateControl{
                      k3x::OfficialKdaStateMode::device_seed, {}}
                : k3x::OfficialKdaStateControl{};
        auto first = k3x::official_layer_cuda(
            *backend.value(), std::span(inputs).first(1), weights, zero,
            config, 16, 4, 25, 1, k3x::ProfilePhase::decode, first_control,
            route_preparation);
        if (!first) return false;
        const k3x::OfficialKdaState no_host_state;
        const auto& second_input_state =
            device_state ? no_host_state : first.value().kda_state;
        const k3x::OfficialKdaStateControl second_control =
            device_state
                ? k3x::OfficialKdaStateControl{
                      k3x::OfficialKdaStateMode::device_publish,
                      first.value().kda_device_state}
                : k3x::OfficialKdaStateControl{};
        auto second = k3x::official_layer_cuda(
            *backend.value(), std::span(inputs).last(1), weights,
            second_input_state, config, 16, 4, 25, 1,
            k3x::ProfilePhase::decode, second_control, route_preparation);
        if (!second ||
            !observe_cuda_result(first.value(), reference.first, maximum_error,
                                 !device_state) ||
            second.value().steps.size() != 1) return false;
        const auto& actual_step = second.value().steps[0];
        const auto& expected_step = reference.full.steps[1];
        if (actual_step.route.expert_ids != expected_step.route.expert_ids ||
            actual_step.route.contributions.size() !=
                expected_step.route.contributions.size() ||
            actual_step.output.size() != expected_step.moe.output.size()) {
            return false;
        }
        for (std::size_t index = 0;
             index < actual_step.route.contributions.size(); ++index) {
            if (std::abs(actual_step.route.contributions[index] -
                         expected_step.route.contributions[index]) >
                kContributionAbsoluteTolerance) return false;
        }
        for (std::size_t index = 0; index < actual_step.output.size(); ++index) {
            const auto error = std::abs(actual_step.output[index] -
                                        expected_step.moe.output[index]);
            if (!std::isfinite(error)) return false;
            maximum_error = std::max(maximum_error, error);
        }
        return second.value().kda_state.conv_q == reference.full.kda.state.conv_q &&
               second.value().kda_state.conv_k == reference.full.kda.state.conv_k &&
               second.value().kda_state.conv_v == reference.full.kda.state.conv_v &&
               f32_error(second.value().kda_state.recurrent_v_first,
                         reference.full.kda.state.recurrent_v_first)
                       .maximum_absolute <= kRecurrentAbsoluteTolerance &&
               publish(output_digest(first.value().steps, second.value().steps),
                       state_digest(second.value().kda_state));
    };
    const auto cold_stats_before = backend.value()->runtime_stats();
    const auto cold_profile_before = profiler.summary();
    const auto cold_start = std::chrono::steady_clock::now();
    if (!execute()) {
        write_error(k3x::ErrorCode::invalid_state,
                    "official layer CUDA cold result mismatch");
        return 4;
    }
    const auto cold_latency = elapsed(cold_start);
    const auto cold_stats_after = backend.value()->runtime_stats();
    const auto cold_profile_after = profiler.summary();
    for (std::uint64_t index = 0; index < arguments->warmups; ++index) {
        if (!execute()) {
            write_error(k3x::ErrorCode::invalid_state,
                        "official layer CUDA warmup mismatch");
            return 4;
        }
    }
    const auto stats_before = backend.value()->runtime_stats();
    const auto profile_before = profiler.summary();
    std::vector<std::uint64_t> samples;
    samples.reserve(arguments->iterations);
    for (std::uint64_t index = 0; index < arguments->iterations; ++index) {
        const auto start = std::chrono::steady_clock::now();
        if (!execute()) {
            write_error(k3x::ErrorCode::invalid_state,
                        "official layer CUDA result mismatch");
            return 4;
        }
        samples.push_back(elapsed(start));
    }
    const auto stats_after = backend.value()->runtime_stats();
    const auto profile_after = profiler.summary();
    const auto memory = backend.value()->memory_stats();
    const auto latency_sum = std::accumulate(
        samples.begin(), samples.end(), std::uint64_t{});
    const auto kernel_time = profile_after.device_nanoseconds -
        profile_before.device_nanoseconds;
    constexpr std::uint64_t kKdaF32Bytes = 640'000;
    constexpr std::uint64_t kKdaBf16Bytes = 887'160'832;
    constexpr std::uint64_t kMoeBf16Bytes = 367'008'768;
    constexpr std::uint64_t kRouteBf16Bytes = 12'888'064;
    constexpr std::uint64_t kExpertBytes = 17'547'264;
    const auto cold_experts = arguments->case_mode == CaseMode::a
        ? 16ULL : static_cast<std::uint64_t>(manifest->selected.size());
    const auto transient = arguments->weight_mode == WeightMode::transient;
    const auto measured_bf16 = transient
        ? arguments->iterations * (kKdaBf16Bytes + kMoeBf16Bytes) : 0ULL;
    const auto measured_f32 = transient
        ? arguments->iterations * kKdaF32Bytes : 0ULL;
    const auto measured_mxfp4 = transient
        ? arguments->iterations * 16ULL * kExpertBytes : 0ULL;
    const auto cold_bf16 = kKdaBf16Bytes + kMoeBf16Bytes +
        (arguments->route_preparation == RoutePreparation::device
             ? kRouteBf16Bytes
             : 0ULL);
    const auto cold_f32 = kKdaF32Bytes;
    const auto cold_mxfp4 = cold_experts * kExpertBytes;
    const auto measured_weight = stats_after.weight_h2d_bytes -
        stats_before.weight_h2d_bytes;
    const auto cold_weight = cold_stats_after.weight_h2d_bytes -
        cold_stats_before.weight_h2d_bytes;
    if (measured_weight != measured_bf16 + measured_f32 + measured_mxfp4 ||
        cold_weight != cold_bf16 + cold_f32 + cold_mxfp4) {
        write_error(k3x::ErrorCode::invalid_state,
                    "official layer CUDA precision traffic mismatch");
        return 4;
    }
    const auto reader_counters = reader.value().counters();
    rusage usage{};
    if (getrusage(RUSAGE_SELF, &usage) != 0 || usage.ru_maxrss <= 0) {
        write_error(k3x::ErrorCode::invalid_state,
                    "official layer process RSS unavailable");
        return 4;
    }
    const auto process_peak_rss =
        static_cast<std::uint64_t>(usage.ru_maxrss) * 1024ULL;
    std::cout << std::setprecision(12)
        << "{\"artifact_kind\":\"official_kimi_k3_kda_layer\""
        << ",\"repository\":\"moonshotai/Kimi-K3\""
        << ",\"resolved_revision\":\"9f62e4e9fffbd0a83ddd60e1c209d828994b3569\""
        << ",\"case\":\""
        << (arguments->case_mode == CaseMode::a ? "a" :
            arguments->case_mode == CaseMode::ab_full ? "ab-full" :
                                                        "ab-incremental")
        << "\",\"weight_mode\":\""
        << (arguments->weight_mode == WeightMode::resident ? "resident" :
                                                             "transient")
        << '"';
    if (arguments->validation_explicit) {
        std::cout << ",\"validation\":\""
                  << (arguments->validation ==
                              k3x::CudaWeightValidationMode::admission
                          ? "admission"
                          : "per-call")
                  << '"';
    }
    if (arguments->state_transfer_explicit) {
        std::cout << ",\"state_transfer\":\""
                  << (arguments->state_transfer == StateTransfer::device
                          ? "device"
                          : "host")
                  << '"';
    }
    if (arguments->route_preparation_explicit) {
        std::cout << ",\"route_preparation\":\""
                  << (arguments->route_preparation == RoutePreparation::device
                          ? "device"
                          : "host")
                  << '"';
    }
    std::cout << ",\"token_semantics\":false,\"routing_semantics\":true"
        << ",\"full_transformer_layer\":true,\"quality_measured\":false"
        << ",\"k3x_root_sha256\":\"" << manifest->root_sha256 << "\""
        << ",\"warmups\":" << arguments->warmups
        << ",\"iterations\":" << arguments->iterations
        << ",\"selected_union\":";
    write_u32_array(std::cout, manifest->selected);
    std::cout << ",\"route_a\":";
    write_u32_array(std::cout, reference.full.steps[0].route.expert_ids);
    std::cout << ",\"route_b\":";
    write_u32_array(std::cout, reference.full.steps[1].route.expert_ids);
    std::cout << ",\"route_a_contributions\":";
    write_float_array(std::cout, reference.full.steps[0].route.contributions);
    std::cout << ",\"route_b_contributions\":";
    write_float_array(std::cout, reference.full.steps[1].route.contributions);
    std::cout << ",\"output_sha256\":\"" << published_output_digest << "\""
        << ",\"state_sha256\":\"" << published_state_digest << "\""
        << ",\"source_bytes\":" << 1'829'256'704ULL
        << ",\"k3x_bytes\":" << reader.value().superblock().file_length
        << ",\"cold_latency_nanoseconds\":" << cold_latency
        << ",\"cold_kernel_nanoseconds\":"
        << cold_profile_after.device_nanoseconds -
               cold_profile_before.device_nanoseconds
        << ",\"cold_weight_h2d_bytes\":"
        << cold_weight
        << ",\"cold_bf16_weight_h2d_bytes\":" << cold_bf16
        << ",\"cold_f32_weight_h2d_bytes\":" << cold_f32
        << ",\"cold_mxfp4_weight_h2d_bytes\":" << cold_mxfp4;
    if (arguments->validation_explicit) {
        std::cout
            << ",\"cold_immutable_validation_scans\":"
            << cold_stats_after.immutable_validation_scans -
                   cold_stats_before.immutable_validation_scans
            << ",\"cold_immutable_validation_hits\":"
            << cold_stats_after.immutable_validation_hits -
                   cold_stats_before.immutable_validation_hits
            << ",\"cold_immutable_validation_bytes\":"
            << cold_stats_after.immutable_validation_bytes -
                   cold_stats_before.immutable_validation_bytes
            << ",\"cold_immutable_validation_nanoseconds\":"
            << cold_stats_after.immutable_validation_nanoseconds -
                   cold_stats_before.immutable_validation_nanoseconds;
    }
    if (arguments->state_transfer_explicit) {
        std::cout
            << ",\"cold_official_kda_device_state_seeds\":"
            << cold_stats_after.official_kda_device_state_seeds -
                   cold_stats_before.official_kda_device_state_seeds
            << ",\"cold_official_kda_device_state_continuations\":"
            << cold_stats_after.official_kda_device_state_continuations -
                   cold_stats_before.official_kda_device_state_continuations
            << ",\"cold_official_kda_device_state_publications\":"
            << cold_stats_after.official_kda_device_state_publications -
                   cold_stats_before.official_kda_device_state_publications
            << ",\"cold_official_kda_device_state_invalidations\":"
            << cold_stats_after.official_kda_device_state_invalidations -
                   cold_stats_before.official_kda_device_state_invalidations;
    }
    if (arguments->route_preparation_explicit) {
        std::cout
            << ",\"cold_official_moe_route_prepare_calls\":"
            << cold_stats_after.official_moe_route_prepare_calls -
                   cold_stats_before.official_moe_route_prepare_calls
            << ",\"cold_official_moe_route_prepare_kernel_launches\":"
            << cold_stats_after.official_moe_route_prepare_kernel_launches -
                   cold_stats_before.official_moe_route_prepare_kernel_launches
            << ",\"cold_official_moe_router_logit_d2h_bytes\":"
            << cold_stats_after.official_moe_router_logit_d2h_bytes -
                   cold_stats_before.official_moe_router_logit_d2h_bytes
            << ",\"cold_official_moe_prepared_seeds\":"
            << cold_stats_after.official_moe_prepared_seeds -
                   cold_stats_before.official_moe_prepared_seeds
            << ",\"cold_official_moe_prepared_consumes\":"
            << cold_stats_after.official_moe_prepared_consumes -
                   cold_stats_before.official_moe_prepared_consumes
            << ",\"cold_official_moe_prepared_discards\":"
            << cold_stats_after.official_moe_prepared_discards -
                   cold_stats_before.official_moe_prepared_discards
            << ",\"cold_official_moe_prepared_invalidations\":"
            << cold_stats_after.official_moe_prepared_invalidations -
                   cold_stats_before.official_moe_prepared_invalidations;
    }
    std::cout << ",\"latency_nanoseconds_p05\":" << percentile(samples, 5)
        << ",\"latency_nanoseconds_median\":" << median(samples)
        << ",\"latency_nanoseconds_p95\":" << percentile(samples, 95)
        << ",\"kernel_nanoseconds\":" << kernel_time
        << ",\"orchestration_nanoseconds\":"
        << latency_sum - std::min(latency_sum, kernel_time)
        << ",\"weight_h2d_bytes\":"
        << measured_weight
        << ",\"bf16_weight_h2d_bytes\":" << measured_bf16
        << ",\"f32_weight_h2d_bytes\":" << measured_f32
        << ",\"mxfp4_weight_h2d_bytes\":" << measured_mxfp4
        << ",\"activation_h2d_bytes\":"
        << stats_after.activation_h2d_bytes - stats_before.activation_h2d_bytes
        << ",\"device_to_host_bytes\":"
        << stats_after.device_to_host_bytes - stats_before.device_to_host_bytes
        << ",\"official_kda_calls\":"
        << stats_after.official_kda_calls - stats_before.official_kda_calls
        << ",\"official_kda_kernel_launches\":"
        << stats_after.official_kda_kernel_launches -
               stats_before.official_kda_kernel_launches
        << ",\"official_kda_state_h2d_bytes\":"
        << stats_after.official_kda_state_h2d_bytes -
               stats_before.official_kda_state_h2d_bytes
        << ",\"official_kda_state_d2h_bytes\":"
        << stats_after.official_kda_state_d2h_bytes -
               stats_before.official_kda_state_d2h_bytes
        << ",\"official_kda_output_d2h_bytes\":"
        << stats_after.official_kda_output_d2h_bytes -
               stats_before.official_kda_output_d2h_bytes
        << ",\"resident_weight_bytes\":" << stats_after.resident_weight_bytes
        << ",\"peak_resident_weight_bytes\":"
        << stats_after.peak_resident_weight_bytes
        << ",\"weight_cache_hits\":"
        << stats_after.weight_cache_hits - stats_before.weight_cache_hits
        << ",\"weight_cache_misses\":"
        << stats_after.weight_cache_misses - stats_before.weight_cache_misses
        << ",\"weight_cache_bypasses\":"
        << stats_after.weight_cache_bypasses - stats_before.weight_cache_bypasses
        << ",\"device_allocation_count\":"
        << stats_after.device_allocation_count - stats_before.device_allocation_count
        << ",\"stream_synchronization_count\":"
        << stats_after.stream_synchronization_count -
               stats_before.stream_synchronization_count;
    if (arguments->state_transfer_explicit) {
        std::cout
            << ",\"official_kda_device_state_seeds\":"
            << stats_after.official_kda_device_state_seeds -
                   stats_before.official_kda_device_state_seeds
            << ",\"official_kda_device_state_continuations\":"
            << stats_after.official_kda_device_state_continuations -
                   stats_before.official_kda_device_state_continuations
            << ",\"official_kda_device_state_publications\":"
            << stats_after.official_kda_device_state_publications -
                   stats_before.official_kda_device_state_publications
            << ",\"official_kda_device_state_invalidations\":"
            << stats_after.official_kda_device_state_invalidations -
                   stats_before.official_kda_device_state_invalidations;
    }
    if (arguments->route_preparation_explicit) {
        std::cout
            << ",\"official_moe_route_prepare_calls\":"
            << stats_after.official_moe_route_prepare_calls -
                   stats_before.official_moe_route_prepare_calls
            << ",\"official_moe_route_prepare_kernel_launches\":"
            << stats_after.official_moe_route_prepare_kernel_launches -
                   stats_before.official_moe_route_prepare_kernel_launches
            << ",\"official_moe_router_logit_d2h_bytes\":"
            << stats_after.official_moe_router_logit_d2h_bytes -
                   stats_before.official_moe_router_logit_d2h_bytes
            << ",\"official_moe_prepared_seeds\":"
            << stats_after.official_moe_prepared_seeds -
                   stats_before.official_moe_prepared_seeds
            << ",\"official_moe_prepared_consumes\":"
            << stats_after.official_moe_prepared_consumes -
                   stats_before.official_moe_prepared_consumes
            << ",\"official_moe_prepared_discards\":"
            << stats_after.official_moe_prepared_discards -
                   stats_before.official_moe_prepared_discards
            << ",\"official_moe_prepared_invalidations\":"
            << stats_after.official_moe_prepared_invalidations -
                   stats_before.official_moe_prepared_invalidations
            << ",\"official_moe_prepared_slot_bytes\":"
            << stats_after.official_moe_prepared_slot_bytes;
    }
    if (arguments->validation_explicit) {
        std::cout
            << ",\"immutable_validation_scans\":"
            << stats_after.immutable_validation_scans -
                   stats_before.immutable_validation_scans
            << ",\"immutable_validation_hits\":"
            << stats_after.immutable_validation_hits -
                   stats_before.immutable_validation_hits
            << ",\"immutable_validation_bytes\":"
            << stats_after.immutable_validation_bytes -
                   stats_before.immutable_validation_bytes
            << ",\"immutable_validation_nanoseconds\":"
            << stats_after.immutable_validation_nanoseconds -
                   stats_before.immutable_validation_nanoseconds;
    }
    std::cout << ",\"peak_vram_bytes\":" << memory.peak_device_bytes
        << ",\"process_peak_rss_bytes\":" << process_peak_rss
        << ",\"reader_read_calls\":" << reader_counters.calls
        << ",\"reader_requested_bytes\":" << reader_counters.requested_bytes
        << ",\"reader_completed_bytes\":" << reader_counters.completed_bytes
        << ",\"reader_storage_submitted_bytes\":"
        << reader_counters.storage_submitted_bytes
        << ",\"reader_storage_completed_bytes\":"
        << reader_counters.storage_completed_bytes
        << ",\"maximum_absolute_error\":" << maximum_error
        << ",\"all_finite\":true}\n";
    return maximum_error <= 2.0e-2F ? 0 : 4;
}
