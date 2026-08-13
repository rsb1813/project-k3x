// 공식 두 레이어 artifact와 실행 옵션을 CUDA 생성 전에 엄격히 검증합니다.
#include "k3x/format.hpp"
#include "k3x/reader.hpp"
#include "k3x/checksums.hpp"
#include "k3x/strict_json.hpp"
#include "k3x/official_two_layer.hpp"
#include "official_layer_fixture.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <iterator>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <span>
#include <string>
#include <string_view>
#include <system_error>
#include <utility>
#include <vector>

namespace {

namespace json = k3x::strict_json;

constexpr float kPortableOutputTolerance = 2.0e-3F;
constexpr float kPortableConvolutionTolerance = 8.0e-3F;
constexpr float kPortableRecurrentTolerance = 5.0e-4F;
constexpr float kPortableContributionTolerance = 2.0e-5F;
constexpr float kCudaMaximumTolerance = 2.0e-3F;

enum class Mode { host_round_trip, device_closure };

struct Arguments {
    std::filesystem::path artifact;
    std::filesystem::path manifest;
    std::filesystem::path oracle;
    Mode mode{Mode::device_closure};
    std::uint64_t resident_bytes{};
    std::uint64_t warmup{};
    std::uint64_t iterations{};
    bool attribution{};
};

std::optional<std::uint64_t> parse_u64(std::string_view text) {
    std::uint64_t value{};
    const auto result = std::from_chars(
        text.data(), text.data() + text.size(), value);
    if (result.ec != std::errc{} || result.ptr != text.data() + text.size()) {
        return std::nullopt;
    }
    return value;
}

std::optional<Arguments> parse_arguments(int argc, char** argv) {
    if (argc < 2 || argc % 2 == 0) {
        std::cerr << "arguments must be key-value pairs\n";
        return std::nullopt;
    }
    Arguments result;
    bool mode_seen = false;
    for (int index = 1; index < argc; index += 2) {
        const std::string_view key = argv[index];
        const std::string_view value = argv[index + 1];
        if (key == "--artifact") result.artifact = value;
        else if (key == "--manifest") result.manifest = value;
        else if (key == "--oracle") result.oracle = value;
        else if (key == "--mode") {
            mode_seen = true;
            if (value == "host-round-trip") {
                result.mode = Mode::host_round_trip;
            } else if (value == "device-closure") {
                result.mode = Mode::device_closure;
            } else {
                std::cerr << "invalid mode\n";
                return std::nullopt;
            }
        } else if (key == "--resident-bytes") {
            const auto parsed = parse_u64(value);
            if (!parsed) {
                std::cerr << "invalid resident bytes\n";
                return std::nullopt;
            }
            result.resident_bytes = *parsed;
        } else if (key == "--warmup") {
            const auto parsed = parse_u64(value);
            if (!parsed) {
                std::cerr << "invalid warmup\n";
                return std::nullopt;
            }
            result.warmup = *parsed;
        } else if (key == "--iterations") {
            const auto parsed = parse_u64(value);
            if (!parsed) {
                std::cerr << "invalid iterations\n";
                return std::nullopt;
            }
            result.iterations = *parsed;
        } else if (key == "--attribution") {
            if (value == "true") {
                result.attribution = true;
            } else if (value == "false") {
                result.attribution = false;
            } else {
                std::cerr << "invalid attribution\n";
                return std::nullopt;
            }
        } else {
            std::cerr << "unknown argument\n";
            return std::nullopt;
        }
    }
    if (!mode_seen || result.artifact.empty() || result.manifest.empty() ||
        result.oracle.empty() || !result.resident_bytes ||
        !result.iterations) {
        std::cerr << "missing required argument\n";
        return std::nullopt;
    }
    return result;
}

constexpr std::uint64_t kOracleBytes = 13'053'992;
constexpr std::array<std::string_view, 17> kKdaSuffixes{
    "self_attention_res_norm.weight", "self_attention_res_proj.weight",
    "input_layernorm.weight", "self_attn.q_proj.weight",
    "self_attn.q_conv1d.weight", "self_attn.k_proj.weight",
    "self_attn.k_conv1d.weight", "self_attn.v_proj.weight",
    "self_attn.v_conv1d.weight", "self_attn.f_a_proj.weight",
    "self_attn.f_b_proj.weight", "self_attn.A_log", "self_attn.dt_bias",
    "self_attn.b_proj.weight", "self_attn.g_proj.weight",
    "self_attn.o_norm.weight", "self_attn.o_proj.weight"};
constexpr std::array<std::string_view, 6> kMoePreExpertSuffixes{
    "mlp_res_norm.weight", "mlp_res_proj.weight",
    "post_attention_layernorm.weight", "block_sparse_moe.gate.weight",
    "block_sparse_moe.gate.e_score_correction_bias",
    "block_sparse_moe.routed_expert_down_proj.weight"};
constexpr std::array<std::string_view, 5> kMoePostExpertSuffixes{
    "block_sparse_moe.routed_expert_norm.weight",
    "block_sparse_moe.routed_expert_up_proj.weight",
    "block_sparse_moe.shared_experts.gate_proj.weight",
    "block_sparse_moe.shared_experts.up_proj.weight",
    "block_sparse_moe.shared_experts.down_proj.weight"};

enum class ManifestStatus { valid, syntax_error, identity_error };

struct Manifest {
    struct Step {
        std::size_t position{};
        std::uint32_t layer_id{};
        std::vector<std::uint32_t> expert_ids;
        std::vector<float> contributions;
        std::string consumes_state_sha256;
        std::string state_sha256;
        std::string kda_output_sha256;
        std::string contribution_sha256;
        std::string output_sha256;
    };

    std::array<std::vector<std::uint32_t>, 2> selected_experts;
    std::array<Step, 4> steps;
    std::array<std::string, 2> final_state_sha256;
    std::string oracle_filename;
    std::string oracle_sha256;
    std::uint64_t oracle_bytes{};
    std::string artifact_root_sha256;
    std::string source_sha256;
    std::map<std::string, std::string> tensor_sha256;
};

struct ManifestResult {
    ManifestStatus status{ManifestStatus::syntax_error};
    Manifest value;

    ManifestResult() = default;
    ManifestResult(ManifestStatus result_status) : status(result_status) {}
    ManifestResult(ManifestStatus result_status, Manifest result_value)
        : status(result_status), value(std::move(result_value)) {}
};

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

bool exact_layer_ids(const json::Value& root) {
    const auto* values = json::array(json::member(root, "layer_ids"));
    if (!values || values->size() != 2) return false;
    for (std::size_t index = 0; index < values->size(); ++index) {
        const auto* value = json::number(&(*values)[index]);
        if (!value || *value < 0 || std::floor(*value) != *value ||
            *value != static_cast<double>(index + 1)) {
            return false;
        }
    }
    return true;
}

bool exact_step_order(const json::Value& root) {
    constexpr std::array expected{"a:1", "a:2", "b:1", "b:2"};
    const auto* values = json::array(json::member(root, "step_order"));
    if (!values || values->size() != expected.size()) return false;
    for (std::size_t index = 0; index < values->size(); ++index) {
        const auto* value = json::string(&(*values)[index]);
        if (!value || *value != expected[index]) return false;
    }
    return true;
}

std::optional<std::vector<std::uint32_t>> expert_ids(
    const json::Value* value) {
    const auto* values = json::array(value);
    if (!values || values->empty() || values->size() > 32) {
        return std::nullopt;
    }
    std::vector<std::uint32_t> result;
    std::set<std::uint32_t> unique;
    for (const auto& item : *values) {
        const auto* number = json::number(&item);
        if (!number || *number < 0 || *number > 895 ||
            std::floor(*number) != *number) {
            return std::nullopt;
        }
        const auto id = static_cast<std::uint32_t>(*number);
        if (!unique.insert(id).second) return std::nullopt;
        result.push_back(id);
    }
    return result;
}

std::optional<std::vector<float>> contributions(const json::Value* value) {
    const auto* values = json::array(value);
    if (!values || values->size() != 16) return std::nullopt;
    std::vector<float> result;
    result.reserve(values->size());
    double total{};
    for (const auto& item : *values) {
        const auto* number = json::number(&item);
        if (!number || !std::isfinite(*number) || *number < 0) {
            return std::nullopt;
        }
        total += *number;
        result.push_back(static_cast<float>(*number));
    }
    if (std::abs(total - 1.0) > 2.0e-5) return std::nullopt;
    return result;
}

ManifestResult load_manifest(const std::filesystem::path& path) {
    std::ifstream stream(path, std::ios::binary);
    if (!stream) return {};
    std::string text{std::istreambuf_iterator<char>(stream), {}};
    if (text.empty() || text.size() > 16 * 1024 * 1024) return {};
    auto root = json::Parser(text).parse();
    if (!root || !json::object(&*root)) return {};
    if (!text_is(*root, "format", "k3x-official-two-layer-v1") ||
        !text_is(*root, "repository", "moonshotai/Kimi-K3") ||
        !text_is(*root, "resolved_revision",
                 "9f62e4e9fffbd0a83ddd60e1c209d828994b3569") ||
        !text_is(*root, "snapshot_sha256",
                 "deaa6394b80afe12976ce8efbbf2463f6808c291d83b029e6b0cfb98de90a4e5") ||
        !text_is(*root, "index_sha256",
                 "a1c5210650ce71d2d3ae9ec5a101ac4afd3cf4b10091be589853437eb967febd") ||
        !text_is(*root, "config_sha256",
                 "9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213") ||
        !text_is(*root, "source_blob_id",
                 "b8c41e8bfce768d74d8da3a37e693f5ee43876a0") ||
        !exact_layer_ids(*root) || !exact_step_order(*root)) {
        return {ManifestStatus::identity_error};
    }
    const auto* shard_paths = json::array(json::member(*root, "shard_paths"));
    if (!shard_paths || shard_paths->size() != 2) {
        return {ManifestStatus::identity_error};
    }
    constexpr std::array expected_shards{
        "model-00002-of-000096.safetensors",
        "model-00003-of-000096.safetensors"};
    for (std::size_t index = 0; index < shard_paths->size(); ++index) {
        const auto* shard = json::string(&(*shard_paths)[index]);
        if (!shard || *shard != expected_shards[index]) {
            return {ManifestStatus::identity_error};
        }
    }
    Manifest manifest;
    const auto* selected = json::array(json::member(*root, "selected_experts"));
    if (!selected || selected->size() != 2) {
        return {ManifestStatus::identity_error};
    }
    for (std::size_t layer = 0; layer < 2; ++layer) {
        auto ids = expert_ids(&(*selected)[layer]);
        if (!ids) return {ManifestStatus::identity_error};
        manifest.selected_experts[layer] = std::move(*ids);
    }
    const auto* steps = json::array(json::member(*root, "steps"));
    if (!steps || steps->size() != manifest.steps.size()) {
        return {ManifestStatus::identity_error};
    }
    std::array<std::set<std::uint32_t>, 2> routed_unions;
    for (std::size_t index = 0; index < steps->size(); ++index) {
        const auto* step = json::object(&(*steps)[index]);
        if (!step) return {ManifestStatus::identity_error};
        const json::Value step_value{*step};
        const auto* position = json::string(json::member(step_value, "position"));
        const auto* layer = json::number(json::member(step_value, "layer_id"));
        auto ids = expert_ids(json::member(step_value, "expert_ids"));
        auto masses = contributions(json::member(step_value, "contributions"));
        const auto* consumes = json::string(
            json::member(step_value, "consumes_state_sha256"));
        const auto* state = json::string(
            json::member(step_value, "state_sha256"));
        const auto* kda_output = json::string(
            json::member(step_value, "kda_output_sha256"));
        const auto* contribution = json::string(
            json::member(step_value, "contribution_sha256"));
        const auto* output = json::string(
            json::member(step_value, "output_sha256"));
        const auto expected_position = index / 2;
        const auto expected_layer = static_cast<std::uint32_t>(index % 2 + 1);
        if (!position || *position != (expected_position ? "b" : "a") ||
            !layer || *layer != expected_layer || !ids || ids->size() != 16 ||
            !masses || !consumes || !hex_string(*consumes, 64) ||
            !state || !hex_string(*state, 64) ||
            !kda_output || !hex_string(*kda_output, 64) ||
            !contribution || !hex_string(*contribution, 64) ||
            !output || !hex_string(*output, 64)) {
            return {ManifestStatus::identity_error};
        }
        manifest.steps[index] = {
            expected_position, expected_layer, std::move(*ids),
            std::move(*masses), *consumes, *state, *kda_output,
            *contribution, *output};
        routed_unions[expected_layer - 1].insert(
            manifest.steps[index].expert_ids.begin(),
            manifest.steps[index].expert_ids.end());
    }
    for (std::size_t layer = 0; layer < 2; ++layer) {
        const std::set<std::uint32_t> selected(
            manifest.selected_experts[layer].begin(),
            manifest.selected_experts[layer].end());
        if (selected != routed_unions[layer]) {
            return {ManifestStatus::identity_error};
        }
        if (manifest.steps[layer + 2].consumes_state_sha256 !=
            manifest.steps[layer].state_sha256) {
            return {ManifestStatus::identity_error};
        }
    }
    const auto* final_states = json::array(
        json::member(*root, "final_state_sha256"));
    if (!final_states || final_states->size() != 2) {
        return {ManifestStatus::identity_error};
    }
    for (std::size_t layer = 0; layer < 2; ++layer) {
        const auto* digest = json::string(&(*final_states)[layer]);
        if (!digest || !hex_string(*digest, 64) ||
            *digest != manifest.steps[layer + 2].state_sha256) {
            return {ManifestStatus::identity_error};
        }
        manifest.final_state_sha256[layer] = *digest;
    }
    const auto* oracle = json::object(json::member(*root, "oracle"));
    const auto* artifact = json::object(json::member(*root, "artifact"));
    if (!oracle || !artifact ||
        !text_is(json::Value{*oracle}, "format",
                 "k3x-official-two-layer-oracle-v1")) {
        return {ManifestStatus::identity_error};
    }
    const json::Value oracle_value{*oracle};
    const json::Value artifact_value{*artifact};
    const auto* filename = json::string(json::member(oracle_value, "filename"));
    const auto* oracle_sha = json::string(json::member(oracle_value, "sha256"));
    const auto* oracle_bytes = json::number(json::member(oracle_value, "bytes"));
    const auto* filename_artifact = json::string(
        json::member(artifact_value, "filename"));
    const auto* root_sha = json::string(
        json::member(artifact_value, "k3x_root_sha256"));
    const auto* source_sha = json::string(
        json::member(artifact_value, "source_sha256"));
    const auto* tensor_sha = json::object(
        json::member(artifact_value, "tensor_sha256"));
    if (!filename || *filename != "official-two-layer-oracle-v1.bin" ||
        !oracle_sha || !hex_string(*oracle_sha, 64) || !oracle_bytes ||
        *oracle_bytes != static_cast<double>(kOracleBytes) ||
        !filename_artifact || *filename_artifact != "official-two-layer.k3x" ||
        !root_sha || !hex_string(*root_sha, 64) ||
        !source_sha || !hex_string(*source_sha, 64) || !tensor_sha) {
        return {ManifestStatus::identity_error};
    }
    manifest.oracle_filename = *filename;
    manifest.oracle_sha256 = *oracle_sha;
    manifest.oracle_bytes = kOracleBytes;
    manifest.artifact_root_sha256 = *root_sha;
    manifest.source_sha256 = *source_sha;
    for (const auto& [name, value] : *tensor_sha) {
        const auto* digest = json::string(&value);
        if (!digest || !hex_string(*digest, 64)) {
            return {ManifestStatus::identity_error};
        }
        manifest.tensor_sha256.emplace(name, *digest);
    }
    return {ManifestStatus::valid, std::move(manifest)};
}

std::string digest_hex(const std::array<std::byte, 32>& digest) {
    static constexpr char alphabet[] = "0123456789abcdef";
    std::string result(64, '0');
    for (std::size_t index = 0; index < digest.size(); ++index) {
        const auto value = std::to_integer<unsigned>(digest[index]);
        result[index * 2] = alphabet[value >> 4U];
        result[index * 2 + 1] = alphabet[value & 0x0fU];
    }
    return result;
}

struct Oracle {
    std::array<std::vector<float>, 2> outputs;
    std::array<k3x::OfficialKdaState, 2> states;
};

std::uint64_t read_u64(std::string_view bytes, std::size_t offset) {
    std::uint64_t result{};
    for (std::size_t index = 0; index < 8; ++index) {
        result |= static_cast<std::uint64_t>(
            static_cast<unsigned char>(bytes[offset + index])) << (index * 8);
    }
    return result;
}

std::optional<Oracle> load_oracle(const std::filesystem::path& path,
                                  const Manifest& manifest) {
    if (path.filename() != manifest.oracle_filename) return std::nullopt;
    std::ifstream stream(path, std::ios::binary);
    if (!stream) return std::nullopt;
    std::string bytes{std::istreambuf_iterator<char>(stream), {}};
    if (bytes.size() != manifest.oracle_bytes ||
        digest_hex(k3x::sha256(std::as_bytes(std::span(bytes)))) !=
            manifest.oracle_sha256) {
        return std::nullopt;
    }
    constexpr std::array<char, 8> magic{'K', '3', 'X', 'O', 'R', 'C', '2', 0};
    if (!std::equal(magic.begin(), magic.end(), bytes.begin()) ||
        read_u64(bytes, 8) != 14'336 || read_u64(bytes, 16) != 2 ||
        read_u64(bytes, 24) != 36'864 ||
        read_u64(bytes, 32) != 1'572'864) {
        return std::nullopt;
    }
    std::size_t cursor = 40;
    const auto read_bf16 = [&](std::size_t count) {
        std::vector<std::uint16_t> result(count);
        std::memcpy(result.data(), bytes.data() + cursor, count * 2);
        cursor += count * 2;
        return result;
    };
    Oracle result;
    const auto output_words = read_bf16(14'336);
    for (std::size_t position = 0; position < 2; ++position) {
        result.outputs[position].resize(7'168);
        for (std::size_t index = 0; index < 7'168; ++index) {
            result.outputs[position][index] = k3x::decode_bf16_word(
                output_words[position * 7'168 + index]);
        }
    }
    for (auto& state : result.states) {
        state.conv_q = read_bf16(36'864);
        state.conv_k = read_bf16(36'864);
        state.conv_v = read_bf16(36'864);
        state.recurrent_v_first.resize(1'572'864);
        std::memcpy(state.recurrent_v_first.data(), bytes.data() + cursor,
                    state.recurrent_v_first.size() * sizeof(float));
        cursor += state.recurrent_v_first.size() * sizeof(float);
        if (!std::all_of(
                state.recurrent_v_first.begin(),
                state.recurrent_v_first.end(),
                [](float value) { return std::isfinite(value); })) {
            return std::nullopt;
        }
    }
    if (cursor != bytes.size()) return std::nullopt;
    return result;
}

std::vector<float> input_values(int multiplier, int increment,
                                int modulus, int offset) {
    std::vector<float> result(7'168);
    for (std::size_t index = 0; index < result.size(); ++index) {
        result[index] = static_cast<float>(
            ((multiplier * static_cast<int>(index) + increment) % modulus) -
            offset) / 1024.0F;
    }
    return result;
}

std::array<k3x::OfficialLayerInput, 2> layer_inputs() {
    return {{
        {input_values(17, 3, 257, 128), input_values(29, 11, 251, 125)},
        {input_values(31, 7, 263, 131), input_values(43, 19, 269, 134)},
    }};
}

bool close(std::span<const float> left, std::span<const float> right,
           float tolerance) {
    if (left.size() != right.size()) return false;
    for (std::size_t index = 0; index < left.size(); ++index) {
        if (!std::isfinite(left[index]) || !std::isfinite(right[index]) ||
            std::abs(left[index] - right[index]) > tolerance) {
            return false;
        }
    }
    return true;
}

float maximum_error(std::span<const float> left,
                    std::span<const float> right) {
    if (left.size() != right.size()) {
        return std::numeric_limits<float>::infinity();
    }
    float result{};
    for (std::size_t index = 0; index < left.size(); ++index) {
        result = std::max(result, std::abs(left[index] - right[index]));
    }
    return result;
}

float maximum_bf16_error(std::span<const std::uint16_t> left,
                         std::span<const std::uint16_t> right) {
    if (left.size() != right.size()) {
        return std::numeric_limits<float>::infinity();
    }
    float result{};
    for (std::size_t index = 0; index < left.size(); ++index) {
        result = std::max(
            result,
            std::abs(k3x::decode_bf16_word(left[index]) -
                     k3x::decode_bf16_word(right[index])));
    }
    return result;
}

bool bf16_close(std::span<const std::uint16_t> left,
                std::span<const std::uint16_t> right, float tolerance) {
    if (left.size() != right.size()) return false;
    for (std::size_t index = 0; index < left.size(); ++index) {
        const auto left_value = k3x::decode_bf16_word(left[index]);
        const auto right_value = k3x::decode_bf16_word(right[index]);
        if (!std::isfinite(left_value) || !std::isfinite(right_value) ||
            std::abs(left_value - right_value) > tolerance) {
            return false;
        }
    }
    return true;
}

bool route_close(const k3x::OfficialRoute& actual,
                 std::span<const std::uint32_t> expected_ids,
                 std::span<const float> expected_contributions,
                 float tolerance) {
    if (actual.expert_ids.size() != expected_ids.size() ||
        actual.contributions.size() != expected_contributions.size()) {
        return false;
    }
    for (std::size_t index = 0; index < expected_ids.size(); ++index) {
        const auto found = std::find(
            actual.expert_ids.begin(), actual.expert_ids.end(),
            expected_ids[index]);
        if (found == actual.expert_ids.end()) return false;
        const auto actual_index = static_cast<std::size_t>(
            std::distance(actual.expert_ids.begin(), found));
        if (!std::isfinite(actual.contributions[actual_index]) ||
            !std::isfinite(expected_contributions[index]) ||
            std::abs(actual.contributions[actual_index] -
                     expected_contributions[index]) > tolerance) {
            return false;
        }
    }
    return true;
}

std::string contribution_digest(const k3x::OfficialRoute& route) {
    k3x::Sha256Hasher digest;
    digest.update(std::as_bytes(std::span(route.expert_ids)));
    digest.update(std::as_bytes(std::span(route.contributions)));
    return digest_hex(digest.finish());
}

std::uint16_t encode_bf16(float value) {
    auto bits = std::bit_cast<std::uint32_t>(value);
    bits += 0x7fffU + ((bits >> 16U) & 1U);
    return static_cast<std::uint16_t>(bits >> 16U);
}

std::string final_output_digest(std::span<const float> values) {
    static constexpr char identity[] =
        "k3x-official-two-layer-final-output-bf16-v1\0";
    std::vector<std::uint16_t> words(values.size());
    std::transform(values.begin(), values.end(), words.begin(), encode_bf16);
    k3x::Sha256Hasher digest;
    digest.update(std::as_bytes(
        std::span(identity).first(sizeof(identity) - 1)));
    digest.update(std::as_bytes(std::span(words)));
    return digest_hex(digest.finish());
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
    static constexpr char identity[] =
        "k3x-official-kda-state-v1\0v-first-fp32\0";
    k3x::Sha256Hasher digest;
    digest.update(std::as_bytes(
        std::span(identity).first(sizeof(identity) - 1)));
    const std::array<std::uint64_t, 3> conv_shape{1, 3, 12'288};
    const std::array<std::uint64_t, 4> recurrent_shape{1, 96, 128, 128};
    update_state_tensor(digest, "conv_q", conv_shape,
                        std::span(state.conv_q));
    update_state_tensor(digest, "conv_k", conv_shape,
                        std::span(state.conv_k));
    update_state_tensor(digest, "conv_v", conv_shape,
                        std::span(state.conv_v));
    update_state_tensor(digest, "recurrent_v_first", recurrent_shape,
                        std::span(state.recurrent_v_first));
    return digest_hex(digest.finish());
}

bool validate_portable(
    const k3x::OfficialTwoLayerResult& result,
    const Manifest& manifest, const Oracle& oracle) {
    if (result.steps.size() != 4) return false;
    bool valid = true;
    for (std::size_t index = 0; index < result.steps.size(); ++index) {
        const auto& actual = result.steps[index];
        const auto& expected = manifest.steps[index];
        const auto fail = [index](std::string_view field) {
            std::cerr << "portable step " << index << ' ' << field
                      << " mismatch\n";
            return false;
        };
        if (actual.position != expected.position ||
            actual.layer_id != expected.layer_id) {
            return fail("identity");
        }
        if (!route_close(actual.result.route, expected.expert_ids,
                         expected.contributions,
                         kPortableContributionTolerance)) {
            std::cerr << "expected route/contribution:";
            for (std::size_t route_index = 0;
                 route_index < expected.expert_ids.size(); ++route_index) {
                std::cerr << ' ' << expected.expert_ids[route_index] << '='
                          << expected.contributions[route_index];
            }
            std::cerr << "\nactual route/contribution:";
            for (std::size_t route_index = 0;
                 route_index < actual.result.route.expert_ids.size();
                 ++route_index) {
                std::cerr << ' '
                          << actual.result.route.expert_ids[route_index] << '='
                          << actual.result.route.contributions[route_index];
            }
            std::cerr << '\n';
            fail("route");
            valid = false;
        }
        const k3x::OfficialRoute manifest_route{
            expected.expert_ids, expected.contributions, {}};
        if (contribution_digest(manifest_route) !=
            expected.contribution_sha256) {
            return fail("manifest contribution digest");
        }
    }
    for (std::size_t position = 0; position < 2; ++position) {
        if (!close(result.outputs[position], oracle.outputs[position],
                   kPortableOutputTolerance)) {
            std::cerr << "portable final output " << position
                      << " mismatch; maximum error="
                      << maximum_error(result.outputs[position],
                                       oracle.outputs[position]) << '\n';
            valid = false;
        }
    }
    for (std::size_t layer = 0; layer < 2; ++layer) {
        const auto& actual = result.final_states[layer];
        const auto& expected = oracle.states[layer];
        if (state_digest(expected) != manifest.final_state_sha256[layer]) {
            std::cerr << "portable oracle final state digest " << layer
                      << " mismatch\n";
            return false;
        }
        if (!bf16_close(actual.conv_q, expected.conv_q,
                        kPortableConvolutionTolerance) ||
            !bf16_close(actual.conv_k, expected.conv_k,
                        kPortableConvolutionTolerance) ||
            !bf16_close(actual.conv_v, expected.conv_v,
                        kPortableConvolutionTolerance) ||
            !close(actual.recurrent_v_first,
                   expected.recurrent_v_first,
                   kPortableRecurrentTolerance)) {
            std::cerr << "portable final state " << layer
                      << " mismatch; conv q/k/v maximum error="
                      << maximum_bf16_error(actual.conv_q, expected.conv_q)
                      << '/'
                      << maximum_bf16_error(actual.conv_k, expected.conv_k)
                      << '/'
                      << maximum_bf16_error(actual.conv_v, expected.conv_v)
                      << "; recurrent maximum error="
                      << maximum_error(actual.recurrent_v_first,
                                       expected.recurrent_v_first) << '\n';
            valid = false;
        }
    }
    return valid;
}

float maximum_cuda_error(const k3x::OfficialTwoLayerCudaResult& actual,
                         const k3x::OfficialTwoLayerResult& expected,
                         float& output_maximum,
                         float& recurrent_maximum) {
    float maximum{};
    output_maximum = 0.0F;
    recurrent_maximum = 0.0F;
    for (std::size_t position = 0; position < 2; ++position) {
        if (actual.outputs[position].size() != expected.outputs[position].size()) {
            return std::numeric_limits<float>::infinity();
        }
        for (std::size_t index = 0; index < actual.outputs[position].size();
             ++index) {
            maximum = std::max(
                maximum,
                std::abs(actual.outputs[position][index] -
                         expected.outputs[position][index]));
            output_maximum = std::max(
                output_maximum,
                std::abs(actual.outputs[position][index] -
                         expected.outputs[position][index]));
        }
    }
    for (std::size_t layer = 0; layer < 2; ++layer) {
        const auto& observed = actual.final_states[layer];
        const auto& reference = expected.final_states[layer];
        if (!bf16_close(observed.conv_q, reference.conv_q,
                        kPortableConvolutionTolerance) ||
            !bf16_close(observed.conv_k, reference.conv_k,
                        kPortableConvolutionTolerance) ||
            !bf16_close(observed.conv_v, reference.conv_v,
                        kPortableConvolutionTolerance) ||
            observed.recurrent_v_first.size() !=
                reference.recurrent_v_first.size()) {
            return std::numeric_limits<float>::infinity();
        }
        for (std::size_t index = 0;
             index < observed.recurrent_v_first.size(); ++index) {
            maximum = std::max(
                maximum,
                std::abs(observed.recurrent_v_first[index] -
                         reference.recurrent_v_first[index]));
            recurrent_maximum = std::max(
                recurrent_maximum,
                std::abs(observed.recurrent_v_first[index] -
                         reference.recurrent_v_first[index]));
        }
    }
    return maximum;
}

float maximum_cuda_oracle_output_error(
    const k3x::OfficialTwoLayerCudaResult& actual, const Oracle& oracle) {
    float result{};
    for (std::size_t position = 0; position < 2; ++position) {
        result = std::max(
            result,
            maximum_error(actual.outputs[position], oracle.outputs[position]));
    }
    return result;
}

float maximum_cuda_oracle_recurrent_error(
    const k3x::OfficialTwoLayerCudaResult& actual, const Oracle& oracle) {
    float result{};
    for (std::size_t layer = 0; layer < 2; ++layer) {
        result = std::max(
            result,
            maximum_error(actual.final_states[layer].recurrent_v_first,
                          oracle.states[layer].recurrent_v_first));
    }
    return result;
}

bool cuda_oracle_state_close(
    const k3x::OfficialTwoLayerCudaResult& actual, const Oracle& oracle) {
    for (std::size_t layer = 0; layer < 2; ++layer) {
        if (!bf16_close(actual.final_states[layer].conv_q,
                        oracle.states[layer].conv_q,
                        kPortableConvolutionTolerance) ||
            !bf16_close(actual.final_states[layer].conv_k,
                        oracle.states[layer].conv_k,
                        kPortableConvolutionTolerance) ||
            !bf16_close(actual.final_states[layer].conv_v,
                        oracle.states[layer].conv_v,
                        kPortableConvolutionTolerance) ||
            !close(actual.final_states[layer].recurrent_v_first,
                   oracle.states[layer].recurrent_v_first,
                   kPortableRecurrentTolerance)) {
            return false;
        }
    }
    return true;
}

k3x::BackendOptions backend_options(std::uint64_t resident_bytes) {
    k3x::BackendOptions result;
    result.kind = k3x::BackendKind::cuda_custom;
    result.cuda_allocation = k3x::CudaAllocationMode::reused;
    result.cuda_weights = k3x::CudaWeightMode::resident;
    result.cuda_batching = k3x::CudaBatchingMode::resident_grid;
    result.cuda_boundary = k3x::CudaBoundaryMode::moe_layer;
    result.cuda_resident_bytes = resident_bytes;
    result.cuda_weight_validation = k3x::CudaWeightValidationMode::admission;
    return result;
}

void write_u64_array(std::ostream& stream,
                     std::span<const std::uint64_t> values) {
    stream << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index) stream << ',';
        stream << values[index];
    }
    stream << ']';
}

void write_u32_arrays(
    std::ostream& stream,
    std::span<const k3x::OfficialTwoLayerCudaStepResult> steps) {
    stream << '[';
    for (std::size_t index = 0; index < steps.size(); ++index) {
        if (index) stream << ',';
        stream << '[';
        for (std::size_t expert = 0;
             expert < steps[index].route.expert_ids.size(); ++expert) {
            if (expert) stream << ',';
            stream << steps[index].route.expert_ids[expert];
        }
        stream << ']';
    }
    stream << ']';
}

void write_string_array(std::ostream& stream,
                        std::span<const std::string> values) {
    stream << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index) stream << ',';
        stream << '\"' << values[index] << '\"';
    }
    stream << ']';
}

struct ExpectedTensor {
    std::string record_name;
    std::string data_name;
    std::string auxiliary_name;
};

std::vector<ExpectedTensor> expected_tensors(const Manifest& manifest) {
    std::vector<ExpectedTensor> expected;
    for (std::uint32_t layer = 1; layer <= 2; ++layer) {
        const auto base = "model.layers." + std::to_string(layer) + ".";
        for (const auto suffix : kKdaSuffixes) {
            const auto name = base + std::string(suffix);
            expected.push_back({name, name, {}});
        }
        for (const auto suffix : kMoePreExpertSuffixes) {
            const auto name = base + std::string(suffix);
            expected.push_back({name, name, {}});
        }
        for (const auto expert_id : manifest.selected_experts[layer - 1]) {
            const auto expert = base + "feed_forward.experts." +
                std::to_string(expert_id) + ".";
            for (const auto role : {"gate", "up", "down"}) {
                const auto record = expert + role;
                expected.push_back({record, record + ".weight_packed",
                                    record + ".weight_scale"});
            }
        }
        for (const auto suffix : kMoePostExpertSuffixes) {
            const auto name = base + std::string(suffix);
            expected.push_back({name, name, {}});
        }
    }
    return expected;
}

bool validate_tensor_identity(k3x::Reader& reader, const Manifest& manifest) {
    const auto expected = expected_tensors(manifest);
    std::size_t expected_digests{};
    for (const auto& item : expected) {
        expected_digests += item.auxiliary_name.empty() ? 1 : 2;
    }
    if (manifest.tensor_sha256.size() != expected_digests) return false;
    std::vector<const k3x::TensorRecord*> actual;
    actual.reserve(reader.tensors().size());
    for (const auto& record : reader.tensors()) actual.push_back(&record);
    std::sort(actual.begin(), actual.end(), [](const auto* left, const auto* right) {
        return left->data_offset < right->data_offset;
    });
    if (actual.size() != expected.size()) return false;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        const auto& item = expected[index];
        if (actual[index]->tensor_id !=
            k3x::fnv1a64(item.record_name.c_str())) {
            return false;
        }
        const auto digest = manifest.tensor_sha256.find(item.data_name);
        if (digest == manifest.tensor_sha256.end()) return false;
        auto data = reader.read_tensor(actual[index]->tensor_id);
        if (!data) return false;
        if (!item.auxiliary_name.empty()) {
            auto auxiliary = reader.read_auxiliary(actual[index]->tensor_id);
            const auto auxiliary_digest = manifest.tensor_sha256.find(
                item.auxiliary_name);
            if (!auxiliary ||
                auxiliary_digest == manifest.tensor_sha256.end() ||
                digest_hex(k3x::sha256(data.value())) != digest->second ||
                digest_hex(k3x::sha256(auxiliary.value())) !=
                    auxiliary_digest->second) {
                return false;
            }
        } else if (digest_hex(k3x::sha256(data.value())) != digest->second) {
            return false;
        }
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    const auto arguments = parse_arguments(argc, argv);
    if (!arguments) return 2;
    const auto manifest = load_manifest(arguments->manifest);
    if (manifest.status == ManifestStatus::syntax_error) {
        std::cerr << "INVALID_EXTENT: invalid two-layer manifest\n";
        return 4;
    }
    if (manifest.status == ManifestStatus::identity_error) {
        std::cerr << "INVALID_EXTENT: two-layer manifest identity mismatch\n";
        return 4;
    }
    const auto oracle = load_oracle(arguments->oracle, manifest.value);
    if (!oracle) {
        std::cerr << "INVALID_EXTENT: invalid two-layer oracle\n";
        return 4;
    }
    auto reader = k3x::Reader::open(
        arguments->artifact, k3x::VerifyMode::checksums);
    if (!reader) {
        std::cerr << k3x::error_code_name(reader.error()) << ": "
                  << reader.message() << '\n';
        return 4;
    }
    if (reader.value().superblock().optional_features !=
        (k3x::optional_storage_fixture |
         k3x::optional_official_moe_fixture)) {
        std::cerr << "INVALID_EXTENT: artifact is not two-layer fixture\n";
        return 4;
    }
    if (digest_hex(reader.value().superblock().root_sha256) !=
        manifest.value.artifact_root_sha256) {
        std::cerr << "ROOT_SHA256_MISMATCH: manifest root mismatch\n";
        return 4;
    }
    std::set<std::int32_t> tensor_layers;
    for (const auto& record : reader.value().tensors()) {
        if (record.layer_id >= 0) tensor_layers.insert(record.layer_id);
    }
    if (tensor_layers != std::set<std::int32_t>{1, 2}) {
        std::cerr << "INVALID_EXTENT: artifact layer identity mismatch\n";
        return 4;
    }
    if (!validate_tensor_identity(reader.value(), manifest.value)) {
        std::cerr << "INVALID_EXTENT: artifact tensor order mismatch\n";
        return 4;
    }
    auto first = k3x::bench::load_official_layer_fixture(
        reader.value(), 1, manifest.value.selected_experts[0]);
    auto second = k3x::bench::load_official_layer_fixture(
        reader.value(), 2, manifest.value.selected_experts[1]);
    if (!first || !second) {
        std::cerr << "INVALID_EXTENT: missing official two-layer tensor\n";
        return 4;
    }

    const k3x::OfficialKdaConfig config{
        7'168, 96, 128, 4, 1.0e-5F, -5.0F};
    const auto zero = k3x::zero_official_kda_state(config);
    const std::array<k3x::OfficialKdaState, 2> states{{zero, zero}};
    const std::array<k3x::OfficialTwoLayerWeights, 2> portable_layers{{
        {1, first->portable_views()}, {2, second->portable_views()}}};
    const auto inputs = layer_inputs();
    const auto portable = k3x::official_two_layer_cpu(
        inputs, portable_layers, states, config, 16, 4, 25);
    if (!portable ||
        !validate_portable(portable.value(), manifest.value, *oracle)) {
        std::cerr << "INVALID_STATE: portable two-layer oracle mismatch\n";
        return 4;
    }

    std::optional<k3x::Profiler> attribution_profiler;
    if (arguments->attribution) attribution_profiler.emplace();
    auto backend = k3x::make_cuda_backend(
        backend_options(arguments->resident_bytes),
        attribution_profiler ? &*attribution_profiler : nullptr);
    if (!backend) {
        std::cerr << k3x::error_code_name(backend.error()) << ": "
                  << backend.message() << '\n';
        return 5;
    }
    const std::array<k3x::OfficialTwoLayerCudaWeights, 2> cuda_layers{{
        {1, first->cuda_views()}, {2, second->cuda_views()}}};
    const auto mode = arguments->mode == Mode::host_round_trip
        ? k3x::OfficialTwoLayerCudaMode::host_round_trip
        : k3x::OfficialTwoLayerCudaMode::device_closure;
    k3x::OfficialTwoLayerCudaResult last;
    for (std::uint64_t index = 0; index < arguments->warmup; ++index) {
        auto warm = k3x::official_two_layer_cuda(
            *backend.value(), inputs, cuda_layers, states, config, 16, 4, 25,
            k3x::ProfilePhase::decode, mode);
        if (!warm) {
            std::cerr << k3x::error_code_name(warm.error()) << ": "
                      << warm.message() << '\n';
            return 5;
        }
    }
    std::vector<std::uint64_t> wall_nanoseconds;
    wall_nanoseconds.reserve(arguments->iterations);
    float maximum_error{};
    k3x::OfficialTwoLayerAttribution attribution;
    for (std::uint64_t index = 0; index < arguments->iterations; ++index) {
        const auto start = std::chrono::steady_clock::now();
        k3x::OfficialTwoLayerAttribution measured_attribution;
        auto measured = k3x::official_two_layer_cuda(
            *backend.value(), inputs, cuda_layers, states, config, 16, 4, 25,
            k3x::ProfilePhase::decode, mode,
            attribution_profiler ? &*attribution_profiler : nullptr,
            arguments->attribution ? &measured_attribution : nullptr);
        const auto elapsed = static_cast<std::uint64_t>(
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - start).count());
        if (!measured || measured.value().steps.size() != 4) {
            std::cerr << (measured ? "INVALID_STATE"
                                  : k3x::error_code_name(measured.error()))
                      << ": two-layer CUDA execution failed\n";
            return 5;
        }
        for (std::size_t step = 0; step < 4; ++step) {
            if (!route_close(
                    measured.value().steps[step].route,
                    manifest.value.steps[step].expert_ids,
                    manifest.value.steps[step].contributions,
                    kPortableContributionTolerance)) {
                std::cerr << "INVALID_STATE: CUDA route mismatch\n";
                return 5;
            }
        }
        float output_error{};
        float recurrent_error{};
        const auto error = maximum_cuda_error(
            measured.value(), portable.value(), output_error,
            recurrent_error);
        const auto oracle_output_error = maximum_cuda_oracle_output_error(
            measured.value(), *oracle);
        const auto oracle_recurrent_error =
            maximum_cuda_oracle_recurrent_error(measured.value(), *oracle);
        if (!std::isfinite(error) || error > kCudaMaximumTolerance ||
            !std::isfinite(oracle_output_error) ||
            oracle_output_error > kPortableOutputTolerance ||
            !std::isfinite(oracle_recurrent_error) ||
            oracle_recurrent_error > kPortableRecurrentTolerance ||
            !cuda_oracle_state_close(measured.value(), *oracle) ||
            (arguments->warmup &&
             measured.value().telemetry.weight_h2d_bytes != 0)) {
            std::cerr << "INVALID_STATE: CUDA parity or warm residency "
                      << "mismatch; maximum error=" << error
                      << "; output error=" << output_error
                      << "; recurrent error=" << recurrent_error
                      << "; oracle output error="
                      << oracle_output_error
                      << "; oracle recurrent error="
                      << oracle_recurrent_error
                      << "; weight H2D bytes="
                      << measured.value().telemetry.weight_h2d_bytes << '\n';
            return 5;
        }
        maximum_error = std::max(maximum_error, error);
        if (arguments->attribution) {
            attribution.total_wall_nanoseconds +=
                measured_attribution.total_wall_nanoseconds;
            attribution.front_wall_nanoseconds +=
                measured_attribution.front_wall_nanoseconds;
            attribution.front_device_nanoseconds +=
                measured_attribution.front_device_nanoseconds;
            attribution.route_wall_nanoseconds +=
                measured_attribution.route_wall_nanoseconds;
            attribution.tail_wall_nanoseconds +=
                measured_attribution.tail_wall_nanoseconds;
            attribution.tail_device_nanoseconds +=
                measured_attribution.tail_device_nanoseconds;
            attribution.unattributed_wall_nanoseconds +=
                measured_attribution.unattributed_wall_nanoseconds;
        }
        wall_nanoseconds.push_back(elapsed);
        last = std::move(measured.value());
    }
    const auto stats = backend.value()->runtime_stats();
    std::array<std::string, 4> measured_contribution_sha256;
    for (std::size_t index = 0; index < last.steps.size(); ++index) {
        measured_contribution_sha256[index] =
            contribution_digest(last.steps[index].route);
    }
    const std::array measured_output_sha256{
        final_output_digest(last.outputs[0]),
        final_output_digest(last.outputs[1])};
    const std::array measured_state_sha256{
        state_digest(last.final_states[0]),
        state_digest(last.final_states[1])};
    std::cout << "{\"schema\":\""
              << (arguments->attribution
                      ? "k3x-official-two-layer-attribution-v1"
                      : "k3x-official-two-layer-bench-v1") << "\""
              << ",\"mode\":\""
              << (arguments->mode == Mode::host_round_trip
                      ? "host-round-trip" : "device-closure") << "\""
              << ",\"warmup\":" << arguments->warmup
              << ",\"iterations\":" << arguments->iterations
              << ",\"wall_nanoseconds\":";
    write_u64_array(std::cout, wall_nanoseconds);
    std::cout << ",\"maximum_absolute_error\":" << maximum_error
              << ",\"weight_h2d_bytes\":"
              << last.telemetry.weight_h2d_bytes
              << ",\"activation_h2d_bytes\":"
              << last.telemetry.activation_h2d_bytes
              << ",\"device_to_host_bytes\":"
              << last.telemetry.device_to_host_bytes
              << ",\"state_h2d_bytes\":" << last.telemetry.state_h2d_bytes
              << ",\"state_d2h_bytes\":" << last.telemetry.state_d2h_bytes
              << ",\"kda_output_d2h_bytes\":"
              << last.telemetry.kda_output_d2h_bytes
              << ",\"router_logit_d2h_bytes\":"
              << last.telemetry.router_logit_d2h_bytes
              << ",\"inter_layer_hidden_h2d_bytes\":"
              << last.telemetry.inter_layer_hidden_h2d_bytes
              << ",\"inter_layer_hidden_d2h_bytes\":"
              << last.telemetry.inter_layer_hidden_d2h_bytes
              << ",\"final_hidden_d2h_bytes\":"
              << last.telemetry.final_hidden_d2h_bytes
              << ",\"layer_front_calls\":"
              << last.telemetry.layer_front_calls
              << ",\"layer_tail_calls\":"
              << last.telemetry.layer_tail_calls
              << (arguments->attribution
                      ? ",\"total_wall_nanoseconds\":" : "")
              << (arguments->attribution
                      ? std::to_string(attribution.total_wall_nanoseconds)
                      : "")
              << (arguments->attribution
                      ? ",\"front_wall_nanoseconds\":" : "")
              << (arguments->attribution
                      ? std::to_string(attribution.front_wall_nanoseconds)
                      : "")
              << (arguments->attribution
                      ? ",\"front_device_nanoseconds\":" : "")
              << (arguments->attribution
                      ? std::to_string(attribution.front_device_nanoseconds)
                      : "")
              << (arguments->attribution
                      ? ",\"route_wall_nanoseconds\":" : "")
              << (arguments->attribution
                      ? std::to_string(attribution.route_wall_nanoseconds)
                      : "")
              << (arguments->attribution
                      ? ",\"tail_wall_nanoseconds\":" : "")
              << (arguments->attribution
                      ? std::to_string(attribution.tail_wall_nanoseconds)
                      : "")
              << (arguments->attribution
                      ? ",\"tail_device_nanoseconds\":" : "")
              << (arguments->attribution
                      ? std::to_string(attribution.tail_device_nanoseconds)
                      : "")
              << (arguments->attribution
                      ? ",\"unattributed_wall_nanoseconds\":" : "")
              << (arguments->attribution
                      ? std::to_string(
                            attribution.unattributed_wall_nanoseconds)
                      : "")
              << ",\"state_seeds\":"
              << stats.official_kda_device_state_seeds
              << ",\"state_continuations\":"
              << stats.official_kda_device_state_continuations
              << ",\"state_publications\":"
              << stats.official_kda_device_state_publications
              << ",\"state_invalidations\":"
              << stats.official_kda_device_state_invalidations
              << ",\"prepared_seeds\":"
              << stats.official_moe_prepared_seeds
              << ",\"prepared_consumes\":"
              << stats.official_moe_prepared_consumes
              << ",\"prepared_discards\":"
              << stats.official_moe_prepared_discards
              << ",\"prepared_invalidations\":"
              << stats.official_moe_prepared_invalidations
              << ",\"resident_weight_bytes\":"
              << stats.resident_weight_bytes
              << ",\"peak_device_bytes\":"
              << backend.value()->memory_stats().peak_device_bytes
              << ",\"k3x_root_sha256\":\""
              << manifest.value.artifact_root_sha256
              << "\",\"route_expert_ids\":";
    write_u32_arrays(std::cout, last.steps);
    std::cout << ",\"route_contribution_sha256\":";
    write_string_array(std::cout, measured_contribution_sha256);
    std::cout << ",\"final_output_sha256\":";
    write_string_array(std::cout, measured_output_sha256);
    std::cout << ",\"final_state_sha256\":";
    write_string_array(std::cout, measured_state_sha256);
    std::cout << "}\n";
    return 0;
}
