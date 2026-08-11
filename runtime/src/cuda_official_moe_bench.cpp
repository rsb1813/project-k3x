// 고정된 공식 Kimi K3 MoE FFN fixture를 엄격히 검증하고 CPU/CUDA parity를 측정합니다.
#include "k3x/backend.hpp"
#include "k3x/format.hpp"
#include "k3x/official_moe.hpp"
#include "k3x/reader.hpp"
#include "k3x/storage_slice.hpp"

#include <algorithm>
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
#include <numeric>
#include <optional>
#include <set>
#include <span>
#include <sstream>
#include <string>
#include <string_view>
#include <variant>
#include <vector>

namespace {

constexpr std::size_t kHidden = 7'168;
constexpr std::size_t kLatent = 3'584;
constexpr std::size_t kExpertWidth = 3'072;
constexpr std::size_t kSharedWidth = 6'144;
constexpr std::size_t kTopK = 16;
constexpr std::uint64_t kExpertBytes = 17'547'264;
constexpr float kEpsilon = 1.0e-5F;
constexpr float kSituBeta = 4.0F;
constexpr float kSituLinear = 25.0F;
constexpr float kMaximumError = 2.0e-2F;

enum class WeightMode { transient, resident };
enum class CaseMode { a, b, alternating };

struct Arguments {
    std::filesystem::path model;
    std::filesystem::path manifest;
    std::string case_name{"a"};
    std::string weight_mode_name{"transient"};
    CaseMode case_mode{CaseMode::a};
    WeightMode weight_mode{WeightMode::transient};
    std::uint64_t warmup{};
    std::uint64_t iterations{1};
};

std::optional<std::uint64_t> parse_u64(std::string_view text) {
    std::uint64_t value{};
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    return !text.empty() && result.ec == std::errc{} &&
                   result.ptr == text.data() + text.size()
        ? std::optional(value)
        : std::nullopt;
}

std::optional<Arguments> parse_arguments(int argc, char** argv) {
    Arguments value;
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc) {
            std::cerr << "missing option value\n";
            return std::nullopt;
        }
        const std::string key = argv[index];
        const std::string argument = argv[index + 1];
        const auto number = parse_u64(argument);
        if (key == "--model") value.model = argument;
        else if (key == "--manifest") value.manifest = argument;
        else if (key == "--case") value.case_name = argument;
        else if (key == "--weight-mode") value.weight_mode_name = argument;
        else if (key == "--warmup" && number) value.warmup = *number;
        else if (key == "--iterations" && number) value.iterations = *number;
        else {
            std::cerr << "invalid option: " << key << '\n';
            return std::nullopt;
        }
    }
    if (value.case_name == "a") value.case_mode = CaseMode::a;
    else if (value.case_name == "b") value.case_mode = CaseMode::b;
    else if (value.case_name == "alternating") value.case_mode = CaseMode::alternating;
    else {
        std::cerr << "unknown case: " << value.case_name << '\n';
        return std::nullopt;
    }
    if (value.weight_mode_name == "transient") value.weight_mode = WeightMode::transient;
    else if (value.weight_mode_name == "resident") value.weight_mode = WeightMode::resident;
    else {
        std::cerr << "unknown weight mode: " << value.weight_mode_name << '\n';
        return std::nullopt;
    }
    if (!value.iterations) {
        std::cerr << "iterations must be positive\n";
        return std::nullopt;
    }
    if (value.model.empty()) {
        std::cerr << "model path is required\n";
        return std::nullopt;
    }
    if (value.manifest.empty()) {
        std::cerr << "manifest path is required\n";
        return std::nullopt;
    }
    return value;
}

struct Json {
    using Array = std::vector<Json>;
    using Object = std::map<std::string, Json>;
    std::variant<std::nullptr_t, bool, double, std::string, Array, Object> value;
};

class JsonParser {
public:
    explicit JsonParser(std::string_view text) : text_(text) {}
    std::optional<Json> parse() {
        auto result = parse_value();
        whitespace();
        return result && position_ == text_.size() ? result : std::nullopt;
    }
private:
    void whitespace() {
        while (position_ < text_.size() &&
               (text_[position_] == ' ' || text_[position_] == '\n' ||
                text_[position_] == '\r' || text_[position_] == '\t')) ++position_;
    }
    bool consume(char value) {
        whitespace();
        if (position_ == text_.size() || text_[position_] != value) return false;
        ++position_;
        return true;
    }
    std::optional<std::string> parse_string() {
        if (!consume('"')) return std::nullopt;
        std::string output;
        while (position_ < text_.size()) {
            const unsigned char current = text_[position_++];
            if (current == '"') return output;
            if (current < 0x20 || current >= 0x80) return std::nullopt;
            if (current != '\\') {
                output.push_back(static_cast<char>(current));
                continue;
            }
            if (position_ == text_.size()) return std::nullopt;
            const char escaped = text_[position_++];
            if (escaped == '"' || escaped == '\\' || escaped == '/') output.push_back(escaped);
            else if (escaped == 'b') output.push_back('\b');
            else if (escaped == 'f') output.push_back('\f');
            else if (escaped == 'n') output.push_back('\n');
            else if (escaped == 'r') output.push_back('\r');
            else if (escaped == 't') output.push_back('\t');
            else return std::nullopt;
        }
        return std::nullopt;
    }
    std::optional<Json> parse_number() {
        whitespace();
        const auto start = position_;
        if (position_ < text_.size() && text_[position_] == '-') ++position_;
        if (position_ == text_.size()) return std::nullopt;
        if (text_[position_] == '0') ++position_;
        else {
            if (text_[position_] < '1' || text_[position_] > '9') return std::nullopt;
            while (position_ < text_.size() && text_[position_] >= '0' && text_[position_] <= '9') ++position_;
        }
        if (position_ < text_.size() && text_[position_] == '.') {
            ++position_;
            const auto fraction = position_;
            while (position_ < text_.size() && text_[position_] >= '0' && text_[position_] <= '9') ++position_;
            if (position_ == fraction) return std::nullopt;
        }
        if (position_ < text_.size() && (text_[position_] == 'e' || text_[position_] == 'E')) {
            ++position_;
            if (position_ < text_.size() && (text_[position_] == '+' || text_[position_] == '-')) ++position_;
            const auto exponent = position_;
            while (position_ < text_.size() && text_[position_] >= '0' && text_[position_] <= '9') ++position_;
            if (position_ == exponent) return std::nullopt;
        }
        double number{};
        const auto parsed = std::from_chars(text_.data() + start, text_.data() + position_, number);
        return parsed.ec == std::errc{} && std::isfinite(number)
            ? std::optional(Json{number}) : std::nullopt;
    }
    std::optional<Json> parse_array() {
        if (!consume('[')) return std::nullopt;
        Json::Array output;
        whitespace();
        if (consume(']')) return Json{std::move(output)};
        while (true) {
            auto item = parse_value();
            if (!item) return std::nullopt;
            output.push_back(std::move(*item));
            if (consume(']')) return Json{std::move(output)};
            if (!consume(',')) return std::nullopt;
        }
    }
    std::optional<Json> parse_object() {
        if (!consume('{')) return std::nullopt;
        Json::Object output;
        whitespace();
        if (consume('}')) return Json{std::move(output)};
        while (true) {
            auto key = parse_string();
            if (!key || !consume(':')) return std::nullopt;
            auto item = parse_value();
            if (!item || !output.emplace(std::move(*key), std::move(*item)).second)
                return std::nullopt;
            if (consume('}')) return Json{std::move(output)};
            if (!consume(',')) return std::nullopt;
        }
    }
    std::optional<Json> parse_value() {
        whitespace();
        if (position_ == text_.size()) return std::nullopt;
        if (text_[position_] == '{') return parse_object();
        if (text_[position_] == '[') return parse_array();
        if (text_[position_] == '"') {
            auto string = parse_string();
            return string ? std::optional(Json{std::move(*string)}) : std::nullopt;
        }
        for (const auto& literal : {std::pair{"true", Json{true}},
                                    std::pair{"false", Json{false}},
                                    std::pair{"null", Json{nullptr}}}) {
            const std::string_view name = literal.first;
            if (text_.substr(position_, name.size()) == name) {
                position_ += name.size();
                return literal.second;
            }
        }
        return parse_number();
    }
    std::string_view text_;
    std::size_t position_{};
};

const Json* member(const Json& value, std::string_view name) {
    const auto* object = std::get_if<Json::Object>(&value.value);
    if (!object) return nullptr;
    const auto found = object->find(std::string(name));
    return found == object->end() ? nullptr : &found->second;
}
const std::string* string_value(const Json* value) {
    return value ? std::get_if<std::string>(&value->value) : nullptr;
}
const Json::Array* array_value(const Json* value) {
    return value ? std::get_if<Json::Array>(&value->value) : nullptr;
}
bool text_is(const Json& root, std::string_view key, std::string_view expected) {
    const auto* value = string_value(member(root, key));
    return value && *value == expected;
}
bool hex_string(const std::string& value, std::size_t size) {
    return value.size() == size && std::all_of(value.begin(), value.end(), [](char c) {
        return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
    });
}

struct ManifestRoute {
    std::string name;
    std::vector<std::uint32_t> ids;
    std::vector<float> contributions;
};
struct Manifest {
    std::string root;
    std::vector<ManifestRoute> routes;
    std::vector<std::uint32_t> selected;
};

std::optional<std::vector<std::uint32_t>> ids(const Json* value) {
    const auto* array = array_value(value);
    if (!array) return std::nullopt;
    std::vector<std::uint32_t> result;
    std::set<std::uint32_t> unique;
    for (const auto& item : *array) {
        const auto* number = std::get_if<double>(&item.value);
        if (!number || *number < 0 || *number > 895 || std::floor(*number) != *number ||
            !unique.insert(static_cast<std::uint32_t>(*number)).second) return std::nullopt;
        result.push_back(static_cast<std::uint32_t>(*number));
    }
    return result;
}

std::optional<Manifest> load_manifest(const std::filesystem::path& path,
                                     bool& syntax_valid) {
    std::error_code error;
    const auto size = std::filesystem::file_size(path, error);
    if (error || size > 16 * 1024 * 1024) return std::nullopt;
    std::ifstream stream(path, std::ios::binary);
    std::string text((std::istreambuf_iterator<char>(stream)), {});
    auto root = JsonParser(text).parse();
    if (!root) return std::nullopt;
    syntax_valid = true;
    if (!text_is(*root, "format", "k3x-official-moe-routes-v1") ||
        !text_is(*root, "converter_version", "k3x-converter-0.1.0") ||
        !text_is(*root, "repository", "moonshotai/Kimi-K3") ||
        !text_is(*root, "requested_revision", "main") ||
        !text_is(*root, "resolved_revision", "9f62e4e9fffbd0a83ddd60e1c209d828994b3569") ||
        !text_is(*root, "snapshot_sha256", "deaa6394b80afe12976ce8efbbf2463f6808c291d83b029e6b0cfb98de90a4e5") ||
        !text_is(*root, "index_sha256", "a1c5210650ce71d2d3ae9ec5a101ac4afd3cf4b10091be589853437eb967febd") ||
        !text_is(*root, "config_sha256", "9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213") ||
        !text_is(*root, "config_git_blob_id", "d7f26ead420b1d967f2759679dbebc65edfcff93") ||
        !text_is(*root, "shard_path", "model-00002-of-000096.safetensors") ||
        !text_is(*root, "shard_lfs_sha256", "26a3284e1d2cb567934ebef002e6a1813551d646739e8bcb1e9e3fe7f878e0f5") ||
        !text_is(*root, "provenance", "transport-pinned-ranges")) return std::nullopt;
    const auto* inputs = array_value(member(*root, "inputs"));
    if (!inputs || inputs->size() != 2 ||
        !text_is((*inputs)[0], "name", "a") ||
        !text_is((*inputs)[0], "prefix_sha256", "acc7746e19fcb6bb17d09ce08d387ca91d3a742c4f671046aaa0184a290d2cc3") ||
        !text_is((*inputs)[0], "block_sha256", "c7d98135ee7f46f4d82822d2e267d368dcdee51411575e578e63385a12e9bc3e") ||
        !text_is((*inputs)[1], "name", "b") ||
        !text_is((*inputs)[1], "prefix_sha256", "9b8f886591586999d0fb6a9661c938e24f2ade01cfdfbe352ea57961a642d566") ||
        !text_is((*inputs)[1], "block_sha256", "323b027923f323953dc12c6bc16618672e84d264891c6ed0a9aa3383b0045046")) return std::nullopt;
    const auto* artifact = member(*root, "artifact");
    if (!artifact) return std::nullopt;
    const auto* root_digest = string_value(member(*artifact, "k3x_root_sha256"));
    const auto* source_digest = string_value(member(*artifact, "source_sha256"));
    const auto* tensor_digests = member(*artifact, "tensor_sha256");
    if (!text_is(*artifact, "filename", "official-moe-l1.k3x") ||
        !root_digest || !hex_string(*root_digest, 64) || !source_digest ||
        !hex_string(*source_digest, 64) || !tensor_digests ||
        !std::holds_alternative<Json::Object>(tensor_digests->value)) return std::nullopt;
    const auto* routes = array_value(member(*root, "routes"));
    auto selected = ids(member(*root, "selected_experts"));
    if (!routes || routes->size() != 2 || !selected || selected->empty() || selected->size() > 32)
        return std::nullopt;
    Manifest result{*root_digest, {}, std::move(*selected)};
    std::vector<std::uint32_t> expected_union;
    std::set<std::uint32_t> union_set;
    for (std::size_t index = 0; index < routes->size(); ++index) {
        const auto* name = string_value(member((*routes)[index], "name"));
        auto route_ids = ids(member((*routes)[index], "expert_ids"));
        const auto* values = array_value(member((*routes)[index], "contributions"));
        if (!name || *name != (index ? "b" : "a") || !route_ids ||
            route_ids->size() != kTopK || !values || values->size() != kTopK) return std::nullopt;
        std::vector<float> contributions;
        double total = 0;
        for (const auto& item : *values) {
            const auto* number = std::get_if<double>(&item.value);
            if (!number || !std::isfinite(*number) || *number <= 0) return std::nullopt;
            contributions.push_back(static_cast<float>(*number));
            total += *number;
        }
        if (std::abs(total - 1.0) > 1.0e-5) return std::nullopt;
        for (const auto id : *route_ids) if (union_set.insert(id).second) expected_union.push_back(id);
        result.routes.push_back({*name, std::move(*route_ids), std::move(contributions)});
    }
    if (result.selected != expected_union) return std::nullopt;
    std::set<std::string> expected_tensors{
        "model.layers.1.mlp_res_norm.weight",
        "model.layers.1.mlp_res_proj.weight",
        "model.layers.1.post_attention_layernorm.weight",
        "model.layers.1.block_sparse_moe.gate.weight",
        "model.layers.1.block_sparse_moe.gate.e_score_correction_bias",
        "model.layers.1.block_sparse_moe.routed_expert_down_proj.weight",
        "model.layers.1.block_sparse_moe.routed_expert_norm.weight",
        "model.layers.1.block_sparse_moe.routed_expert_up_proj.weight",
        "model.layers.1.block_sparse_moe.shared_experts.gate_proj.weight",
        "model.layers.1.block_sparse_moe.shared_experts.up_proj.weight",
        "model.layers.1.block_sparse_moe.shared_experts.down_proj.weight"};
    for (const auto id : result.selected) {
        const auto base = "model.layers.1.feed_forward.experts." +
            std::to_string(id) + ".";
        for (const auto role : {"gate", "up", "down"}) {
            expected_tensors.insert(base + role + ".weight_packed");
            expected_tensors.insert(base + role + ".weight_scale");
        }
    }
    const auto& observed_tensors = std::get<Json::Object>(tensor_digests->value);
    if (observed_tensors.size() != expected_tensors.size()) return std::nullopt;
    for (const auto& name : expected_tensors) {
        const auto found = observed_tensors.find(name);
        const auto* digest = found == observed_tensors.end()
            ? nullptr : std::get_if<std::string>(&found->second.value);
        if (!digest || !hex_string(*digest, 64)) return std::nullopt;
    }
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

void write_error(k3x::ErrorCode code, std::string_view message) {
    std::cerr << k3x::error_code_name(code) << ": " << message << '\n';
}

const k3x::TensorRecord* find_record(const k3x::Reader& reader, const std::string& name) {
    const auto id = k3x::fnv1a64(name.c_str());
    const auto found = std::find_if(reader.tensors().begin(), reader.tensors().end(),
        [id](const auto& value) { return value.tensor_id == id; });
    return found == reader.tensors().end() ? nullptr : &*found;
}

bool shape(const k3x::TensorRecord& value, std::uint16_t dtype,
           std::span<const std::uint64_t> dimensions) {
    if (value.dtype != dtype || value.quantization != 0 || value.rank != dimensions.size() ||
        value.layer_id != 1 || value.expert_id != -1) return false;
    for (std::size_t index = 0; index < 4; ++index)
        if (value.dimensions[index] != (index < dimensions.size() ? dimensions[index] : 0)) return false;
    return true;
}

struct OwnedBf16 {
    std::uint64_t id{};
    std::vector<std::uint16_t> values;
    std::size_t rows{};
    std::size_t cols{};
    k3x::Bf16WeightView matrix() const { return {values, rows, cols, id}; }
    k3x::Bf16VectorView vector() const { return {values, id}; }
};

std::optional<OwnedBf16> load_bf16(k3x::Reader& reader, const std::string& name,
                                   std::size_t rows, std::size_t cols,
                                   bool vector = false) {
    const auto* record = find_record(reader, name);
    const std::array<std::uint64_t, 2> matrix_dims{rows, cols};
    const std::array<std::uint64_t, 1> vector_dims{cols};
    const auto dimensions = vector ? std::span<const std::uint64_t>(vector_dims)
                                   : std::span<const std::uint64_t>(matrix_dims);
    if (!record || !shape(*record, 3, dimensions) ||
        record->data_length != rows * cols * 2 || record->auxiliary_length) return std::nullopt;
    auto bytes = reader.read_tensor(record->tensor_id);
    if (!bytes || bytes.value().size() != record->data_length) return std::nullopt;
    OwnedBf16 result{record->tensor_id, std::vector<std::uint16_t>(rows * cols), rows, cols};
    std::memcpy(result.values.data(), bytes.value().data(), bytes.value().size());
    return result;
}

struct OwnedExpert {
    std::uint32_t id{};
    std::array<std::vector<std::byte>, 6> extents;
    k3x::Mxfp4MlpView view() const {
        const auto base = "model.layers.1.feed_forward.experts." + std::to_string(id) + ".";
        return {
            {k3x::fnv1a64((base + "gate").c_str()), extents[0], extents[1], kExpertWidth, kLatent, 32},
            {k3x::fnv1a64((base + "up").c_str()), extents[2], extents[3], kExpertWidth, kLatent, 32},
            {k3x::fnv1a64((base + "down").c_str()), extents[4], extents[5], kLatent, kExpertWidth, 32},
        };
    }
};

struct Loaded {
    OwnedBf16 residual_norm, residual_proj, post_norm, router;
    std::vector<float> correction;
    OwnedBf16 routed_down, routed_norm, routed_up, shared_gate, shared_up, shared_down;
    std::vector<OwnedExpert> experts;
};

std::optional<Loaded> load_weights(k3x::Reader& reader, const Manifest& manifest) {
    const std::string base = "model.layers.1.";
    auto rn = load_bf16(reader, base + "mlp_res_norm.weight", 1, kHidden, true);
    auto rp = load_bf16(reader, base + "mlp_res_proj.weight", 1, kHidden);
    auto pn = load_bf16(reader, base + "post_attention_layernorm.weight", 1, kHidden, true);
    auto router = load_bf16(reader, base + "block_sparse_moe.gate.weight", 896, kHidden);
    auto rd = load_bf16(reader, base + "block_sparse_moe.routed_expert_down_proj.weight", kLatent, kHidden);
    auto rnorm = load_bf16(reader, base + "block_sparse_moe.routed_expert_norm.weight", 1, kLatent, true);
    auto ru = load_bf16(reader, base + "block_sparse_moe.routed_expert_up_proj.weight", kHidden, kLatent);
    auto sg = load_bf16(reader, base + "block_sparse_moe.shared_experts.gate_proj.weight", kSharedWidth, kHidden);
    auto su = load_bf16(reader, base + "block_sparse_moe.shared_experts.up_proj.weight", kSharedWidth, kHidden);
    auto sd = load_bf16(reader, base + "block_sparse_moe.shared_experts.down_proj.weight", kHidden, kSharedWidth);
    const auto correction_name = base + "block_sparse_moe.gate.e_score_correction_bias";
    const auto* correction_record = find_record(reader, correction_name);
    const std::array<std::uint64_t, 1> correction_dims{896};
    if (!rn || !rp || !pn || !router || !rd || !rnorm || !ru || !sg || !su || !sd ||
        !correction_record || !shape(*correction_record, 1, correction_dims) ||
        correction_record->data_length != 896 * sizeof(float)) return std::nullopt;
    auto correction_bytes = reader.read_tensor(correction_record->tensor_id);
    if (!correction_bytes) return std::nullopt;
    Loaded result{std::move(*rn), std::move(*rp), std::move(*pn), std::move(*router),
                  std::vector<float>(896), std::move(*rd), std::move(*rnorm), std::move(*ru),
                  std::move(*sg), std::move(*su), std::move(*sd), {}};
    std::memcpy(result.correction.data(), correction_bytes.value().data(), correction_bytes.value().size());
    result.experts.reserve(manifest.selected.size());
    for (const auto id : manifest.selected) {
        auto loaded = k3x::load_storage_expert(reader, 1, id);
        if (!loaded) return std::nullopt;
        result.experts.push_back({id, std::move(loaded.value().extents)});
    }
    std::vector<std::string> expected_order{
        base + "mlp_res_norm.weight",
        base + "mlp_res_proj.weight",
        base + "post_attention_layernorm.weight",
        base + "block_sparse_moe.gate.weight",
        base + "block_sparse_moe.gate.e_score_correction_bias",
        base + "block_sparse_moe.routed_expert_down_proj.weight"};
    for (const auto id : manifest.selected) {
        const auto expert = base + "feed_forward.experts." + std::to_string(id) + ".";
        expected_order.push_back(expert + "gate");
        expected_order.push_back(expert + "up");
        expected_order.push_back(expert + "down");
    }
    expected_order.insert(expected_order.end(), {
        base + "block_sparse_moe.routed_expert_norm.weight",
        base + "block_sparse_moe.routed_expert_up_proj.weight",
        base + "block_sparse_moe.shared_experts.gate_proj.weight",
        base + "block_sparse_moe.shared_experts.up_proj.weight",
        base + "block_sparse_moe.shared_experts.down_proj.weight"});
    if (reader.tensors().size() != expected_order.size()) return std::nullopt;
    for (std::size_t index = 0; index < expected_order.size(); ++index)
        if (reader.tensors()[index].tensor_id != k3x::fnv1a64(expected_order[index].c_str()))
            return std::nullopt;
    return result;
}

k3x::OfficialMoeWeights cpu_views(const Loaded& value,
                                  const std::vector<k3x::OfficialExpertView>& experts) {
    return {value.residual_norm.vector(), value.residual_proj.matrix(), value.post_norm.vector(),
            value.router.matrix(), value.correction, value.routed_down.matrix(),
            value.routed_norm.vector(), value.routed_up.matrix(),
            {value.shared_gate.matrix(), value.shared_up.matrix(), value.shared_down.matrix()}, experts};
}
k3x::OfficialMoeFfnView cuda_views(const Loaded& value) {
    return {value.routed_down.matrix(), value.routed_norm.vector(), value.routed_up.matrix(),
            {value.shared_gate.matrix(), value.shared_up.matrix(), value.shared_down.matrix()}};
}

std::uint64_t elapsed(std::chrono::steady_clock::time_point start) {
    return std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now() - start).count();
}
std::uint64_t percentile(std::vector<std::uint64_t> values, std::size_t percent) {
    std::sort(values.begin(), values.end());
    return values[(values.size() - 1) * percent / 100];
}
std::uint64_t median(std::vector<std::uint64_t> values) {
    std::sort(values.begin(), values.end());
    const auto middle = values.size() / 2;
    return values.size() % 2 ? values[middle]
        : values[middle - 1] + (values[middle] - values[middle - 1]) / 2;
}

bool compare(std::span<const float> actual, std::span<const float> expected, float& maximum) {
    if (actual.size() != expected.size()) return false;
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (!std::isfinite(actual[index]) || !std::isfinite(expected[index])) return false;
        maximum = std::max(maximum, std::abs(actual[index] - expected[index]));
    }
    return maximum <= kMaximumError;
}

void write_u32_array(std::ostream& output, std::span<const std::uint32_t> values) {
    output << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index) output << ',';
        output << values[index];
    }
    output << ']';
}

void write_float_array(std::ostream& output, std::span<const float> values) {
    output << '[';
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index) output << ',';
        output << std::setprecision(12) << values[index];
    }
    output << ']';
}

k3x::BackendOptions backend_options(WeightMode mode, std::uint64_t capacity) {
    k3x::BackendOptions value;
    value.kind = k3x::BackendKind::cuda_custom;
    value.cuda_allocation = k3x::CudaAllocationMode::reused;
    value.cuda_weights = mode == WeightMode::resident ? k3x::CudaWeightMode::resident
                                                       : k3x::CudaWeightMode::transient;
    value.cuda_batching = k3x::CudaBatchingMode::resident_grid;
    value.cuda_boundary = k3x::CudaBoundaryMode::moe_layer;
    value.cuda_transfer = k3x::CudaTransferMode::synchronous;
    value.cuda_resident_bytes = mode == WeightMode::resident ? capacity : 0;
    return value;
}

}  // namespace

int main(int argc, char** argv) {
    const auto arguments = parse_arguments(argc, argv);
    if (!arguments) return 2;
    bool syntax_valid = false;
    const auto manifest = load_manifest(arguments->manifest, syntax_valid);
    if (!manifest) {
        write_error(k3x::ErrorCode::invalid_extent,
                    syntax_valid ? "official MoE manifest identity mismatch"
                                 : "invalid official MoE manifest");
        return 4;
    }
    auto reader = k3x::Reader::open(arguments->model, k3x::VerifyMode::checksums);
    if (!reader) {
        write_error(reader.error(), reader.message());
        return 4;
    }
    if (reader.value().superblock().optional_features !=
            (k3x::optional_storage_fixture | k3x::optional_official_moe_fixture)) {
        write_error(k3x::ErrorCode::invalid_extent, "artifact is not official MoE fixture");
        return 4;
    }
    if (digest_hex(reader.value().superblock().root_sha256) != manifest->root) {
        write_error(k3x::ErrorCode::invalid_extent, "official MoE artifact root mismatch");
        return 4;
    }
    auto loaded = load_weights(reader.value(), *manifest);
    if (!loaded) {
        write_error(k3x::ErrorCode::invalid_extent, "official MoE tensor identity mismatch");
        return 4;
    }
    std::vector<k3x::Mxfp4MlpView> expert_views;
    std::vector<k3x::OfficialExpertView> cpu_experts;
    expert_views.reserve(loaded->experts.size());
    cpu_experts.reserve(loaded->experts.size());
    for (const auto& expert : loaded->experts) {
        expert_views.push_back(expert.view());
        cpu_experts.push_back({expert.id, expert_views.back()});
    }
    const auto weights = cpu_views(*loaded, cpu_experts);
    const auto inputs = k3x::official_moe_inputs();
    struct Prepared {
        k3x::OfficialRoute route;
        k3x::OfficialMoeResult oracle;
    };
    std::array<Prepared, 2> prepared;
    std::uint64_t attention_residual_nanoseconds{};
    std::uint64_t router_nanoseconds{};
    std::uint64_t cpu_oracle_nanoseconds{};
    for (std::size_t index = 0; index < 2; ++index) {
        auto start = std::chrono::steady_clock::now();
        auto hidden = k3x::prepare_official_moe_input(inputs[index], weights, kEpsilon);
        attention_residual_nanoseconds += elapsed(start);
        if (!hidden) return 4;
        start = std::chrono::steady_clock::now();
        auto route = k3x::route_official_moe(hidden.value(), weights.router,
                                             weights.correction, kTopK);
        router_nanoseconds += elapsed(start);
        if (!route || route.value().expert_ids != manifest->routes[index].ids) {
            write_error(k3x::ErrorCode::invalid_state, "official MoE route mismatch");
            return 4;
        }
        for (std::size_t slot = 0; slot < kTopK; ++slot) {
            if (std::abs(route.value().contributions[slot] -
                         manifest->routes[index].contributions[slot]) > 1.0e-6F) {
                write_error(k3x::ErrorCode::invalid_state, "official MoE contribution mismatch");
                return 4;
            }
        }
        start = std::chrono::steady_clock::now();
        auto oracle = k3x::official_moe_cpu(inputs[index], weights, route.value(),
                                            kEpsilon, kSituBeta, kSituLinear);
        cpu_oracle_nanoseconds += elapsed(start);
        if (!oracle) return 4;
        prepared[index] = {std::move(route.value()), std::move(oracle.value())};
    }
    const auto common_bytes = loaded->routed_down.values.size() * 2ULL +
        loaded->routed_norm.values.size() * 2ULL + loaded->routed_up.values.size() * 2ULL +
        loaded->shared_gate.values.size() * 2ULL + loaded->shared_up.values.size() * 2ULL +
        loaded->shared_down.values.size() * 2ULL;
    const auto resident_capacity = common_bytes + manifest->selected.size() * kExpertBytes;
    k3x::Profiler profiler;
    auto backend = k3x::make_cuda_backend(
        backend_options(arguments->weight_mode, resident_capacity), &profiler);
    if (!backend) return 4;
    const auto execute_one = [&](std::size_t case_index, float& maximum) {
        std::vector<k3x::Mxfp4MlpView> selected;
        for (const auto id : prepared[case_index].route.expert_ids) {
            const auto found = std::find(manifest->selected.begin(), manifest->selected.end(), id);
            selected.push_back(expert_views[found - manifest->selected.begin()]);
        }
        auto result = backend.value()->official_mxfp4_moe_ffn(
            prepared[case_index].oracle.hidden, inputs[case_index].prefix_sum,
            cuda_views(*loaded), selected, prepared[case_index].route.expert_ids,
            prepared[case_index].route.contributions, kEpsilon, kSituBeta,
            kSituLinear, 1, k3x::ProfilePhase::decode);
        return result && compare(result.value().output,
                                 prepared[case_index].oracle.output, maximum);
    };
    const std::array<std::size_t, 2> sequence{0, 1};
    const auto sequence_size = arguments->case_mode == CaseMode::alternating ? 2U : 1U;
    const auto first_case = arguments->case_mode == CaseMode::b ? 1U : 0U;
    const auto execute_sequence = [&](float& maximum) {
        if (sequence_size == 1) return execute_one(first_case, maximum);
        return execute_one(sequence[0], maximum) && execute_one(sequence[1], maximum);
    };
    float maximum_error{};
    const auto cold_before = backend.value()->runtime_stats();
    const auto cold_profile_before = profiler.summary();
    auto start = std::chrono::steady_clock::now();
    if (!execute_sequence(maximum_error)) return 4;
    const auto cold_latency = elapsed(start);
    const auto cold_after = backend.value()->runtime_stats();
    const auto cold_profile_after = profiler.summary();
    for (std::uint64_t index = 0; index < arguments->warmup; ++index)
        if (!execute_sequence(maximum_error)) return 4;
    const auto measured_before = backend.value()->runtime_stats();
    const auto profile_before = profiler.summary();
    std::vector<std::uint64_t> samples;
    for (std::uint64_t index = 0; index < arguments->iterations; ++index) {
        start = std::chrono::steady_clock::now();
        if (!execute_sequence(maximum_error)) return 4;
        samples.push_back(elapsed(start));
    }
    const auto measured_after = backend.value()->runtime_stats();
    const auto profile_after = profiler.summary();
    const auto memory = backend.value()->memory_stats();
    const auto calls = arguments->iterations * sequence_size;
    const auto measured_weight = measured_after.weight_h2d_bytes - measured_before.weight_h2d_bytes;
    std::uint64_t sequence_weight{};
    if (sequence_size == 1) sequence_weight = common_bytes + kTopK * kExpertBytes;
    else sequence_weight = 2 * common_bytes + 2 * kTopK * kExpertBytes;
    const auto expected_d2h = calls * kHidden * sizeof(float);
    const auto measured_bf16 = arguments->weight_mode == WeightMode::transient
        ? calls * common_bytes : 0;
    const auto measured_mxfp4 = arguments->weight_mode == WeightMode::transient
        ? calls * kTopK * kExpertBytes : 0;
    const auto cold_bf16 = (arguments->weight_mode == WeightMode::transient
        ? sequence_size : 1U) * common_bytes;
    const auto cold_mxfp4 = (arguments->weight_mode == WeightMode::resident &&
                             sequence_size == 2
        ? manifest->selected.size() : sequence_size * kTopK) * kExpertBytes;
    const auto cold_weight = cold_after.weight_h2d_bytes - cold_before.weight_h2d_bytes;
    if (measured_after.weight_cache_bypasses != measured_before.weight_cache_bypasses ||
        measured_after.device_to_host_bytes - measured_before.device_to_host_bytes != expected_d2h ||
        cold_weight != cold_bf16 + cold_mxfp4 ||
        (arguments->weight_mode == WeightMode::transient
             ? measured_weight != arguments->iterations * sequence_weight ||
                   measured_weight != measured_bf16 + measured_mxfp4
             : measured_weight != 0)) {
        write_error(k3x::ErrorCode::invalid_state, "official MoE CUDA traffic invariant failure");
        return 4;
    }
    std::cout << std::setprecision(12)
        << "{\"artifact_kind\":\"official_kimi_k3_moe_ffn\""
        << ",\"repository\":\"moonshotai/Kimi-K3\""
        << ",\"resolved_revision\":\"9f62e4e9fffbd0a83ddd60e1c209d828994b3569\""
        << ",\"case\":\"" << arguments->case_name << "\""
        << ",\"weight_mode\":\"" << arguments->weight_mode_name << "\""
        << ",\"token_semantics\":false,\"routing_semantics\":true"
        << ",\"full_moe_ffn\":true,\"full_transformer_layer\":false,\"quality_measured\":false"
        << ",\"k3x_root_sha256\":\"" << manifest->root << "\""
        << ",\"warmup\":" << arguments->warmup << ",\"iterations\":" << arguments->iterations
        << ",\"input_elements\":" << kHidden << ",\"output_elements\":" << kHidden
        << ",\"selected_union\":";
    write_u32_array(std::cout, manifest->selected);
    std::cout << ",\"route_a\":"; write_u32_array(std::cout, prepared[0].route.expert_ids);
    std::cout << ",\"route_b\":"; write_u32_array(std::cout, prepared[1].route.expert_ids);
    std::cout << ",\"route_a_contributions\":";
    write_float_array(std::cout, prepared[0].route.contributions);
    std::cout << ",\"route_b_contributions\":";
    write_float_array(std::cout, prepared[1].route.contributions);
    std::cout << ",\"source_bytes\":" << 379'900'416ULL + manifest->selected.size() * kExpertBytes
        << ",\"k3x_bytes\":" << reader.value().superblock().file_length
        << ",\"cpu_oracle_nanoseconds\":" << cpu_oracle_nanoseconds
        << ",\"attention_residual_nanoseconds\":" << attention_residual_nanoseconds
        << ",\"router_nanoseconds\":" << router_nanoseconds
        << ",\"cold_latency_nanoseconds\":" << cold_latency
        << ",\"cold_kernel_nanoseconds\":" << cold_profile_after.device_nanoseconds - cold_profile_before.device_nanoseconds
        << ",\"cold_weight_h2d_bytes\":" << cold_weight
        << ",\"cold_bf16_weight_h2d_bytes\":" << cold_bf16
        << ",\"cold_mxfp4_weight_h2d_bytes\":" << cold_mxfp4
        << ",\"latency_nanoseconds_p05\":" << percentile(samples, 5)
        << ",\"latency_nanoseconds_median\":" << median(samples)
        << ",\"latency_nanoseconds_p95\":" << percentile(samples, 95)
        << ",\"kernel_nanoseconds\":" << profile_after.device_nanoseconds - profile_before.device_nanoseconds
        << ",\"orchestration_nanoseconds\":" <<
            (std::accumulate(samples.begin(), samples.end(), std::uint64_t{}) -
             std::min(std::accumulate(samples.begin(), samples.end(), std::uint64_t{}),
                      profile_after.device_nanoseconds - profile_before.device_nanoseconds))
        << ",\"weight_h2d_bytes\":" << measured_weight
        << ",\"bf16_weight_h2d_bytes\":" << measured_bf16
        << ",\"mxfp4_weight_h2d_bytes\":" << measured_mxfp4
        << ",\"activation_h2d_bytes\":" << measured_after.activation_h2d_bytes - measured_before.activation_h2d_bytes
        << ",\"device_to_host_bytes\":" << measured_after.device_to_host_bytes - measured_before.device_to_host_bytes
        << ",\"resident_weight_bytes\":" << measured_after.resident_weight_bytes
        << ",\"peak_resident_weight_bytes\":" << measured_after.peak_resident_weight_bytes
        << ",\"weight_cache_hits\":" << measured_after.weight_cache_hits - measured_before.weight_cache_hits
        << ",\"weight_cache_misses\":" << measured_after.weight_cache_misses - measured_before.weight_cache_misses
        << ",\"weight_cache_bypasses\":" << measured_after.weight_cache_bypasses - measured_before.weight_cache_bypasses
        << ",\"device_allocation_count\":" << measured_after.device_allocation_count - measured_before.device_allocation_count
        << ",\"stream_synchronization_count\":" << measured_after.stream_synchronization_count - measured_before.stream_synchronization_count
        << ",\"peak_vram_bytes\":" << memory.peak_device_bytes
        << ",\"maximum_absolute_error\":" << maximum_error << ",\"all_finite\":true}\n";
    return 0;
}
