// 고정된 공식 Kimi K3 expert를 CPU oracle과 RTX CUDA에서 비교 측정합니다.
#include "k3x/backend.hpp"
#include "k3x/official_expert.hpp"
#include "k3x/reader.hpp"
#include "k3x/storage_slice.hpp"

#include <algorithm>
#include <charconv>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <limits>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

namespace {

constexpr std::size_t kInputElements = 3584;
constexpr std::size_t kOutputElements = 3584;
constexpr float kSituBeta = 4.0F;
constexpr float kSituLinear = 25.0F;
constexpr float kMaximumError = 1.0e-6F;

enum class WeightMode { transient, resident };

struct Arguments {
    std::filesystem::path model;
    std::string weight_mode_name{"transient"};
    WeightMode weight_mode{WeightMode::transient};
    std::uint64_t warmup{};
    std::uint64_t iterations{1};
};

std::optional<std::uint64_t> parse_u64(std::string_view text) {
    std::uint64_t value{};
    const auto result =
        std::from_chars(text.data(), text.data() + text.size(), value);
    if (text.empty() || result.ec != std::errc{} ||
        result.ptr != text.data() + text.size()) {
        return std::nullopt;
    }
    return value;
}

std::optional<Arguments> parse_arguments(int argc, char** argv) {
    Arguments arguments;
    for (int index = 1; index < argc; index += 2) {
        if (index + 1 >= argc) {
            std::cerr << "missing option value\n";
            return std::nullopt;
        }
        const std::string key = argv[index];
        const std::string value = argv[index + 1];
        const auto number = parse_u64(value);
        if (key == "--model") {
            arguments.model = value;
        } else if (key == "--weight-mode") {
            arguments.weight_mode_name = value;
        } else if (key == "--warmup" && number) {
            arguments.warmup = *number;
        } else if (key == "--iterations" && number) {
            arguments.iterations = *number;
        } else {
            std::cerr << "invalid option: " << key << '\n';
            return std::nullopt;
        }
    }
    if (arguments.weight_mode_name == "transient") {
        arguments.weight_mode = WeightMode::transient;
    } else if (arguments.weight_mode_name == "resident") {
        arguments.weight_mode = WeightMode::resident;
    } else {
        std::cerr << "unknown weight mode: " << arguments.weight_mode_name
                  << '\n';
        return std::nullopt;
    }
    if (arguments.iterations == 0) {
        std::cerr << "iterations must be positive\n";
        return std::nullopt;
    }
    if (arguments.model.empty()) {
        std::cerr << "model path is required\n";
        return std::nullopt;
    }
    const auto& identity = k3x::official_kimi_k3_expert();
    if (arguments.iterations >
        std::numeric_limits<std::uint64_t>::max() / identity.payload_bytes) {
        std::cerr << "iterations out of range\n";
        return std::nullopt;
    }
    return arguments;
}

void write_error(k3x::ErrorCode code, const std::string& message) {
    std::cerr << k3x::error_code_name(code);
    if (!message.empty()) std::cerr << ": " << message;
    std::cerr << '\n';
}

std::string digest_hex(std::span<const std::byte> digest) {
    static constexpr char digits[] = "0123456789abcdef";
    std::string result(digest.size() * 2, '0');
    for (std::size_t index = 0; index < digest.size(); ++index) {
        const auto value = std::to_integer<unsigned int>(digest[index]);
        result[index * 2] = digits[value >> 4];
        result[index * 2 + 1] = digits[value & 0x0f];
    }
    return result;
}

std::uint64_t elapsed_nanoseconds(
    std::chrono::steady_clock::time_point start) {
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::steady_clock::now() - start)
            .count());
}

std::uint64_t median(std::vector<std::uint64_t> values) {
    std::sort(values.begin(), values.end());
    const auto middle = values.size() / 2;
    return values.size() % 2 != 0
        ? values[middle]
        : values[middle - 1] +
              (values[middle] - values[middle - 1]) / 2;
}

std::uint64_t percentile(
    std::vector<std::uint64_t> values, std::size_t numerator) {
    std::sort(values.begin(), values.end());
    return values[(values.size() - 1) * numerator / 100];
}

bool compare_output(
    std::span<const float> actual,
    std::span<const float> expected,
    float& maximum_error) {
    if (actual.size() != kOutputElements || expected.size() != kOutputElements) {
        return false;
    }
    for (std::size_t index = 0; index < actual.size(); ++index) {
        if (!std::isfinite(actual[index]) || !std::isfinite(expected[index])) {
            return false;
        }
        maximum_error = std::max(
            maximum_error, std::abs(actual[index] - expected[index]));
    }
    return maximum_error <= kMaximumError;
}

k3x::BackendOptions backend_options(
    WeightMode mode, std::uint64_t payload_bytes) {
    k3x::BackendOptions options;
    options.kind = k3x::BackendKind::cuda_custom;
    options.cuda_allocation = k3x::CudaAllocationMode::reused;
    options.cuda_weights = mode == WeightMode::resident
        ? k3x::CudaWeightMode::resident
        : k3x::CudaWeightMode::transient;
    options.cuda_boundary = k3x::CudaBoundaryMode::ffn_block;
    options.cuda_transfer = k3x::CudaTransferMode::synchronous;
    options.cuda_moe_fusion = k3x::CudaMoeFusionMode::none;
    options.cuda_resident_bytes =
        mode == WeightMode::resident ? payload_bytes : 0;
    return options;
}

}  // namespace

int main(int argc, char** argv) {
    const auto arguments = parse_arguments(argc, argv);
    if (!arguments) return 2;

    auto reader = k3x::Reader::open(
        arguments->model, k3x::VerifyMode::checksums);
    if (!reader) {
        write_error(reader.error(), reader.message());
        return 4;
    }
    auto loaded = k3x::load_storage_expert(reader.value(), 1, 0);
    if (!loaded) {
        write_error(loaded.error(), loaded.message());
        return 4;
    }
    const auto& pinned = k3x::official_kimi_k3_expert();
    const k3x::OfficialExpertObservation observation{
        reader.value().superblock().root_sha256,
        loaded.value().ordered_sha256,
        reader.value().superblock().optional_features,
        1,
        0,
        loaded.value().logical_bytes,
        {{{3072, 3584}, {3072, 3584}, {3584, 3072}}},
    };
    const auto verified = k3x::verify_official_kimi_k3_expert(observation);
    if (!verified) {
        write_error(verified.error(), verified.message());
        return 4;
    }

    const auto& extents = loaded.value().extents;
    const k3x::Mxfp4MlpView expert{
        {101, extents[0], extents[1], 3072, 3584, 32},
        {102, extents[2], extents[3], 3072, 3584, 32},
        {103, extents[4], extents[5], 3584, 3072, 32},
    };
    const std::vector<k3x::Mxfp4MlpView> experts{expert};
    std::vector<float> input(kInputElements);
    for (std::size_t index = 0; index < input.size(); ++index) {
        input[index] =
            static_cast<float>(static_cast<int>(index % 17) - 8) * 0.01F;
    }

    auto cpu = k3x::make_cpu_backend();
    const auto cpu_start = std::chrono::steady_clock::now();
    auto oracle_result = cpu->mxfp4_situ_mlp_group(
        input, experts, kSituBeta, kSituLinear, 1,
        k3x::ProfilePhase::decode);
    const auto cpu_oracle_nanoseconds = elapsed_nanoseconds(cpu_start);
    if (!oracle_result || oracle_result.value().size() != 1) {
        write_error(
            oracle_result ? k3x::ErrorCode::invalid_state : oracle_result.error(),
            oracle_result ? "invalid CPU oracle output" : oracle_result.message());
        return 4;
    }
    const auto& oracle = oracle_result.value().front();
    float maximum_error{};
    if (!compare_output(oracle, oracle, maximum_error)) {
        write_error(k3x::ErrorCode::invalid_state, "invalid CPU oracle output");
        return 4;
    }

    k3x::Profiler profiler;
    auto backend = k3x::make_cuda_backend(
        backend_options(arguments->weight_mode, pinned.payload_bytes),
        &profiler);
    if (!backend) {
        write_error(backend.error(), backend.message());
        return 4;
    }
    const auto execute = [&]() {
        return backend.value()->mxfp4_situ_mlp_group(
            input, experts, kSituBeta, kSituLinear, 1,
            k3x::ProfilePhase::decode);
    };

    const auto cold_runtime_before = backend.value()->runtime_stats();
    const auto cold_profile_before = profiler.summary();
    const auto cold_start = std::chrono::steady_clock::now();
    auto cold = execute();
    const auto cold_latency_nanoseconds = elapsed_nanoseconds(cold_start);
    const auto cold_profile_after = profiler.summary();
    const auto cold_runtime_after = backend.value()->runtime_stats();
    if (!cold || cold.value().size() != 1 ||
        !compare_output(cold.value().front(), oracle, maximum_error)) {
        write_error(
            cold ? k3x::ErrorCode::invalid_state : cold.error(),
            cold ? "official expert CUDA parity failure" : cold.message());
        return 4;
    }

    for (std::uint64_t index = 0; index < arguments->warmup; ++index) {
        auto result = execute();
        if (!result || result.value().size() != 1 ||
            !compare_output(result.value().front(), oracle, maximum_error)) {
            write_error(
                result ? k3x::ErrorCode::invalid_state : result.error(),
                result ? "official expert CUDA parity failure" : result.message());
            return 4;
        }
    }

    const auto runtime_before = backend.value()->runtime_stats();
    const auto profile_before = profiler.summary();
    std::vector<std::uint64_t> samples;
    samples.reserve(static_cast<std::size_t>(arguments->iterations));
    for (std::uint64_t index = 0; index < arguments->iterations; ++index) {
        const auto start = std::chrono::steady_clock::now();
        auto result = execute();
        samples.push_back(elapsed_nanoseconds(start));
        if (!result || result.value().size() != 1 ||
            !compare_output(result.value().front(), oracle, maximum_error)) {
            write_error(
                result ? k3x::ErrorCode::invalid_state : result.error(),
                result ? "official expert CUDA parity failure" : result.message());
            return 4;
        }
    }
    const auto profile_after = profiler.summary();
    const auto runtime_after = backend.value()->runtime_stats();
    const auto memory = backend.value()->memory_stats();

    const auto cold_weight_h2d = cold_runtime_after.weight_h2d_bytes -
        cold_runtime_before.weight_h2d_bytes;
    const auto measured_weight_h2d = runtime_after.weight_h2d_bytes -
        runtime_before.weight_h2d_bytes;
    const auto expected_measured_h2d =
        arguments->iterations * pinned.payload_bytes;
    const bool traffic_valid =
        cold_weight_h2d == pinned.payload_bytes &&
        runtime_after.weight_cache_bypasses ==
            runtime_before.weight_cache_bypasses &&
        (arguments->weight_mode == WeightMode::transient
             ? measured_weight_h2d == expected_measured_h2d &&
                   runtime_after.resident_weight_bytes == 0
             : measured_weight_h2d == 0 &&
                   runtime_after.resident_weight_bytes == pinned.payload_bytes &&
                   runtime_after.peak_resident_weight_bytes == pinned.payload_bytes);
    if (!traffic_valid) {
        write_error(
            k3x::ErrorCode::invalid_state,
            "official expert CUDA traffic invariant failure");
        return 4;
    }

    std::cout << std::setprecision(12)
              << "{\"artifact_kind\":\"official_kimi_k3_expert\""
              << ",\"repository\":\"moonshotai/Kimi-K3\""
              << ",\"resolved_revision\":\"9f62e4e9fffbd0a83ddd60e1c209d828994b3569\""
              << ",\"token_semantics\":false"
              << ",\"routing_semantics\":false"
              << ",\"full_moe_layer\":false"
              << ",\"layer_id\":" << pinned.layer_id
              << ",\"expert_id\":" << pinned.expert_id
              << ",\"weight_mode\":\"" << arguments->weight_mode_name << "\""
              << ",\"k3x_root_sha256\":\""
              << digest_hex(pinned.k3x_root_sha256) << "\""
              << ",\"ordered_sha256\":\""
              << digest_hex(pinned.ordered_sha256) << "\""
              << ",\"expert_payload_bytes\":" << pinned.payload_bytes
              << ",\"input_elements\":" << input.size()
              << ",\"output_elements\":" << oracle.size()
              << ",\"warmup\":" << arguments->warmup
              << ",\"iterations\":" << arguments->iterations
              << ",\"cpu_oracle_nanoseconds\":" << cpu_oracle_nanoseconds
              << ",\"cold_latency_nanoseconds\":" << cold_latency_nanoseconds
              << ",\"cold_kernel_nanoseconds\":"
              << cold_profile_after.device_nanoseconds -
                     cold_profile_before.device_nanoseconds
              << ",\"cold_weight_h2d_bytes\":" << cold_weight_h2d
              << ",\"cold_activation_h2d_bytes\":"
              << cold_runtime_after.activation_h2d_bytes -
                     cold_runtime_before.activation_h2d_bytes
              << ",\"cold_device_to_host_bytes\":"
              << cold_runtime_after.device_to_host_bytes -
                     cold_runtime_before.device_to_host_bytes
              << ",\"latency_nanoseconds_median\":" << median(samples)
              << ",\"latency_nanoseconds_p05\":" << percentile(samples, 5)
              << ",\"latency_nanoseconds_p95\":" << percentile(samples, 95)
              << ",\"kernel_nanoseconds\":"
              << profile_after.device_nanoseconds - profile_before.device_nanoseconds
              << ",\"weight_h2d_bytes\":" << measured_weight_h2d
              << ",\"activation_h2d_bytes\":"
              << runtime_after.activation_h2d_bytes -
                     runtime_before.activation_h2d_bytes
              << ",\"device_to_host_bytes\":"
              << runtime_after.device_to_host_bytes -
                     runtime_before.device_to_host_bytes
              << ",\"device_allocation_count\":"
              << runtime_after.device_allocation_count -
                     runtime_before.device_allocation_count
              << ",\"stream_synchronization_count\":"
              << runtime_after.stream_synchronization_count -
                     runtime_before.stream_synchronization_count
              << ",\"weight_cache_hits\":"
              << runtime_after.weight_cache_hits - runtime_before.weight_cache_hits
              << ",\"weight_cache_misses\":"
              << runtime_after.weight_cache_misses -
                     runtime_before.weight_cache_misses
              << ",\"weight_cache_bypasses\":"
              << runtime_after.weight_cache_bypasses -
                     runtime_before.weight_cache_bypasses
              << ",\"resident_weight_bytes\":"
              << runtime_after.resident_weight_bytes
              << ",\"peak_resident_weight_bytes\":"
              << runtime_after.peak_resident_weight_bytes
              << ",\"peak_vram_bytes\":" << memory.peak_device_bytes
              << ",\"maximum_absolute_error\":" << maximum_error
              << ",\"all_finite\":true}\n";
    return 0;
}
