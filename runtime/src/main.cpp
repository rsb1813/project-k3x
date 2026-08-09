// K3X synthetic runtime을 실행하고 bounded JSON metrics를 기록합니다.
#include "k3x/model.hpp"

#include <charconv>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>

namespace {

void write_json_string(std::ostream& output, std::string_view value) {
    output << '"';
    for (const auto character : value) {
        if (character == '"' || character == '\\') output << '\\';
        output << character;
    }
    output << '"';
}

void write_error(k3x::ErrorCode code, const std::string& message) {
    std::cerr << k3x::error_code_name(code);
    if (!message.empty()) std::cerr << ": " << message;
    std::cerr << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    std::filesystem::path model_path, output_path;
    std::string prompt_text, mode = "incremental";
    std::string backend_name = "cpu", dense_precision_name = "fp32";
    std::string cuda_allocation_name = "per-operation";
    std::string cuda_weights_name = "transient";
    std::string cuda_batching_name = "scalar";
    std::string cuda_resident_bytes_text = "0";
    bool diagnostics = false;
    std::size_t count = 0;
    for (int index = 1; index + 1 < argc; index += 2) {
        const std::string key = argv[index];
        const std::string value = argv[index + 1];
        if (key == "--model") model_path = value;
        else if (key == "--prompt-ids") prompt_text = value;
        else if (key == "--generate") count = std::stoull(value);
        else if (key == "--mode") mode = value;
        else if (key == "--diagnostics") diagnostics = value == "true";
        else if (key == "--json") output_path = value;
        else if (key == "--backend") backend_name = value;
        else if (key == "--dense-precision") dense_precision_name = value;
        else if (key == "--cuda-allocation") cuda_allocation_name = value;
        else if (key == "--cuda-weights") cuda_weights_name = value;
        else if (key == "--cuda-batching") cuda_batching_name = value;
        else if (key == "--cuda-resident-bytes") cuda_resident_bytes_text = value;
        else { std::cerr << "unknown argument: " << key << '\n'; return 2; }
    }

    k3x::BackendOptions backend_options;
    if (backend_name == "cpu") {
        backend_options.kind = k3x::BackendKind::cpu;
    } else if (backend_name == "cuda-dense") {
        backend_options.kind = k3x::BackendKind::cuda_dense;
    } else if (backend_name == "cuda-custom") {
        backend_options.kind = k3x::BackendKind::cuda_custom;
    } else {
        std::cerr << "unknown backend: " << backend_name << '\n';
        return 2;
    }
    if (dense_precision_name == "fp32") {
        backend_options.dense_precision = k3x::DensePrecision::fp32;
    } else if (dense_precision_name == "bf16") {
        backend_options.dense_precision = k3x::DensePrecision::bf16_rounded;
    } else {
        std::cerr << "unknown dense precision: " << dense_precision_name << '\n';
        return 2;
    }
    if (cuda_allocation_name == "per-operation") {
        backend_options.cuda_allocation = k3x::CudaAllocationMode::per_operation;
    } else if (cuda_allocation_name == "reused") {
        backend_options.cuda_allocation = k3x::CudaAllocationMode::reused;
    } else {
        std::cerr << "unknown CUDA allocation mode: " << cuda_allocation_name << '\n';
        return 2;
    }
    if (cuda_weights_name == "transient") {
        backend_options.cuda_weights = k3x::CudaWeightMode::transient;
    } else if (cuda_weights_name == "resident") {
        backend_options.cuda_weights = k3x::CudaWeightMode::resident;
    } else {
        std::cerr << "unknown CUDA weight mode: " << cuda_weights_name << '\n';
        return 2;
    }
    if (cuda_batching_name == "scalar") {
        backend_options.cuda_batching = k3x::CudaBatchingMode::scalar;
    } else if (cuda_batching_name == "grouped") {
        backend_options.cuda_batching = k3x::CudaBatchingMode::grouped;
    } else {
        std::cerr << "unknown CUDA batching mode: " << cuda_batching_name << '\n';
        return 2;
    }
    const auto* resident_begin = cuda_resident_bytes_text.data();
    const auto* resident_end = resident_begin + cuda_resident_bytes_text.size();
    const auto resident_parse = std::from_chars(
        resident_begin, resident_end, backend_options.cuda_resident_bytes);
    if (cuda_resident_bytes_text.empty() || resident_parse.ec != std::errc{} ||
        resident_parse.ptr != resident_end) {
        std::cerr << "invalid CUDA resident byte capacity: "
                  << cuda_resident_bytes_text << '\n';
        return 2;
    }
    if (backend_options.kind == k3x::BackendKind::cpu &&
        backend_options.dense_precision != k3x::DensePrecision::fp32) {
        std::cerr << "bf16 dense precision requires a CUDA backend\n";
        return 2;
    }
    if (backend_options.kind == k3x::BackendKind::cpu &&
        (backend_options.cuda_allocation != k3x::CudaAllocationMode::per_operation ||
         backend_options.cuda_weights != k3x::CudaWeightMode::transient ||
         backend_options.cuda_batching != k3x::CudaBatchingMode::scalar ||
         backend_options.cuda_resident_bytes != 0)) {
        std::cerr << "CUDA execution options require a CUDA backend\n";
        return 2;
    }
    if (backend_options.kind != k3x::BackendKind::cpu &&
        backend_options.cuda_weights == k3x::CudaWeightMode::resident &&
        backend_options.cuda_resident_bytes == 0) {
        std::cerr << "resident CUDA weights require a positive resident byte capacity\n";
        return 2;
    }
    if (backend_options.kind != k3x::BackendKind::cpu &&
        backend_options.cuda_weights == k3x::CudaWeightMode::transient &&
        backend_options.cuda_resident_bytes != 0) {
        std::cerr << "transient CUDA weights require a zero resident byte capacity\n";
        return 2;
    }

    k3x::Profiler profiler;
    std::unique_ptr<k3x::ComputeBackend> backend;
    if (backend_options.kind == k3x::BackendKind::cpu) {
        backend = k3x::make_cpu_backend(&profiler);
    } else {
        auto cuda_backend = k3x::make_cuda_backend(backend_options, &profiler);
        if (!cuda_backend) {
            write_error(cuda_backend.error(), cuda_backend.message());
            return 4;
        }
        backend = std::move(cuda_backend.value());
    }

    std::vector<std::uint32_t> prompt;
    std::stringstream parser(prompt_text);
    std::string item;
    while (std::getline(parser, item, ',')) prompt.push_back(static_cast<std::uint32_t>(std::stoul(item)));
    auto reader = k3x::Reader::open(model_path, k3x::VerifyMode::checksums);
    if (!reader) {
        std::cerr << (reader.message().empty() ? k3x::error_code_name(reader.error())
                                               : reader.message()) << '\n';
        return 3;
    }
    auto result = k3x::generate_greedy(
        reader.value(), *backend, prompt, count, mode == "incremental", diagnostics);
    if (!result) {
        std::cerr << (result.message().empty() ? k3x::error_code_name(result.error())
                                               : result.message()) << '\n';
        return 4;
    }
    std::ofstream output(output_path);
    if (!output) return 5;
    const auto profile = profiler.summary();
    const auto memory = backend->memory_stats();
    const auto runtime = backend->runtime_stats();
    const auto& effective_options = backend->options();
    output << std::setprecision(9);
    output << "{\"backend\":";
    write_json_string(output, backend_name);
    output << ",\"device\":";
    write_json_string(output, backend->device_name());
    output << ",\"dense_precision\":";
    write_json_string(output, dense_precision_name);
    output << ",\"cuda_allocation\":";
    write_json_string(output, cuda_allocation_name);
    output << ",\"cuda_weights\":";
    write_json_string(output, cuda_weights_name);
    output << ",\"cuda_batching\":";
    write_json_string(output, cuda_batching_name);
    output << ",\"cuda_resident_bytes\":"
           << effective_options.cuda_resident_bytes;
    output << ",\"kernel_nanoseconds\":" << profile.device_nanoseconds
           << ",\"host_to_device_bytes\":" << profile.host_to_device_bytes
           << ",\"weight_h2d_bytes\":"
           << profile.weight_host_to_device_bytes
           << ",\"activation_h2d_bytes\":"
           << profile.activation_host_to_device_bytes
           << ",\"device_to_host_bytes\":" << profile.device_to_host_bytes
           << ",\"peak_vram_bytes\":" << memory.peak_device_bytes
           << ",\"device_allocation_count\":" << runtime.device_allocation_count
           << ",\"device_free_count\":" << runtime.device_free_count
           << ",\"stream_synchronization_count\":"
           << runtime.stream_synchronization_count
           << ",\"weight_cache_hits\":" << runtime.weight_cache_hits
           << ",\"weight_cache_misses\":" << runtime.weight_cache_misses
           << ",\"weight_cache_bypasses\":" << runtime.weight_cache_bypasses
           << ",\"resident_weight_bytes\":" << runtime.resident_weight_bytes
           << ",\"peak_resident_weight_bytes\":"
           << runtime.peak_resident_weight_bytes
           << ",\"scratch_bytes\":" << runtime.scratch_bytes
           << ",\"peak_scratch_bytes\":" << runtime.peak_scratch_bytes
           << ",\"grouped_projection_calls\":"
           << runtime.grouped_projection_calls
           << ",\"grouped_projection_members\":"
           << runtime.grouped_projection_members
           << ",\"profile_wall_nanoseconds\":" << profile.wall_nanoseconds
           << ",\"profile_logical_bytes\":" << profile.logical_bytes
           << ",\"failed_operations\":" << profile.failed_operations
           << ",\"decode_nanoseconds\":" << result.value().decode_nanoseconds
           << ",\"prefill_nanoseconds\":" << result.value().prefill_nanoseconds
           << ",\"read_bytes\":" << reader.value().counters().completed_bytes
           << ",\"read_calls\":" << reader.value().counters().calls
           << ",\"per_layer_nanoseconds\":[";
    for (std::size_t index = 0; index < result.value().per_layer_nanoseconds.size(); ++index) {
        if (index) output << ',';
        output << result.value().per_layer_nanoseconds[index];
    }
    output << "],\"prefill_layer_outputs\":[";
    for (std::size_t layer = 0; layer < result.value().prefill_layer_outputs.size(); ++layer) {
        if (layer) output << ',';
        output << '[';
        for (std::size_t index = 0; index < result.value().prefill_layer_outputs[layer].size(); ++index) {
            if (index) output << ',';
            output << result.value().prefill_layer_outputs[layer][index];
        }
        output << ']';
    }
    output << "],\"prefill_logits\":[";
    for (std::size_t index = 0; index < result.value().prefill_logits.size(); ++index) {
        if (index) output << ',';
        output << result.value().prefill_logits[index];
    }
    output << "],\"prefill_state\":[";
    for (std::size_t index = 0; index < result.value().prefill_state.size(); ++index) {
        if (index) output << ',';
        output << result.value().prefill_state[index];
    }
    output << "],\"token_ids\":[";
    for (std::size_t index = 0; index < result.value().token_ids.size(); ++index) {
        if (index) output << ',';
        output << result.value().token_ids[index];
    }
    output << "]}\n";
    return 0;
}
