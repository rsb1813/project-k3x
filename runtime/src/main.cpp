// K3X synthetic runtime을 실행하고 bounded JSON metrics를 기록합니다.
#include "k3x/model.hpp"

#include <charconv>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <string>
#include <string_view>

namespace {

struct ProcessIoSnapshot {
    bool available{};
    std::uint64_t rchar{};
    std::uint64_t read_bytes{};
};

ProcessIoSnapshot process_io_snapshot() {
#ifdef __linux__
    std::ifstream input("/proc/self/io");
    std::string key;
    std::uint64_t value = 0;
    ProcessIoSnapshot snapshot;
    bool found_rchar = false;
    bool found_read_bytes = false;
    while (input >> key >> value) {
        if (key == "rchar:") {
            snapshot.rchar = value;
            found_rchar = true;
        } else if (key == "read_bytes:") {
            snapshot.read_bytes = value;
            found_read_bytes = true;
        }
    }
    snapshot.available = found_rchar && found_read_bytes;
    return snapshot;
#else
    return {};
#endif
}

ProcessIoSnapshot process_io_delta(const ProcessIoSnapshot& before,
                                   const ProcessIoSnapshot& after) {
    if (!before.available || !after.available ||
        after.rchar < before.rchar || after.read_bytes < before.read_bytes) {
        return {};
    }
    return ProcessIoSnapshot{
        true, after.rchar - before.rchar, after.read_bytes - before.read_bytes};
}

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
    std::string cuda_boundary_name = "operation";
    std::string cuda_transfer_name = "synchronous";
    std::string cuda_resident_bytes_text = "0";
    std::string cuda_pinned_bytes_text = "0";
    std::string l1_expert_cache_name = "disabled";
    std::string l1_expert_cache_bytes_text = "0";
    std::string l2_io_name = "pread";
    std::string l2_cache_name = "buffered";
    std::string l2_queue_depth_text = "8";
    std::string l2_schedule_name = "blocking";
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
        else if (key == "--cuda-boundary") cuda_boundary_name = value;
        else if (key == "--cuda-transfer") cuda_transfer_name = value;
        else if (key == "--cuda-resident-bytes") cuda_resident_bytes_text = value;
        else if (key == "--cuda-pinned-bytes") cuda_pinned_bytes_text = value;
        else if (key == "--l1-expert-cache") l1_expert_cache_name = value;
        else if (key == "--l1-expert-cache-bytes") l1_expert_cache_bytes_text = value;
        else if (key == "--l2-io") l2_io_name = value;
        else if (key == "--l2-cache") l2_cache_name = value;
        else if (key == "--l2-queue-depth") l2_queue_depth_text = value;
        else if (key == "--l2-schedule") l2_schedule_name = value;
        else { std::cerr << "unknown argument: " << key << '\n'; return 2; }
    }

    k3x::BackendOptions backend_options;
    k3x::RuntimeOptions runtime_options;
    k3x::ReaderOptions reader_options;
    runtime_options.incremental = mode == "incremental";
    runtime_options.diagnostics = diagnostics;
    if (l1_expert_cache_name == "disabled") {
        runtime_options.l1_expert_cache = k3x::L1ExpertCacheMode::disabled;
    } else if (l1_expert_cache_name == "static") {
        runtime_options.l1_expert_cache = k3x::L1ExpertCacheMode::static_admission;
    } else if (l1_expert_cache_name == "lru") {
        runtime_options.l1_expert_cache = k3x::L1ExpertCacheMode::lru;
    } else if (l1_expert_cache_name == "lfu") {
        runtime_options.l1_expert_cache = k3x::L1ExpertCacheMode::lfu;
    } else if (l1_expert_cache_name == "least-stale") {
        runtime_options.l1_expert_cache = k3x::L1ExpertCacheMode::least_stale;
    } else {
        std::cerr << "unknown L1 expert cache mode: " << l1_expert_cache_name << '\n';
        return 2;
    }
    const auto* l1_begin = l1_expert_cache_bytes_text.data();
    const auto* l1_end = l1_begin + l1_expert_cache_bytes_text.size();
    const auto l1_parse = std::from_chars(
        l1_begin, l1_end, runtime_options.l1_expert_cache_bytes);
    if (l1_expert_cache_bytes_text.empty() || l1_parse.ec != std::errc{} ||
        l1_parse.ptr != l1_end) {
        std::cerr << "invalid L1 expert cache byte capacity: "
                  << l1_expert_cache_bytes_text << '\n';
        return 2;
    }
    if (runtime_options.l1_expert_cache != k3x::L1ExpertCacheMode::disabled &&
        runtime_options.l1_expert_cache_bytes == 0) {
        std::cerr << l1_expert_cache_name
                  << " L1 expert cache requires a positive byte capacity\n";
        return 2;
    }
    if (runtime_options.l1_expert_cache == k3x::L1ExpertCacheMode::disabled &&
        runtime_options.l1_expert_cache_bytes != 0) {
        std::cerr << "disabled L1 expert cache requires a zero byte capacity\n";
        return 2;
    }
    if (l2_schedule_name == "blocking") {
        runtime_options.l2_expert_schedule =
            k3x::L2ExpertScheduleMode::blocking;
    } else if (l2_schedule_name == "deadline") {
        runtime_options.l2_expert_schedule =
            k3x::L2ExpertScheduleMode::deadline;
    } else {
        std::cerr << "unknown L2 expert schedule mode: "
                  << l2_schedule_name << '\n';
        return 2;
    }
    if (l2_io_name == "pread") {
        reader_options.io_engine = k3x::L2IoEngine::pread;
    } else if (l2_io_name == "io-uring") {
        reader_options.io_engine = k3x::L2IoEngine::io_uring;
    } else {
        std::cerr << "unknown L2 I/O engine: " << l2_io_name << '\n';
        return 2;
    }
    if (l2_cache_name == "buffered") {
        reader_options.cache_mode = k3x::L2CacheMode::buffered;
    } else if (l2_cache_name == "direct") {
        reader_options.cache_mode = k3x::L2CacheMode::direct;
    } else {
        std::cerr << "unknown L2 cache mode: " << l2_cache_name << '\n';
        return 2;
    }
    const auto* l2_queue_begin = l2_queue_depth_text.data();
    const auto* l2_queue_end = l2_queue_begin + l2_queue_depth_text.size();
    const auto l2_queue_parse = std::from_chars(
        l2_queue_begin, l2_queue_end, reader_options.queue_depth);
    if (l2_queue_depth_text.empty() || l2_queue_parse.ec != std::errc{} ||
        l2_queue_parse.ptr != l2_queue_end) {
        std::cerr << "invalid L2 queue depth: " << l2_queue_depth_text << '\n';
        return 2;
    }
    if (reader_options.queue_depth == 0) {
        std::cerr << "L2 queue depth must be positive\n";
        return 2;
    }
    if (reader_options.queue_depth > k3x::maximum_l2_queue_depth) {
        std::cerr << "L2 queue depth exceeds maximum: "
                  << reader_options.queue_depth << '\n';
        return 2;
    }
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
    if (cuda_boundary_name == "operation") {
        backend_options.cuda_boundary = k3x::CudaBoundaryMode::operation;
    } else if (cuda_boundary_name == "ffn-block") {
        backend_options.cuda_boundary = k3x::CudaBoundaryMode::ffn_block;
    } else {
        std::cerr << "unknown CUDA boundary mode: " << cuda_boundary_name << '\n';
        return 2;
    }
    if (cuda_transfer_name == "synchronous") {
        backend_options.cuda_transfer = k3x::CudaTransferMode::synchronous;
    } else if (cuda_transfer_name == "prefetch") {
        backend_options.cuda_transfer = k3x::CudaTransferMode::prefetch;
    } else {
        std::cerr << "unknown CUDA transfer mode: " << cuda_transfer_name << '\n';
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
    const auto* pinned_begin = cuda_pinned_bytes_text.data();
    const auto* pinned_end = pinned_begin + cuda_pinned_bytes_text.size();
    const auto pinned_parse = std::from_chars(
        pinned_begin, pinned_end, backend_options.cuda_pinned_bytes);
    if (cuda_pinned_bytes_text.empty() || pinned_parse.ec != std::errc{} ||
        pinned_parse.ptr != pinned_end) {
        std::cerr << "invalid CUDA pinned byte capacity: "
                  << cuda_pinned_bytes_text << '\n';
        return 2;
    }
    if (backend_options.kind == k3x::BackendKind::cpu &&
        backend_options.dense_precision != k3x::DensePrecision::fp32) {
        std::cerr << "bf16 dense precision requires a CUDA backend\n";
        return 2;
    }
    if (backend_options.cuda_boundary == k3x::CudaBoundaryMode::ffn_block &&
        backend_options.kind != k3x::BackendKind::cuda_custom) {
        std::cerr << "ffn-block boundary requires cuda-custom\n";
        return 2;
    }
    if (backend_options.kind == k3x::BackendKind::cpu &&
        (backend_options.cuda_allocation != k3x::CudaAllocationMode::per_operation ||
         backend_options.cuda_weights != k3x::CudaWeightMode::transient ||
         backend_options.cuda_batching != k3x::CudaBatchingMode::scalar ||
         backend_options.cuda_transfer != k3x::CudaTransferMode::synchronous ||
         backend_options.cuda_resident_bytes != 0 ||
         backend_options.cuda_pinned_bytes != 0)) {
        std::cerr << "CUDA execution options require a CUDA backend\n";
        return 2;
    }
    if (backend_options.cuda_transfer == k3x::CudaTransferMode::synchronous &&
        backend_options.cuda_pinned_bytes != 0) {
        std::cerr << "synchronous CUDA transfer requires a zero pinned byte capacity\n";
        return 2;
    }
    if (backend_options.cuda_transfer == k3x::CudaTransferMode::prefetch) {
        if (backend_options.cuda_pinned_bytes == 0) {
            std::cerr << "prefetch CUDA transfer requires a positive pinned byte capacity\n";
            return 2;
        }
        if (backend_options.kind != k3x::BackendKind::cuda_custom) {
            std::cerr << "prefetch CUDA transfer requires cuda-custom\n";
            return 2;
        }
        if (backend_options.cuda_boundary != k3x::CudaBoundaryMode::ffn_block) {
            std::cerr << "prefetch CUDA transfer requires ffn-block boundary\n";
            return 2;
        }
        if (backend_options.cuda_allocation != k3x::CudaAllocationMode::reused) {
            std::cerr << "prefetch CUDA transfer requires reused allocation\n";
            return 2;
        }
        if (backend_options.cuda_weights != k3x::CudaWeightMode::transient) {
            std::cerr << "prefetch CUDA transfer requires transient weights\n";
            return 2;
        }
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
    auto reader = k3x::Reader::open(model_path, reader_options);
    if (!reader) {
        write_error(reader.error(), reader.message());
        return 3;
    }
    k3x::RuntimeSession session(runtime_options);
    const auto process_io_before = process_io_snapshot();
    auto result = k3x::generate_greedy(
        reader.value(), *backend, prompt, count, session);
    const auto process_io = process_io_delta(
        process_io_before, process_io_snapshot());
    if (!result) {
        write_error(result.error(), result.message());
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
    output << ",\"cuda_boundary\":";
    write_json_string(output, cuda_boundary_name);
    output << ",\"cuda_transfer\":";
    write_json_string(output, cuda_transfer_name);
    output << ",\"cuda_resident_bytes\":"
           << effective_options.cuda_resident_bytes;
    output << ",\"cuda_pinned_bytes\":"
           << effective_options.cuda_pinned_bytes;
    output << ",\"l1_expert_cache_mode\":";
    write_json_string(output, l1_expert_cache_name);
    output << ",\"l1_expert_cache_bytes\":"
           << runtime_options.l1_expert_cache_bytes;
    output << ",\"l1_expert_cache_hits\":"
           << result.value().l1_expert_cache.hits
           << ",\"l1_expert_cache_misses\":"
           << result.value().l1_expert_cache.misses
           << ",\"l1_expert_cache_bypasses\":"
           << result.value().l1_expert_cache.bypasses
           << ",\"l1_expert_cache_resident_bytes\":"
           << result.value().l1_expert_cache.resident_bytes
           << ",\"peak_l1_expert_cache_resident_bytes\":"
           << result.value().l1_expert_cache.peak_resident_bytes;
    output << ",\"l2_io_engine\":";
    write_json_string(output, l2_io_name);
    output << ",\"l2_cache_mode\":";
    write_json_string(output, l2_cache_name);
    output << ",\"l2_queue_depth\":" << reader_options.queue_depth
           << ",\"l2_direct_memory_alignment\":"
           << reader.value().direct_memory_alignment()
           << ",\"l2_direct_offset_alignment\":"
           << reader.value().direct_offset_alignment();
    output << ",\"l2_expert_schedule\":";
    write_json_string(output, l2_schedule_name);
    const auto& expert_load = result.value().expert_load_scheduler;
    output << ",\"expert_load_submissions\":"
           << expert_load.submissions
           << ",\"expert_load_inline_resident_hits\":"
           << expert_load.inline_resident_hits
           << ",\"expert_load_completions\":"
           << expert_load.completions
           << ",\"expert_load_ready_before_use\":"
           << expert_load.ready_before_use
           << ",\"expert_load_late_at_use\":"
           << expert_load.late_at_use
           << ",\"expert_load_estimated_deadline_misses\":"
           << expert_load.estimated_deadline_misses
           << ",\"expert_load_requested_bytes\":"
           << expert_load.requested_bytes
           << ",\"expert_load_queue_high_water\":"
           << expert_load.queue_high_water
           << ",\"expert_load_worker_nanoseconds\":"
           << expert_load.worker_nanoseconds
           << ",\"expert_load_exposed_wait_nanoseconds\":"
           << expert_load.exposed_wait_nanoseconds;
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
           << ",\"ffn_block_calls\":" << runtime.ffn_block_calls
           << ",\"ffn_block_experts\":" << runtime.ffn_block_experts
           << ",\"pinned_host_bytes\":" << runtime.pinned_host_bytes
           << ",\"peak_pinned_host_bytes\":"
           << runtime.peak_pinned_host_bytes
           << ",\"async_prefetch_calls\":" << runtime.async_prefetch_calls
           << ",\"async_prefetch_bytes\":" << runtime.async_prefetch_bytes
           << ",\"async_prefetch_ready_before_use\":"
           << runtime.async_prefetch_ready_before_use
           << ",\"async_prefetch_late_at_use\":"
           << runtime.async_prefetch_late_at_use
           << ",\"transfer_stream_wait_count\":"
           << runtime.transfer_stream_wait_count
           << ",\"pinned_staging_nanoseconds\":"
           << runtime.pinned_staging_nanoseconds
           << ",\"transfer_device_nanoseconds\":"
           << runtime.transfer_device_nanoseconds
           << ",\"transfer_stall_nanoseconds\":"
           << runtime.transfer_stall_nanoseconds
           << ",\"async_engine_count\":" << runtime.async_engine_count
           << ",\"device_overlap\":"
           << (runtime.device_overlap ? "true" : "false")
           << ",\"profile_wall_nanoseconds\":" << profile.wall_nanoseconds
           << ",\"profile_logical_bytes\":" << profile.logical_bytes
           << ",\"failed_operations\":" << profile.failed_operations
           << ",\"decode_nanoseconds\":" << result.value().decode_nanoseconds
           << ",\"prefill_nanoseconds\":" << result.value().prefill_nanoseconds
           << ",\"read_bytes\":" << reader.value().counters().completed_bytes
           << ",\"read_calls\":" << reader.value().counters().calls
           << ",\"reader_read_calls\":" << reader.value().counters().calls
           << ",\"reader_requested_bytes\":"
           << reader.value().counters().requested_bytes
           << ",\"reader_completed_bytes\":"
           << reader.value().counters().completed_bytes
           << ",\"reader_batch_submissions\":"
           << reader.value().counters().batch_submissions
           << ",\"reader_storage_submitted_bytes\":"
           << reader.value().counters().storage_submitted_bytes
           << ",\"reader_storage_completed_bytes\":"
           << reader.value().counters().storage_completed_bytes
           << ",\"reader_completions\":"
           << reader.value().counters().completions
           << ",\"reader_short_reads\":"
           << reader.value().counters().short_reads
           << ",\"reader_failures\":"
           << reader.value().counters().failures
           << ",\"reader_storage_nanoseconds\":"
           << reader.value().counters().storage_nanoseconds
           << ",\"process_io_available\":"
           << (process_io.available ? "true" : "false")
           << ",\"process_rchar_bytes\":";
    if (process_io.available) output << process_io.rchar;
    else output << "null";
    output << ",\"process_read_bytes\":";
    if (process_io.available) output << process_io.read_bytes;
    else output << "null";
    output
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
    output << "],\"prefill_routed_experts\":[";
    for (std::size_t index = 0;
         index < result.value().prefill_routed_experts.size(); ++index) {
        if (index) output << ',';
        output << result.value().prefill_routed_experts[index];
    }
    output << "],\"token_ids\":[";
    for (std::size_t index = 0; index < result.value().token_ids.size(); ++index) {
        if (index) output << ',';
        output << result.value().token_ids[index];
    }
    output << "]}\n";
    return 0;
}
