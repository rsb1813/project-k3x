// 동일 runtime session의 연속 generation이 L1 expert residency를 재사용하는지 검증합니다.
#include "k3x/model.hpp"

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {
void require(bool condition) {
    if (!condition) throw std::runtime_error("session cache requirement failed");
}
}

int main(int argc, char** argv) {
    if (argc != 2 && argc != 3) return 2;
    k3x::ReaderOptions reader_options;
    if (argc == 3) {
        const auto mode = std::string_view(argv[2]);
        if (mode != "io-uring" && mode != "direct" &&
            mode != "io-uring-direct") {
            return 2;
        }
        if (mode != "direct") {
            reader_options.io_engine = k3x::L2IoEngine::io_uring;
        }
        if (mode != "io-uring") {
            reader_options.cache_mode = k3x::L2CacheMode::direct;
        }
    }
    auto reader = k3x::Reader::open(
        std::filesystem::path(argv[1]), reader_options);
    if (!reader) return 3;
    auto backend = k3x::make_cpu_backend();
    k3x::RuntimeOptions options;
    options.incremental = true;
    options.diagnostics = true;
    options.l1_expert_cache = k3x::L1ExpertCacheMode::static_admission;
    options.l1_expert_cache_bytes = 65536;
    k3x::RuntimeSession session(options);
    const std::vector<std::uint32_t> prompt{1, 7, 3, 9};

    auto first = k3x::generate_greedy(
        reader.value(), *backend, prompt, 6, session);
    require(static_cast<bool>(first));
    const auto first_reads = reader.value().counters();
    const auto first_cache = first.value().l1_expert_cache;
    require(first.value().expert_load_scheduler.submissions == 0);
    require(first_cache.hits == 36);
    require(first_cache.misses == 18);
    require(first_reads.calls - first_reads.batch_submissions ==
            first_cache.misses * 5);
    if (reader_options.cache_mode == k3x::L2CacheMode::direct) {
        require(first_reads.storage_submitted_bytes >
                first_reads.requested_bytes);
    }

    auto second = k3x::generate_greedy(
        reader.value(), *backend, prompt, 6, session);
    require(static_cast<bool>(second));
    const auto second_reads = reader.value().counters();
    const auto second_cache = second.value().l1_expert_cache;
    require(second.value().token_ids == first.value().token_ids);
    require(second.value().prefill_routed_experts ==
            first.value().prefill_routed_experts);
    require(second_cache.misses == first_cache.misses);
    require(second_cache.hits == first_cache.hits + 54);
    require(second_cache.resident_bytes == first_cache.resident_bytes);
    require(second_reads.calls - first_reads.calls < first_reads.calls);
    require(second_reads.completed_bytes - first_reads.completed_bytes <
            first_reads.completed_bytes);
    require(second_reads.calls - first_reads.calls ==
            second_reads.batch_submissions - first_reads.batch_submissions);

    auto deadline_reader = k3x::Reader::open(
        std::filesystem::path(argv[1]), reader_options);
    require(static_cast<bool>(deadline_reader));
    auto deadline_backend = k3x::make_cpu_backend();
    auto deadline_options = options;
    deadline_options.l2_expert_schedule =
        k3x::L2ExpertScheduleMode::deadline;
    k3x::RuntimeSession deadline_session(deadline_options);
    auto deadline = k3x::generate_greedy(
        deadline_reader.value(), *deadline_backend, prompt, 6,
        deadline_session);
    require(static_cast<bool>(deadline));
    require(deadline.value().token_ids == first.value().token_ids);
    require(deadline.value().prefill_routed_experts ==
            first.value().prefill_routed_experts);
    require(deadline.value().l1_expert_cache.hits == first_cache.hits);
    require(deadline.value().l1_expert_cache.misses == first_cache.misses);
    const auto scheduled = deadline.value().expert_load_scheduler;
    require(scheduled.submissions ==
            scheduled.inline_resident_hits + first_cache.misses);
    require(scheduled.inline_resident_hits == first_cache.hits);
    require(scheduled.completions == scheduled.submissions);
    require(scheduled.ready_before_use + scheduled.late_at_use ==
            scheduled.submissions);
    const auto deadline_reads = deadline_reader.value().counters();
    require(deadline_reads.calls == first_reads.calls);
    require(deadline_reads.requested_bytes == first_reads.requested_bytes);
    require(deadline_reads.completed_bytes == first_reads.completed_bytes);
    return 0;
}
