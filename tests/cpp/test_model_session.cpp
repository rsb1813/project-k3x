// 동일 runtime session의 연속 generation이 L1 expert residency를 재사용하는지 검증합니다.
#include "k3x/model.hpp"

#include <cstdint>
#include <filesystem>
#include <stdexcept>
#include <vector>

namespace {
void require(bool condition) {
    if (!condition) throw std::runtime_error("session cache requirement failed");
}
}

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    auto reader = k3x::Reader::open(
        std::filesystem::path(argv[1]), k3x::VerifyMode::checksums);
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
    require(first_cache.hits == 36);
    require(first_cache.misses == 18);

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
    return 0;
}
