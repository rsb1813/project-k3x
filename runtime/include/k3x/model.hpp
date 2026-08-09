// 합성 K3 graph의 C++ greedy generation 인터페이스를 선언합니다.
#pragma once

#include "k3x/backend.hpp"
#include "k3x/host_expert_store.hpp"
#include "k3x/reader.hpp"
#include "k3x/status.hpp"

#include <cstddef>
#include <cstdint>
#include <span>
#include <vector>

namespace k3x {
struct RuntimeOptions {
    bool incremental{true};
    bool diagnostics{};
    L1ExpertCacheMode l1_expert_cache{L1ExpertCacheMode::disabled};
    std::size_t l1_expert_cache_bytes{};
};

struct GenerationResult {
    std::vector<std::uint32_t> token_ids;
    std::vector<std::uint64_t> per_layer_nanoseconds;
    std::vector<std::vector<float>> prefill_layer_outputs;
    std::vector<float> prefill_logits;
    std::vector<float> prefill_state;
    std::vector<std::uint32_t> prefill_routed_experts;
    std::uint64_t prefill_nanoseconds{};
    std::uint64_t decode_nanoseconds{};
    L1ExpertCacheStats l1_expert_cache;
};

Result<GenerationResult> generate_greedy(Reader& reader,
                                         ComputeBackend& backend,
                                         std::span<const std::uint32_t> prompt,
                                         std::size_t count,
                                         RuntimeOptions options);

Result<GenerationResult> generate_greedy(Reader& reader,
                                         ComputeBackend& backend,
                                         std::span<const std::uint32_t> prompt,
                                         std::size_t count,
                                         bool incremental,
                                         bool diagnostics = false);

Result<GenerationResult> generate_greedy(Reader& reader,
                                         std::span<const std::uint32_t> prompt,
                                         std::size_t count,
                                         bool incremental,
                                         bool diagnostics = false);

Result<GenerationResult> generate_greedy(Reader& reader,
                                         std::span<const std::uint32_t> prompt,
                                         std::size_t count,
                                         RuntimeOptions options);
}
