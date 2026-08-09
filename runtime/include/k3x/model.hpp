// 합성 K3 graph의 C++ greedy generation 인터페이스를 선언합니다.
#pragma once

#include "k3x/backend.hpp"
#include "k3x/expert_scheduler.hpp"
#include "k3x/host_expert_store.hpp"
#include "k3x/reader.hpp"
#include "k3x/runtime_profile.hpp"
#include "k3x/status.hpp"

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <mutex>
#include <span>
#include <utility>
#include <vector>

namespace k3x {
enum class L2ExpertScheduleMode { blocking, deadline };

struct RuntimeOptions {
    bool incremental{true};
    bool diagnostics{};
    L1ExpertCacheMode l1_expert_cache{L1ExpertCacheMode::disabled};
    std::size_t l1_expert_cache_bytes{};
    std::uint64_t profile_prior_strength{64};
    bool profile_observation{};
    L2ExpertScheduleMode l2_expert_schedule{L2ExpertScheduleMode::blocking};
};

class RuntimeSession {
public:
    explicit RuntimeSession(RuntimeOptions options)
        : RuntimeSession(options, RuntimeProfile{}) {}

    RuntimeSession(RuntimeOptions options, RuntimeProfile profile)
        : options_(options), profile_(std::move(profile)),
          expert_store_(options.l1_expert_cache,
                        options.l1_expert_cache_bytes,
                        options.profile_observation ||
                                options.l1_expert_cache ==
                                    L1ExpertCacheMode::profiled
                            ? &profile_
                            : nullptr,
                        options.profile_prior_strength) {
        if (options.l2_expert_schedule == L2ExpertScheduleMode::deadline) {
            expert_loader_ = std::make_unique<DeadlineExpertLoader>(64);
        }
    }

    const RuntimeOptions& options() const noexcept { return options_; }
    HostExpertStore& expert_store() noexcept { return expert_store_; }
    RuntimeProfile& profile() noexcept { return profile_; }
    const RuntimeProfile& profile() const noexcept { return profile_; }
    L1ExpertCacheStats l1_expert_cache_stats() const {
        return expert_store_.stats();
    }
    std::uint64_t acquire_forward_cycle() noexcept {
        return next_forward_cycle_.fetch_add(1, std::memory_order_relaxed);
    }
    std::unique_lock<std::mutex> acquire_generation_guard() {
        return std::unique_lock(generation_mutex_);
    }
    DeadlineExpertLoader* expert_loader() noexcept {
        return expert_loader_.get();
    }
    ExpertLoadSchedulerStats expert_load_scheduler_stats() const {
        return expert_loader_ ? expert_loader_->stats()
                              : ExpertLoadSchedulerStats{};
    }

private:
    RuntimeOptions options_;
    RuntimeProfile profile_;
    HostExpertStore expert_store_;
    std::unique_ptr<DeadlineExpertLoader> expert_loader_;
    std::atomic<std::uint64_t> next_forward_cycle_{};
    std::mutex generation_mutex_;
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
    ExpertLoadSchedulerStats expert_load_scheduler;
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
                                         RuntimeSession& session);

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
