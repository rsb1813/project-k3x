// 동일 runtime session의 연속 generation이 L1 expert residency를 재사용하는지 검증합니다.
#include "k3x/model.hpp"

#include <chrono>
#include <cstdint>
#include <deque>
#include <filesystem>
#include <future>
#include <stdexcept>
#include <string_view>
#include <vector>

namespace {
void require(bool condition) {
    if (!condition) throw std::runtime_error("session cache requirement failed");
}

class ScriptedDraftProvider final : public k3x::DraftProvider {
public:
    explicit ScriptedDraftProvider(std::deque<k3x::DraftProposal> proposals)
        : proposals_(std::move(proposals)) {}

    k3x::Result<k3x::DraftProposal> propose(
        const k3x::DraftRequest& request) override {
        requests.push_back({request.anchor_token, request.max_draft_tokens,
                            request.generated_position,
                            std::vector<std::uint32_t>(
                                request.generated_tokens.begin(),
                                request.generated_tokens.end())});
        if (proposals_.empty()) {
            return k3x::Result<k3x::DraftProposal>::failure(
                k3x::ErrorCode::invalid_state, "script exhausted");
        }
        auto proposal = std::move(proposals_.front());
        proposals_.pop_front();
        return k3x::Result<k3x::DraftProposal>::success(std::move(proposal));
    }

    void update(const k3x::DraftVerification& verification) override {
        verifications.push_back(verification);
    }

    struct RequestRecord {
        std::uint32_t anchor_token{};
        std::size_t max_draft_tokens{};
        std::size_t generated_position{};
        std::vector<std::uint32_t> generated_tokens;
    };
    std::vector<RequestRecord> requests;
    std::vector<k3x::DraftVerification> verifications;

private:
    std::deque<k3x::DraftProposal> proposals_;
};
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

    auto protected_reader = k3x::Reader::open(
        std::filesystem::path(argv[1]), reader_options);
    require(static_cast<bool>(protected_reader));
    auto protected_backend = k3x::make_cpu_backend();
    auto protected_options = options;
    protected_options.l1_expert_cache =
        k3x::L1ExpertCacheMode::least_stale;
    protected_options.l1_expert_cache_bytes = 1632;
    k3x::RuntimeSession protected_session(protected_options);
    auto protected_result = k3x::generate_greedy(
        protected_reader.value(), *protected_backend,
        std::vector<std::uint32_t>{1}, 0, protected_session);
    require(static_cast<bool>(protected_result));
    const auto& routed = protected_result.value().prefill_routed_experts;
    require(routed.size() >= 2);
    require(protected_session.expert_store().contains({3, routed[routed.size() - 2]}));
    require(!protected_session.expert_store().contains({3, routed.back()}));

    auto serialized_reader = k3x::Reader::open(
        std::filesystem::path(argv[1]), reader_options);
    require(static_cast<bool>(serialized_reader));
    auto serialized_backend = k3x::make_cpu_backend();
    k3x::RuntimeSession serialized_session(options);
    auto generation_guard = serialized_session.acquire_generation_guard();
    auto concurrent_generation = std::async(std::launch::async, [&] {
        return k3x::generate_greedy(
            serialized_reader.value(), *serialized_backend,
            std::vector<std::uint32_t>{1}, 0, serialized_session);
    });
    require(concurrent_generation.wait_for(std::chrono::milliseconds(50)) ==
            std::future_status::timeout);
    generation_guard.unlock();
    require(static_cast<bool>(concurrent_generation.get()));

    k3x::RuntimeProfile runtime_profile;
    require(static_cast<bool>(runtime_profile.set_metadata("TASK", "coding")));
    require(static_cast<bool>(runtime_profile.set_metadata("REPO", "k3x")));
    auto profiled_options = options;
    profiled_options.l1_expert_cache = k3x::L1ExpertCacheMode::profiled;
    profiled_options.profile_prior_strength = 4;
    k3x::RuntimeSession profiled_session(
        profiled_options, std::move(runtime_profile));
    auto profiled = k3x::generate_greedy(
        serialized_reader.value(), *serialized_backend, prompt, 6,
        profiled_session);
    require(static_cast<bool>(profiled));
    require(profiled.value().token_ids == first.value().token_ids);
    require(profiled.value().prefill_routed_experts ==
            first.value().prefill_routed_experts);
    require(profiled_session.profile().metadata().at("TASK") == "coding");
    require(profiled_session.profile().live_route_observations() > 0);

    auto run_speculative = [&](ScriptedDraftProvider& provider,
                               std::size_t count,
                               std::size_t block_size) {
        auto speculative_reader = k3x::Reader::open(
            std::filesystem::path(argv[1]), reader_options);
        require(static_cast<bool>(speculative_reader));
        auto speculative_backend = k3x::make_cpu_backend();
        k3x::RuntimeSession speculative_session(options);
        auto generated = k3x::generate_speculative(
            speculative_reader.value(), *speculative_backend, prompt, count,
            speculative_session, provider, block_size);
        return std::pair{std::move(generated),
                         speculative_reader.value().counters()};
    };

    const auto& greedy_tokens = first.value().token_ids;
    ScriptedDraftProvider perfect({
        {.anchor_token = greedy_tokens[0],
         .candidate_tokens = {greedy_tokens[1], greedy_tokens[2]}},
        {.anchor_token = greedy_tokens[3],
         .candidate_tokens = {greedy_tokens[4]}},
    });
    auto [perfect_result, perfect_reads] = run_speculative(perfect, 6, 2);
    require(static_cast<bool>(perfect_result));
    require(perfect_result.value().token_ids == greedy_tokens);
    require(perfect_result.value().prefill_state == first.value().prefill_state);
    require(perfect_result.value().final_state == first.value().final_state);
    require(perfect_result.value().routed_experts == first.value().routed_experts);
    require(perfect_result.value().routed_k == first.value().routed_k);
    require(perfect_result.value().speculative_verification_blocks == 2);
    require(perfect_result.value().speculative_proposed_draft_tokens == 3);
    require(perfect_result.value().speculative_accepted_draft_tokens == 3);
    require(perfect_result.value().speculative_committed_tokens == 5);
    require(perfect_result.value().speculative_max_proposal_tokens == 2);
    require(perfect_result.value().l1_expert_cache.hits == first_cache.hits);
    require(perfect_result.value().l1_expert_cache.misses == first_cache.misses);
    require(perfect_reads.calls == first_reads.calls);
    require(perfect_reads.requested_bytes == first_reads.requested_bytes);
    require(perfect.verifications.size() == 2);
    require(perfect.verifications[0].accepted_draft_tokens == 2);
    require(perfect.verifications[1].accepted_draft_tokens == 1);

    ScriptedDraftProvider mixed({
        {.anchor_token = greedy_tokens[0],
         .candidate_tokens = {greedy_tokens[1] ^ 1U, greedy_tokens[2]}},
        {.anchor_token = greedy_tokens[1], .candidate_tokens = {}},
        {.anchor_token = greedy_tokens[2],
         .candidate_tokens = {greedy_tokens[3], greedy_tokens[4] ^ 1U}},
        {.anchor_token = greedy_tokens[4], .candidate_tokens = {}},
    });
    auto [mixed_result, mixed_reads] = run_speculative(mixed, 6, 2);
    require(static_cast<bool>(mixed_result));
    require(mixed_result.value().token_ids == greedy_tokens);
    require(mixed_result.value().final_state == first.value().final_state);
    require(mixed_result.value().routed_experts == first.value().routed_experts);
    require(mixed_result.value().routed_k == first.value().routed_k);
    require(mixed_result.value().speculative_verification_blocks == 4);
    require(mixed_result.value().speculative_proposed_draft_tokens == 4);
    require(mixed_result.value().speculative_accepted_draft_tokens == 1);
    require(mixed_result.value().speculative_committed_tokens == 5);
    require(mixed_reads.calls == first_reads.calls);
    require(mixed_reads.requested_bytes == first_reads.requested_bytes);
    require(mixed.verifications.size() == 4);

    ScriptedDraftProvider unused({});
    auto [single_result, single_reads] = run_speculative(unused, 1, 2);
    require(static_cast<bool>(single_result));
    require(single_result.value().token_ids ==
            std::vector<std::uint32_t>{greedy_tokens[0]});
    require(single_result.value().speculative_verification_blocks == 0);
    require(unused.requests.empty());

    ScriptedDraftProvider exhausted({});
    auto [exhausted_result, exhausted_reads] =
        run_speculative(exhausted, 2, 2);
    require(!exhausted_result);
    require(exhausted_result.error() == k3x::ErrorCode::invalid_state);
    require(exhausted.verifications.empty());

    ScriptedDraftProvider zero_block({});
    auto [zero_block_result, zero_block_reads] =
        run_speculative(zero_block, 2, 0);
    require(!zero_block_result);
    require(zero_block_result.error() == k3x::ErrorCode::invalid_state);
    require(zero_block.requests.empty());

    ScriptedDraftProvider wrong_anchor({
        {.anchor_token = greedy_tokens[0] ^ 1U, .candidate_tokens = {}},
    });
    auto [wrong_anchor_result, wrong_anchor_reads] =
        run_speculative(wrong_anchor, 2, 2);
    require(!wrong_anchor_result);
    require(wrong_anchor_result.error() == k3x::ErrorCode::invalid_state);
    require(wrong_anchor.verifications.empty());

    ScriptedDraftProvider oversized({
        {.anchor_token = greedy_tokens[0],
         .candidate_tokens = {greedy_tokens[1], greedy_tokens[2]}},
    });
    auto [oversized_result, oversized_reads] =
        run_speculative(oversized, 2, 2);
    require(!oversized_result);
    require(oversized_result.error() == k3x::ErrorCode::invalid_extent);
    require(oversized.verifications.empty());

    auto nonincremental_reader = k3x::Reader::open(
        std::filesystem::path(argv[1]), reader_options);
    require(static_cast<bool>(nonincremental_reader));
    auto nonincremental_backend = k3x::make_cpu_backend();
    auto nonincremental_options = options;
    nonincremental_options.incremental = false;
    k3x::RuntimeSession nonincremental_session(nonincremental_options);
    ScriptedDraftProvider nonincremental_provider({});
    auto nonincremental = k3x::generate_speculative(
        nonincremental_reader.value(), *nonincremental_backend, prompt, 2,
        nonincremental_session, nonincremental_provider, 2);
    require(!nonincremental);
    require(nonincremental.error() == k3x::ErrorCode::invalid_state);
    require(nonincremental_provider.requests.empty());
    return 0;
}
