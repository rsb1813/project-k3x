// AURORA draft 후보를 committed prefix 전체 replay로 생성합니다.
#include "k3x/aurora.hpp"

#include <algorithm>
#include <utility>

namespace k3x {
namespace {
bool allowed_draft_k(std::size_t value) {
    return value == 4 || value == 6 || value == 8 || value == 12;
}

bool same_tokens(std::span<const std::uint32_t> left,
                 std::span<const std::uint32_t> right) {
    return left.size() == right.size() &&
        std::equal(left.begin(), left.end(), right.begin());
}
}

AuroraReplayDraftProvider::AuroraReplayDraftProvider(
    Reader& reader, ComputeBackend& backend,
    std::vector<std::uint32_t> prompt,
    RuntimeOptions draft_options,
    AdaptiveDraftScheduler scheduler)
    : reader_(reader), backend_(backend), session_(draft_options),
      prompt_(std::move(prompt)), scheduler_(std::move(scheduler)) {}

Result<std::unique_ptr<AuroraReplayDraftProvider>>
AuroraReplayDraftProvider::create(
    Reader& reader, ComputeBackend& backend,
    std::span<const std::uint32_t> prompt,
    RuntimeOptions draft_options, AuroraReplayConfig config) {
    if (backend.kind() != BackendKind::cpu || !draft_options.incremental ||
        draft_options.routing_policy.mode != RoutingMode::fixed ||
        !allowed_draft_k(draft_options.routing_policy.fixed_k) ||
        draft_options.l1_expert_cache != L1ExpertCacheMode::disabled ||
        draft_options.l1_expert_cache_bytes != 0 ||
        draft_options.l2_expert_schedule != L2ExpertScheduleMode::blocking ||
        draft_options.profile_observation || prompt.empty()) {
        return Result<std::unique_ptr<AuroraReplayDraftProvider>>::failure(
            ErrorCode::invalid_state,
            "unsupported AURORA replay runtime combination");
    }
    auto scheduler = AdaptiveDraftScheduler::create(config.scheduler);
    if (!scheduler) {
        return Result<std::unique_ptr<AuroraReplayDraftProvider>>::failure(
            scheduler.error(), scheduler.message());
    }
    return Result<std::unique_ptr<AuroraReplayDraftProvider>>::success(
        std::unique_ptr<AuroraReplayDraftProvider>(
            new AuroraReplayDraftProvider(
                reader, backend,
                std::vector<std::uint32_t>(prompt.begin(), prompt.end()),
                draft_options, std::move(scheduler.value()))));
}

Result<DraftProposal> AuroraReplayDraftProvider::propose(
    const DraftRequest& request) {
    if (lifecycle_error_ || pending_ || request.generated_tokens.empty() ||
        request.generated_position != request.generated_tokens.size() ||
        request.anchor_token != request.generated_tokens.back() ||
        (initialized_ &&
         !same_tokens(request.generated_tokens, expected_generated_))) {
        return Result<DraftProposal>::failure(
            ErrorCode::invalid_state, "invalid AURORA draft lifecycle");
    }

    auto selected = scheduler_.select(request.max_draft_tokens);
    if (!selected) {
        return Result<DraftProposal>::failure(
            selected.error(), selected.message());
    }

    DraftProposal proposal{.anchor_token = request.anchor_token};
    if (selected.value() != 0) {
        std::vector<std::uint32_t> sequence;
        sequence.reserve(prompt_.size() + request.generated_tokens.size());
        sequence.insert(sequence.end(), prompt_.begin(), prompt_.end());
        sequence.insert(sequence.end(), request.generated_tokens.begin(),
                        request.generated_tokens.end());
        const auto reads_before = reader_.counters();
        auto generated = generate_greedy(
            reader_, backend_, sequence, selected.value(), session_);
        const auto reads_after = reader_.counters();
        if (!generated) {
            return Result<DraftProposal>::failure(
                generated.error(), generated.message());
        }
        if (generated.value().token_ids.size() != selected.value() ||
            reads_after.calls < reads_before.calls ||
            reads_after.completed_bytes < reads_before.completed_bytes) {
            return Result<DraftProposal>::failure(
                ErrorCode::invalid_state, "invalid AURORA replay result");
        }
        proposal.candidate_tokens = std::move(generated.value().token_ids);
        stats_.replayed_context_tokens += sequence.size();
        stats_.generation_nanoseconds +=
            generated.value().prefill_nanoseconds +
            generated.value().decode_nanoseconds;
        stats_.reader_calls += reads_after.calls - reads_before.calls;
        stats_.reader_bytes +=
            reads_after.completed_bytes - reads_before.completed_bytes;
        stats_.routing_decisions += generated.value().routing_decisions;
        stats_.routing_selected_experts +=
            generated.value().routing_selected_experts;
    }

    if (!initialized_) {
        expected_generated_.assign(request.generated_tokens.begin(),
                                   request.generated_tokens.end());
        initialized_ = true;
    }
    pending_anchor_ = request.anchor_token;
    pending_candidate_tokens_ = proposal.candidate_tokens;
    pending_ = true;
    ++stats_.proposal_calls;
    stats_.candidate_tokens += proposal.candidate_tokens.size();
    return Result<DraftProposal>::success(std::move(proposal));
}

void AuroraReplayDraftProvider::update(
    const DraftVerification& verification) {
    const auto proposed = pending_candidate_tokens_.size();
    const auto accepted = verification.accepted_draft_tokens;
    bool valid = pending_ && verification.anchor_token == pending_anchor_ &&
        verification.proposed_draft_tokens == proposed &&
        accepted <= proposed &&
        verification.committed_tokens.size() == accepted + 1 &&
        verification.all_draft_tokens_accepted == (accepted == proposed);
    for (std::size_t index = 0; valid && index < accepted; ++index) {
        valid = verification.committed_tokens[index] ==
            pending_candidate_tokens_[index];
    }
    auto observed = valid
        ? scheduler_.observe(verification)
        : Result<bool>::failure(ErrorCode::invalid_state);
    if (!valid || !observed) {
        lifecycle_error_ = true;
        pending_ = false;
        pending_candidate_tokens_.clear();
        return;
    }
    expected_generated_.insert(expected_generated_.end(),
                               verification.committed_tokens.begin(),
                               verification.committed_tokens.end());
    pending_ = false;
    pending_candidate_tokens_.clear();
}

DraftProviderStats AuroraReplayDraftProvider::stats() const noexcept {
    auto result = stats_;
    const auto scheduler_stats = scheduler_.stats();
    result.selected_length_1 = scheduler_stats.selected_length_1;
    result.selected_length_2 = scheduler_stats.selected_length_2;
    result.selected_length_4 = scheduler_stats.selected_length_4;
    result.scheduler_growths = scheduler_stats.scheduler_growths;
    result.scheduler_backoffs = scheduler_stats.scheduler_backoffs;
    return result;
}
}
