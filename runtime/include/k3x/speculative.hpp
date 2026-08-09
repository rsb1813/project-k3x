// speculative draft proposal과 strict greedy target 검증 계약을 선언합니다.
#pragma once

#include "k3x/status.hpp"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <span>
#include <vector>

namespace k3x {
struct DraftRequest {
    std::uint32_t anchor_token{};
    std::size_t max_draft_tokens{};
    std::size_t generated_position{};
    std::span<const std::uint32_t> generated_tokens;
};

struct DraftProposal {
    std::uint32_t anchor_token{};
    std::vector<std::uint32_t> candidate_tokens;
};

struct DraftVerification {
    std::uint32_t anchor_token{};
    std::size_t proposed_draft_tokens{};
    std::size_t accepted_draft_tokens{};
    std::vector<std::uint32_t> committed_tokens;
    bool all_draft_tokens_accepted{};
};

class DraftProvider {
public:
    virtual ~DraftProvider() = default;
    virtual Result<DraftProposal> propose(const DraftRequest& request) = 0;
    virtual void update(const DraftVerification& verification) = 0;
};

using GreedyTargetStep =
    std::function<Result<std::uint32_t>(std::uint32_t input_token)>;

Result<DraftVerification> verify_greedy_draft(
    const DraftProposal& proposal, std::size_t max_draft_tokens,
    std::size_t vocabulary_size, const GreedyTargetStep& target_step);
}
