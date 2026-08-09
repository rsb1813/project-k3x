// AURORA replay drafter의 실제 후보와 lifecycle 경계를 검증합니다.
#include "k3x/aurora.hpp"

#include <filesystem>
#include <iostream>
#include <source_location>
#include <stdexcept>
#include <vector>

namespace {
void require(
    bool condition,
    const std::source_location location = std::source_location::current()) {
    if (!condition) {
        std::cerr << "AURORA requirement failed at line "
                  << location.line() << '\n';
        throw std::runtime_error("AURORA replay requirement failed");
    }
}

k3x::RuntimeOptions draft_options() {
    k3x::RuntimeOptions options;
    options.incremental = true;
    options.routing_policy.mode = k3x::RoutingMode::fixed;
    options.routing_policy.fixed_k = 4;
    return options;
}
}

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    auto reader = k3x::Reader::open(std::filesystem::path(argv[1]));
    require(static_cast<bool>(reader));
    auto backend = k3x::make_cpu_backend();
    const std::vector<std::uint32_t> prompt{1, 7, 3, 9};

    auto provider = k3x::AuroraReplayDraftProvider::create(
        reader.value(), *backend, prompt, draft_options(),
        {.scheduler = {.policy = k3x::AuroraBlockPolicy::fixed,
                       .maximum_length = 2}});
    require(static_cast<bool>(provider));

    const std::vector<std::uint32_t> initial{43};
    auto proposal = provider.value()->propose({
        .anchor_token = 43,
        .max_draft_tokens = 2,
        .generated_position = 1,
        .generated_tokens = initial,
    });
    require(static_cast<bool>(proposal));
    require(proposal.value().anchor_token == 43);
    require(proposal.value().candidate_tokens.size() == 2);
    require(provider.value()->stats().proposal_calls == 1);
    require(provider.value()->stats().candidate_tokens == 2);
    require(provider.value()->stats().replayed_context_tokens == 5);
    require(provider.value()->stats().reader_calls > 0);
    require(provider.value()->stats().reader_bytes > 0);
    require(provider.value()->stats().routing_decisions > 0);

    auto oracle_reader = k3x::Reader::open(std::filesystem::path(argv[1]));
    require(static_cast<bool>(oracle_reader));
    auto oracle_backend = k3x::make_cpu_backend();
    k3x::RuntimeSession oracle_session(draft_options());
    const std::vector<std::uint32_t> replay_sequence{1, 7, 3, 9, 43};
    auto oracle = k3x::generate_greedy(
        oracle_reader.value(), *oracle_backend, replay_sequence, 2,
        oracle_session);
    require(static_cast<bool>(oracle));
    require(proposal.value().candidate_tokens == oracle.value().token_ids);

    auto outstanding = provider.value()->propose({
        .anchor_token = 43,
        .max_draft_tokens = 2,
        .generated_position = 1,
        .generated_tokens = initial,
    });
    require(!outstanding);

    const auto first_candidates = proposal.value().candidate_tokens;
    provider.value()->update({
        .anchor_token = 43,
        .proposed_draft_tokens = 2,
        .accepted_draft_tokens = 2,
        .committed_tokens = {
            first_candidates[0], first_candidates[1], 17},
        .all_draft_tokens_accepted = true,
    });
    const std::vector<std::uint32_t> committed{
        43, first_candidates[0], first_candidates[1], 17};
    const auto reads_before_zero = reader.value().counters();
    auto zero = provider.value()->propose({
        .anchor_token = 17,
        .max_draft_tokens = 0,
        .generated_position = committed.size(),
        .generated_tokens = committed,
    });
    require(static_cast<bool>(zero));
    require(zero.value().candidate_tokens.empty());
    const auto reads_after_zero = reader.value().counters();
    require(reads_after_zero.calls == reads_before_zero.calls);
    require(reads_after_zero.requested_bytes ==
            reads_before_zero.requested_bytes);
    provider.value()->update({
        .anchor_token = 17,
        .proposed_draft_tokens = 0,
        .accepted_draft_tokens = 0,
        .committed_tokens = {18},
        .all_draft_tokens_accepted = true,
    });

    const auto bad_position = provider.value()->propose({
        .anchor_token = 18,
        .max_draft_tokens = 1,
        .generated_position = 4,
        .generated_tokens = std::vector<std::uint32_t>{
            43, first_candidates[0], first_candidates[1], 17, 18},
    });
    require(!bad_position);
    const auto changed_history = provider.value()->propose({
        .anchor_token = 18,
        .max_draft_tokens = 1,
        .generated_position = 5,
        .generated_tokens = std::vector<std::uint32_t>{
            43, first_candidates[0], first_candidates[1], 16, 18},
    });
    require(!changed_history);
    const auto bad_anchor = provider.value()->propose({
        .anchor_token = 99,
        .max_draft_tokens = 1,
        .generated_position = committed.size() + 1,
        .generated_tokens = std::vector<std::uint32_t>{
            43, first_candidates[0], first_candidates[1], 17, 18},
    });
    require(!bad_anchor);

    auto corrected_provider = k3x::AuroraReplayDraftProvider::create(
        reader.value(), *backend, prompt, draft_options(),
        {.scheduler = {.policy = k3x::AuroraBlockPolicy::fixed,
                       .maximum_length = 1}});
    require(static_cast<bool>(corrected_provider));
    auto mismatching = corrected_provider.value()->propose({
        .anchor_token = 43,
        .max_draft_tokens = 1,
        .generated_position = 1,
        .generated_tokens = initial,
    });
    require(static_cast<bool>(mismatching));
    corrected_provider.value()->update({
        .anchor_token = 43,
        .proposed_draft_tokens = 1,
        .accepted_draft_tokens = 0,
        .committed_tokens = {17},
    });
    const std::vector<std::uint32_t> corrected_history{43, 17};
    require(static_cast<bool>(corrected_provider.value()->propose({
        .anchor_token = 17,
        .max_draft_tokens = 1,
        .generated_position = 2,
        .generated_tokens = corrected_history,
    })));

    auto invalid_provider = k3x::AuroraReplayDraftProvider::create(
        reader.value(), *backend, prompt, draft_options(),
        {.scheduler = {.policy = k3x::AuroraBlockPolicy::fixed,
                       .maximum_length = 2}});
    require(static_cast<bool>(invalid_provider));
    auto pending = invalid_provider.value()->propose({
        .anchor_token = 43,
        .max_draft_tokens = 2,
        .generated_position = 1,
        .generated_tokens = initial,
    });
    require(static_cast<bool>(pending));
    invalid_provider.value()->update({
        .anchor_token = 43,
        .proposed_draft_tokens = 1,
        .accepted_draft_tokens = 1,
        .committed_tokens = {pending.value().candidate_tokens[0], 17},
        .all_draft_tokens_accepted = true,
    });
    const auto reads_before_error = reader.value().counters();
    require(!invalid_provider.value()->propose({
        .anchor_token = 17,
        .max_draft_tokens = 1,
        .generated_position = 3,
        .generated_tokens = std::vector<std::uint32_t>{43, 17, 17},
    }));
    const auto reads_after_error = reader.value().counters();
    require(reads_after_error.calls == reads_before_error.calls);
    require(reads_after_error.requested_bytes ==
            reads_before_error.requested_bytes);

    auto invalid_options = draft_options();
    invalid_options.incremental = false;
    require(!k3x::AuroraReplayDraftProvider::create(
        reader.value(), *backend, prompt, invalid_options,
        {.scheduler = {.maximum_length = 1}}));
    invalid_options = draft_options();
    invalid_options.routing_policy.mode = k3x::RoutingMode::natural;
    require(!k3x::AuroraReplayDraftProvider::create(
        reader.value(), *backend, prompt, invalid_options,
        {.scheduler = {.maximum_length = 1}}));
    invalid_options = draft_options();
    invalid_options.routing_policy.fixed_k = 16;
    require(!k3x::AuroraReplayDraftProvider::create(
        reader.value(), *backend, prompt, invalid_options,
        {.scheduler = {.maximum_length = 1}}));
    return 0;
}
