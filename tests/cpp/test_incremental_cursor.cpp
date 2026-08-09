// 증분 draft cursor의 초기 prefill과 실제 reduced-Top-K proposal을 검증합니다.
#include "k3x/incremental_cursor.hpp"
#include "k3x/model.hpp"

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
        std::cerr << "incremental cursor requirement failed at line "
                  << location.line() << '\n';
        throw std::runtime_error("incremental cursor requirement failed");
    }
}

k3x::RuntimeOptions draft_options() {
    k3x::RuntimeOptions options;
    options.incremental = true;
    options.diagnostics = true;
    options.routing_policy.mode = k3x::RoutingMode::fixed;
    options.routing_policy.fixed_k = 4;
    return options;
}
}

int main(int argc, char** argv) {
    if (argc != 2) return 2;
    const auto artifact = std::filesystem::path(argv[1]);
    auto reader = k3x::Reader::open(artifact);
    require(static_cast<bool>(reader));
    auto backend = k3x::make_cpu_backend();
    k3x::RuntimeSession session(draft_options());
    const std::vector<std::uint32_t> context{1, 7, 3, 9, 43};

    auto cursor = k3x::IncrementalDraftCursor::create(
        reader.value(), *backend, context, session);
    require(static_cast<bool>(cursor));
    require(cursor.value()->diagnostics().mla_length == context.size());

    auto proposal = cursor.value()->propose(2);
    require(static_cast<bool>(proposal));
    require(proposal.value().size() == 2);
    require(cursor.value()->stats().context_prefill_tokens == context.size());
    require(cursor.value()->stats().incremental_forward_calls == 1);
    require(cursor.value()->diagnostics().mla_length == context.size() + 1);

    auto oracle_reader = k3x::Reader::open(artifact);
    require(static_cast<bool>(oracle_reader));
    auto oracle_backend = k3x::make_cpu_backend();
    k3x::RuntimeSession oracle_session(draft_options());
    auto oracle = k3x::generate_greedy(
        oracle_reader.value(), *oracle_backend, context, 2, oracle_session);
    require(static_cast<bool>(oracle));
    require(proposal.value() == oracle.value().token_ids);
    return 0;
}
