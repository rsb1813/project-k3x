// 고정된 공식 Kimi K3 expert 식별자 계약과 모든 실패 경계를 검증합니다.
#include "k3x/format.hpp"
#include "k3x/official_expert.hpp"

#include <cstddef>

namespace {

bool rejected(const k3x::OfficialExpertObservation& observation) {
    const auto result = k3x::verify_official_kimi_k3_expert(observation);
    return !result && result.error() == k3x::ErrorCode::invalid_mxfp4 &&
        result.message() == "official Kimi K3 expert identity mismatch";
}

}  // namespace

int main() {
    const auto& identity = k3x::official_kimi_k3_expert();
    k3x::OfficialExpertObservation observation = identity;
    const auto accepted = k3x::verify_official_kimi_k3_expert(observation);
    if (!accepted) return 1;

    auto bad = observation;
    bad.k3x_root_sha256[0] ^= std::byte{1};
    if (!rejected(bad)) return 2;

    bad = observation;
    bad.ordered_sha256[31] ^= std::byte{1};
    if (!rejected(bad)) return 3;

    bad = observation;
    bad.optional_features = 0;
    if (!rejected(bad)) return 4;

    bad = observation;
    ++bad.layer_id;
    if (!rejected(bad)) return 5;

    bad = observation;
    ++bad.expert_id;
    if (!rejected(bad)) return 6;

    bad = observation;
    --bad.payload_bytes;
    if (!rejected(bad)) return 7;

    for (std::size_t index = 0; index < observation.shapes.size(); ++index) {
        bad = observation;
        ++bad.shapes[index].rows;
        if (!rejected(bad)) return static_cast<int>(8 + index * 2);
        bad = observation;
        ++bad.shapes[index].columns;
        if (!rejected(bad)) return static_cast<int>(9 + index * 2);
    }
    return 0;
}
