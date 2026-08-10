// 공식 Kimi K3 단일 expert의 고정된 K3X 및 tensor 식별자 계약을 정의합니다.
#pragma once

#include "k3x/status.hpp"

#include <array>
#include <cstddef>
#include <cstdint>

namespace k3x {

struct OfficialExpertShape {
    std::uint64_t rows{};
    std::uint64_t columns{};
    bool operator==(const OfficialExpertShape&) const = default;
};

struct OfficialExpertIdentity {
    std::array<std::byte, 32> k3x_root_sha256{};
    std::array<std::byte, 32> ordered_sha256{};
    std::uint64_t optional_features{};
    std::uint32_t layer_id{};
    std::uint32_t expert_id{};
    std::uint64_t payload_bytes{};
    std::array<OfficialExpertShape, 3> shapes{};
    bool operator==(const OfficialExpertIdentity&) const = default;
};

using OfficialExpertObservation = OfficialExpertIdentity;

const OfficialExpertIdentity& official_kimi_k3_expert();

Result<OfficialExpertIdentity> verify_official_kimi_k3_expert(
    const OfficialExpertObservation& observation);

}  // namespace k3x
