// 공식 Kimi K3 단일 expert의 pinned identity를 순수 비교로 검증합니다.
#include "k3x/official_expert.hpp"

#include "k3x/format.hpp"

#include <array>
#include <cstddef>
#include <string_view>

namespace k3x {
namespace {

constexpr std::byte hex_nibble(char value) {
    return static_cast<std::byte>(
        value >= '0' && value <= '9' ? value - '0' : value - 'a' + 10);
}

constexpr std::array<std::byte, 32> digest(std::string_view value) {
    std::array<std::byte, 32> result{};
    for (std::size_t index = 0; index < result.size(); ++index) {
        result[index] = static_cast<std::byte>(
            (std::to_integer<unsigned int>(hex_nibble(value[index * 2])) << 4) |
            std::to_integer<unsigned int>(hex_nibble(value[index * 2 + 1])));
    }
    return result;
}

constexpr OfficialExpertIdentity identity{
    digest("d585d283325e13e1316a0194c2d6274dd89ef75a28b96b02f02733290b7658be"),
    digest("4e23bd960dfb5e8b10def10e12a94bac1119500f72918698986bd332d56d33ff"),
    optional_storage_fixture,
    1,
    0,
    17'547'264,
    {{{3072, 3584}, {3072, 3584}, {3584, 3072}}},
};

}  // namespace

const OfficialExpertIdentity& official_kimi_k3_expert() {
    return identity;
}

Result<OfficialExpertIdentity> verify_official_kimi_k3_expert(
    const OfficialExpertObservation& observation) {
    if (observation != identity) {
        return Result<OfficialExpertIdentity>::failure(
            ErrorCode::invalid_mxfp4,
            "official Kimi K3 expert identity mismatch");
    }
    return Result<OfficialExpertIdentity>::success(identity);
}

}  // namespace k3x
