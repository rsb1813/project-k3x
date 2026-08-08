// 합성 K3 graph의 C++ greedy generation 인터페이스를 선언합니다.
#pragma once

#include "k3x/reader.hpp"
#include "k3x/status.hpp"

#include <cstdint>
#include <span>
#include <vector>

namespace k3x {
struct GenerationResult {
    std::vector<std::uint32_t> token_ids;
    std::uint64_t decode_nanoseconds{};
};

Result<GenerationResult> generate_greedy(Reader& reader,
                                         std::span<const std::uint32_t> prompt,
                                         std::size_t count,
                                         bool incremental);
}
