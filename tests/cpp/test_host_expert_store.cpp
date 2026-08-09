// 영속적 L1 expert store의 원자적 admission과 exact bypass 계약을 검증합니다.
#include "k3x/host_expert_store.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <vector>

#ifdef assert
#undef assert
#endif
#define assert(condition)                                                        \
    do {                                                                         \
        if (!(condition)) throw std::runtime_error("test requirement failed"); \
    } while (false)

namespace {
k3x::ExpertProjection projection(std::uint64_t id, std::size_t bytes) {
    return {id, std::vector<std::byte>(bytes, std::byte{0x2a}),
            std::vector<std::byte>(bytes, std::byte{0x7f}), 2, bytes};
}

k3x::ExpertMlpPayload payload(std::uint64_t base, std::size_t bytes = 2) {
    return {projection(base, bytes), projection(base + 1, bytes),
            projection(base + 2, bytes)};
}
}

int main() {
    using k3x::ErrorCode;
    using k3x::ExpertKey;
    using k3x::HostExpertStore;
    using k3x::L1ExpertCacheMode;
    using k3x::Result;

    HostExpertStore store(L1ExpertCacheMode::static_admission, 24);
    std::size_t loads = 0;
    const auto loader = [&]() {
        ++loads;
        return Result<k3x::ExpertMlpPayload>::success(payload(10));
    };
    auto first = store.get_or_load({0, 3}, loader);
    assert(first);
    assert(loads == 1);
    assert(store.stats().hits == 0);
    assert(store.stats().misses == 1);
    assert(store.stats().resident_bytes == 12);
    assert(store.stats().peak_resident_bytes == 12);

    auto hit = store.get_or_load({0, 3}, loader);
    assert(hit);
    assert(loads == 1);
    assert(hit.value().get() == first.value().get());
    assert(store.stats().hits == 1);
    assert(store.stats().misses == 1);

    auto exact_fit = store.get_or_load({0, 4}, [&]() {
        return Result<k3x::ExpertMlpPayload>::success(payload(20));
    });
    assert(exact_fit);
    assert(store.stats().resident_bytes == 24);

    auto no_room = store.get_or_load({0, 5}, [&]() {
        return Result<k3x::ExpertMlpPayload>::success(payload(30));
    });
    assert(no_room);
    assert(store.stats().misses == 3);
    assert(store.stats().bypasses == 1);
    assert(store.stats().resident_bytes == 24);
    assert(no_room.value().get() != first.value().get());

    HostExpertStore oversized(L1ExpertCacheMode::static_admission, 11);
    auto bypass = oversized.get_or_load({1, 0}, [&]() {
        return Result<k3x::ExpertMlpPayload>::success(payload(40));
    });
    assert(bypass);
    assert(oversized.stats().misses == 1);
    assert(oversized.stats().bypasses == 1);
    assert(oversized.stats().resident_bytes == 0);

    HostExpertStore disabled(L1ExpertCacheMode::disabled, 0);
    auto transient_one = disabled.get_or_load({2, 0}, loader);
    auto transient_two = disabled.get_or_load({2, 0}, loader);
    assert(transient_one && transient_two);
    assert(transient_one.value().get() != transient_two.value().get());
    assert(disabled.stats().hits == 0);
    assert(disabled.stats().misses == 0);
    assert(disabled.stats().bypasses == 0);
    assert(disabled.stats().resident_bytes == 0);

    HostExpertStore failure_atomic(L1ExpertCacheMode::static_admission, 24);
    auto failed = failure_atomic.get_or_load({3, 0}, []() {
        return Result<k3x::ExpertMlpPayload>::failure(ErrorCode::io_error);
    });
    assert(!failed);
    assert(failure_atomic.stats().hits == 0);
    assert(failure_atomic.stats().misses == 0);
    assert(failure_atomic.stats().resident_bytes == 0);

    auto invalid = failure_atomic.get_or_load({3, 1}, []() {
        return Result<k3x::ExpertMlpPayload>::success(k3x::ExpertMlpPayload{});
    });
    assert(!invalid);
    assert(invalid.error() == ErrorCode::invalid_mxfp4);
    assert(failure_atomic.stats().misses == 0);
    assert(failure_atomic.stats().resident_bytes == 0);

    return 0;
}
