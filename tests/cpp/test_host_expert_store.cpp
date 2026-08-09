// 영속적 L1 expert store의 원자적 admission과 exact bypass 계약을 검증합니다.
#include "k3x/host_expert_store.hpp"

#include <atomic>
#include <barrier>
#include <concepts>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <stdexcept>
#include <thread>
#include <utility>
#include <utility>
#include <vector>

#ifdef assert
#undef assert
#endif
#define assert(condition)                                                        \
    do {                                                                         \
        if (!(condition)) {                                                       \
            throw std::runtime_error("test requirement failed: " #condition);   \
        }                                                                        \
    } while (false)

namespace {
k3x::ExpertProjection projection(std::uint64_t id) {
    return {id, std::vector<std::byte>(512, std::byte{0x2a}),
            std::vector<std::byte>(32, std::byte{0x7f}), 32, 32};
}

k3x::ExpertMlpPayload payload(std::uint64_t base) {
    return {projection(base), projection(base + 1), projection(base + 2)};
}
}

int main() {
    static_assert(std::same_as<
                  decltype(std::declval<const k3x::HostExpertStore&>().stats()),
                  k3x::L1ExpertCacheStats>);
    using k3x::ErrorCode;
    using k3x::ExpertKey;
    using k3x::HostExpertStore;
    using k3x::L1ExpertCacheMode;
    using k3x::Result;

    HostExpertStore store(L1ExpertCacheMode::static_admission, 3264);
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
    assert(store.stats().resident_bytes == 1632);
    assert(store.stats().peak_resident_bytes == 1632);

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
    assert(store.stats().resident_bytes == 3264);

    auto no_room = store.get_or_load({0, 5}, [&]() {
        return Result<k3x::ExpertMlpPayload>::success(payload(30));
    });
    assert(no_room);
    assert(store.stats().misses == 3);
    assert(store.stats().bypasses == 1);
    assert(store.stats().resident_bytes == 3264);
    assert(no_room.value().get() != first.value().get());

    HostExpertStore oversized(L1ExpertCacheMode::static_admission, 1631);
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

    HostExpertStore failure_atomic(L1ExpertCacheMode::static_admission, 3264);
    auto failed = failure_atomic.get_or_load({3, 0}, []() {
        return Result<k3x::ExpertMlpPayload>::failure(ErrorCode::io_error);
    });
    assert(!failed);
    assert(failure_atomic.stats().hits == 0);
    assert(failure_atomic.stats().misses == 0);
    assert(failure_atomic.stats().resident_bytes == 0);

    auto malformed_payload = payload(50);
    malformed_payload.gate.packed.pop_back();
    auto malformed = failure_atomic.get_or_load(
        {3, 2}, [value = std::move(malformed_payload)]() mutable {
            return Result<k3x::ExpertMlpPayload>::success(std::move(value));
        });
    assert(!malformed);
    assert(malformed.error() == ErrorCode::invalid_mxfp4);
    assert(failure_atomic.stats().misses == 0);
    assert(failure_atomic.stats().resident_bytes == 0);

    auto reserved_payload = payload(60);
    reserved_payload.up.scales[0] = std::byte{0xff};
    auto reserved = failure_atomic.get_or_load(
        {3, 3}, [value = std::move(reserved_payload)]() mutable {
            return Result<k3x::ExpertMlpPayload>::success(std::move(value));
        });
    assert(!reserved);
    assert(reserved.error() == ErrorCode::invalid_mxfp4);
    assert(failure_atomic.stats().misses == 0);
    assert(failure_atomic.stats().resident_bytes == 0);

    auto invalid = failure_atomic.get_or_load({3, 1}, []() {
        return Result<k3x::ExpertMlpPayload>::success(k3x::ExpertMlpPayload{});
    });
    assert(!invalid);
    assert(invalid.error() == ErrorCode::invalid_mxfp4);
    assert(failure_atomic.stats().misses == 0);
    assert(failure_atomic.stats().resident_bytes == 0);

    constexpr std::size_t thread_count = 8;
    HostExpertStore concurrent(L1ExpertCacheMode::static_admission, 3264);
    std::atomic<std::size_t> concurrent_loads{};
    std::barrier start(static_cast<std::ptrdiff_t>(thread_count));
    std::vector<k3x::ExpertPayloadHandle> handles(thread_count);
    std::vector<std::thread> threads;
    for (std::size_t index = 0; index < thread_count; ++index) {
        threads.emplace_back([&, index] {
            start.arrive_and_wait();
            auto result = concurrent.get_or_load({4, 7}, [&] {
                ++concurrent_loads;
                std::this_thread::sleep_for(std::chrono::milliseconds{20});
                return Result<k3x::ExpertMlpPayload>::success(payload(70));
            });
            assert(result);
            handles[index] = std::move(result.value());
        });
    }
    for (auto& thread : threads) thread.join();
    assert(concurrent_loads == 1);
    assert(concurrent.stats().misses == 1);
    assert(concurrent.stats().hits == thread_count - 1);
    for (const auto& handle : handles) {
        assert(handle.get() == handles.front().get());
    }

    return 0;
}
