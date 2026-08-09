// bounded L1 expert store의 atomic admission과 exact bypass를 구현합니다.
#include "k3x/host_expert_store.hpp"

#include <algorithm>
#include <limits>

namespace k3x {
namespace {
bool valid_projection(const ExpertProjection& projection) {
    if (projection.rows == 0 || projection.cols == 0 ||
        projection.packed.empty() || projection.scales.empty()) {
        return false;
    }
    return projection.rows <=
           std::numeric_limits<std::size_t>::max() / projection.cols;
}

Result<std::size_t> charged_bytes(const ExpertMlpPayload& payload) {
    if (!valid_projection(payload.gate) || !valid_projection(payload.up) ||
        !valid_projection(payload.down)) {
        return Result<std::size_t>::failure(
            ErrorCode::invalid_mxfp4, "invalid expert payload");
    }
    std::size_t total = 0;
    for (const auto* projection : {&payload.gate, &payload.up, &payload.down}) {
        for (const auto size : {projection->packed.size(),
                                projection->scales.size()}) {
            if (size > std::numeric_limits<std::size_t>::max() - total) {
                return Result<std::size_t>::failure(
                    ErrorCode::invalid_mxfp4, "expert payload size overflow");
            }
            total += size;
        }
    }
    return Result<std::size_t>::success(total);
}
}

HostExpertStore::HostExpertStore(L1ExpertCacheMode mode,
                                 std::size_t capacity_bytes)
    : mode_(mode), capacity_bytes_(capacity_bytes) {}

std::size_t HostExpertStore::KeyHash::operator()(
    const ExpertKey& key) const noexcept {
    const auto layer = std::hash<std::size_t>{}(key.layer);
    const auto expert = std::hash<std::size_t>{}(key.expert);
    return layer ^ (expert + 0x9e3779b9U + (layer << 6U) + (layer >> 2U));
}

Result<ExpertPayloadHandle> HostExpertStore::get_or_load(
    ExpertKey key, const ExpertPayloadLoader& loader) {
    if (mode_ == L1ExpertCacheMode::static_admission) {
        if (const auto found = entries_.find(key); found != entries_.end()) {
            ++stats_.hits;
            return Result<ExpertPayloadHandle>::success(found->second);
        }
    }

    auto loaded = loader();
    if (!loaded) {
        return Result<ExpertPayloadHandle>::failure(
            loaded.error(), loaded.message());
    }
    auto bytes = charged_bytes(loaded.value());
    if (!bytes) {
        return Result<ExpertPayloadHandle>::failure(bytes.error(), bytes.message());
    }
    ExpertPayloadHandle handle =
        std::make_shared<const ExpertMlpPayload>(std::move(loaded.value()));
    if (mode_ == L1ExpertCacheMode::disabled) {
        return Result<ExpertPayloadHandle>::success(std::move(handle));
    }

    ++stats_.misses;
    if (bytes.value() > capacity_bytes_ - stats_.resident_bytes) {
        ++stats_.bypasses;
        return Result<ExpertPayloadHandle>::success(std::move(handle));
    }
    entries_.emplace(key, handle);
    stats_.resident_bytes += bytes.value();
    stats_.peak_resident_bytes =
        std::max(stats_.peak_resident_bytes, stats_.resident_bytes);
    return Result<ExpertPayloadHandle>::success(std::move(handle));
}
}
