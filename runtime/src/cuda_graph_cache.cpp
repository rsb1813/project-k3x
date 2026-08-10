// CUDA Graph ordered identity의 bounded LRU 선택을 구현합니다.
#include "k3x/cuda_graph_cache.hpp"

#include <algorithm>
#include <stdexcept>

namespace k3x {

BoundedCudaGraphIndex::BoundedCudaGraphIndex(std::size_t capacity)
    : capacity_(capacity) {
    if (capacity_ == 0) {
        throw std::invalid_argument("CUDA Graph cache capacity must be positive");
    }
}

CudaGraphCacheDecision BoundedCudaGraphIndex::touch(
    const CudaGraphKey& key) {
    const auto found = last_use_.find(key);
    if (found != last_use_.end()) {
        found->second = ++sequence_;
        return {true, std::nullopt};
    }

    std::optional<CudaGraphKey> evicted;
    if (last_use_.size() == capacity_) {
        const auto victim = std::min_element(
            last_use_.begin(), last_use_.end(),
            [](const auto& left, const auto& right) {
                if (left.second != right.second) {
                    return left.second < right.second;
                }
                return left.first < right.first;
            });
        evicted = victim->first;
        last_use_.erase(victim);
    }
    last_use_.emplace(key, ++sequence_);
    peak_size_ = std::max(peak_size_, last_use_.size());
    return {false, std::move(evicted)};
}

bool BoundedCudaGraphIndex::erase(const CudaGraphKey& key) {
    return last_use_.erase(key) != 0;
}

void BoundedCudaGraphIndex::clear() noexcept {
    last_use_.clear();
}

bool BoundedCudaGraphIndex::contains(const CudaGraphKey& key) const {
    return last_use_.contains(key);
}

}  // namespace k3x
