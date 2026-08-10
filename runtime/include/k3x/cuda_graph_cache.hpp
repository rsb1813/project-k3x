// CUDA Graph ordered identity와 bounded LRU index 계약을 정의합니다.
#pragma once

#include <cstddef>
#include <cstdint>
#include <map>
#include <optional>
#include <vector>

namespace k3x {

struct CudaGraphKey {
    std::vector<std::uint64_t> words;

    bool operator==(const CudaGraphKey&) const = default;
    bool operator<(const CudaGraphKey& other) const noexcept {
        return words < other.words;
    }
};

struct CudaGraphCacheDecision {
    bool hit{};
    std::optional<CudaGraphKey> evicted;
};

class BoundedCudaGraphIndex {
public:
    explicit BoundedCudaGraphIndex(std::size_t capacity);

    CudaGraphCacheDecision touch(const CudaGraphKey& key);
    bool erase(const CudaGraphKey& key);
    void clear() noexcept;
    bool contains(const CudaGraphKey& key) const;
    std::size_t size() const noexcept { return last_use_.size(); }
    std::size_t peak_size() const noexcept { return peak_size_; }

private:
    std::size_t capacity_{};
    std::uint64_t sequence_{};
    std::size_t peak_size_{};
    std::map<CudaGraphKey, std::uint64_t> last_use_;
};

}  // namespace k3x
