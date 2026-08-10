// CUDA Graph ordered identity와 bounded LRU index를 검증합니다.
#include "k3x/cuda_graph_cache.hpp"

#include <stdexcept>

int main() {
    bool rejected_zero = false;
    try {
        static_cast<void>(k3x::BoundedCudaGraphIndex(0));
    } catch (const std::invalid_argument&) {
        rejected_zero = true;
    }
    if (!rejected_zero) return 1;

    k3x::BoundedCudaGraphIndex index(2);
    const k3x::CudaGraphKey a{{1, 10}};
    const k3x::CudaGraphKey b{{2, 20}};
    const k3x::CudaGraphKey c{{3, 30}};
    if (index.touch(a).hit || index.touch(b).hit) return 2;
    if (!index.touch(a).hit) return 3;
    const auto miss = index.touch(c);
    if (miss.hit || !miss.evicted || *miss.evicted != b) return 4;
    if (!index.contains(a) || !index.contains(c) || index.contains(b)) {
        return 5;
    }
    if (index.size() != 2 || index.peak_size() != 2) return 6;
    if (!index.erase(a) || index.erase(a) || index.contains(a)) return 7;
    index.clear();
    return index.size() == 0 ? 0 : 8;
}
