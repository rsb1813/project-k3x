// dense와 native MXFP4 연산을 실행하는 명시적 compute backend 계약을 정의합니다.
#pragma once

#include "k3x/profile.hpp"
#include "k3x/status.hpp"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <span>
#include <string_view>
#include <vector>

namespace k3x {

enum class BackendKind { cpu, cuda_dense, cuda_custom };
enum class DensePrecision { fp32, bf16_rounded };
enum class CudaAllocationMode { per_operation, reused };
enum class CudaWeightMode { transient, resident };
enum class CudaBatchingMode { scalar, grouped };

struct BackendOptions {
    BackendKind kind{BackendKind::cpu};
    DensePrecision dense_precision{DensePrecision::fp32};
    CudaAllocationMode cuda_allocation{CudaAllocationMode::per_operation};
    CudaWeightMode cuda_weights{CudaWeightMode::transient};
    CudaBatchingMode cuda_batching{CudaBatchingMode::scalar};
    std::uint64_t cuda_resident_bytes{};
};

struct BackendMemoryStats {
    std::uint64_t current_device_bytes{};
    std::uint64_t peak_device_bytes{};
};

struct BackendRuntimeStats {
    std::uint64_t device_allocation_count{};
    std::uint64_t device_free_count{};
    std::uint64_t stream_synchronization_count{};
    std::uint64_t weight_cache_hits{};
    std::uint64_t weight_cache_misses{};
    std::uint64_t weight_cache_bypasses{};
    std::uint64_t resident_weight_bytes{};
    std::uint64_t peak_resident_weight_bytes{};
    std::uint64_t scratch_bytes{};
    std::uint64_t peak_scratch_bytes{};
    std::uint64_t weight_h2d_bytes{};
    std::uint64_t activation_h2d_bytes{};
    std::uint64_t grouped_projection_calls{};
    std::uint64_t grouped_projection_members{};
};

class ComputeBackend {
public:
    virtual ~ComputeBackend() = default;
    virtual BackendKind kind() const noexcept = 0;
    virtual const BackendOptions& options() const noexcept = 0;
    virtual BackendRuntimeStats runtime_stats() const noexcept = 0;
    virtual Result<std::vector<float>> dense_matvec(
        std::span<const float> input, std::span<const float> weight,
        std::size_t rows, std::size_t cols, std::uint32_t layer,
        ProfilePhase phase) = 0;
    virtual Result<std::vector<float>> mxfp4_matvec(
        std::span<const float> input, std::span<const std::byte> packed,
        std::span<const std::byte> scales, std::size_t rows,
        std::size_t cols, std::size_t group_size,
        std::uint32_t layer, ProfilePhase phase) = 0;
    virtual BackendMemoryStats memory_stats() const noexcept = 0;
    virtual std::string_view device_name() const noexcept = 0;
};

std::unique_ptr<ComputeBackend> make_cpu_backend(Profiler* profiler = nullptr);
Result<std::unique_ptr<ComputeBackend>> make_cuda_backend(
    const BackendOptions& options, Profiler* profiler = nullptr);

}  // namespace k3x
