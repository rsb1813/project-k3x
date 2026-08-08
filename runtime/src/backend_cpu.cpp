// 기존 double 누산 dense와 native MXFP4 연산을 exact CPU backend로 제공합니다.
#include "k3x/backend.hpp"

#include "k3x/ops.hpp"

#include <chrono>
#include <utility>

namespace k3x {
namespace {

class CpuBackend final : public ComputeBackend {
public:
    explicit CpuBackend(Profiler* profiler) : profiler_(profiler) {}

    BackendKind kind() const noexcept override { return BackendKind::cpu; }
    const BackendOptions& options() const noexcept override { return options_; }
    BackendRuntimeStats runtime_stats() const noexcept override { return {}; }

    Result<std::vector<float>> dense_matvec(
        std::span<const float> input, std::span<const float> weight,
        std::size_t rows, std::size_t cols, std::uint32_t layer,
        ProfilePhase phase) override {
        const auto start = std::chrono::steady_clock::now();
        if (input.size() != cols || rows > weight.size() ||
            cols != 0 && rows > weight.size() / cols ||
            weight.size() != rows * cols) {
            record(phase, ProfileOperation::dense_matvec, NumericPrecision::fp32,
                   layer, start, weight.size_bytes(), false);
            return Result<std::vector<float>>::failure(ErrorCode::invalid_extent);
        }

        std::vector<float> output(rows);
        for (std::size_t row = 0; row < rows; ++row) {
            double sum = 0.0;
            for (std::size_t column = 0; column < cols; ++column) {
                sum += static_cast<double>(weight[row * cols + column]) *
                       input[column];
            }
            output[row] = static_cast<float>(sum);
        }
        record(phase, ProfileOperation::dense_matvec, NumericPrecision::fp32,
               layer, start, weight.size_bytes(), true);
        return Result<std::vector<float>>::success(std::move(output));
    }

    Result<std::vector<float>> mxfp4_matvec(
        std::span<const float> input, std::span<const std::byte> packed,
        std::span<const std::byte> scales, std::size_t rows,
        std::size_t cols, std::size_t group_size,
        std::uint32_t layer, ProfilePhase phase) override {
        const auto start = std::chrono::steady_clock::now();
        auto result = mxfp4_matmul(input, packed, scales, rows, cols, group_size);
        record(phase, ProfileOperation::mxfp4_matvec,
               NumericPrecision::mxfp4_e2m1_e8m0, layer, start,
               packed.size_bytes() + scales.size_bytes(), static_cast<bool>(result));
        return result;
    }

    BackendMemoryStats memory_stats() const noexcept override { return {}; }
    std::string_view device_name() const noexcept override { return "CPU"; }

private:
    void record(ProfilePhase phase, ProfileOperation operation,
                NumericPrecision precision, std::uint32_t layer,
                std::chrono::steady_clock::time_point start,
                std::uint64_t logical_bytes, bool success) {
        if (!profiler_) return;
        const auto wall_nanoseconds =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - start)
                .count();
        profiler_->record({phase, operation, precision, layer,
                           static_cast<std::uint64_t>(wall_nanoseconds), 0,
                           logical_bytes, 0, success});
    }

    BackendOptions options_{};
    Profiler* profiler_;
};

}  // namespace

std::unique_ptr<ComputeBackend> make_cpu_backend(Profiler* profiler) {
    return std::make_unique<CpuBackend>(profiler);
}

}  // namespace k3x
