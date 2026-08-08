// RTX 5080용 CUDA stream과 cuBLASLt handle의 수명 및 device capability를 관리합니다.
#include "k3x/backend.hpp"

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <memory>
#include <span>
#include <string>
#include <utility>
#include <vector>

namespace k3x {
namespace {

class StreamOwner {
public:
    ~StreamOwner() {
        if (stream_) cudaStreamDestroy(stream_);
    }
    StreamOwner(const StreamOwner&) = delete;
    StreamOwner& operator=(const StreamOwner&) = delete;
    StreamOwner() = default;

    cudaStream_t* out() { return &stream_; }
    cudaStream_t release() { return std::exchange(stream_, nullptr); }

private:
    cudaStream_t stream_{};
};

class CublasLtOwner {
public:
    ~CublasLtOwner() {
        if (handle_) cublasLtDestroy(handle_);
    }
    CublasLtOwner(const CublasLtOwner&) = delete;
    CublasLtOwner& operator=(const CublasLtOwner&) = delete;
    CublasLtOwner() = default;

    cublasLtHandle_t* out() { return &handle_; }
    cublasLtHandle_t release() { return std::exchange(handle_, nullptr); }

private:
    cublasLtHandle_t handle_{};
};

class DeviceBuffer {
public:
    DeviceBuffer() = default;
    ~DeviceBuffer() { reset(); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;

    cudaError_t allocate(std::size_t bytes, BackendMemoryStats* stats) {
        if (bytes == 0) return cudaSuccess;
        const auto status = cudaMalloc(&pointer_, bytes);
        if (status != cudaSuccess) return status;
        bytes_ = bytes;
        stats_ = stats;
        stats_->current_device_bytes += bytes_;
        stats_->peak_device_bytes =
            std::max(stats_->peak_device_bytes, stats_->current_device_bytes);
        return cudaSuccess;
    }

    void* get() const noexcept { return pointer_; }

private:
    void reset() noexcept {
        if (!pointer_) return;
        cudaFree(pointer_);
        stats_->current_device_bytes -= bytes_;
        pointer_ = nullptr;
        bytes_ = 0;
    }

    void* pointer_{};
    std::size_t bytes_{};
    BackendMemoryStats* stats_{};
};

class MatmulDescOwner {
public:
    ~MatmulDescOwner() {
        if (descriptor_) cublasLtMatmulDescDestroy(descriptor_);
    }
    MatmulDescOwner(const MatmulDescOwner&) = delete;
    MatmulDescOwner& operator=(const MatmulDescOwner&) = delete;
    MatmulDescOwner() = default;
    cublasLtMatmulDesc_t* out() { return &descriptor_; }
    cublasLtMatmulDesc_t get() const { return descriptor_; }

private:
    cublasLtMatmulDesc_t descriptor_{};
};

class MatrixLayoutOwner {
public:
    ~MatrixLayoutOwner() {
        if (layout_) cublasLtMatrixLayoutDestroy(layout_);
    }
    MatrixLayoutOwner(const MatrixLayoutOwner&) = delete;
    MatrixLayoutOwner& operator=(const MatrixLayoutOwner&) = delete;
    MatrixLayoutOwner() = default;
    cublasLtMatrixLayout_t* out() { return &layout_; }
    cublasLtMatrixLayout_t get() const { return layout_; }

private:
    cublasLtMatrixLayout_t layout_{};
};

class MatmulPreferenceOwner {
public:
    ~MatmulPreferenceOwner() {
        if (preference_) cublasLtMatmulPreferenceDestroy(preference_);
    }
    MatmulPreferenceOwner(const MatmulPreferenceOwner&) = delete;
    MatmulPreferenceOwner& operator=(const MatmulPreferenceOwner&) = delete;
    MatmulPreferenceOwner() = default;
    cublasLtMatmulPreference_t* out() { return &preference_; }
    cublasLtMatmulPreference_t get() const { return preference_; }

private:
    cublasLtMatmulPreference_t preference_{};
};

class EventOwner {
public:
    ~EventOwner() {
        if (event_) cudaEventDestroy(event_);
    }
    EventOwner(const EventOwner&) = delete;
    EventOwner& operator=(const EventOwner&) = delete;
    EventOwner() = default;
    cudaEvent_t* out() { return &event_; }
    cudaEvent_t get() const { return event_; }

private:
    cudaEvent_t event_{};
};

class CudaBackend final : public ComputeBackend {
public:
    CudaBackend(BackendOptions options, Profiler* profiler,
                cudaStream_t stream, cublasLtHandle_t handle,
                std::string device_name)
        : options_(options), profiler_(profiler), stream_(stream), handle_(handle),
          device_name_(std::move(device_name)) {}

    ~CudaBackend() override {
        if (handle_) cublasLtDestroy(handle_);
        if (stream_) cudaStreamDestroy(stream_);
    }

    BackendKind kind() const noexcept override { return options_.kind; }

    Result<std::vector<float>> dense_matvec(
        std::span<const float> input, std::span<const float> weight,
        std::size_t rows, std::size_t cols, std::uint32_t layer,
        ProfilePhase phase) override {
        const auto operation_start = std::chrono::steady_clock::now();
        const auto precision = numeric_precision();
        if (input.size() != cols || rows > weight.size() ||
            (cols != 0 && rows > weight.size() / cols) ||
            weight.size() != rows * cols) {
            record(phase, ProfileOperation::dense_matvec, precision, layer,
                   operation_start, weight.size_bytes(), 0, 0, false);
            return Result<std::vector<float>>::failure(ErrorCode::invalid_extent);
        }
        std::vector<__nv_bfloat16> bf16_input;
        std::vector<__nv_bfloat16> bf16_weight;
        const auto logical_weight_bytes = weight.size_bytes();
        const void* host_input = input.data();
        const void* host_weight = weight.data();
        cudaDataType_t input_type = CUDA_R_32F;
        cudaDataType_t weight_type = CUDA_R_32F;
        std::size_t input_bytes = input.size_bytes();
        std::size_t weight_bytes = weight.size_bytes();
        if (options_.dense_precision == DensePrecision::bf16_rounded) {
            bf16_input.reserve(input.size());
            bf16_weight.reserve(weight.size());
            for (const auto value : input) {
                bf16_input.push_back(__float2bfloat16_rn(value));
            }
            for (const auto value : weight) {
                bf16_weight.push_back(__float2bfloat16_rn(value));
            }
            host_input = bf16_input.data();
            host_weight = bf16_weight.data();
            input_type = CUDA_R_16BF;
            weight_type = CUDA_R_16BF;
            input_bytes = bf16_input.size() * sizeof(__nv_bfloat16);
            weight_bytes = bf16_weight.size() * sizeof(__nv_bfloat16);
        }

        const auto output_bytes = rows * sizeof(float);
        DeviceBuffer device_input;
        DeviceBuffer device_weight;
        DeviceBuffer device_output;
        if (device_input.allocate(input_bytes, &memory_stats_) != cudaSuccess ||
            device_weight.allocate(weight_bytes, &memory_stats_) != cudaSuccess ||
            device_output.allocate(output_bytes, &memory_stats_) != cudaSuccess) {
            record(phase, ProfileOperation::dense_matvec, precision, layer,
                   operation_start, logical_weight_bytes, 0, 0, false);
            return Result<std::vector<float>>::failure(
                ErrorCode::backend_unavailable, "CUDA device allocation failed");
        }

        const auto h2d_start = std::chrono::steady_clock::now();
        if (cudaMemcpyAsync(device_input.get(), host_input, input_bytes,
                            cudaMemcpyHostToDevice, stream_) != cudaSuccess ||
            cudaMemcpyAsync(device_weight.get(), host_weight, weight_bytes,
                            cudaMemcpyHostToDevice, stream_) != cudaSuccess) {
            record(phase, ProfileOperation::host_to_device, precision, layer,
                   h2d_start, 0, input_bytes + weight_bytes, 0, false);
            return Result<std::vector<float>>::failure(
                ErrorCode::backend_unavailable, "CUDA host-to-device copy failed");
        }
        record(phase, ProfileOperation::host_to_device, precision, layer,
               h2d_start, 0, input_bytes + weight_bytes, 0, true);

        MatmulDescOwner operation;
        MatrixLayoutOwner input_layout;
        MatrixLayoutOwner weight_layout;
        MatrixLayoutOwner output_layout;
        MatmulPreferenceOwner preference;
        if (cublasLtMatmulDescCreate(operation.out(), CUBLAS_COMPUTE_32F,
                                     CUDA_R_32F) != CUBLAS_STATUS_SUCCESS ||
            !create_row_major_layout(weight_layout, weight_type, rows, cols,
                                     cols) ||
            !create_row_major_layout(input_layout, input_type, cols, 1, 1) ||
            !create_row_major_layout(output_layout, CUDA_R_32F, rows, 1, 1) ||
            cublasLtMatmulPreferenceCreate(preference.out()) !=
                CUBLAS_STATUS_SUCCESS) {
            record(phase, ProfileOperation::dense_matvec, precision, layer,
                   operation_start, logical_weight_bytes, 0, 0, false);
            return Result<std::vector<float>>::failure(
                ErrorCode::backend_unavailable,
                "cuBLASLt descriptor creation failed");
        }

        cublasLtMatmulHeuristicResult_t heuristic{};
        int returned_results = 0;
        if (cublasLtMatmulAlgoGetHeuristic(
                handle_, operation.get(), weight_layout.get(), input_layout.get(),
                output_layout.get(), output_layout.get(), preference.get(), 1,
                &heuristic, &returned_results) != CUBLAS_STATUS_SUCCESS ||
            returned_results != 1 || heuristic.state != CUBLAS_STATUS_SUCCESS) {
            record(phase, ProfileOperation::dense_matvec, precision, layer,
                   operation_start, logical_weight_bytes, 0, 0, false);
            return Result<std::vector<float>>::failure(
                ErrorCode::backend_unavailable,
                "cuBLASLt found no zero-workspace dense algorithm");
        }

        EventOwner event_start;
        EventOwner event_end;
        if (cudaEventCreate(event_start.out()) != cudaSuccess ||
            cudaEventCreate(event_end.out()) != cudaSuccess ||
            cudaEventRecord(event_start.get(), stream_) != cudaSuccess) {
            record(phase, ProfileOperation::dense_matvec, precision, layer,
                   operation_start, logical_weight_bytes, 0, 0, false);
            return Result<std::vector<float>>::failure(
                ErrorCode::backend_unavailable, "CUDA event creation failed");
        }

        constexpr float alpha = 1.0F;
        constexpr float beta = 0.0F;
        const auto matmul_status = cublasLtMatmul(
            handle_, operation.get(), &alpha, device_weight.get(),
            weight_layout.get(), device_input.get(), input_layout.get(), &beta,
            device_output.get(), output_layout.get(), device_output.get(),
            output_layout.get(), &heuristic.algo, nullptr, 0, stream_);
        if (matmul_status != CUBLAS_STATUS_SUCCESS ||
            cudaEventRecord(event_end.get(), stream_) != cudaSuccess) {
            record(phase, ProfileOperation::dense_matvec, precision, layer,
                   operation_start, logical_weight_bytes, 0, 0, false);
            return Result<std::vector<float>>::failure(
                ErrorCode::backend_unavailable, "cuBLASLt dense matvec failed");
        }

        std::vector<float> output(rows);
        const auto d2h_start = std::chrono::steady_clock::now();
        if (cudaMemcpyAsync(output.data(), device_output.get(), output_bytes,
                            cudaMemcpyDeviceToHost, stream_) != cudaSuccess ||
            cudaStreamSynchronize(stream_) != cudaSuccess) {
            record(phase, ProfileOperation::device_to_host, precision, layer,
                   d2h_start, 0, output_bytes, 0, false);
            record(phase, ProfileOperation::dense_matvec, precision, layer,
                   operation_start, logical_weight_bytes, 0, 0, false);
            return Result<std::vector<float>>::failure(
                ErrorCode::backend_unavailable, "CUDA device-to-host copy failed");
        }
        record(phase, ProfileOperation::device_to_host, precision, layer,
               d2h_start, 0, output_bytes, 0, true);

        float elapsed_milliseconds = 0.0F;
        if (cudaEventElapsedTime(&elapsed_milliseconds, event_start.get(),
                                 event_end.get()) != cudaSuccess) {
            record(phase, ProfileOperation::dense_matvec, precision, layer,
                   operation_start, logical_weight_bytes, 0, 0, false);
            return Result<std::vector<float>>::failure(
                ErrorCode::backend_unavailable, "CUDA event timing failed");
        }
        const auto device_nanoseconds = static_cast<std::uint64_t>(
            std::llround(static_cast<double>(elapsed_milliseconds) * 1.0e6));
        record(phase, ProfileOperation::dense_matvec, precision, layer,
               operation_start, logical_weight_bytes, 0, device_nanoseconds, true);
        return Result<std::vector<float>>::success(std::move(output));
    }

    Result<std::vector<float>> mxfp4_matvec(
        std::span<const float>, std::span<const std::byte>,
        std::span<const std::byte>, std::size_t, std::size_t,
        std::size_t, std::uint32_t, ProfilePhase) override {
        return Result<std::vector<float>>::failure(
            ErrorCode::backend_unavailable,
            "CUDA MXFP4 matvec is not implemented in the backend shell");
    }

    BackendMemoryStats memory_stats() const noexcept override { return memory_stats_; }
    std::string_view device_name() const noexcept override { return device_name_; }

private:
    NumericPrecision numeric_precision() const noexcept {
        return options_.dense_precision == DensePrecision::fp32
                   ? NumericPrecision::fp32
                   : NumericPrecision::bf16_rounded;
    }

    static bool create_row_major_layout(MatrixLayoutOwner& owner,
                                        cudaDataType_t type,
                                        std::size_t rows, std::size_t cols,
                                        std::int64_t leading_dimension) {
        if (cublasLtMatrixLayoutCreate(owner.out(), type, rows, cols,
                                       leading_dimension) !=
            CUBLAS_STATUS_SUCCESS) {
            return false;
        }
        constexpr cublasLtOrder_t order = CUBLASLT_ORDER_ROW;
        return cublasLtMatrixLayoutSetAttribute(
                   owner.get(), CUBLASLT_MATRIX_LAYOUT_ORDER, &order,
                   sizeof(order)) == CUBLAS_STATUS_SUCCESS;
    }

    void record(ProfilePhase phase, ProfileOperation operation,
                NumericPrecision precision, std::uint32_t layer,
                std::chrono::steady_clock::time_point start,
                std::uint64_t logical_bytes, std::uint64_t transfer_bytes,
                std::uint64_t device_nanoseconds, bool success) {
        if (!profiler_) return;
        const auto wall_nanoseconds =
            std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now() - start)
                .count();
        profiler_->record({phase, operation, precision, layer,
                           static_cast<std::uint64_t>(wall_nanoseconds),
                           device_nanoseconds, logical_bytes, transfer_bytes,
                           success});
    }

    BackendOptions options_;
    Profiler* profiler_;
    cudaStream_t stream_{};
    cublasLtHandle_t handle_{};
    std::string device_name_;
    BackendMemoryStats memory_stats_{};
};

Result<std::unique_ptr<ComputeBackend>> cuda_failure(
    ErrorCode code, const char* message) {
    return Result<std::unique_ptr<ComputeBackend>>::failure(code, message);
}

}  // namespace

Result<std::unique_ptr<ComputeBackend>> make_cuda_backend(
    const BackendOptions& options, Profiler* profiler) {
    if (options.kind == BackendKind::cpu) {
        return cuda_failure(ErrorCode::backend_unavailable,
                            "CUDA factory requires a CUDA backend kind");
    }

    int device = -1;
    if (cudaGetDevice(&device) != cudaSuccess) {
        return cuda_failure(ErrorCode::backend_unavailable,
                            "CUDA device selection failed");
    }

    int major = 0;
    int minor = 0;
    if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                               device) != cudaSuccess ||
        cudaDeviceGetAttribute(&minor, cudaDevAttrComputeCapabilityMinor,
                               device) != cudaSuccess) {
        return cuda_failure(ErrorCode::backend_unavailable,
                            "CUDA capability query failed");
    }
    if (major < 12) {
        return cuda_failure(ErrorCode::unsupported_architecture,
                            "CUDA backend requires compute capability 12.0 or newer");
    }

    cudaDeviceProp properties{};
    if (cudaGetDeviceProperties(&properties, device) != cudaSuccess) {
        return cuda_failure(ErrorCode::backend_unavailable,
                            "CUDA device property query failed");
    }

    StreamOwner stream;
    if (cudaStreamCreateWithFlags(stream.out(), cudaStreamNonBlocking) != cudaSuccess) {
        return cuda_failure(ErrorCode::backend_unavailable,
                            "CUDA stream creation failed");
    }

    CublasLtOwner handle;
    if (cublasLtCreate(handle.out()) != CUBLAS_STATUS_SUCCESS) {
        return cuda_failure(ErrorCode::backend_unavailable,
                            "cuBLASLt handle creation failed");
    }

    std::unique_ptr<ComputeBackend> backend = std::make_unique<CudaBackend>(
        options, profiler, stream.release(), handle.release(), properties.name);
    return Result<std::unique_ptr<ComputeBackend>>::success(std::move(backend));
}

}  // namespace k3x
