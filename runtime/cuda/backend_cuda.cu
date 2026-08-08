// RTX 5080용 CUDA stream과 cuBLASLt handle의 수명 및 device capability를 관리합니다.
#include "k3x/backend.hpp"

#include <cublasLt.h>
#include <cuda_runtime_api.h>

#include <memory>
#include <string>
#include <utility>

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
        std::span<const float>, std::span<const float>, std::size_t,
        std::size_t, std::uint32_t, ProfilePhase) override {
        return Result<std::vector<float>>::failure(
            ErrorCode::backend_unavailable,
            "CUDA dense matvec is not implemented in the backend shell");
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
