// RTX 5080용 CUDA stream과 cuBLASLt handle의 수명 및 device capability를 관리합니다.
#include "k3x/backend.hpp"
#include "k3x/ops.hpp"

#include "device_memory.cuh"
#include "mxfp4.cuh"
#include "resident_weights.cuh"
#include "situ.cuh"

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <tuple>
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
    cudaError_t ensure() {
        return event_ ? cudaSuccess : cudaEventCreate(&event_);
    }
    cudaEvent_t get() const { return event_; }

private:
    cudaEvent_t event_{};
};

struct DensePlan {
    MatmulDescOwner operation;
    MatrixLayoutOwner input_layout;
    MatrixLayoutOwner weight_layout;
    MatrixLayoutOwner output_layout;
    MatmulPreferenceOwner preference;
    cublasLtMatmulHeuristicResult_t heuristic{};
};

class CudaBackend final : public ComputeBackend {
public:
    CudaBackend(BackendOptions options, Profiler* profiler,
                cudaStream_t stream, cublasLtHandle_t handle,
                std::string device_name)
        : options_(options), profiler_(profiler), stream_(stream), handle_(handle),
          device_name_(std::move(device_name)),
          dense_input_scratch_(&memory_stats_, &runtime_stats_),
          dense_weight_scratch_(&memory_stats_, &runtime_stats_),
          dense_output_scratch_(&memory_stats_, &runtime_stats_),
          dense_group_output_scratch_(&memory_stats_, &runtime_stats_),
          ffn_input_scratch_(&memory_stats_, &runtime_stats_),
          ffn_weight_scratch_(&memory_stats_, &runtime_stats_),
          ffn_gate_scratch_(&memory_stats_, &runtime_stats_),
          ffn_up_scratch_(&memory_stats_, &runtime_stats_),
          ffn_activation_scratch_(&memory_stats_, &runtime_stats_),
          ffn_output_scratch_(&memory_stats_, &runtime_stats_),
          mxfp4_input_scratch_(&memory_stats_, &runtime_stats_),
          mxfp4_packed_scratch_(&memory_stats_, &runtime_stats_),
          mxfp4_scales_scratch_(&memory_stats_, &runtime_stats_),
          mxfp4_output_scratch_(&memory_stats_, &runtime_stats_),
          mxfp4_group_output_scratch_(&memory_stats_, &runtime_stats_) {
        if (options_.cuda_weights == CudaWeightMode::resident) {
            resident_weights_ = std::make_unique<cuda::ResidentWeightTable>(
                options_.cuda_resident_bytes, &memory_stats_, &runtime_stats_,
                stream_);
        }
    }

    ~CudaBackend() override {
        if (handle_) cublasLtDestroy(handle_);
        if (stream_) cudaStreamDestroy(stream_);
    }

    BackendKind kind() const noexcept override { return options_.kind; }
    const BackendOptions& options() const noexcept override { return options_; }
    BackendRuntimeStats runtime_stats() const noexcept override {
        return runtime_stats_;
    }

    Result<std::vector<float>> dense_matvec(
        std::span<const float> input, DenseWeightView weight_view,
        std::uint32_t layer, ProfilePhase phase) override {
        const auto weight = weight_view.values;
        const auto rows = weight_view.rows;
        const auto cols = weight_view.cols;
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
        bool has_resident_weight = false;
        std::uint64_t weight_transfer_bytes = weight_bytes;
        const void* resident_weight = nullptr;
        if (resident_weights_) {
            const auto representation =
                options_.dense_precision == DensePrecision::fp32
                    ? cuda::WeightRepresentation::dense_fp32
                    : cuda::WeightRepresentation::dense_bf16;
            const auto host_bytes = std::span(
                static_cast<const std::byte*>(host_weight), weight_bytes);
            const auto acquisition = resident_weights_->acquire(
                {weight_view.tensor_id, representation, rows, cols, 0},
                host_bytes, {});
            if (!acquisition) {
                record(phase, ProfileOperation::dense_matvec, precision, layer,
                       operation_start, logical_weight_bytes, 0, 0, false);
                return Result<std::vector<float>>::failure(
                    acquisition.error(), acquisition.message());
            }
            has_resident_weight =
                acquisition.value().disposition !=
                cuda::ResidentDisposition::bypass;
            resident_weight = acquisition.value().primary;
            weight_transfer_bytes = acquisition.value().uploaded_bytes;
            if (!has_resident_weight) weight_transfer_bytes = weight_bytes;
        }
        cuda::DeviceAllocation local_input(&memory_stats_, &runtime_stats_);
        cuda::DeviceAllocation local_weight(&memory_stats_, &runtime_stats_);
        cuda::DeviceAllocation local_output(&memory_stats_, &runtime_stats_);
        void* device_input = nullptr;
        void* device_weight = nullptr;
        void* device_output = nullptr;
        if (options_.cuda_allocation == CudaAllocationMode::reused) {
            if (dense_input_scratch_.reserve(input_bytes) != cudaSuccess ||
                (!has_resident_weight &&
                 dense_weight_scratch_.reserve(weight_bytes) != cudaSuccess) ||
                dense_output_scratch_.reserve(output_bytes) != cudaSuccess) {
                record(phase, ProfileOperation::dense_matvec, precision, layer,
                       operation_start, logical_weight_bytes, 0, 0, false);
                return Result<std::vector<float>>::failure(
                    ErrorCode::backend_unavailable,
                    "CUDA reusable device allocation failed");
            }
            device_input = dense_input_scratch_.get();
            device_weight = has_resident_weight
                                ? const_cast<void*>(resident_weight)
                                : dense_weight_scratch_.get();
            device_output = dense_output_scratch_.get();
        } else if (local_input.allocate(input_bytes) != cudaSuccess ||
                   (!has_resident_weight &&
                    local_weight.allocate(weight_bytes) != cudaSuccess) ||
                   local_output.allocate(output_bytes) != cudaSuccess) {
            record(phase, ProfileOperation::dense_matvec, precision, layer,
                   operation_start, logical_weight_bytes, 0, 0, false);
            return Result<std::vector<float>>::failure(
                ErrorCode::backend_unavailable, "CUDA device allocation failed");
        } else {
            device_input = local_input.get();
            device_weight = has_resident_weight
                                ? const_cast<void*>(resident_weight)
                                : local_weight.get();
            device_output = local_output.get();
        }

        const auto h2d_start = std::chrono::steady_clock::now();
        if (cudaMemcpyAsync(device_input, host_input, input_bytes,
                            cudaMemcpyHostToDevice, stream_) != cudaSuccess ||
            (!has_resident_weight &&
             cudaMemcpyAsync(device_weight, host_weight, weight_bytes,
                             cudaMemcpyHostToDevice, stream_) != cudaSuccess)) {
            record(phase, ProfileOperation::host_to_device, precision, layer,
                   h2d_start, 0, input_bytes + weight_transfer_bytes, 0, false);
            return Result<std::vector<float>>::failure(
                ErrorCode::backend_unavailable, "CUDA host-to-device copy failed");
        }
        record(phase, ProfileOperation::activation_host_to_device, precision,
               layer, h2d_start, 0, input_bytes, 0, true);
        record(phase, ProfileOperation::weight_host_to_device, precision, layer,
               h2d_start, 0, weight_transfer_bytes, 0, true);
        runtime_stats_.activation_h2d_bytes += input_bytes;
        runtime_stats_.weight_h2d_bytes += weight_transfer_bytes;

        DensePlan local_plan;
        DensePlan* plan = &local_plan;
        if (options_.cuda_allocation == CudaAllocationMode::reused) {
            const DensePlanKey key{rows, cols, static_cast<int>(input_type),
                                   static_cast<int>(weight_type)};
            const auto found = dense_plans_.find(key);
            if (found != dense_plans_.end()) {
                plan = found->second.get();
            } else {
                auto candidate = std::make_unique<DensePlan>();
                if (!initialize_dense_plan(
                        *candidate, rows, cols, input_type, weight_type)) {
                    record(phase, ProfileOperation::dense_matvec, precision,
                           layer, operation_start, logical_weight_bytes, 0, 0,
                           false);
                    return Result<std::vector<float>>::failure(
                        ErrorCode::backend_unavailable,
                        "cuBLASLt plan creation failed");
                }
                plan = candidate.get();
                dense_plans_.emplace(key, std::move(candidate));
            }
        } else if (!initialize_dense_plan(
                       local_plan, rows, cols, input_type, weight_type)) {
            record(phase, ProfileOperation::dense_matvec, precision, layer,
                   operation_start, logical_weight_bytes, 0, 0, false);
            return Result<std::vector<float>>::failure(
                ErrorCode::backend_unavailable,
                "cuBLASLt plan creation failed");
        }

        EventOwner local_event_start;
        EventOwner local_event_end;
        auto* event_start = &local_event_start;
        auto* event_end = &local_event_end;
        if (options_.cuda_allocation == CudaAllocationMode::reused) {
            event_start = &dense_event_start_;
            event_end = &dense_event_end_;
        }
        if (event_start->ensure() != cudaSuccess ||
            event_end->ensure() != cudaSuccess ||
            cudaEventRecord(event_start->get(), stream_) != cudaSuccess) {
            record(phase, ProfileOperation::dense_matvec, precision, layer,
                   operation_start, logical_weight_bytes, 0, 0, false);
            return Result<std::vector<float>>::failure(
                ErrorCode::backend_unavailable, "CUDA event creation failed");
        }

        constexpr float alpha = 1.0F;
        constexpr float beta = 0.0F;
        const auto matmul_status = cublasLtMatmul(
            handle_, plan->operation.get(), &alpha, device_weight,
            plan->weight_layout.get(), device_input, plan->input_layout.get(),
            &beta, device_output, plan->output_layout.get(), device_output,
            plan->output_layout.get(), &plan->heuristic.algo, nullptr, 0,
            stream_);
        if (matmul_status != CUBLAS_STATUS_SUCCESS ||
            cudaEventRecord(event_end->get(), stream_) != cudaSuccess) {
            record(phase, ProfileOperation::dense_matvec, precision, layer,
                   operation_start, logical_weight_bytes, 0, 0, false);
            return Result<std::vector<float>>::failure(
                ErrorCode::backend_unavailable, "cuBLASLt dense matvec failed");
        }

        std::vector<float> output(rows);
        const auto d2h_start = std::chrono::steady_clock::now();
        if (cudaMemcpyAsync(output.data(), device_output, output_bytes,
                            cudaMemcpyDeviceToHost, stream_) != cudaSuccess ||
            cudaStreamSynchronize(stream_) != cudaSuccess) {
            record(phase, ProfileOperation::device_to_host, precision, layer,
                   d2h_start, 0, output_bytes, 0, false);
            record(phase, ProfileOperation::dense_matvec, precision, layer,
                   operation_start, logical_weight_bytes, 0, 0, false);
            return Result<std::vector<float>>::failure(
                ErrorCode::backend_unavailable, "CUDA device-to-host copy failed");
        }
        ++runtime_stats_.stream_synchronization_count;
        record(phase, ProfileOperation::device_to_host, precision, layer,
               d2h_start, 0, output_bytes, 0, true);

        float elapsed_milliseconds = 0.0F;
        if (cudaEventElapsedTime(&elapsed_milliseconds, event_start->get(),
                                 event_end->get()) != cudaSuccess) {
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
        std::span<const float> input, Mxfp4WeightView weight,
        std::uint32_t layer, ProfilePhase phase) override {
        const auto packed = weight.packed;
        const auto scales = weight.scales;
        const auto rows = weight.rows;
        const auto cols = weight.cols;
        const auto group_size = weight.group_size;
        const auto operation_start = std::chrono::steady_clock::now();
        constexpr auto precision = NumericPrecision::mxfp4_e2m1_e8m0;
        const auto logical_bytes = packed.size_bytes() + scales.size_bytes();
        if (options_.kind == BackendKind::cuda_dense) {
            auto result = k3x::mxfp4_matmul(
                input, packed, scales, rows, cols, group_size);
            record(phase, ProfileOperation::mxfp4_matvec, precision, layer,
                   operation_start, logical_bytes, 0, 0,
                   static_cast<bool>(result));
            return result;
        }
        if (options_.kind != BackendKind::cuda_custom) {
            record(phase, ProfileOperation::mxfp4_matvec, precision, layer,
                   operation_start, logical_bytes, 0, 0, false);
            return Result<std::vector<float>>::failure(
                ErrorCode::backend_unavailable,
                "native K3 MXFP4 requires the cuda_custom backend");
        }
        if (input.size() != cols || rows == 0 || cols == 0 ||
            group_size != 32 || cols % group_size != 0 || cols % 2 != 0 ||
            rows > std::numeric_limits<std::size_t>::max() / cols) {
            record(phase, ProfileOperation::mxfp4_matvec, precision, layer,
                   operation_start, logical_bytes, 0, 0, false);
            return Result<std::vector<float>>::failure(ErrorCode::invalid_mxfp4);
        }
        const auto elements = rows * cols;
        if (packed.size() != elements / 2 ||
            scales.size() != elements / group_size ||
            std::any_of(scales.begin(), scales.end(), [](std::byte scale) {
                return std::to_integer<std::uint8_t>(scale) == 0xFFU;
            })) {
            record(phase, ProfileOperation::mxfp4_matvec, precision, layer,
                   operation_start, logical_bytes, 0, 0, false);
            return Result<std::vector<float>>::failure(ErrorCode::invalid_mxfp4);
        }

        const auto input_bytes = input.size_bytes();
        const auto output_bytes = rows * sizeof(float);
        bool has_resident_weight = false;
        std::uint64_t weight_transfer_bytes = logical_bytes;
        const void* resident_packed = nullptr;
        const void* resident_scales = nullptr;
        if (resident_weights_) {
            const auto acquisition = resident_weights_->acquire(
                {weight.tensor_id, cuda::WeightRepresentation::mxfp4, rows,
                 cols, group_size},
                packed, scales);
            if (!acquisition) {
                record(phase, ProfileOperation::mxfp4_matvec, precision, layer,
                       operation_start, logical_bytes, 0, 0, false);
                return Result<std::vector<float>>::failure(
                    acquisition.error(), acquisition.message());
            }
            has_resident_weight =
                acquisition.value().disposition !=
                cuda::ResidentDisposition::bypass;
            resident_packed = acquisition.value().primary;
            resident_scales = acquisition.value().secondary;
            weight_transfer_bytes = acquisition.value().uploaded_bytes;
            if (!has_resident_weight) weight_transfer_bytes = logical_bytes;
        }
        cuda::DeviceAllocation local_input(&memory_stats_, &runtime_stats_);
        cuda::DeviceAllocation local_packed(&memory_stats_, &runtime_stats_);
        cuda::DeviceAllocation local_scales(&memory_stats_, &runtime_stats_);
        cuda::DeviceAllocation local_output(&memory_stats_, &runtime_stats_);
        void* device_input = nullptr;
        void* device_packed = nullptr;
        void* device_scales = nullptr;
        void* device_output = nullptr;
        if (options_.cuda_allocation == CudaAllocationMode::reused) {
            if (mxfp4_input_scratch_.reserve(input_bytes) != cudaSuccess ||
                (!has_resident_weight &&
                 mxfp4_packed_scratch_.reserve(packed.size_bytes()) !=
                     cudaSuccess) ||
                (!has_resident_weight &&
                 mxfp4_scales_scratch_.reserve(scales.size_bytes()) !=
                     cudaSuccess) ||
                mxfp4_output_scratch_.reserve(output_bytes) != cudaSuccess) {
                record(phase, ProfileOperation::mxfp4_matvec, precision, layer,
                       operation_start, logical_bytes, 0, 0, false);
                return Result<std::vector<float>>::failure(
                    ErrorCode::backend_unavailable,
                    "CUDA reusable device allocation failed");
            }
            device_input = mxfp4_input_scratch_.get();
            device_packed = has_resident_weight
                                ? const_cast<void*>(resident_packed)
                                : mxfp4_packed_scratch_.get();
            device_scales = has_resident_weight
                                ? const_cast<void*>(resident_scales)
                                : mxfp4_scales_scratch_.get();
            device_output = mxfp4_output_scratch_.get();
        } else if (local_input.allocate(input_bytes) != cudaSuccess ||
                   (!has_resident_weight &&
                    local_packed.allocate(packed.size_bytes()) != cudaSuccess) ||
                   (!has_resident_weight &&
                    local_scales.allocate(scales.size_bytes()) != cudaSuccess) ||
                   local_output.allocate(output_bytes) != cudaSuccess) {
            record(phase, ProfileOperation::mxfp4_matvec, precision, layer,
                   operation_start, logical_bytes, 0, 0, false);
            return Result<std::vector<float>>::failure(
                ErrorCode::backend_unavailable, "CUDA device allocation failed");
        } else {
            device_input = local_input.get();
            device_packed = has_resident_weight
                                ? const_cast<void*>(resident_packed)
                                : local_packed.get();
            device_scales = has_resident_weight
                                ? const_cast<void*>(resident_scales)
                                : local_scales.get();
            device_output = local_output.get();
        }

        const auto h2d_start = std::chrono::steady_clock::now();
        if (cudaMemcpyAsync(device_input, input.data(), input_bytes,
                            cudaMemcpyHostToDevice, stream_) != cudaSuccess ||
            (!has_resident_weight &&
             cudaMemcpyAsync(device_packed, packed.data(), packed.size_bytes(),
                             cudaMemcpyHostToDevice, stream_) != cudaSuccess) ||
            (!has_resident_weight &&
             cudaMemcpyAsync(device_scales, scales.data(), scales.size_bytes(),
                             cudaMemcpyHostToDevice, stream_) != cudaSuccess)) {
            record(phase, ProfileOperation::mxfp4_matvec, precision, layer,
                   operation_start, logical_bytes, 0, 0, false);
            return Result<std::vector<float>>::failure(
                ErrorCode::backend_unavailable, "CUDA host-to-device copy failed");
        }
        record(phase, ProfileOperation::activation_host_to_device, precision,
               layer, h2d_start, 0, input_bytes, 0, true);
        record(phase, ProfileOperation::weight_host_to_device, precision, layer,
               h2d_start, 0, weight_transfer_bytes, 0, true);
        runtime_stats_.activation_h2d_bytes += input_bytes;
        runtime_stats_.weight_h2d_bytes += weight_transfer_bytes;

        EventOwner local_event_start;
        EventOwner local_event_end;
        auto* event_start = &local_event_start;
        auto* event_end = &local_event_end;
        if (options_.cuda_allocation == CudaAllocationMode::reused) {
            event_start = &mxfp4_event_start_;
            event_end = &mxfp4_event_end_;
        }
        if (event_start->ensure() != cudaSuccess ||
            event_end->ensure() != cudaSuccess ||
            cudaEventRecord(event_start->get(), stream_) != cudaSuccess ||
            cuda::launch_mxfp4_matvec(
                static_cast<const float*>(device_input),
                static_cast<const std::uint8_t*>(device_packed),
                static_cast<const std::uint8_t*>(device_scales),
                static_cast<float*>(device_output), rows, cols, stream_) !=
                cudaSuccess ||
            cudaEventRecord(event_end->get(), stream_) != cudaSuccess) {
            record(phase, ProfileOperation::mxfp4_matvec, precision, layer,
                   operation_start, logical_bytes, 0, 0, false);
            return Result<std::vector<float>>::failure(
                ErrorCode::backend_unavailable, "CUDA MXFP4 kernel launch failed");
        }

        std::vector<float> output(rows);
        const auto d2h_start = std::chrono::steady_clock::now();
        if (cudaMemcpyAsync(output.data(), device_output, output_bytes,
                            cudaMemcpyDeviceToHost, stream_) != cudaSuccess ||
            cudaStreamSynchronize(stream_) != cudaSuccess) {
            record(phase, ProfileOperation::mxfp4_matvec, precision, layer,
                   operation_start, logical_bytes, 0, 0, false);
            return Result<std::vector<float>>::failure(
                ErrorCode::backend_unavailable,
                "CUDA MXFP4 execution or device-to-host copy failed");
        }
        ++runtime_stats_.stream_synchronization_count;

        float elapsed_milliseconds = 0.0F;
        if (cudaEventElapsedTime(&elapsed_milliseconds, event_start->get(),
                                 event_end->get()) != cudaSuccess) {
            record(phase, ProfileOperation::mxfp4_matvec, precision, layer,
                   operation_start, logical_bytes, 0, 0, false);
            return Result<std::vector<float>>::failure(
                ErrorCode::backend_unavailable, "CUDA event timing failed");
        }
        record(phase, ProfileOperation::device_to_host, precision, layer,
               d2h_start, 0, output_bytes, 0, true);
        const auto device_nanoseconds = static_cast<std::uint64_t>(
            std::llround(static_cast<double>(elapsed_milliseconds) * 1.0e6));
        record(phase, ProfileOperation::mxfp4_matvec, precision, layer,
               operation_start, logical_bytes, 0, device_nanoseconds, true);
        return Result<std::vector<float>>::success(std::move(output));
    }

    Result<std::vector<std::vector<float>>> dense_matvec_group(
        std::span<const float> input, std::span<const DenseWeightView> weights,
        std::uint32_t layer, ProfilePhase phase) override {
        for (const auto& weight : weights) {
            if (!valid_dense(input, weight)) {
                return Result<std::vector<std::vector<float>>>::failure(
                    ErrorCode::invalid_extent);
            }
        }
        if (options_.cuda_batching != CudaBatchingMode::grouped) {
            std::vector<std::vector<float>> outputs;
            outputs.reserve(weights.size());
            for (const auto& weight : weights) {
                auto output = dense_matvec(input, weight, layer, phase);
                if (!output) {
                    return Result<std::vector<std::vector<float>>>::failure(
                        output.error(), output.message());
                }
                outputs.push_back(std::move(output.value()));
            }
            return Result<std::vector<std::vector<float>>>::success(
                std::move(outputs));
        }
        if (weights.empty()) {
            return Result<std::vector<std::vector<float>>>::success({});
        }

        const auto precision = numeric_precision();
        std::vector<__nv_bfloat16> bf16_input;
        const void* host_input = input.data();
        std::size_t input_bytes = input.size_bytes();
        cudaDataType_t input_type = CUDA_R_32F;
        if (options_.dense_precision == DensePrecision::bf16_rounded) {
            bf16_input.reserve(input.size());
            for (const auto value : input) {
                bf16_input.push_back(__float2bfloat16_rn(value));
            }
            host_input = bf16_input.data();
            input_bytes = bf16_input.size() * sizeof(__nv_bfloat16);
            input_type = CUDA_R_16BF;
        }

        struct GroupMember {
            const void* host_weight{};
            std::size_t weight_bytes{};
            void* device_weight{};
            std::size_t output_offset{};
            std::uint64_t transfer_bytes{};
            DensePlan* plan{};
        };
        std::vector<std::vector<__nv_bfloat16>> bf16_weights(weights.size());
        std::vector<GroupMember> members(weights.size());
        std::size_t maximum_weight_bytes = 0;
        std::size_t total_output_bytes = 0;
        std::uint64_t total_weight_transfer = 0;
        const auto weight_type =
            options_.dense_precision == DensePrecision::fp32
                ? CUDA_R_32F
                : CUDA_R_16BF;
        for (std::size_t index = 0; index < weights.size(); ++index) {
            auto& member = members[index];
            const auto& weight = weights[index];
            member.host_weight = weight.values.data();
            member.weight_bytes = weight.values.size_bytes();
            if (options_.dense_precision == DensePrecision::bf16_rounded) {
                auto& converted = bf16_weights[index];
                converted.reserve(weight.values.size());
                for (const auto value : weight.values) {
                    converted.push_back(__float2bfloat16_rn(value));
                }
                member.host_weight = converted.data();
                member.weight_bytes = converted.size() * sizeof(__nv_bfloat16);
            }
            member.output_offset = total_output_bytes;
            total_output_bytes += weight.rows * sizeof(float);
            maximum_weight_bytes =
                std::max(maximum_weight_bytes, member.weight_bytes);
            member.transfer_bytes = member.weight_bytes;
            if (resident_weights_) {
                const auto representation =
                    options_.dense_precision == DensePrecision::fp32
                        ? cuda::WeightRepresentation::dense_fp32
                        : cuda::WeightRepresentation::dense_bf16;
                const auto host_bytes = std::span(
                    static_cast<const std::byte*>(member.host_weight),
                    member.weight_bytes);
                const auto acquisition = resident_weights_->acquire(
                    {weight.tensor_id, representation, weight.rows,
                     weight.cols, 0},
                    host_bytes, {});
                if (!acquisition) {
                    return Result<std::vector<std::vector<float>>>::failure(
                        acquisition.error(), acquisition.message());
                }
                if (acquisition.value().disposition !=
                    cuda::ResidentDisposition::bypass) {
                    member.device_weight =
                        const_cast<void*>(acquisition.value().primary);
                    member.transfer_bytes = acquisition.value().uploaded_bytes;
                }
            }
            total_weight_transfer += member.transfer_bytes;
        }

        cuda::DeviceAllocation local_input(&memory_stats_, &runtime_stats_);
        cuda::DeviceAllocation local_weight(&memory_stats_, &runtime_stats_);
        cuda::DeviceAllocation local_output(&memory_stats_, &runtime_stats_);
        void* device_input = nullptr;
        void* transient_weight = nullptr;
        void* device_output = nullptr;
        if (options_.cuda_allocation == CudaAllocationMode::reused) {
            if (dense_input_scratch_.reserve(input_bytes) != cudaSuccess ||
                dense_weight_scratch_.reserve(maximum_weight_bytes) !=
                    cudaSuccess ||
                dense_group_output_scratch_.reserve(total_output_bytes) !=
                    cudaSuccess) {
                return Result<std::vector<std::vector<float>>>::failure(
                    ErrorCode::backend_unavailable,
                    "CUDA grouped dense allocation failed");
            }
            device_input = dense_input_scratch_.get();
            transient_weight = dense_weight_scratch_.get();
            device_output = dense_group_output_scratch_.get();
        } else if (local_input.allocate(input_bytes) != cudaSuccess ||
                   local_weight.allocate(maximum_weight_bytes) != cudaSuccess ||
                   local_output.allocate(total_output_bytes) != cudaSuccess) {
            return Result<std::vector<std::vector<float>>>::failure(
                ErrorCode::backend_unavailable,
                "CUDA grouped dense allocation failed");
        } else {
            device_input = local_input.get();
            transient_weight = local_weight.get();
            device_output = local_output.get();
        }

        if (cudaMemcpyAsync(device_input, host_input, input_bytes,
                            cudaMemcpyHostToDevice, stream_) != cudaSuccess) {
            return Result<std::vector<std::vector<float>>>::failure(
                ErrorCode::backend_unavailable,
                "CUDA grouped activation upload failed");
        }
        std::vector<std::vector<float>> outputs(weights.size());
        std::vector<std::unique_ptr<DensePlan>> local_plans;
        std::vector<std::unique_ptr<EventOwner>> local_starts;
        std::vector<std::unique_ptr<EventOwner>> local_ends;
        local_plans.reserve(weights.size());
        local_starts.reserve(weights.size());
        local_ends.reserve(weights.size());
        if (options_.cuda_allocation == CudaAllocationMode::reused) {
            while (dense_group_event_starts_.size() < weights.size()) {
                dense_group_event_starts_.push_back(
                    std::make_unique<EventOwner>());
                dense_group_event_ends_.push_back(
                    std::make_unique<EventOwner>());
            }
        }
        const auto d2h_start = std::chrono::steady_clock::now();
        constexpr float alpha = 1.0F;
        constexpr float beta = 0.0F;
        for (std::size_t index = 0; index < weights.size(); ++index) {
            const auto& weight = weights[index];
            auto& member = members[index];
            if (!member.device_weight) {
                member.device_weight = transient_weight;
                if (cudaMemcpyAsync(member.device_weight, member.host_weight,
                                    member.weight_bytes,
                                    cudaMemcpyHostToDevice, stream_) !=
                    cudaSuccess) {
                    return Result<std::vector<std::vector<float>>>::failure(
                        ErrorCode::backend_unavailable,
                        "CUDA grouped weight upload failed");
                }
            }

            const DensePlanKey key{
                weight.rows, weight.cols, static_cast<int>(input_type),
                static_cast<int>(weight_type)};
            if (options_.cuda_allocation == CudaAllocationMode::reused) {
                const auto found = dense_plans_.find(key);
                if (found != dense_plans_.end()) {
                    member.plan = found->second.get();
                } else {
                    auto candidate = std::make_unique<DensePlan>();
                    if (!initialize_dense_plan(*candidate, weight.rows,
                                               weight.cols, input_type,
                                               weight_type)) {
                        return Result<std::vector<std::vector<float>>>::failure(
                            ErrorCode::backend_unavailable,
                            "cuBLASLt grouped plan creation failed");
                    }
                    member.plan = candidate.get();
                    dense_plans_.emplace(key, std::move(candidate));
                }
            } else {
                auto candidate = std::make_unique<DensePlan>();
                if (!initialize_dense_plan(*candidate, weight.rows, weight.cols,
                                           input_type, weight_type)) {
                    return Result<std::vector<std::vector<float>>>::failure(
                        ErrorCode::backend_unavailable,
                        "cuBLASLt grouped plan creation failed");
                }
                member.plan = candidate.get();
                local_plans.push_back(std::move(candidate));
            }

            EventOwner* event_start = nullptr;
            EventOwner* event_end = nullptr;
            if (options_.cuda_allocation == CudaAllocationMode::reused) {
                event_start = dense_group_event_starts_[index].get();
                event_end = dense_group_event_ends_[index].get();
            } else {
                local_starts.push_back(std::make_unique<EventOwner>());
                local_ends.push_back(std::make_unique<EventOwner>());
                event_start = local_starts.back().get();
                event_end = local_ends.back().get();
            }
            if (event_start->ensure() != cudaSuccess ||
                event_end->ensure() != cudaSuccess ||
                cudaEventRecord(event_start->get(), stream_) != cudaSuccess) {
                return Result<std::vector<std::vector<float>>>::failure(
                    ErrorCode::backend_unavailable,
                    "CUDA grouped event creation failed");
            }
            auto* member_output = static_cast<std::byte*>(device_output) +
                                  member.output_offset;
            const auto status = cublasLtMatmul(
                handle_, member.plan->operation.get(), &alpha,
                member.device_weight, member.plan->weight_layout.get(),
                device_input, member.plan->input_layout.get(), &beta,
                member_output, member.plan->output_layout.get(), member_output,
                member.plan->output_layout.get(), &member.plan->heuristic.algo,
                nullptr, 0, stream_);
            if (status != CUBLAS_STATUS_SUCCESS ||
                cudaEventRecord(event_end->get(), stream_) != cudaSuccess) {
                return Result<std::vector<std::vector<float>>>::failure(
                    ErrorCode::backend_unavailable,
                    "cuBLASLt grouped dense matvec failed");
            }
            outputs[index].resize(weight.rows);
            if (cudaMemcpyAsync(outputs[index].data(), member_output,
                                weight.rows * sizeof(float),
                                cudaMemcpyDeviceToHost, stream_) != cudaSuccess) {
                return Result<std::vector<std::vector<float>>>::failure(
                    ErrorCode::backend_unavailable,
                    "CUDA grouped output copy failed");
            }
        }
        if (cudaStreamSynchronize(stream_) != cudaSuccess) {
            return Result<std::vector<std::vector<float>>>::failure(
                ErrorCode::backend_unavailable,
                "CUDA grouped synchronization failed");
        }
        ++runtime_stats_.stream_synchronization_count;
        runtime_stats_.activation_h2d_bytes += input_bytes;
        runtime_stats_.weight_h2d_bytes += total_weight_transfer;
        ++runtime_stats_.grouped_projection_calls;
        runtime_stats_.grouped_projection_members += weights.size();
        record(phase, ProfileOperation::activation_host_to_device, precision,
               layer, d2h_start, 0, input_bytes, 0, true);
        record(phase, ProfileOperation::weight_host_to_device, precision, layer,
               d2h_start, 0, total_weight_transfer, 0, true);
        record(phase, ProfileOperation::device_to_host, precision, layer,
               d2h_start, 0, total_output_bytes, 0, true);
        for (std::size_t index = 0; index < weights.size(); ++index) {
            auto* event_start =
                options_.cuda_allocation == CudaAllocationMode::reused
                    ? dense_group_event_starts_[index].get()
                    : local_starts[index].get();
            auto* event_end =
                options_.cuda_allocation == CudaAllocationMode::reused
                    ? dense_group_event_ends_[index].get()
                    : local_ends[index].get();
            float elapsed_milliseconds = 0.0F;
            if (cudaEventElapsedTime(&elapsed_milliseconds, event_start->get(),
                                     event_end->get()) != cudaSuccess) {
                return Result<std::vector<std::vector<float>>>::failure(
                    ErrorCode::backend_unavailable,
                    "CUDA grouped event timing failed");
            }
            record(phase, ProfileOperation::dense_matvec, precision, layer,
                   d2h_start, weights[index].values.size_bytes(), 0,
                   static_cast<std::uint64_t>(std::llround(
                       static_cast<double>(elapsed_milliseconds) * 1.0e6)),
                   true);
        }
        return Result<std::vector<std::vector<float>>>::success(
            std::move(outputs));

    }

    Result<std::vector<std::vector<float>>> mxfp4_matvec_group(
        std::span<const float> input, std::span<const Mxfp4WeightView> weights,
        std::uint32_t layer, ProfilePhase phase) override {
        for (const auto& weight : weights) {
            if (!valid_mxfp4(input, weight) ||
                (options_.kind == BackendKind::cuda_custom &&
                 weight.group_size != 32)) {
                return Result<std::vector<std::vector<float>>>::failure(
                    ErrorCode::invalid_mxfp4);
            }
        }
        if (options_.kind != BackendKind::cuda_custom ||
            options_.cuda_batching != CudaBatchingMode::grouped) {
            std::vector<std::vector<float>> outputs;
            outputs.reserve(weights.size());
            for (const auto& weight : weights) {
                auto output = mxfp4_matvec(input, weight, layer, phase);
                if (!output) {
                    return Result<std::vector<std::vector<float>>>::failure(
                        output.error(), output.message());
                }
                outputs.push_back(std::move(output.value()));
            }
            return Result<std::vector<std::vector<float>>>::success(
                std::move(outputs));
        }
        if (weights.empty()) {
            return Result<std::vector<std::vector<float>>>::success({});
        }

        struct GroupMember {
            void* device_packed{};
            void* device_scales{};
            std::size_t output_offset{};
            std::uint64_t transfer_bytes{};
        };
        std::vector<GroupMember> members(weights.size());
        std::size_t maximum_packed_bytes = 0;
        std::size_t maximum_scale_bytes = 0;
        std::size_t total_output_bytes = 0;
        std::uint64_t total_weight_transfer = 0;
        for (std::size_t index = 0; index < weights.size(); ++index) {
            const auto& weight = weights[index];
            auto& member = members[index];
            member.output_offset = total_output_bytes;
            total_output_bytes += weight.rows * sizeof(float);
            maximum_packed_bytes =
                std::max(maximum_packed_bytes, weight.packed.size_bytes());
            maximum_scale_bytes =
                std::max(maximum_scale_bytes, weight.scales.size_bytes());
            member.transfer_bytes =
                weight.packed.size_bytes() + weight.scales.size_bytes();
            if (resident_weights_) {
                const auto acquisition = resident_weights_->acquire(
                    {weight.tensor_id, cuda::WeightRepresentation::mxfp4,
                     weight.rows, weight.cols, weight.group_size},
                    weight.packed, weight.scales);
                if (!acquisition) {
                    return Result<std::vector<std::vector<float>>>::failure(
                        acquisition.error(), acquisition.message());
                }
                if (acquisition.value().disposition !=
                    cuda::ResidentDisposition::bypass) {
                    member.device_packed =
                        const_cast<void*>(acquisition.value().primary);
                    member.device_scales =
                        const_cast<void*>(acquisition.value().secondary);
                    member.transfer_bytes = acquisition.value().uploaded_bytes;
                }
            }
            total_weight_transfer += member.transfer_bytes;
        }

        const auto input_bytes = input.size_bytes();
        cuda::DeviceAllocation local_input(&memory_stats_, &runtime_stats_);
        cuda::DeviceAllocation local_packed(&memory_stats_, &runtime_stats_);
        cuda::DeviceAllocation local_scales(&memory_stats_, &runtime_stats_);
        cuda::DeviceAllocation local_output(&memory_stats_, &runtime_stats_);
        void* device_input = nullptr;
        void* transient_packed = nullptr;
        void* transient_scales = nullptr;
        void* device_output = nullptr;
        if (options_.cuda_allocation == CudaAllocationMode::reused) {
            if (mxfp4_input_scratch_.reserve(input_bytes) != cudaSuccess ||
                mxfp4_packed_scratch_.reserve(maximum_packed_bytes) !=
                    cudaSuccess ||
                mxfp4_scales_scratch_.reserve(maximum_scale_bytes) !=
                    cudaSuccess ||
                mxfp4_group_output_scratch_.reserve(total_output_bytes) !=
                    cudaSuccess) {
                return Result<std::vector<std::vector<float>>>::failure(
                    ErrorCode::backend_unavailable,
                    "CUDA grouped MXFP4 allocation failed");
            }
            device_input = mxfp4_input_scratch_.get();
            transient_packed = mxfp4_packed_scratch_.get();
            transient_scales = mxfp4_scales_scratch_.get();
            device_output = mxfp4_group_output_scratch_.get();
        } else if (local_input.allocate(input_bytes) != cudaSuccess ||
                   local_packed.allocate(maximum_packed_bytes) != cudaSuccess ||
                   local_scales.allocate(maximum_scale_bytes) != cudaSuccess ||
                   local_output.allocate(total_output_bytes) != cudaSuccess) {
            return Result<std::vector<std::vector<float>>>::failure(
                ErrorCode::backend_unavailable,
                "CUDA grouped MXFP4 allocation failed");
        } else {
            device_input = local_input.get();
            transient_packed = local_packed.get();
            transient_scales = local_scales.get();
            device_output = local_output.get();
        }
        if (cudaMemcpyAsync(device_input, input.data(), input_bytes,
                            cudaMemcpyHostToDevice, stream_) != cudaSuccess) {
            return Result<std::vector<std::vector<float>>>::failure(
                ErrorCode::backend_unavailable,
                "CUDA grouped MXFP4 activation upload failed");
        }

        std::vector<std::vector<float>> outputs;
        outputs.resize(weights.size());
        std::vector<std::unique_ptr<EventOwner>> local_starts;
        std::vector<std::unique_ptr<EventOwner>> local_ends;
        local_starts.reserve(weights.size());
        local_ends.reserve(weights.size());
        if (options_.cuda_allocation == CudaAllocationMode::reused) {
            while (mxfp4_group_event_starts_.size() < weights.size()) {
                mxfp4_group_event_starts_.push_back(
                    std::make_unique<EventOwner>());
                mxfp4_group_event_ends_.push_back(
                    std::make_unique<EventOwner>());
            }
        }
        const auto group_start = std::chrono::steady_clock::now();
        for (std::size_t index = 0; index < weights.size(); ++index) {
            const auto& weight = weights[index];
            auto& member = members[index];
            if (!member.device_packed) {
                member.device_packed = transient_packed;
                member.device_scales = transient_scales;
                if (cudaMemcpyAsync(
                        member.device_packed, weight.packed.data(),
                        weight.packed.size_bytes(), cudaMemcpyHostToDevice,
                        stream_) != cudaSuccess ||
                    cudaMemcpyAsync(
                        member.device_scales, weight.scales.data(),
                        weight.scales.size_bytes(), cudaMemcpyHostToDevice,
                        stream_) != cudaSuccess) {
                    return Result<std::vector<std::vector<float>>>::failure(
                        ErrorCode::backend_unavailable,
                        "CUDA grouped MXFP4 weight upload failed");
                }
            }
            EventOwner* event_start = nullptr;
            EventOwner* event_end = nullptr;
            if (options_.cuda_allocation == CudaAllocationMode::reused) {
                event_start = mxfp4_group_event_starts_[index].get();
                event_end = mxfp4_group_event_ends_[index].get();
            } else {
                local_starts.push_back(std::make_unique<EventOwner>());
                local_ends.push_back(std::make_unique<EventOwner>());
                event_start = local_starts.back().get();
                event_end = local_ends.back().get();
            }
            auto* member_output = static_cast<std::byte*>(device_output) +
                                  member.output_offset;
            if (event_start->ensure() != cudaSuccess ||
                event_end->ensure() != cudaSuccess ||
                cudaEventRecord(event_start->get(), stream_) != cudaSuccess ||
                cuda::launch_mxfp4_matvec(
                    static_cast<const float*>(device_input),
                    static_cast<const std::uint8_t*>(member.device_packed),
                    static_cast<const std::uint8_t*>(member.device_scales),
                    static_cast<float*>(static_cast<void*>(member_output)),
                    weight.rows, weight.cols, stream_) != cudaSuccess ||
                cudaEventRecord(event_end->get(), stream_) != cudaSuccess) {
                return Result<std::vector<std::vector<float>>>::failure(
                    ErrorCode::backend_unavailable,
                    "CUDA grouped MXFP4 launch failed");
            }
            outputs[index].resize(weight.rows);
            if (cudaMemcpyAsync(outputs[index].data(), member_output,
                                weight.rows * sizeof(float),
                                cudaMemcpyDeviceToHost, stream_) != cudaSuccess) {
                return Result<std::vector<std::vector<float>>>::failure(
                    ErrorCode::backend_unavailable,
                    "CUDA grouped MXFP4 output copy failed");
            }
        }
        if (cudaStreamSynchronize(stream_) != cudaSuccess) {
            return Result<std::vector<std::vector<float>>>::failure(
                ErrorCode::backend_unavailable,
                "CUDA grouped MXFP4 synchronization failed");
        }
        ++runtime_stats_.stream_synchronization_count;
        runtime_stats_.activation_h2d_bytes += input_bytes;
        runtime_stats_.weight_h2d_bytes += total_weight_transfer;
        ++runtime_stats_.grouped_projection_calls;
        runtime_stats_.grouped_projection_members += weights.size();
        record(phase, ProfileOperation::activation_host_to_device,
               NumericPrecision::mxfp4_e2m1_e8m0, layer, group_start, 0,
               input_bytes, 0, true);
        record(phase, ProfileOperation::weight_host_to_device,
               NumericPrecision::mxfp4_e2m1_e8m0, layer, group_start, 0,
               total_weight_transfer, 0, true);
        record(phase, ProfileOperation::device_to_host,
               NumericPrecision::mxfp4_e2m1_e8m0, layer, group_start, 0,
               total_output_bytes, 0, true);
        for (std::size_t index = 0; index < weights.size(); ++index) {
            auto* event_start =
                options_.cuda_allocation == CudaAllocationMode::reused
                    ? mxfp4_group_event_starts_[index].get()
                    : local_starts[index].get();
            auto* event_end =
                options_.cuda_allocation == CudaAllocationMode::reused
                    ? mxfp4_group_event_ends_[index].get()
                    : local_ends[index].get();
            float elapsed_milliseconds = 0.0F;
            if (cudaEventElapsedTime(&elapsed_milliseconds, event_start->get(),
                                     event_end->get()) != cudaSuccess) {
                return Result<std::vector<std::vector<float>>>::failure(
                    ErrorCode::backend_unavailable,
                    "CUDA grouped MXFP4 event timing failed");
            }
            record(phase, ProfileOperation::mxfp4_matvec,
                   NumericPrecision::mxfp4_e2m1_e8m0, layer, group_start,
                   weights[index].packed.size_bytes() +
                       weights[index].scales.size_bytes(),
                   0,
                   static_cast<std::uint64_t>(std::llround(
                       static_cast<double>(elapsed_milliseconds) * 1.0e6)),
                   true);
        }
        return Result<std::vector<std::vector<float>>>::success(
            std::move(outputs));
    }

    Result<std::vector<float>> dense_situ_mlp(
        std::span<const float> input, DenseMlpView weights, float situ_beta,
        std::optional<float> situ_linear, std::uint32_t layer,
        ProfilePhase phase) override {
        const auto operation_start = std::chrono::steady_clock::now();
        const auto precision = numeric_precision();
        if (options_.kind != BackendKind::cuda_custom ||
            options_.cuda_boundary != CudaBoundaryMode::ffn_block ||
            !valid_dense_mlp(input, weights) || !std::isfinite(situ_beta) ||
            situ_beta <= 0.0F ||
            (situ_linear &&
             (!std::isfinite(*situ_linear) || *situ_linear <= 0.0F))) {
            return Result<std::vector<float>>::failure(ErrorCode::invalid_extent);
        }

        std::vector<__nv_bfloat16> bf16_input;
        std::array<std::vector<__nv_bfloat16>, 3> bf16_weights;
        const void* host_input = input.data();
        std::size_t input_bytes = input.size_bytes();
        const auto input_type =
            options_.dense_precision == DensePrecision::fp32
                ? CUDA_R_32F
                : CUDA_R_16BF;
        const auto weight_type = input_type;
        if (options_.dense_precision == DensePrecision::bf16_rounded) {
            bf16_input.reserve(input.size());
            for (const auto value : input) {
                bf16_input.push_back(__float2bfloat16_rn(value));
            }
            host_input = bf16_input.data();
            input_bytes = bf16_input.size() * sizeof(__nv_bfloat16);
        }

        struct WeightMember {
            DenseWeightView view;
            const void* host{};
            std::size_t bytes{};
            void* device{};
            std::uint64_t transfer_bytes{};
        };
        std::array<WeightMember, 3> members{{
            {weights.gate}, {weights.up}, {weights.down}}};
        std::size_t maximum_weight_bytes = 0;
        std::uint64_t total_weight_transfer = 0;
        for (std::size_t index = 0; index < members.size(); ++index) {
            auto& member = members[index];
            member.host = member.view.values.data();
            member.bytes = member.view.values.size_bytes();
            if (options_.dense_precision == DensePrecision::bf16_rounded) {
                auto& converted = bf16_weights[index];
                converted.reserve(member.view.values.size());
                for (const auto value : member.view.values) {
                    converted.push_back(__float2bfloat16_rn(value));
                }
                member.host = converted.data();
                member.bytes = converted.size() * sizeof(__nv_bfloat16);
            }
            member.transfer_bytes = member.bytes;
            maximum_weight_bytes = std::max(maximum_weight_bytes, member.bytes);
            if (resident_weights_) {
                const auto representation =
                    options_.dense_precision == DensePrecision::fp32
                        ? cuda::WeightRepresentation::dense_fp32
                        : cuda::WeightRepresentation::dense_bf16;
                const auto acquisition = resident_weights_->acquire(
                    {member.view.tensor_id, representation, member.view.rows,
                     member.view.cols, 0},
                    std::span(static_cast<const std::byte*>(member.host),
                              member.bytes),
                    {});
                if (!acquisition) {
                    return Result<std::vector<float>>::failure(
                        acquisition.error(), acquisition.message());
                }
                if (acquisition.value().disposition !=
                    cuda::ResidentDisposition::bypass) {
                    member.device =
                        const_cast<void*>(acquisition.value().primary);
                    member.transfer_bytes = acquisition.value().uploaded_bytes;
                }
            }
            total_weight_transfer += member.transfer_bytes;
        }

        const auto intermediate_count = weights.gate.rows;
        const auto gate_bytes = intermediate_count * sizeof(float);
        const auto activation_bytes = intermediate_count *
            (options_.dense_precision == DensePrecision::fp32
                 ? sizeof(float)
                 : sizeof(__nv_bfloat16));
        const auto output_bytes = weights.down.rows * sizeof(float);
        cuda::DeviceAllocation local_input(&memory_stats_, &runtime_stats_);
        cuda::DeviceAllocation local_weight(&memory_stats_, &runtime_stats_);
        cuda::DeviceAllocation local_gate(&memory_stats_, &runtime_stats_);
        cuda::DeviceAllocation local_up(&memory_stats_, &runtime_stats_);
        cuda::DeviceAllocation local_activation(&memory_stats_, &runtime_stats_);
        cuda::DeviceAllocation local_output(&memory_stats_, &runtime_stats_);
        void* device_input = nullptr;
        void* transient_weight = nullptr;
        void* device_gate = nullptr;
        void* device_up = nullptr;
        void* device_activation = nullptr;
        void* device_output = nullptr;
        if (options_.cuda_allocation == CudaAllocationMode::reused) {
            if (ffn_input_scratch_.reserve(input_bytes) != cudaSuccess ||
                ffn_weight_scratch_.reserve(maximum_weight_bytes) != cudaSuccess ||
                ffn_gate_scratch_.reserve(gate_bytes) != cudaSuccess ||
                ffn_up_scratch_.reserve(gate_bytes) != cudaSuccess ||
                ffn_activation_scratch_.reserve(activation_bytes) != cudaSuccess ||
                ffn_output_scratch_.reserve(output_bytes) != cudaSuccess) {
                return Result<std::vector<float>>::failure(
                    ErrorCode::backend_unavailable,
                    "CUDA FFN reusable allocation failed");
            }
            device_input = ffn_input_scratch_.get();
            transient_weight = ffn_weight_scratch_.get();
            device_gate = ffn_gate_scratch_.get();
            device_up = ffn_up_scratch_.get();
            device_activation = ffn_activation_scratch_.get();
            device_output = ffn_output_scratch_.get();
        } else if (local_input.allocate(input_bytes) != cudaSuccess ||
                   local_weight.allocate(maximum_weight_bytes) != cudaSuccess ||
                   local_gate.allocate(gate_bytes) != cudaSuccess ||
                   local_up.allocate(gate_bytes) != cudaSuccess ||
                   local_activation.allocate(activation_bytes) != cudaSuccess ||
                   local_output.allocate(output_bytes) != cudaSuccess) {
            return Result<std::vector<float>>::failure(
                ErrorCode::backend_unavailable, "CUDA FFN allocation failed");
        } else {
            device_input = local_input.get();
            transient_weight = local_weight.get();
            device_gate = local_gate.get();
            device_up = local_up.get();
            device_activation = local_activation.get();
            device_output = local_output.get();
        }

        std::vector<std::unique_ptr<DensePlan>> local_plans;
        const auto resolve_plan = [&](std::size_t rows, std::size_t cols)
            -> DensePlan* {
            const DensePlanKey key{rows, cols, static_cast<int>(input_type),
                                   static_cast<int>(weight_type)};
            if (options_.cuda_allocation == CudaAllocationMode::reused) {
                const auto found = dense_plans_.find(key);
                if (found != dense_plans_.end()) return found->second.get();
                auto candidate = std::make_unique<DensePlan>();
                if (!initialize_dense_plan(*candidate, rows, cols, input_type,
                                           weight_type)) return nullptr;
                auto* result = candidate.get();
                dense_plans_.emplace(key, std::move(candidate));
                return result;
            }
            auto candidate = std::make_unique<DensePlan>();
            if (!initialize_dense_plan(*candidate, rows, cols, input_type,
                                       weight_type)) return nullptr;
            auto* result = candidate.get();
            local_plans.push_back(std::move(candidate));
            return result;
        };
        auto* gate_plan = resolve_plan(weights.gate.rows, weights.gate.cols);
        auto* up_plan = resolve_plan(weights.up.rows, weights.up.cols);
        auto* down_plan = resolve_plan(weights.down.rows, weights.down.cols);
        if (!gate_plan || !up_plan || !down_plan) {
            return Result<std::vector<float>>::failure(
                ErrorCode::backend_unavailable, "CUDA FFN plan creation failed");
        }

        EventOwner gate_start, gate_end, up_start, up_end;
        EventOwner situ_start, situ_end, down_start, down_end;
        for (auto* event : {&gate_start, &gate_end, &up_start, &up_end,
                            &situ_start, &situ_end, &down_start, &down_end}) {
            if (event->ensure() != cudaSuccess) {
                return Result<std::vector<float>>::failure(
                    ErrorCode::backend_unavailable,
                    "CUDA FFN event creation failed");
            }
        }
        if (cudaMemcpyAsync(device_input, host_input, input_bytes,
                            cudaMemcpyHostToDevice, stream_) != cudaSuccess) {
            return Result<std::vector<float>>::failure(
                ErrorCode::backend_unavailable,
                "CUDA FFN activation upload failed");
        }

        constexpr float alpha = 1.0F;
        constexpr float beta = 0.0F;
        const auto run_matmul = [&](WeightMember& member, DensePlan* plan,
                                    const void* matmul_input, void* output,
                                    EventOwner& start, EventOwner& end) {
            if (!member.device) {
                member.device = transient_weight;
                if (cudaMemcpyAsync(member.device, member.host, member.bytes,
                                    cudaMemcpyHostToDevice, stream_) !=
                    cudaSuccess) return false;
            }
            return cudaEventRecord(start.get(), stream_) == cudaSuccess &&
                   cublasLtMatmul(
                       handle_, plan->operation.get(), &alpha, member.device,
                       plan->weight_layout.get(), matmul_input,
                       plan->input_layout.get(), &beta, output,
                       plan->output_layout.get(), output,
                       plan->output_layout.get(), &plan->heuristic.algo,
                       nullptr, 0, stream_) == CUBLAS_STATUS_SUCCESS &&
                   cudaEventRecord(end.get(), stream_) == cudaSuccess;
        };
        if (!run_matmul(members[0], gate_plan, device_input, device_gate,
                        gate_start, gate_end)) {
            return Result<std::vector<float>>::failure(
                ErrorCode::backend_unavailable, "CUDA FFN gate failed");
        }
        if (!run_matmul(members[1], up_plan, device_input, device_up,
                        up_start, up_end)) {
            return Result<std::vector<float>>::failure(
                ErrorCode::backend_unavailable, "CUDA FFN up failed");
        }
        if (cudaEventRecord(situ_start.get(), stream_) != cudaSuccess ||
            cuda::launch_situ_glu(
                static_cast<const float*>(device_gate),
                static_cast<const float*>(device_up), device_activation,
                intermediate_count, situ_beta, situ_linear.has_value(),
                situ_linear.value_or(0.0F),
                options_.dense_precision == DensePrecision::bf16_rounded,
                stream_) != cudaSuccess ||
            cudaEventRecord(situ_end.get(), stream_) != cudaSuccess) {
            return Result<std::vector<float>>::failure(
                ErrorCode::backend_unavailable, "CUDA FFN SiTU failed");
        }
        if (!run_matmul(members[2], down_plan, device_activation, device_output,
                        down_start, down_end)) {
            return Result<std::vector<float>>::failure(
                ErrorCode::backend_unavailable, "CUDA FFN down failed");
        }

        std::vector<float> output(weights.down.rows);
        const auto d2h_start = std::chrono::steady_clock::now();
        if (cudaMemcpyAsync(output.data(), device_output, output_bytes,
                            cudaMemcpyDeviceToHost, stream_) != cudaSuccess ||
            cudaStreamSynchronize(stream_) != cudaSuccess) {
            return Result<std::vector<float>>::failure(
                ErrorCode::backend_unavailable,
                "CUDA FFN output copy or synchronization failed");
        }

        const auto elapsed = [](EventOwner& start, EventOwner& end,
                                std::uint64_t& nanoseconds) {
            float milliseconds = 0.0F;
            if (cudaEventElapsedTime(&milliseconds, start.get(), end.get()) !=
                cudaSuccess) return false;
            nanoseconds = static_cast<std::uint64_t>(std::llround(
                static_cast<double>(milliseconds) * 1.0e6));
            return true;
        };
        std::uint64_t gate_ns = 0, up_ns = 0, situ_ns = 0, down_ns = 0;
        if (!elapsed(gate_start, gate_end, gate_ns) ||
            !elapsed(up_start, up_end, up_ns) ||
            !elapsed(situ_start, situ_end, situ_ns) ||
            !elapsed(down_start, down_end, down_ns)) {
            return Result<std::vector<float>>::failure(
                ErrorCode::backend_unavailable, "CUDA FFN event timing failed");
        }

        ++runtime_stats_.stream_synchronization_count;
        runtime_stats_.activation_h2d_bytes += input_bytes;
        runtime_stats_.weight_h2d_bytes += total_weight_transfer;
        ++runtime_stats_.ffn_block_calls;
        record(phase, ProfileOperation::activation_host_to_device, precision,
               layer, operation_start, 0, input_bytes, 0, true);
        record(phase, ProfileOperation::weight_host_to_device, precision, layer,
               operation_start, 0, total_weight_transfer, 0, true);
        record(phase, ProfileOperation::device_to_host, precision, layer,
               d2h_start, 0, output_bytes, 0, true);
        record(phase, ProfileOperation::dense_matvec, precision, layer,
               operation_start, weights.gate.values.size_bytes(), 0, gate_ns,
               true);
        record(phase, ProfileOperation::dense_matvec, precision, layer,
               operation_start, weights.up.values.size_bytes(), 0, up_ns, true);
        record(phase, ProfileOperation::dense_matvec, precision, layer,
               operation_start, weights.down.values.size_bytes(), 0, down_ns,
               true);
        static_cast<void>(situ_ns);
        return Result<std::vector<float>>::success(std::move(output));
    }

    Result<std::vector<std::vector<float>>> mxfp4_situ_mlp_group(
        std::span<const float>, std::span<const Mxfp4MlpView>, float,
        std::optional<float>, std::uint32_t, ProfilePhase) override {
        return Result<std::vector<std::vector<float>>>::failure(
            ErrorCode::invalid_mxfp4);
    }

    BackendMemoryStats memory_stats() const noexcept override { return memory_stats_; }
    std::string_view device_name() const noexcept override { return device_name_; }

private:
    using DensePlanKey =
        std::tuple<std::size_t, std::size_t, int, int>;

    bool initialize_dense_plan(DensePlan& plan, std::size_t rows,
                               std::size_t cols, cudaDataType_t input_type,
                               cudaDataType_t weight_type) {
        if (cublasLtMatmulDescCreate(plan.operation.out(), CUBLAS_COMPUTE_32F,
                                     CUDA_R_32F) != CUBLAS_STATUS_SUCCESS ||
            !create_row_major_layout(plan.weight_layout, weight_type, rows,
                                     cols, cols) ||
            !create_row_major_layout(plan.input_layout, input_type, cols, 1,
                                     1) ||
            !create_row_major_layout(plan.output_layout, CUDA_R_32F, rows, 1,
                                     1) ||
            cublasLtMatmulPreferenceCreate(plan.preference.out()) !=
                CUBLAS_STATUS_SUCCESS) {
            return false;
        }
        int returned_results = 0;
        return cublasLtMatmulAlgoGetHeuristic(
                   handle_, plan.operation.get(), plan.weight_layout.get(),
                   plan.input_layout.get(), plan.output_layout.get(),
                   plan.output_layout.get(), plan.preference.get(), 1,
                   &plan.heuristic, &returned_results) ==
                   CUBLAS_STATUS_SUCCESS &&
               returned_results == 1 &&
               plan.heuristic.state == CUBLAS_STATUS_SUCCESS;
    }

    static bool valid_dense(std::span<const float> input,
                            DenseWeightView weight) {
        if (input.size() != weight.cols ||
            weight.rows > weight.values.size() ||
            (weight.cols != 0 &&
             weight.rows > weight.values.size() / weight.cols)) {
            return false;
        }
        return weight.values.size() == weight.rows * weight.cols;
    }

    static bool valid_dense_size(std::size_t input_size,
                                 DenseWeightView weight) {
        if (input_size != weight.cols || !weight.rows || !weight.cols ||
            weight.rows > weight.values.size() ||
            weight.rows > weight.values.size() / weight.cols) {
            return false;
        }
        return weight.values.size() == weight.rows * weight.cols;
    }

    static bool valid_dense_mlp(std::span<const float> input,
                                DenseMlpView weights) {
        return valid_dense(input, weights.gate) &&
               valid_dense(input, weights.up) && weights.gate.rows != 0 &&
               weights.gate.rows == weights.up.rows &&
               valid_dense_size(weights.gate.rows, weights.down);
    }

    static bool valid_mxfp4(std::span<const float> input,
                            Mxfp4WeightView weight) {
        if (input.size() != weight.cols || !weight.rows || !weight.cols ||
            !weight.group_size || weight.cols % weight.group_size ||
            weight.cols % 2 ||
            weight.rows > std::numeric_limits<std::size_t>::max() /
                              weight.cols) {
            return false;
        }
        const auto elements = weight.rows * weight.cols;
        return weight.packed.size() == elements / 2 &&
               weight.scales.size() == elements / weight.group_size &&
               std::none_of(
                   weight.scales.begin(), weight.scales.end(),
                   [](std::byte scale) {
                       return std::to_integer<std::uint8_t>(scale) == 0xFFU;
                   });
    }

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
    BackendRuntimeStats runtime_stats_{};
    cuda::ScratchBuffer dense_input_scratch_;
    cuda::ScratchBuffer dense_weight_scratch_;
    cuda::ScratchBuffer dense_output_scratch_;
    cuda::ScratchBuffer dense_group_output_scratch_;
    cuda::ScratchBuffer ffn_input_scratch_;
    cuda::ScratchBuffer ffn_weight_scratch_;
    cuda::ScratchBuffer ffn_gate_scratch_;
    cuda::ScratchBuffer ffn_up_scratch_;
    cuda::ScratchBuffer ffn_activation_scratch_;
    cuda::ScratchBuffer ffn_output_scratch_;
    cuda::ScratchBuffer mxfp4_input_scratch_;
    cuda::ScratchBuffer mxfp4_packed_scratch_;
    cuda::ScratchBuffer mxfp4_scales_scratch_;
    cuda::ScratchBuffer mxfp4_output_scratch_;
    cuda::ScratchBuffer mxfp4_group_output_scratch_;
    EventOwner dense_event_start_;
    EventOwner dense_event_end_;
    EventOwner mxfp4_event_start_;
    EventOwner mxfp4_event_end_;
    std::vector<std::unique_ptr<EventOwner>> dense_group_event_starts_;
    std::vector<std::unique_ptr<EventOwner>> dense_group_event_ends_;
    std::vector<std::unique_ptr<EventOwner>> mxfp4_group_event_starts_;
    std::vector<std::unique_ptr<EventOwner>> mxfp4_group_event_ends_;
    std::map<DensePlanKey, std::unique_ptr<DensePlan>> dense_plans_;
    std::unique_ptr<cuda::ResidentWeightTable> resident_weights_;
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
