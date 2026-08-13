// RTX 5080용 CUDA stream과 cuBLASLt handle의 수명 및 device capability를 관리합니다.
#include "k3x/backend.hpp"
#include "k3x/ops.hpp"

#include "async_mxfp4_pipeline.cuh"
#include "device_memory.cuh"
#include "graph_resources.cuh"
#include "mxfp4.cuh"
#include "moe_layer.cuh"
#include "official_kda.cuh"
#include "official_moe_route.cuh"
#include "quant3.cuh"
#include "resident_weights.cuh"
#include "situ.cuh"

#include <cublasLt.h>
#include <cuda_bf16.h>
#include <cuda_runtime_api.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstring>
#include <cstdint>
#include <limits>
#include <map>
#include <memory>
#include <span>
#include <string>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace k3x {
namespace {

std::atomic<std::uint64_t> next_kda_state_owner{1};
std::atomic<std::uint64_t> next_moe_prepared_owner{1};
std::atomic<std::uint64_t> next_layer_hidden_owner{1};

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

struct OfficialKdaDeviceStateSlot {
    std::uint64_t generation{};
    OfficialKdaCudaConfig config{};
    bool active{};
};

enum class OfficialLayerHiddenSlotState { free, live, prepared };

struct OfficialLayerHiddenSlot {
    std::uint64_t generation{};
    std::uint32_t producing_layer{};
    std::size_t width{};
    OfficialLayerHiddenSlotState state{OfficialLayerHiddenSlotState::free};
};

class CudaBackend final : public ComputeBackend {
public:
    CudaBackend(BackendOptions options, Profiler* profiler,
                cudaStream_t stream, cublasLtHandle_t handle,
                std::string device_name, std::uint64_t async_engine_count,
                bool device_overlap)
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
          mxfp4_group_output_scratch_(&memory_stats_, &runtime_stats_),
          mxfp4_descriptor_scratch_(&memory_stats_, &runtime_stats_),
          layer_input_scratch_(&memory_stats_, &runtime_stats_),
          layer_routed_latent_scratch_(&memory_stats_, &runtime_stats_),
          layer_descriptor_scratch_(&memory_stats_, &runtime_stats_),
          layer_expert_gate_scratch_(&memory_stats_, &runtime_stats_),
          layer_expert_up_scratch_(&memory_stats_, &runtime_stats_),
          layer_expert_activation_scratch_(&memory_stats_, &runtime_stats_),
          layer_expert_output_scratch_(&memory_stats_, &runtime_stats_),
          layer_contribution_scratch_(&memory_stats_, &runtime_stats_),
          layer_mixed_scratch_(&memory_stats_, &runtime_stats_),
          layer_normalized_scratch_(&memory_stats_, &runtime_stats_),
          layer_routed_hidden_scratch_(&memory_stats_, &runtime_stats_),
          layer_shared_gate_scratch_(&memory_stats_, &runtime_stats_),
          layer_shared_up_scratch_(&memory_stats_, &runtime_stats_),
          layer_shared_activation_scratch_(&memory_stats_, &runtime_stats_),
          layer_shared_hidden_scratch_(&memory_stats_, &runtime_stats_),
          layer_final_hidden_scratch_(&memory_stats_, &runtime_stats_),
          official_kda_scratch_(&memory_stats_, &runtime_stats_),
          official_kda_state_(&memory_stats_, &runtime_stats_),
          official_kda_state_one_(&memory_stats_, &runtime_stats_),
          official_kda_state_two_(&memory_stats_, &runtime_stats_),
          official_moe_prepared_(&memory_stats_, &runtime_stats_),
          official_moe_route_scratch_(&memory_stats_, &runtime_stats_),
          official_layer_front_scratch_(&memory_stats_, &runtime_stats_),
          official_layer_hidden_one_(&memory_stats_, &runtime_stats_),
          official_layer_hidden_two_(&memory_stats_, &runtime_stats_) {
        device_state_owner_ = next_kda_state_owner.fetch_add(
            1, std::memory_order_relaxed);
        if (!device_state_owner_) {
            device_state_owner_ = next_kda_state_owner.fetch_add(
                1, std::memory_order_relaxed);
        }
        moe_prepared_owner_ = next_moe_prepared_owner.fetch_add(
            1, std::memory_order_relaxed);
        if (!moe_prepared_owner_) {
            moe_prepared_owner_ = next_moe_prepared_owner.fetch_add(
                1, std::memory_order_relaxed);
        }
        layer_hidden_owner_ = next_layer_hidden_owner.fetch_add(
            1, std::memory_order_relaxed);
        if (!layer_hidden_owner_) {
            layer_hidden_owner_ = next_layer_hidden_owner.fetch_add(
                1, std::memory_order_relaxed);
        }
        runtime_stats_.async_engine_count = async_engine_count;
        runtime_stats_.device_overlap = device_overlap;
        if (options_.cuda_weights == CudaWeightMode::resident) {
            resident_weights_ = std::make_unique<cuda::ResidentWeightTable>(
                options_.cuda_resident_bytes, &memory_stats_, &runtime_stats_,
                stream_);
        }
        if (options_.cuda_graph == CudaGraphMode::cache) {
            graph_index_ = std::make_unique<BoundedCudaGraphIndex>(
                options_.cuda_graph_entries);
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

    cudaError_t initialize_async_pipeline() {
        if (options_.cuda_transfer == CudaTransferMode::synchronous) {
            return cudaSuccess;
        }
        async_pipeline_ = std::make_unique<cuda::AsyncMxfp4Pipeline>(
            &memory_stats_, &runtime_stats_);
        return async_pipeline_->initialize(options_.cuda_pinned_bytes);
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
        runtime_stats_.device_to_host_bytes += output_bytes;
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

    Result<std::vector<float>> quant3_matvec(
        std::span<const float> input, Quant3WeightView weight,
        std::uint32_t layer, ProfilePhase phase) override {
        const auto operation_start = std::chrono::steady_clock::now();
        constexpr auto precision = NumericPrecision::groupwise_signed_3bit;
        const auto logical_bytes =
            weight.packed.size_bytes() + weight.scales_bf16.size_bytes();
        const auto fail = [&](ErrorCode code, const char* message) {
            record(phase, ProfileOperation::mxfp4_matvec, precision, layer,
                   operation_start, logical_bytes, 0, 0, false);
            return Result<std::vector<float>>::failure(code, message);
        };
        if (options_.kind != BackendKind::cuda_custom ||
            input.size() != weight.cols || weight.rows == 0 ||
            weight.cols == 0 || weight.group_size != 32 ||
            weight.cols % 32 != 0 ||
            weight.rows > std::numeric_limits<std::size_t>::max() /
                              weight.cols) {
            return fail(ErrorCode::invalid_quant3, "invalid 3-bit shape");
        }
        const auto elements = weight.rows * weight.cols;
        const auto groups = elements / 32;
        if (groups > std::numeric_limits<std::size_t>::max() / 12 ||
            weight.packed.size() != groups * 12 ||
            weight.scales_bf16.size() != groups * 2) {
            return fail(ErrorCode::invalid_quant3, "invalid 3-bit extent");
        }
        for (std::size_t group = 0; group < groups; ++group) {
            const auto low = std::to_integer<std::uint16_t>(
                weight.scales_bf16[group * 2]);
            const auto high = std::to_integer<std::uint16_t>(
                weight.scales_bf16[group * 2 + 1]);
            const auto bits = static_cast<std::uint16_t>(low | (high << 8U));
            const auto scale = std::bit_cast<float>(
                static_cast<std::uint32_t>(bits) << 16U);
            if (!std::isfinite(scale) || scale <= 0.0F) {
                return fail(ErrorCode::invalid_quant3, "invalid 3-bit scale");
            }
            for (std::size_t block = 0; block < 4; ++block) {
                const auto offset = group * 12 + block * 3;
                const auto word =
                    std::to_integer<std::uint32_t>(weight.packed[offset]) |
                    (std::to_integer<std::uint32_t>(weight.packed[offset + 1]) << 8U) |
                    (std::to_integer<std::uint32_t>(weight.packed[offset + 2]) << 16U);
                for (std::size_t index = 0; index < 8; ++index) {
                    if (((word >> (index * 3U)) & 7U) == 7U) {
                        return fail(ErrorCode::invalid_quant3,
                                    "reserved 3-bit code");
                    }
                }
            }
        }

        cuda::DeviceAllocation device_input(&memory_stats_, &runtime_stats_);
        cuda::DeviceAllocation device_packed(&memory_stats_, &runtime_stats_);
        cuda::DeviceAllocation device_scales(&memory_stats_, &runtime_stats_);
        cuda::DeviceAllocation device_output(&memory_stats_, &runtime_stats_);
        const auto input_bytes = input.size_bytes();
        const auto output_bytes = weight.rows * sizeof(float);
        if (device_input.allocate(input_bytes) != cudaSuccess ||
            device_packed.allocate(weight.packed.size_bytes()) != cudaSuccess ||
            device_scales.allocate(weight.scales_bf16.size_bytes()) != cudaSuccess ||
            device_output.allocate(output_bytes) != cudaSuccess) {
            return fail(ErrorCode::backend_unavailable,
                        "CUDA 3-bit allocation failed");
        }
        const auto h2d_start = std::chrono::steady_clock::now();
        if (cudaMemcpyAsync(device_input.get(), input.data(), input_bytes,
                            cudaMemcpyHostToDevice, stream_) != cudaSuccess ||
            cudaMemcpyAsync(device_packed.get(), weight.packed.data(),
                            weight.packed.size_bytes(), cudaMemcpyHostToDevice,
                            stream_) != cudaSuccess ||
            cudaMemcpyAsync(device_scales.get(), weight.scales_bf16.data(),
                            weight.scales_bf16.size_bytes(), cudaMemcpyHostToDevice,
                            stream_) != cudaSuccess) {
            return fail(ErrorCode::backend_unavailable,
                        "CUDA 3-bit upload failed");
        }
        runtime_stats_.activation_h2d_bytes += input_bytes;
        runtime_stats_.weight_h2d_bytes += logical_bytes;
        record(phase, ProfileOperation::activation_host_to_device, precision,
               layer, h2d_start, 0, input_bytes, 0, true);
        record(phase, ProfileOperation::weight_host_to_device, precision,
               layer, h2d_start, 0, logical_bytes, 0, true);

        EventOwner event_start;
        EventOwner event_end;
        if (event_start.ensure() != cudaSuccess ||
            event_end.ensure() != cudaSuccess ||
            cudaEventRecord(event_start.get(), stream_) != cudaSuccess ||
            cuda::launch_quant3_matvec(
                static_cast<const float*>(device_input.get()),
                static_cast<const std::uint8_t*>(device_packed.get()),
                static_cast<const std::uint16_t*>(device_scales.get()),
                static_cast<float*>(device_output.get()),
                weight.rows, weight.cols, stream_) != cudaSuccess ||
            cudaEventRecord(event_end.get(), stream_) != cudaSuccess) {
            return fail(ErrorCode::backend_unavailable,
                        "CUDA 3-bit launch failed");
        }
        std::vector<float> output(weight.rows);
        const auto d2h_start = std::chrono::steady_clock::now();
        if (cudaMemcpyAsync(output.data(), device_output.get(), output_bytes,
                            cudaMemcpyDeviceToHost, stream_) != cudaSuccess ||
            cudaStreamSynchronize(stream_) != cudaSuccess) {
            return fail(ErrorCode::backend_unavailable,
                        "CUDA 3-bit result failed");
        }
        ++runtime_stats_.stream_synchronization_count;
        runtime_stats_.device_to_host_bytes += output_bytes;
        record(phase, ProfileOperation::device_to_host, precision, layer,
               d2h_start, 0, output_bytes, 0, true);
        float elapsed_milliseconds = 0.0F;
        if (cudaEventElapsedTime(&elapsed_milliseconds, event_start.get(),
                                 event_end.get()) != cudaSuccess) {
            return fail(ErrorCode::backend_unavailable,
                        "CUDA 3-bit timing failed");
        }
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
            (options_.cuda_boundary != CudaBoundaryMode::ffn_block &&
             options_.cuda_boundary != CudaBoundaryMode::moe_layer) ||
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
        runtime_stats_.device_to_host_bytes += output_bytes;
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
        record(phase, ProfileOperation::situ_glu, precision, layer,
               operation_start, intermediate_count * sizeof(float), 0,
               situ_ns, true);
        return Result<std::vector<float>>::success(std::move(output));
    }

    Result<std::vector<std::vector<float>>> mxfp4_situ_mlp_group(
        std::span<const float> input, std::span<const Mxfp4MlpView> experts,
        float situ_beta, std::optional<float> situ_linear,
        std::uint32_t layer, ProfilePhase phase) override {
        return mxfp4_situ_mlp_group_impl(
            input, experts, {}, situ_beta, situ_linear, layer, phase);
    }

    Result<std::vector<std::vector<float>>> mxfp4_situ_mlp_grid(
        std::span<const float> inputs, std::size_t token_count,
        std::span<const Mxfp4MlpView> experts, float situ_beta,
        std::optional<float> situ_linear, std::uint32_t layer,
        ProfilePhase phase) override {
        constexpr auto precision = NumericPrecision::mxfp4_e2m1_e8m0;
        const auto operation_start = std::chrono::steady_clock::now();
        const auto multiply_fits = [](std::size_t left, std::size_t right) {
            return right == 0 ||
                   left <= std::numeric_limits<std::size_t>::max() / right;
        };
        if (options_.kind != BackendKind::cuda_custom ||
            (options_.cuda_boundary != CudaBoundaryMode::ffn_block &&
             options_.cuda_boundary != CudaBoundaryMode::moe_layer) ||
            options_.cuda_allocation != CudaAllocationMode::reused ||
            options_.cuda_weights != CudaWeightMode::resident ||
            options_.cuda_transfer != CudaTransferMode::synchronous ||
            options_.cuda_batching != CudaBatchingMode::resident_grid ||
            options_.cuda_moe_fusion != CudaMoeFusionMode::none ||
            options_.cuda_resident_bytes == 0 || resident_weights_ == nullptr ||
            token_count == 0 ||
            token_count > 65535 || experts.empty() || experts.size() > 65535 ||
            experts.front().gate.cols == 0 ||
            !multiply_fits(token_count, experts.front().gate.cols) ||
            inputs.size() != token_count * experts.front().gate.cols ||
            !std::isfinite(situ_beta) || situ_beta <= 0.0F ||
            (situ_linear &&
             (!std::isfinite(*situ_linear) || *situ_linear <= 0.0F))) {
            return Result<std::vector<std::vector<float>>>::failure(
                ErrorCode::invalid_mxfp4);
        }

        const auto input_width = experts.front().gate.cols;
        const auto intermediate_width = experts.front().gate.rows;
        const auto output_width = experts.front().down.rows;
        std::unordered_set<std::uint64_t> tensor_ids;
        for (const auto& expert : experts) {
            if (expert.gate.cols != input_width ||
                expert.gate.rows != intermediate_width ||
                expert.up.cols != input_width ||
                expert.up.rows != intermediate_width ||
                expert.down.cols != intermediate_width ||
                expert.down.rows != output_width ||
                expert.gate.group_size != 32 || expert.up.group_size != 32 ||
                expert.down.group_size != 32 ||
                !valid_mxfp4_size(input_width, expert.gate) ||
                !valid_mxfp4_size(input_width, expert.up) ||
                !valid_mxfp4_size(intermediate_width, expert.down)) {
                return Result<std::vector<std::vector<float>>>::failure(
                    ErrorCode::invalid_mxfp4);
            }
            for (const auto tensor_id : {
                     expert.gate.tensor_id, expert.up.tensor_id,
                     expert.down.tensor_id}) {
                if (tensor_id == 0 || !tensor_ids.insert(tensor_id).second) {
                    return Result<std::vector<std::vector<float>>>::failure(
                        ErrorCode::invalid_mxfp4);
                }
            }
        }
        if (!multiply_fits(experts.size(), token_count) ||
            !multiply_fits(experts.size() * token_count, intermediate_width) ||
            !multiply_fits(experts.size() * token_count, output_width) ||
            intermediate_width > 65535 || output_width > 65535) {
            return Result<std::vector<std::vector<float>>>::failure(
                ErrorCode::invalid_mxfp4);
        }

        std::vector<cuda::Mxfp4DeviceMatrix> descriptors(experts.size() * 3);
        std::uint64_t weight_transfer_bytes = 0;
        bool bypass = false;
        for (std::size_t expert_index = 0;
             expert_index < experts.size(); ++expert_index) {
            const auto weights = std::array{
                experts[expert_index].gate,
                experts[expert_index].up,
                experts[expert_index].down,
            };
            for (std::size_t projection = 0; projection < weights.size();
                 ++projection) {
                const auto& weight = weights[projection];
                const auto acquisition = resident_weights_->acquire(
                    {weight.tensor_id, cuda::WeightRepresentation::mxfp4,
                     weight.rows, weight.cols, weight.group_size},
                    weight.packed, weight.scales);
                if (!acquisition) {
                    return Result<std::vector<std::vector<float>>>::failure(
                        acquisition.error(), acquisition.message());
                }
                if (acquisition.value().disposition ==
                    cuda::ResidentDisposition::bypass) {
                    bypass = true;
                    continue;
                }
                descriptors[projection * experts.size() + expert_index] = {
                    static_cast<const std::uint8_t*>(
                        acquisition.value().primary),
                    static_cast<const std::uint8_t*>(
                        acquisition.value().secondary),
                };
                weight_transfer_bytes += acquisition.value().uploaded_bytes;
            }
        }
        if (bypass) {
            ++runtime_stats_.resident_grid_fallbacks;
            if (weight_transfer_bytes != 0) {
                runtime_stats_.weight_h2d_bytes += weight_transfer_bytes;
                record(phase, ProfileOperation::weight_host_to_device,
                       precision, layer, operation_start, 0,
                       weight_transfer_bytes, 0, true);
            }
            std::vector<std::vector<float>> outputs(experts.size());
            for (std::size_t token = 0; token < token_count; ++token) {
                const auto token_input = inputs.subspan(
                    token * input_width, input_width);
                auto token_outputs = mxfp4_situ_mlp_group_impl(
                    token_input, experts, {}, situ_beta, situ_linear, layer,
                    phase);
                if (!token_outputs) {
                    return Result<std::vector<std::vector<float>>>::failure(
                        token_outputs.error(), token_outputs.message());
                }
                for (std::size_t expert = 0; expert < experts.size(); ++expert) {
                    outputs[expert].insert(outputs[expert].end(),
                                           token_outputs.value()[expert].begin(),
                                           token_outputs.value()[expert].end());
                }
            }
            return Result<std::vector<std::vector<float>>>::success(
                std::move(outputs));
        }

        const auto expert_tokens = experts.size() * token_count;
        const auto input_bytes = inputs.size_bytes();
        const auto descriptor_bytes =
            descriptors.size() * sizeof(cuda::Mxfp4DeviceMatrix);
        const auto intermediate_count = expert_tokens * intermediate_width;
        const auto intermediate_bytes = intermediate_count * sizeof(float);
        const auto output_count = expert_tokens * output_width;
        const auto output_bytes = output_count * sizeof(float);
        if (ffn_input_scratch_.reserve(input_bytes) != cudaSuccess ||
            mxfp4_descriptor_scratch_.reserve(descriptor_bytes) != cudaSuccess ||
            ffn_gate_scratch_.reserve(intermediate_bytes) != cudaSuccess ||
            ffn_up_scratch_.reserve(intermediate_bytes) != cudaSuccess ||
            ffn_activation_scratch_.reserve(intermediate_bytes) != cudaSuccess ||
            ffn_output_scratch_.reserve(output_bytes) != cudaSuccess) {
            return Result<std::vector<std::vector<float>>>::failure(
                ErrorCode::backend_unavailable,
                "CUDA resident expert grid allocation failed");
        }
        for (auto& event : resident_grid_events_) {
            if (event.ensure() != cudaSuccess) {
                return Result<std::vector<std::vector<float>>>::failure(
                    ErrorCode::backend_unavailable,
                    "CUDA resident expert grid event creation failed");
            }
        }

        auto* device_input = static_cast<float*>(ffn_input_scratch_.get());
        auto* device_descriptors = static_cast<cuda::Mxfp4DeviceMatrix*>(
            mxfp4_descriptor_scratch_.get());
        auto* device_gate = static_cast<float*>(ffn_gate_scratch_.get());
        auto* device_up = static_cast<float*>(ffn_up_scratch_.get());
        auto* device_activation = static_cast<float*>(
            ffn_activation_scratch_.get());
        auto* device_output = static_cast<float*>(ffn_output_scratch_.get());
        const auto h2d_start = std::chrono::steady_clock::now();
        if (cudaMemcpyAsync(device_input, inputs.data(), input_bytes,
                            cudaMemcpyHostToDevice, stream_) != cudaSuccess ||
            cudaMemcpyAsync(device_descriptors, descriptors.data(),
                            descriptor_bytes, cudaMemcpyHostToDevice,
                            stream_) != cudaSuccess) {
            return Result<std::vector<std::vector<float>>>::failure(
                ErrorCode::backend_unavailable,
                "CUDA resident expert grid upload failed");
        }
        const auto launch_grid = [&](std::size_t event_pair,
                                     std::size_t descriptor_projection,
                                     const float* grid_input, float* grid_output,
                                     std::size_t rows, std::size_t cols,
                                     cuda::ExpertGridInputLayout layout) {
            auto& start = resident_grid_events_[event_pair * 2];
            auto& end = resident_grid_events_[event_pair * 2 + 1];
            return cudaEventRecord(start.get(), stream_) == cudaSuccess &&
                   cuda::launch_mxfp4_matvec_grid(
                       grid_input,
                       device_descriptors +
                           descriptor_projection * experts.size(),
                       grid_output, rows, cols, experts.size(), token_count,
                       layout, stream_) == cudaSuccess &&
                   cudaEventRecord(end.get(), stream_) == cudaSuccess;
        };
        if (!launch_grid(0, 0, device_input, device_gate, intermediate_width,
                         input_width,
                         cuda::ExpertGridInputLayout::shared_token_major) ||
            !launch_grid(1, 1, device_input, device_up, intermediate_width,
                         input_width,
                         cuda::ExpertGridInputLayout::shared_token_major) ||
            cudaEventRecord(resident_grid_events_[4].get(), stream_) !=
                cudaSuccess ||
            cuda::launch_situ_glu(
                device_gate, device_up, device_activation, intermediate_count,
                situ_beta, situ_linear.has_value(), situ_linear.value_or(0.0F),
                false, stream_) != cudaSuccess ||
            cudaEventRecord(resident_grid_events_[5].get(), stream_) !=
                cudaSuccess ||
            !launch_grid(3, 2, device_activation, device_output, output_width,
                         intermediate_width,
                         cuda::ExpertGridInputLayout::expert_token_major)) {
            return Result<std::vector<std::vector<float>>>::failure(
                ErrorCode::backend_unavailable,
                "CUDA resident expert grid launch failed");
        }

        std::vector<float> flat_output(output_count);
        const auto d2h_start = std::chrono::steady_clock::now();
        if (cudaMemcpyAsync(flat_output.data(), device_output, output_bytes,
                            cudaMemcpyDeviceToHost, stream_) != cudaSuccess ||
            cudaStreamSynchronize(stream_) != cudaSuccess) {
            return Result<std::vector<std::vector<float>>>::failure(
                ErrorCode::backend_unavailable,
                "CUDA resident expert grid output or synchronization failed");
        }
        std::array<std::uint64_t, 4> durations{};
        for (std::size_t pair = 0; pair < durations.size(); ++pair) {
            float milliseconds = 0.0F;
            if (cudaEventElapsedTime(&milliseconds,
                                     resident_grid_events_[pair * 2].get(),
                                     resident_grid_events_[pair * 2 + 1].get()) !=
                cudaSuccess) {
                return Result<std::vector<std::vector<float>>>::failure(
                    ErrorCode::backend_unavailable,
                    "CUDA resident expert grid timing failed");
            }
            durations[pair] = static_cast<std::uint64_t>(std::llround(
                static_cast<double>(milliseconds) * 1.0e6));
        }

        std::vector<std::vector<float>> outputs(experts.size());
        const auto values_per_expert = token_count * output_width;
        for (std::size_t expert = 0; expert < experts.size(); ++expert) {
            outputs[expert].assign(
                flat_output.begin() + expert * values_per_expert,
                flat_output.begin() + (expert + 1) * values_per_expert);
        }
        std::array<std::uint64_t, 3> logical_projection_bytes{};
        for (const auto& expert : experts) {
            const auto weights = std::array{expert.gate, expert.up, expert.down};
            for (std::size_t projection = 0; projection < weights.size();
                 ++projection) {
                logical_projection_bytes[projection] +=
                    weights[projection].packed.size_bytes() +
                    weights[projection].scales.size_bytes();
            }
        }
        ++runtime_stats_.stream_synchronization_count;
        runtime_stats_.activation_h2d_bytes += input_bytes;
        runtime_stats_.weight_h2d_bytes += weight_transfer_bytes;
        runtime_stats_.device_to_host_bytes += output_bytes;
        ++runtime_stats_.ffn_block_calls;
        runtime_stats_.ffn_block_experts += experts.size();
        ++runtime_stats_.resident_grid_calls;
        runtime_stats_.resident_grid_experts += experts.size();
        runtime_stats_.resident_grid_tokens += token_count;
        runtime_stats_.resident_grid_expert_tokens += expert_tokens;
        runtime_stats_.resident_grid_kernel_launches += 4;
        runtime_stats_.resident_grid_descriptor_h2d_bytes += descriptor_bytes;
        record(phase, ProfileOperation::activation_host_to_device, precision,
               layer, h2d_start, 0, input_bytes + descriptor_bytes, 0, true);
        record(phase, ProfileOperation::weight_host_to_device, precision, layer,
               operation_start, 0, weight_transfer_bytes, 0, true);
        record(phase, ProfileOperation::device_to_host, precision, layer,
               d2h_start, 0, output_bytes, 0, true);
        record(phase, ProfileOperation::mxfp4_matvec, precision, layer,
               operation_start, logical_projection_bytes[0], 0,
               durations[0], true);
        record(phase, ProfileOperation::mxfp4_matvec, precision, layer,
               operation_start, logical_projection_bytes[1], 0,
               durations[1], true);
        record(phase, ProfileOperation::situ_glu, NumericPrecision::fp32,
               layer, operation_start, intermediate_bytes, 0, durations[2],
               true);
        record(phase, ProfileOperation::mxfp4_matvec, precision, layer,
               operation_start, logical_projection_bytes[2], 0,
               durations[3], true);
        return Result<std::vector<std::vector<float>>>::success(
            std::move(outputs));
    }

    Result<ResidentMoeLayerResult> resident_mxfp4_moe_layer(
        std::span<const float> input, ResidentMoeLayerView weights,
        std::span<const Mxfp4MlpView> experts,
        std::span<const float> contributions, float epsilon, float situ_beta,
        std::optional<float> situ_linear, std::uint32_t layer,
        ProfilePhase phase) override {
        const auto operation_start = std::chrono::steady_clock::now();
        constexpr auto dense_precision = NumericPrecision::fp32;
        constexpr auto expert_precision = NumericPrecision::mxfp4_e2m1_e8m0;
        const auto all_finite = [](std::span<const float> values) {
            return std::all_of(values.begin(), values.end(),
                               [](float value) { return std::isfinite(value); });
        };
        const auto multiply_fits = [](std::size_t left, std::size_t right) {
            return right == 0 ||
                   left <= std::numeric_limits<std::size_t>::max() / right;
        };
        if (options_.kind != BackendKind::cuda_custom ||
            options_.dense_precision != DensePrecision::fp32 ||
            options_.cuda_boundary != CudaBoundaryMode::moe_layer ||
            options_.cuda_allocation != CudaAllocationMode::reused ||
            options_.cuda_weights != CudaWeightMode::resident ||
            options_.cuda_transfer != CudaTransferMode::synchronous ||
            options_.cuda_batching != CudaBatchingMode::resident_grid ||
            options_.cuda_moe_fusion != CudaMoeFusionMode::none ||
            options_.cuda_resident_bytes == 0 || resident_weights_ == nullptr ||
            input.empty() || experts.empty() || experts.size() > 65535 ||
            experts.size() != contributions.size() ||
            !std::isfinite(epsilon) || epsilon <= 0.0F ||
            !std::isfinite(situ_beta) || situ_beta <= 0.0F ||
            (situ_linear &&
             (!std::isfinite(*situ_linear) || *situ_linear <= 0.0F)) ||
            !all_finite(input) || !all_finite(contributions) ||
            !valid_dense(input, weights.routed_down) ||
            !valid_dense_mlp(input, weights.shared)) {
            return Result<ResidentMoeLayerResult>::failure(
                ErrorCode::invalid_mxfp4);
        }

        const auto hidden_width = input.size();
        const auto latent_width = weights.routed_down.rows;
        const auto routed_width = experts.front().down.rows;
        const auto intermediate_width = experts.front().gate.rows;
        const auto shared_width = weights.shared.gate.rows;
        if (latent_width == 0 || routed_width == 0 ||
            intermediate_width == 0 || shared_width == 0 ||
            latent_width > 65535 || routed_width > 65535 ||
            intermediate_width > 65535 ||
            weights.routed_norm.values.size() != routed_width ||
            !valid_dense_size(routed_width, weights.routed_up) ||
            weights.routed_up.rows != hidden_width ||
            weights.shared.down.rows != hidden_width ||
            !multiply_fits(experts.size(), intermediate_width) ||
            !multiply_fits(experts.size(), routed_width)) {
            return Result<ResidentMoeLayerResult>::failure(
                ErrorCode::invalid_mxfp4);
        }

        std::unordered_set<std::uint64_t> tensor_ids;
        const auto insert_id = [&tensor_ids](std::uint64_t tensor_id) {
            return tensor_id != 0 && tensor_ids.insert(tensor_id).second;
        };
        if (!insert_id(weights.routed_down.tensor_id) ||
            !insert_id(weights.routed_norm.tensor_id) ||
            !insert_id(weights.routed_up.tensor_id) ||
            !insert_id(weights.shared.gate.tensor_id) ||
            !insert_id(weights.shared.up.tensor_id) ||
            !insert_id(weights.shared.down.tensor_id)) {
            return Result<ResidentMoeLayerResult>::failure(
                ErrorCode::invalid_mxfp4);
        }
        for (const auto& expert : experts) {
            if (expert.gate.cols != latent_width ||
                expert.gate.rows != intermediate_width ||
                expert.up.cols != latent_width ||
                expert.up.rows != intermediate_width ||
                expert.down.cols != intermediate_width ||
                expert.down.rows != routed_width ||
                expert.gate.group_size != 32 || expert.up.group_size != 32 ||
                expert.down.group_size != 32 ||
                !valid_mxfp4_size(latent_width, expert.gate) ||
                !valid_mxfp4_size(latent_width, expert.up) ||
                !valid_mxfp4_size(intermediate_width, expert.down) ||
                !insert_id(expert.gate.tensor_id) ||
                !insert_id(expert.up.tensor_id) ||
                !insert_id(expert.down.tensor_id)) {
                return Result<ResidentMoeLayerResult>::failure(
                    ErrorCode::invalid_mxfp4);
            }
        }

        const std::array dense_views{
            weights.routed_down,
            DenseWeightView{weights.routed_norm.tensor_id,
                            weights.routed_norm.values, 1, routed_width},
            weights.routed_up,
            weights.shared.gate,
            weights.shared.up,
            weights.shared.down,
        };
        std::array<bool, 6> validation_hits{};
        std::array<bool, 6> validation_scans{};
        if (options_.cuda_weight_validation ==
            CudaWeightValidationMode::admission) {
            for (std::size_t index = 0; index < dense_views.size(); ++index) {
                const auto& view = dense_views[index];
                const ImmutableWeightIdentity identity{
                    view.values.data(), view.values.size_bytes(),
                    view.rows, view.cols};
                const auto found = immutable_weights_.find(view.tensor_id);
                if (found == immutable_weights_.end()) {
                    validation_scans[index] = true;
                } else if (found->second == identity) {
                    validation_hits[index] = true;
                } else {
                    return Result<ResidentMoeLayerResult>::failure(
                        ErrorCode::invalid_mxfp4);
                }
            }
        } else {
            validation_scans.fill(true);
        }
        const auto validation_start = std::chrono::steady_clock::now();
        bool immutable_finite = true;
        for (std::size_t index = 0; index < dense_views.size(); ++index) {
            if (!validation_scans[index]) continue;
            ++runtime_stats_.immutable_validation_scans;
            runtime_stats_.immutable_validation_bytes +=
                dense_views[index].values.size_bytes();
            if (!all_finite(dense_views[index].values)) immutable_finite = false;
        }
        runtime_stats_.immutable_validation_nanoseconds +=
            static_cast<std::uint64_t>(std::chrono::duration_cast<
                std::chrono::nanoseconds>(std::chrono::steady_clock::now() -
                                         validation_start).count());
        if (!immutable_finite) {
            return Result<ResidentMoeLayerResult>::failure(
                ErrorCode::invalid_mxfp4);
        }
        if (options_.cuda_weight_validation ==
            CudaWeightValidationMode::admission) {
            for (std::size_t index = 0; index < dense_views.size(); ++index) {
                if (validation_hits[index]) {
                    ++runtime_stats_.immutable_validation_hits;
                } else {
                    const auto& view = dense_views[index];
                    immutable_weights_.emplace(
                        view.tensor_id,
                        ImmutableWeightIdentity{
                            view.values.data(), view.values.size_bytes(),
                            view.rows, view.cols});
                }
            }
        }
        std::array<cuda::ResidentAcquisition, 6> dense_members{};
        std::uint64_t weight_transfer_bytes = 0;
        bool bypass = false;
        for (std::size_t index = 0; index < dense_views.size(); ++index) {
            const auto& view = dense_views[index];
            const auto acquired = resident_weights_->acquire(
                {view.tensor_id, cuda::WeightRepresentation::dense_fp32,
                 view.rows, view.cols, 0},
                std::as_bytes(view.values), {});
            if (!acquired) {
                return Result<ResidentMoeLayerResult>::failure(
                    acquired.error(), acquired.message());
            }
            dense_members[index] = acquired.value();
            bypass |= acquired.value().disposition ==
                      cuda::ResidentDisposition::bypass;
            weight_transfer_bytes += acquired.value().uploaded_bytes;
        }

        std::vector<cuda::Mxfp4DeviceMatrix> descriptors(experts.size() * 3);
        for (std::size_t expert_index = 0;
             expert_index < experts.size(); ++expert_index) {
            const std::array expert_weights{
                experts[expert_index].gate,
                experts[expert_index].up,
                experts[expert_index].down,
            };
            for (std::size_t projection = 0;
                 projection < expert_weights.size(); ++projection) {
                const auto& view = expert_weights[projection];
                const auto acquired = resident_weights_->acquire(
                    {view.tensor_id, cuda::WeightRepresentation::mxfp4,
                     view.rows, view.cols, view.group_size},
                    view.packed, view.scales);
                if (!acquired) {
                    return Result<ResidentMoeLayerResult>::failure(
                        acquired.error(), acquired.message());
                }
                bypass |= acquired.value().disposition ==
                          cuda::ResidentDisposition::bypass;
                weight_transfer_bytes += acquired.value().uploaded_bytes;
                if (acquired.value().disposition !=
                    cuda::ResidentDisposition::bypass) {
                    descriptors[projection * experts.size() + expert_index] = {
                        static_cast<const std::uint8_t*>(
                            acquired.value().primary),
                        static_cast<const std::uint8_t*>(
                            acquired.value().secondary),
                    };
                }
            }
        }
        if (bypass) {
            ++runtime_stats_.resident_moe_layer_fallbacks;
            if (weight_transfer_bytes != 0) {
                runtime_stats_.weight_h2d_bytes += weight_transfer_bytes;
                record(phase, ProfileOperation::weight_host_to_device,
                       dense_precision, layer, operation_start, 0,
                       weight_transfer_bytes, 0, true);
            }
            return Result<ResidentMoeLayerResult>::success({false, {}});
        }

        std::array<DensePlan*, 5> plans{};
        const std::array plan_views{
            dense_views[0], dense_views[2], dense_views[3], dense_views[4],
            dense_views[5],
        };
        for (std::size_t index = 0; index < plan_views.size(); ++index) {
            const auto& view = plan_views[index];
            const DensePlanKey key{view.rows, view.cols,
                                   static_cast<int>(CUDA_R_32F),
                                   static_cast<int>(CUDA_R_32F)};
            const auto found = dense_plans_.find(key);
            if (found != dense_plans_.end()) {
                plans[index] = found->second.get();
            } else {
                auto candidate = std::make_unique<DensePlan>();
                if (!initialize_dense_plan(*candidate, view.rows, view.cols,
                                           CUDA_R_32F, CUDA_R_32F)) {
                    return Result<ResidentMoeLayerResult>::failure(
                        ErrorCode::backend_unavailable,
                        "resident MoE layer dense plan creation failed");
                }
                plans[index] = candidate.get();
                dense_plans_.emplace(key, std::move(candidate));
            }
        }

        const auto input_bytes = input.size_bytes();
        const auto latent_bytes = latent_width * sizeof(float);
        const auto descriptor_bytes =
            descriptors.size() * sizeof(cuda::Mxfp4DeviceMatrix);
        const auto expert_intermediate_count =
            experts.size() * intermediate_width;
        const auto expert_intermediate_bytes =
            expert_intermediate_count * sizeof(float);
        const auto expert_output_count = experts.size() * routed_width;
        const auto expert_output_bytes = expert_output_count * sizeof(float);
        const auto contribution_bytes = contributions.size_bytes();
        const auto routed_bytes = routed_width * sizeof(float);
        const auto hidden_bytes = hidden_width * sizeof(float);
        const auto shared_bytes = shared_width * sizeof(float);
        if (layer_input_scratch_.reserve(input_bytes) != cudaSuccess ||
            layer_routed_latent_scratch_.reserve(latent_bytes) != cudaSuccess ||
            layer_descriptor_scratch_.reserve(descriptor_bytes) != cudaSuccess ||
            layer_expert_gate_scratch_.reserve(expert_intermediate_bytes) !=
                cudaSuccess ||
            layer_expert_up_scratch_.reserve(expert_intermediate_bytes) !=
                cudaSuccess ||
            layer_expert_activation_scratch_.reserve(
                expert_intermediate_bytes) != cudaSuccess ||
            layer_expert_output_scratch_.reserve(expert_output_bytes) !=
                cudaSuccess ||
            layer_contribution_scratch_.reserve(contribution_bytes) !=
                cudaSuccess ||
            layer_mixed_scratch_.reserve(routed_bytes) != cudaSuccess ||
            layer_normalized_scratch_.reserve(routed_bytes) != cudaSuccess ||
            layer_routed_hidden_scratch_.reserve(hidden_bytes) != cudaSuccess ||
            layer_shared_gate_scratch_.reserve(shared_bytes) != cudaSuccess ||
            layer_shared_up_scratch_.reserve(shared_bytes) != cudaSuccess ||
            layer_shared_activation_scratch_.reserve(shared_bytes) !=
                cudaSuccess ||
            layer_shared_hidden_scratch_.reserve(hidden_bytes) != cudaSuccess ||
            layer_final_hidden_scratch_.reserve(hidden_bytes) != cudaSuccess) {
            return Result<ResidentMoeLayerResult>::failure(
                ErrorCode::backend_unavailable,
                "resident MoE layer scratch allocation failed");
        }
        for (auto& event : resident_moe_layer_events_) {
            if (event.ensure() != cudaSuccess) {
                return Result<ResidentMoeLayerResult>::failure(
                    ErrorCode::backend_unavailable,
                    "resident MoE layer event creation failed");
            }
        }

        auto* device_input = static_cast<float*>(layer_input_scratch_.get());
        auto* device_latent =
            static_cast<float*>(layer_routed_latent_scratch_.get());
        auto* device_descriptors = static_cast<cuda::Mxfp4DeviceMatrix*>(
            layer_descriptor_scratch_.get());
        auto* device_expert_gate =
            static_cast<float*>(layer_expert_gate_scratch_.get());
        auto* device_expert_up =
            static_cast<float*>(layer_expert_up_scratch_.get());
        auto* device_expert_activation =
            static_cast<float*>(layer_expert_activation_scratch_.get());
        auto* device_expert_output =
            static_cast<float*>(layer_expert_output_scratch_.get());
        auto* device_contributions =
            static_cast<float*>(layer_contribution_scratch_.get());
        auto* device_mixed = static_cast<float*>(layer_mixed_scratch_.get());
        auto* device_normalized =
            static_cast<float*>(layer_normalized_scratch_.get());
        auto* device_routed =
            static_cast<float*>(layer_routed_hidden_scratch_.get());
        auto* device_shared_gate =
            static_cast<float*>(layer_shared_gate_scratch_.get());
        auto* device_shared_up =
            static_cast<float*>(layer_shared_up_scratch_.get());
        auto* device_shared_activation =
            static_cast<float*>(layer_shared_activation_scratch_.get());
        auto* device_shared =
            static_cast<float*>(layer_shared_hidden_scratch_.get());
        auto* device_final =
            static_cast<float*>(layer_final_hidden_scratch_.get());
        if (options_.cuda_graph != CudaGraphMode::disabled) {
            std::vector<std::uint64_t> scratch_identity;
            scratch_identity.reserve(32);
            const auto append_scratch = [&scratch_identity](const auto& scratch) {
                scratch_identity.push_back(static_cast<std::uint64_t>(
                    reinterpret_cast<std::uintptr_t>(scratch.get())));
                scratch_identity.push_back(scratch.capacity());
            };
            append_scratch(layer_input_scratch_);
            append_scratch(layer_routed_latent_scratch_);
            append_scratch(layer_descriptor_scratch_);
            append_scratch(layer_expert_gate_scratch_);
            append_scratch(layer_expert_up_scratch_);
            append_scratch(layer_expert_activation_scratch_);
            append_scratch(layer_expert_output_scratch_);
            append_scratch(layer_contribution_scratch_);
            append_scratch(layer_mixed_scratch_);
            append_scratch(layer_normalized_scratch_);
            append_scratch(layer_routed_hidden_scratch_);
            append_scratch(layer_shared_gate_scratch_);
            append_scratch(layer_shared_up_scratch_);
            append_scratch(layer_shared_activation_scratch_);
            append_scratch(layer_shared_hidden_scratch_);
            append_scratch(layer_final_hidden_scratch_);
            if (!graph_scratch_identity_.empty() &&
                graph_scratch_identity_ != scratch_identity) {
                graph_entries_.clear();
                if (graph_index_) graph_index_->clear();
                update_graph_entry_.reset();
                runtime_stats_.cuda_graph_resident_entries = 0;
                ++runtime_stats_.cuda_graph_invalidations;
            }
            graph_scratch_identity_ = std::move(scratch_identity);
        }
        std::vector<float> output(hidden_width);
        CudaGraphKey graph_key;
        cuda::GraphEntry* graph_entry = nullptr;
        std::unique_ptr<cuda::GraphEntry> graph_candidate;
        bool graph_cache_hit = false;
        const auto graph_host_start = std::chrono::steady_clock::now();
        if (options_.cuda_graph != CudaGraphMode::disabled) {
            graph_key.words = {
                hidden_width, latent_width, routed_width, intermediate_width,
                shared_width, experts.size(), weights.routed_down.tensor_id,
                weights.routed_norm.tensor_id, weights.routed_up.tensor_id,
                weights.shared.gate.tensor_id, weights.shared.up.tensor_id,
                weights.shared.down.tensor_id,
                std::bit_cast<std::uint32_t>(epsilon),
                std::bit_cast<std::uint32_t>(situ_beta),
                situ_linear.has_value() ? 1ULL : 0ULL,
                situ_linear ? std::bit_cast<std::uint32_t>(*situ_linear) : 0ULL,
            };
            for (const auto& expert : experts) {
                graph_key.words.push_back(expert.gate.tensor_id);
                graph_key.words.push_back(expert.up.tensor_id);
                graph_key.words.push_back(expert.down.tensor_id);
            }
            if (options_.cuda_graph == CudaGraphMode::cache) {
                const auto found = graph_entries_.find(graph_key);
                if (found != graph_entries_.end()) {
                    graph_entry = found->second.get();
                    graph_cache_hit = true;
                    ++runtime_stats_.cuda_graph_cache_hits;
                    graph_index_->touch(graph_key);
                } else {
                    ++runtime_stats_.cuda_graph_cache_misses;
                }
            }
            if (!graph_entry) {
                graph_candidate = std::make_unique<cuda::GraphEntry>(
                    &runtime_stats_, graph_key);
                const auto align_up = [](std::size_t value) {
                    constexpr auto alignment = alignof(std::max_align_t);
                    return (value + alignment - 1) & ~(alignment - 1);
                };
                auto& layout = graph_candidate->layout;
                layout.input_offset = 0;
                layout.contribution_offset = align_up(input_bytes);
                layout.descriptor_offset = align_up(
                    layout.contribution_offset + contribution_bytes);
                layout.output_offset = align_up(
                    layout.descriptor_offset + descriptor_bytes);
                layout.total_bytes = layout.output_offset + hidden_bytes;
                if (graph_candidate->staging.allocate(layout.total_bytes) !=
                    cudaSuccess) {
                    return Result<ResidentMoeLayerResult>::failure(
                        ErrorCode::backend_unavailable,
                        "resident MoE graph staging allocation failed");
                }
                graph_entry = graph_candidate.get();
            }
            auto* staging = static_cast<std::byte*>(graph_entry->staging.get());
            std::memcpy(staging + graph_entry->layout.input_offset,
                        input.data(), input_bytes);
            std::memcpy(staging + graph_entry->layout.contribution_offset,
                        contributions.data(), contribution_bytes);
            std::memcpy(staging + graph_entry->layout.descriptor_offset,
                        descriptors.data(), descriptor_bytes);
        }
        auto* staging = graph_entry
            ? static_cast<std::byte*>(graph_entry->staging.get())
            : nullptr;
        const void* host_input = graph_entry
            ? staging + graph_entry->layout.input_offset
            : static_cast<const void*>(input.data());
        const void* host_contributions = graph_entry
            ? staging + graph_entry->layout.contribution_offset
            : static_cast<const void*>(contributions.data());
        const void* host_descriptors = graph_entry
            ? staging + graph_entry->layout.descriptor_offset
            : static_cast<const void*>(descriptors.data());
        void* host_output = graph_entry
            ? staging + graph_entry->layout.output_offset
            : static_cast<void*>(output.data());
        const bool capture_graph = graph_entry && !graph_cache_hit;
        if (capture_graph &&
            cudaStreamBeginCapture(stream_, cudaStreamCaptureModeThreadLocal) !=
                cudaSuccess) {
            return Result<ResidentMoeLayerResult>::failure(
                ErrorCode::backend_unavailable,
                "resident MoE graph capture start failed");
        }
        const auto abort_graph_capture = [&] {
            if (!capture_graph) return;
            cudaGraph_t discarded{};
            cudaStreamEndCapture(stream_, &discarded);
            if (discarded) cudaGraphDestroy(discarded);
        };
        const auto h2d_start = std::chrono::steady_clock::now();
        if (!graph_cache_hit &&
            (cudaMemcpyAsync(device_input, host_input, input_bytes,
                            cudaMemcpyHostToDevice, stream_) != cudaSuccess ||
            cudaMemcpyAsync(device_contributions, host_contributions,
                            contribution_bytes, cudaMemcpyHostToDevice,
                            stream_) != cudaSuccess ||
            cudaMemcpyAsync(device_descriptors, host_descriptors,
                            descriptor_bytes, cudaMemcpyHostToDevice,
                            stream_) != cudaSuccess)) {
            abort_graph_capture();
            return Result<ResidentMoeLayerResult>::failure(
                ErrorCode::backend_unavailable,
                "resident MoE layer upload failed");
        }

        constexpr float alpha = 1.0F;
        constexpr float beta = 0.0F;
        const bool inline_timing = !capture_graph;
        const auto launch_dense = [&](std::size_t pair, DensePlan* plan,
                                      const void* weight, const float* source,
                                      float* output) {
            auto& start = resident_moe_layer_events_[pair * 2];
            auto& end = resident_moe_layer_events_[pair * 2 + 1];
            return (!inline_timing ||
                    cudaEventRecord(start.get(), stream_) == cudaSuccess) &&
                   cublasLtMatmul(
                       handle_, plan->operation.get(), &alpha, weight,
                       plan->weight_layout.get(), source,
                       plan->input_layout.get(), &beta, output,
                       plan->output_layout.get(), output,
                       plan->output_layout.get(), &plan->heuristic.algo,
                       nullptr, 0, stream_) == CUBLAS_STATUS_SUCCESS &&
                   (!inline_timing ||
                    cudaEventRecord(end.get(), stream_) == cudaSuccess);
        };
        const auto launch_grid = [&](std::size_t pair,
                                     std::size_t projection,
                                     const float* source, float* output,
                                     std::size_t rows, std::size_t cols,
                                     cuda::ExpertGridInputLayout layout) {
            auto& start = resident_moe_layer_events_[pair * 2];
            auto& end = resident_moe_layer_events_[pair * 2 + 1];
            return (!inline_timing ||
                    cudaEventRecord(start.get(), stream_) == cudaSuccess) &&
                   cuda::launch_mxfp4_matvec_grid(
                       source,
                       device_descriptors + projection * experts.size(),
                       output, rows, cols, experts.size(), 1, layout,
                       stream_) == cudaSuccess &&
                   (!inline_timing ||
                    cudaEventRecord(end.get(), stream_) == cudaSuccess);
        };
        const auto launch_simple = [&](std::size_t pair, auto&& launch) {
            auto& start = resident_moe_layer_events_[pair * 2];
            auto& end = resident_moe_layer_events_[pair * 2 + 1];
            return (!inline_timing ||
                    cudaEventRecord(start.get(), stream_) == cudaSuccess) &&
                   launch() == cudaSuccess &&
                   (!inline_timing ||
                    cudaEventRecord(end.get(), stream_) == cudaSuccess);
        };
        if (!graph_cache_hit &&
            (!launch_dense(0, plans[0], dense_members[0].primary,
                          device_input, device_latent) ||
            !launch_grid(1, 0, device_latent, device_expert_gate,
                         intermediate_width, latent_width,
                         cuda::ExpertGridInputLayout::shared_token_major) ||
            !launch_grid(2, 1, device_latent, device_expert_up,
                         intermediate_width, latent_width,
                         cuda::ExpertGridInputLayout::shared_token_major) ||
            !launch_simple(3, [&] {
                return cuda::launch_situ_glu(
                    device_expert_gate, device_expert_up,
                    device_expert_activation, expert_intermediate_count,
                    situ_beta, situ_linear.has_value(),
                    situ_linear.value_or(0.0F), false, stream_);
            }) ||
            !launch_grid(4, 2, device_expert_activation,
                         device_expert_output, routed_width,
                         intermediate_width,
                         cuda::ExpertGridInputLayout::expert_token_major) ||
            !launch_simple(5, [&] {
                return cuda::launch_ordered_expert_mix(
                    device_expert_output, device_contributions,
                    contributions, device_mixed, routed_width, stream_);
            }) ||
            !launch_simple(6, [&] {
                return cuda::launch_strict_rms_norm(
                    device_mixed,
                    static_cast<const float*>(dense_members[1].primary),
                    device_normalized, routed_width, epsilon, stream_);
            }) ||
            !launch_dense(7, plans[1], dense_members[2].primary,
                          device_normalized, device_routed) ||
            !launch_dense(8, plans[2], dense_members[3].primary,
                          device_input, device_shared_gate) ||
            !launch_dense(9, plans[3], dense_members[4].primary,
                          device_input, device_shared_up) ||
            !launch_simple(10, [&] {
                return cuda::launch_situ_glu(
                    device_shared_gate, device_shared_up,
                    device_shared_activation, shared_width, situ_beta,
                    situ_linear.has_value(), situ_linear.value_or(0.0F),
                    false, stream_);
            }) ||
            !launch_dense(11, plans[4], dense_members[5].primary,
                          device_shared_activation, device_shared) ||
            !launch_simple(12, [&] {
                return cuda::launch_vector_add(
                    device_routed, device_shared, device_final,
                    hidden_width, stream_);
            }))) {
            abort_graph_capture();
            return Result<ResidentMoeLayerResult>::failure(
                ErrorCode::backend_unavailable,
                "resident MoE layer launch failed");
        }

        const auto d2h_start = std::chrono::steady_clock::now();
        if (!graph_cache_hit &&
            cudaMemcpyAsync(host_output, device_final, hidden_bytes,
                            cudaMemcpyDeviceToHost, stream_) != cudaSuccess) {
            abort_graph_capture();
            return Result<ResidentMoeLayerResult>::failure(
                ErrorCode::backend_unavailable,
                "resident MoE layer output or synchronization failed");
        }
        if (capture_graph) {
            std::array<cudaEvent_t, 26> timing_events{};
            for (std::size_t index = 0; index < timing_events.size(); ++index) {
                timing_events[index] = resident_moe_layer_events_[index].get();
            }
            if (cudaStreamEndCapture(stream_, graph_entry->graph.out()) !=
                    cudaSuccess ||
                !graph_entry->graph.get() ||
                cuda::instrument_linear_graph(
                    graph_entry->graph.get(), timing_events, 3, 13) !=
                    cudaSuccess) {
                return Result<ResidentMoeLayerResult>::failure(
                    ErrorCode::backend_unavailable,
                    "resident MoE graph capture finalization failed");
            }
            bool instantiate = true;
            if (options_.cuda_graph == CudaGraphMode::update &&
                update_graph_entry_) {
                ++runtime_stats_.cuda_graph_update_attempts;
                cudaGraphExecUpdateResultInfo update_info{};
                if (cudaGraphExecUpdate(
                        update_graph_entry_->executable.get(),
                        graph_entry->graph.get(), &update_info) == cudaSuccess &&
                    update_info.result == cudaGraphExecUpdateSuccess) {
                    ++runtime_stats_.cuda_graph_update_successes;
                    graph_entry->executable =
                        std::move(update_graph_entry_->executable);
                    instantiate = false;
                } else {
                    ++runtime_stats_.cuda_graph_update_failures;
                }
            }
            if (instantiate &&
                cudaGraphInstantiate(graph_entry->executable.out(),
                                     graph_entry->graph.get(), nullptr,
                                     nullptr, 0) != cudaSuccess) {
                return Result<ResidentMoeLayerResult>::failure(
                    ErrorCode::backend_unavailable,
                    "resident MoE graph instantiation failed");
            }
            if (instantiate) ++runtime_stats_.cuda_graph_instantiations;
        }
        if (graph_entry &&
            cudaGraphLaunch(graph_entry->executable.get(), stream_) !=
                cudaSuccess) {
            return Result<ResidentMoeLayerResult>::failure(
                ErrorCode::backend_unavailable,
                "resident MoE graph launch failed");
        }
        const auto graph_wait_start = std::chrono::steady_clock::now();
        const auto graph_host_before_wait = graph_entry
            ? static_cast<std::uint64_t>(std::chrono::duration_cast<
                  std::chrono::nanoseconds>(graph_wait_start -
                                           graph_host_start).count())
            : 0;
        if (cudaStreamSynchronize(stream_) != cudaSuccess) {
            return Result<ResidentMoeLayerResult>::failure(
                ErrorCode::backend_unavailable,
                "resident MoE layer output or synchronization failed");
        }
        const auto graph_wait_end = std::chrono::steady_clock::now();
        if (graph_entry) {
            ++runtime_stats_.cuda_graph_launches;
            std::memcpy(output.data(), host_output, hidden_bytes);
        }
        std::array<std::uint64_t, 13> durations{};
        for (std::size_t pair = 0; pair < durations.size(); ++pair) {
            float milliseconds = 0.0F;
            const auto timing_status = cudaEventElapsedTime(
                &milliseconds,
                resident_moe_layer_events_[pair * 2].get(),
                resident_moe_layer_events_[pair * 2 + 1].get());
            if (timing_status != cudaSuccess) {
                return Result<ResidentMoeLayerResult>::failure(
                    ErrorCode::backend_unavailable,
                    std::string("resident MoE layer timing failed: ") +
                        cudaGetErrorString(timing_status));
            }
            durations[pair] = static_cast<std::uint64_t>(std::llround(
                static_cast<double>(milliseconds) * 1.0e6));
        }
        if (graph_entry) {
            if (graph_candidate) {
                if (options_.cuda_graph == CudaGraphMode::update) {
                    update_graph_entry_ = std::move(graph_candidate);
                } else {
                    const auto decision = graph_index_->touch(graph_key);
                    if (decision.evicted) {
                        graph_entries_.erase(*decision.evicted);
                        ++runtime_stats_.cuda_graph_cache_evictions;
                    }
                    graph_entries_.emplace(
                        graph_key, std::move(graph_candidate));
                }
                runtime_stats_.cuda_graph_resident_entries =
                    options_.cuda_graph == CudaGraphMode::update
                        ? 1
                        : graph_entries_.size();
                runtime_stats_.cuda_graph_peak_entries = std::max(
                    runtime_stats_.cuda_graph_peak_entries,
                    runtime_stats_.cuda_graph_resident_entries);
            }
            runtime_stats_.cuda_graph_host_nanoseconds +=
                graph_host_before_wait + static_cast<std::uint64_t>(
                    std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - graph_wait_end)
                        .count());
        }

        std::array<std::uint64_t, 3> expert_logical_bytes{};
        for (const auto& expert : experts) {
            const std::array expert_weights{
                expert.gate, expert.up, expert.down};
            for (std::size_t projection = 0; projection < 3; ++projection) {
                expert_logical_bytes[projection] +=
                    expert_weights[projection].packed.size_bytes() +
                    expert_weights[projection].scales.size_bytes();
            }
        }
        const auto activation_transfer_bytes =
            input_bytes + contribution_bytes + descriptor_bytes;
        ++runtime_stats_.stream_synchronization_count;
        runtime_stats_.activation_h2d_bytes += activation_transfer_bytes;
        runtime_stats_.weight_h2d_bytes += weight_transfer_bytes;
        runtime_stats_.device_to_host_bytes += hidden_bytes;
        ++runtime_stats_.resident_moe_layer_calls;
        runtime_stats_.resident_moe_layer_experts += experts.size();
        runtime_stats_.resident_moe_layer_kernel_launches += 13;
        runtime_stats_.resident_moe_layer_contribution_h2d_bytes +=
            contribution_bytes;
        ++runtime_stats_.resident_grid_calls;
        runtime_stats_.resident_grid_experts += experts.size();
        ++runtime_stats_.resident_grid_tokens;
        runtime_stats_.resident_grid_expert_tokens += experts.size();
        runtime_stats_.resident_grid_kernel_launches += 4;
        runtime_stats_.resident_grid_descriptor_h2d_bytes += descriptor_bytes;
        record(phase, ProfileOperation::activation_host_to_device,
               dense_precision, layer, h2d_start, 0,
               activation_transfer_bytes, 0, true);
        record(phase, ProfileOperation::weight_host_to_device,
               dense_precision, layer, operation_start, 0,
               weight_transfer_bytes, 0, true);
        record(phase, ProfileOperation::device_to_host, dense_precision, layer,
               d2h_start, 0, hidden_bytes, 0, true);
        const std::array operations{
            ProfileOperation::dense_matvec,
            ProfileOperation::mxfp4_matvec,
            ProfileOperation::mxfp4_matvec,
            ProfileOperation::situ_glu,
            ProfileOperation::mxfp4_matvec,
            ProfileOperation::moe_mix,
            ProfileOperation::rms_norm,
            ProfileOperation::dense_matvec,
            ProfileOperation::dense_matvec,
            ProfileOperation::dense_matvec,
            ProfileOperation::situ_glu,
            ProfileOperation::dense_matvec,
            ProfileOperation::residual_add,
        };
        const std::array logical_bytes{
            dense_views[0].values.size_bytes(), expert_logical_bytes[0],
            expert_logical_bytes[1], expert_intermediate_bytes,
            expert_logical_bytes[2], expert_output_bytes + contribution_bytes,
            routed_bytes + dense_views[1].values.size_bytes(),
            dense_views[2].values.size_bytes(),
            dense_views[3].values.size_bytes(),
            dense_views[4].values.size_bytes(), shared_bytes,
            dense_views[5].values.size_bytes(), hidden_bytes * 2,
        };
        for (std::size_t index = 0; index < operations.size(); ++index) {
            const auto precision = index >= 1 && index <= 4 && index != 3
                                       ? expert_precision
                                       : dense_precision;
            record(phase, operations[index], precision, layer,
                   operation_start, logical_bytes[index], 0,
                   durations[index], true);
        }
        return Result<ResidentMoeLayerResult>::success(
            {true, std::move(output)});
    }

    Result<OfficialKdaCudaResult> official_kda(
        std::span<const float> hidden, OfficialKdaCudaView weights,
        OfficialKdaCudaStateView state, OfficialKdaCudaConfig config,
        std::uint32_t layer, ProfilePhase phase,
        OfficialKdaStateControl state_control) override {
        const auto operation_start = std::chrono::steady_clock::now();
        const bool device_io = official_kda_device_input_ != nullptr;
        const auto product = [](std::size_t left, std::size_t right,
                                std::size_t& output) {
            if (right && left > std::numeric_limits<std::size_t>::max() / right) {
                return false;
            }
            output = left * right;
            return true;
        };
        std::size_t projection{};
        std::size_t history_count{};
        std::size_t head_square{};
        std::size_t recurrent_count{};
        const auto state_mode = state_control.mode;
        const bool valid_state_mode =
            state_mode == OfficialKdaStateMode::host_roundtrip ||
            state_mode == OfficialKdaStateMode::device_seed ||
            state_mode == OfficialKdaStateMode::device_continue ||
            state_mode == OfficialKdaStateMode::device_publish;
        const bool host_state_input =
            state_mode == OfficialKdaStateMode::host_roundtrip ||
            state_mode == OfficialKdaStateMode::device_seed;
        const bool device_continuation =
            state_mode == OfficialKdaStateMode::device_continue ||
            state_mode == OfficialKdaStateMode::device_publish;
        const bool publish_state =
            state_mode == OfficialKdaStateMode::host_roundtrip ||
            state_mode == OfficialKdaStateMode::device_publish;
        const bool retain_state =
            state_mode == OfficialKdaStateMode::device_seed ||
            state_mode == OfficialKdaStateMode::device_continue;
        const bool device_state_mode =
            state_mode != OfficialKdaStateMode::host_roundtrip;
        const auto device_state_slot_index =
            layer == 1 ? std::size_t{0} : layer == 2 ? std::size_t{1}
                                                   : std::size_t{2};
        if (!valid_state_mode ||
            options_.kind != BackendKind::cuda_custom ||
            options_.cuda_boundary != CudaBoundaryMode::moe_layer ||
            options_.cuda_allocation != CudaAllocationMode::reused ||
            options_.cuda_transfer != CudaTransferMode::synchronous ||
            options_.cuda_batching != CudaBatchingMode::resident_grid ||
            (options_.cuda_weight_validation ==
                 CudaWeightValidationMode::admission &&
             options_.cuda_weights != CudaWeightMode::resident) ||
            (device_state_mode &&
             (options_.cuda_weights != CudaWeightMode::resident ||
              options_.cuda_weight_validation !=
                  CudaWeightValidationMode::admission)) ||
            !config.hidden_size || !config.heads || !config.head_dim ||
            config.conv_width < 2 || config.head_dim > 1024 ||
            !std::isfinite(config.rms_norm_epsilon) ||
            config.rms_norm_epsilon <= 0.0F ||
            !std::isfinite(config.gate_lower_bound) ||
            config.gate_lower_bound >= 0.0F || hidden.empty() ||
            hidden.size() % config.hidden_size != 0 ||
            !product(config.heads, config.head_dim, projection) ||
            !product(config.conv_width - 1, projection, history_count) ||
            !product(config.head_dim, config.head_dim, head_square) ||
            !product(config.heads, head_square, recurrent_count) ||
            (host_state_input &&
             (state.conv_q.size() != history_count ||
              state.conv_k.size() != history_count ||
              state.conv_v.size() != history_count ||
              state.recurrent_v_first.size() != recurrent_count)) ||
            (!host_state_input &&
             (!state.conv_q.empty() || !state.conv_k.empty() ||
              !state.conv_v.empty() ||
              !state.recurrent_v_first.empty())) ||
            ((state_mode == OfficialKdaStateMode::host_roundtrip ||
              state_mode == OfficialKdaStateMode::device_seed) &&
             (state_control.token.owner ||
              state_control.token.generation))) {
            return Result<OfficialKdaCudaResult>::failure(ErrorCode::invalid_extent);
        }
        if (device_state_mode && device_state_slot_index >= 2) {
            return Result<OfficialKdaCudaResult>::failure(
                ErrorCode::invalid_state);
        }
        const auto finite_f32 = [](std::span<const float> values) {
            return std::all_of(values.begin(), values.end(),
                               [](float value) { return std::isfinite(value); });
        };
        const auto valid_bf16_shape = [](Bf16WeightView view,
                                         std::size_t rows, std::size_t cols) {
            return view.tensor_id && view.rows == rows && view.cols == cols &&
                   view.values.size() == rows * cols;
        };
        const auto valid_bf16_state = [](std::span<const std::uint16_t> values) {
            return std::all_of(values.begin(), values.end(),
                               [](std::uint16_t value) {
                                   return (value & 0x7f80U) != 0x7f80U;
                               });
        };
        const auto valid_f32_shape = [](DenseWeightView view,
                                        std::size_t rows,
                                        std::size_t cols) {
            return view.tensor_id && view.rows == rows && view.cols == cols &&
                   view.values.size() == rows * cols;
        };
        const auto valid_vector_shape = [](DenseVectorView view,
                                           std::size_t size) {
            return view.tensor_id && view.values.size() == size;
        };
        if ((!device_io && !finite_f32(hidden)) ||
            (host_state_input &&
             (!finite_f32(state.recurrent_v_first) ||
              !valid_bf16_state(state.conv_q) ||
              !valid_bf16_state(state.conv_k) ||
              !valid_bf16_state(state.conv_v))) ||
            !valid_bf16_shape(weights.q_proj, projection, config.hidden_size) ||
            !valid_bf16_shape(weights.k_proj, projection, config.hidden_size) ||
            !valid_bf16_shape(weights.v_proj, projection, config.hidden_size) ||
            !valid_f32_shape(weights.q_conv, projection, config.conv_width) ||
            !valid_f32_shape(weights.k_conv, projection, config.conv_width) ||
            !valid_f32_shape(weights.v_conv, projection, config.conv_width) ||
            !valid_bf16_shape(weights.f_a_proj, config.head_dim,
                              config.hidden_size) ||
            !valid_bf16_shape(weights.f_b_proj, projection, config.head_dim) ||
            !valid_vector_shape(weights.a_log, config.head_dim) ||
            !valid_vector_shape(weights.dt_bias, projection) ||
            !valid_bf16_shape(weights.b_proj, config.heads,
                              config.hidden_size) ||
            !valid_bf16_shape(weights.g_proj, projection,
                              config.hidden_size) ||
            !valid_vector_shape(weights.o_norm, config.head_dim) ||
            !valid_bf16_shape(weights.o_proj, config.hidden_size, projection)) {
            return Result<OfficialKdaCudaResult>::failure(ErrorCode::invalid_extent);
        }
        const auto same_config = [](OfficialKdaCudaConfig left,
                                    OfficialKdaCudaConfig right) {
            return left.hidden_size == right.hidden_size &&
                   left.heads == right.heads &&
                   left.head_dim == right.head_dim &&
                   left.conv_width == right.conv_width &&
                   left.rms_norm_epsilon == right.rms_norm_epsilon &&
                   left.gate_lower_bound == right.gate_lower_bound;
        };
        auto* device_state_slot = device_state_slot_index < 2
                                      ? &device_state_slots_[device_state_slot_index]
                                      : nullptr;
        if ((state_mode == OfficialKdaStateMode::device_seed &&
             device_state_slot->active) ||
            (device_continuation &&
             (!device_state_slot->active ||
              state_control.token.owner != device_state_owner_ ||
              state_control.token.generation !=
                  device_state_slot->generation ||
              !same_config(device_state_slot->config, config)))) {
            return Result<OfficialKdaCudaResult>::failure(
                ErrorCode::invalid_state);
        }
        const std::array bf16_views{
            weights.q_proj, weights.k_proj, weights.v_proj,
            weights.f_a_proj, weights.f_b_proj, weights.b_proj,
            weights.g_proj, weights.o_proj};
        const std::array f32_views{
            weights.q_conv, weights.k_conv, weights.v_conv,
            DenseWeightView{weights.a_log.tensor_id, weights.a_log.values,
                            1, config.head_dim},
            DenseWeightView{weights.dt_bias.tensor_id, weights.dt_bias.values,
                            1, projection},
            DenseWeightView{weights.o_norm.tensor_id, weights.o_norm.values,
                            1, config.head_dim}};
        std::unordered_set<std::uint64_t> tensor_ids;
        std::uint64_t total_weight_bytes{};
        for (const auto& view : bf16_views) {
            if (!tensor_ids.insert(view.tensor_id).second) {
                return Result<OfficialKdaCudaResult>::failure(ErrorCode::invalid_extent);
            }
            total_weight_bytes += view.values.size_bytes();
        }
        for (const auto& view : f32_views) {
            if (!tensor_ids.insert(view.tensor_id).second) {
                return Result<OfficialKdaCudaResult>::failure(ErrorCode::invalid_extent);
            }
            total_weight_bytes += view.values.size_bytes();
        }
        std::array<bool, 8> bf16_validation_hits{};
        std::array<bool, 8> bf16_validation_scans{};
        std::array<bool, 6> f32_validation_hits{};
        std::array<bool, 6> f32_validation_scans{};
        const auto classify = [this](std::uint64_t tensor_id,
                                     const void* data, std::size_t bytes,
                                     std::size_t rows, std::size_t cols,
                                     bool& hit, bool& scan) {
            if (options_.cuda_weight_validation ==
                CudaWeightValidationMode::per_call) {
                scan = true;
                return true;
            }
            const ImmutableWeightIdentity identity{data, bytes, rows, cols};
            const auto found = immutable_weights_.find(tensor_id);
            if (found == immutable_weights_.end()) {
                scan = true;
                return true;
            }
            if (found->second == identity) {
                hit = true;
                return true;
            }
            return false;
        };
        for (std::size_t index = 0; index < bf16_views.size(); ++index) {
            const auto& view = bf16_views[index];
            if (!classify(view.tensor_id, view.values.data(),
                          view.values.size_bytes(), view.rows, view.cols,
                          bf16_validation_hits[index],
                          bf16_validation_scans[index])) {
                return Result<OfficialKdaCudaResult>::failure(
                    ErrorCode::invalid_extent);
            }
        }
        for (std::size_t index = 0; index < f32_views.size(); ++index) {
            const auto& view = f32_views[index];
            if (!classify(view.tensor_id, view.values.data(),
                          view.values.size_bytes(), view.rows, view.cols,
                          f32_validation_hits[index],
                          f32_validation_scans[index])) {
                return Result<OfficialKdaCudaResult>::failure(
                    ErrorCode::invalid_extent);
            }
        }
        const auto validation_start = std::chrono::steady_clock::now();
        bool immutable_finite = true;
        bool validation_scanned = false;
        for (std::size_t index = 0; index < bf16_views.size(); ++index) {
            if (!bf16_validation_scans[index]) continue;
            validation_scanned = true;
            ++runtime_stats_.immutable_validation_scans;
            runtime_stats_.immutable_validation_bytes +=
                bf16_views[index].values.size_bytes();
            if (!std::all_of(
                    bf16_views[index].values.begin(),
                    bf16_views[index].values.end(), [](std::uint16_t value) {
                        return (value & 0x7f80U) != 0x7f80U;
                    })) {
                immutable_finite = false;
            }
        }
        for (std::size_t index = 0; index < f32_views.size(); ++index) {
            if (!f32_validation_scans[index]) continue;
            validation_scanned = true;
            ++runtime_stats_.immutable_validation_scans;
            runtime_stats_.immutable_validation_bytes +=
                f32_views[index].values.size_bytes();
            if (!finite_f32(f32_views[index].values)) immutable_finite = false;
        }
        if (validation_scanned) {
            runtime_stats_.immutable_validation_nanoseconds +=
                static_cast<std::uint64_t>(std::chrono::duration_cast<
                    std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - validation_start)
                                               .count());
        }
        if (!immutable_finite) {
            return Result<OfficialKdaCudaResult>::failure(
                ErrorCode::invalid_extent);
        }
        if (options_.cuda_weight_validation ==
            CudaWeightValidationMode::admission) {
            for (std::size_t index = 0; index < bf16_views.size(); ++index) {
                const auto& view = bf16_views[index];
                if (bf16_validation_hits[index]) {
                    ++runtime_stats_.immutable_validation_hits;
                } else {
                    immutable_weights_.emplace(
                        view.tensor_id,
                        ImmutableWeightIdentity{
                            view.values.data(), view.values.size_bytes(),
                            view.rows, view.cols});
                }
            }
            for (std::size_t index = 0; index < f32_views.size(); ++index) {
                const auto& view = f32_views[index];
                if (f32_validation_hits[index]) {
                    ++runtime_stats_.immutable_validation_hits;
                } else {
                    immutable_weights_.emplace(
                        view.tensor_id,
                        ImmutableWeightIdentity{
                            view.values.data(), view.values.size_bytes(),
                            view.rows, view.cols});
                }
            }
        }
        if (options_.cuda_weights == CudaWeightMode::resident &&
            (!resident_weights_ || total_weight_bytes > options_.cuda_resident_bytes)) {
            return Result<OfficialKdaCudaResult>::failure(ErrorCode::invalid_extent);
        }
        std::vector<std::unique_ptr<cuda::DeviceAllocation>> transient;
        transient.reserve(bf16_views.size() + f32_views.size());
        std::uint64_t uploaded_weight_bytes{};
        const auto acquire = [&](cuda::ResidentWeightKey key,
                                 std::span<const std::byte> bytes)
            -> Result<const void*> {
            if (resident_weights_) {
                auto value = resident_weights_->acquire(key, bytes, {});
                if (!value || value.value().disposition ==
                                  cuda::ResidentDisposition::bypass) {
                    return Result<const void*>::failure(ErrorCode::invalid_extent);
                }
                uploaded_weight_bytes += value.value().uploaded_bytes;
                return Result<const void*>::success(value.value().primary);
            }
            auto allocation = std::make_unique<cuda::DeviceAllocation>(
                &memory_stats_, &runtime_stats_);
            if (allocation->allocate(bytes.size()) != cudaSuccess ||
                cudaMemcpyAsync(allocation->get(), bytes.data(), bytes.size(),
                                cudaMemcpyHostToDevice, stream_) != cudaSuccess) {
                return Result<const void*>::failure(ErrorCode::backend_unavailable);
            }
            const auto* pointer = allocation->get();
            uploaded_weight_bytes += bytes.size();
            transient.push_back(std::move(allocation));
            return Result<const void*>::success(pointer);
        };
        std::array<const std::uint16_t*, 8> device_bf16{};
        for (std::size_t index = 0; index < bf16_views.size(); ++index) {
            const auto& view = bf16_views[index];
            auto value = acquire(
                {view.tensor_id, cuda::WeightRepresentation::dense_bf16,
                 view.rows, view.cols, 0}, std::as_bytes(view.values));
            if (!value) return Result<OfficialKdaCudaResult>::failure(
                value.error(), value.message());
            device_bf16[index] = static_cast<const std::uint16_t*>(value.value());
        }
        std::array<const float*, 6> device_f32{};
        for (std::size_t index = 0; index < f32_views.size(); ++index) {
            const auto& view = f32_views[index];
            auto value = acquire(
                {view.tensor_id, cuda::WeightRepresentation::dense_fp32,
                 view.rows, view.cols, 0}, std::as_bytes(view.values));
            if (!value) return Result<OfficialKdaCudaResult>::failure(
                value.error(), value.message());
            device_f32[index] = static_cast<const float*>(value.value());
        }
        const auto sequence = hidden.size() / config.hidden_size;
        std::size_t sequence_projection{};
        std::size_t sequence_heads{};
        std::size_t sequence_hidden{};
        if (!product(sequence, projection, sequence_projection) ||
            !product(sequence, config.heads, sequence_heads) ||
            !product(sequence, config.hidden_size, sequence_hidden)) {
            return Result<OfficialKdaCudaResult>::failure(ErrorCode::invalid_extent);
        }
        const auto float_count = 2 * sequence_hidden + 13 * sequence_projection +
                                 sequence * config.head_dim + 2 * sequence_heads;
        const auto float_bytes = float_count * sizeof(float);
        const auto conv_state_bytes = history_count * sizeof(std::uint16_t);
        const auto conv_total_bytes = 3 * conv_state_bytes;
        const auto recurrent_offset =
            (conv_total_bytes + alignof(float) - 1) & ~(alignof(float) - 1);
        const auto state_allocation_bytes =
            recurrent_offset + recurrent_count * sizeof(float);
        auto* state_buffer = &official_kda_state_;
        if (device_state_slot_index == 0 && device_state_mode) {
            state_buffer = &official_kda_state_one_;
        } else if (device_state_slot_index == 1 && device_state_mode) {
            state_buffer = &official_kda_state_two_;
        }
        if (official_kda_scratch_.reserve(float_bytes) != cudaSuccess ||
            state_buffer->reserve(state_allocation_bytes) != cudaSuccess) {
            return Result<OfficialKdaCudaResult>::failure(
                ErrorCode::backend_unavailable);
        }
        auto* cursor = static_cast<float*>(official_kda_scratch_.get());
        const auto take = [&cursor](std::size_t count) {
            auto* result = cursor;
            cursor += count;
            return result;
        };
        auto* device_hidden = take(sequence_hidden);
        auto* projected_q = take(sequence_projection);
        auto* projected_k = take(sequence_projection);
        auto* projected_v = take(sequence_projection);
        auto* forget_low = take(sequence * config.head_dim);
        auto* forget = take(sequence_projection);
        auto* beta_projection = take(sequence_heads);
        auto* output_gate = take(sequence_projection);
        auto* convolved_q = take(sequence_projection);
        auto* convolved_k = take(sequence_projection);
        auto* convolved_v = take(sequence_projection);
        auto* q = take(sequence_projection);
        auto* k = take(sequence_projection);
        auto* log_decay = take(sequence_projection);
        auto* beta = take(sequence_heads);
        auto* recurrent_output = take(sequence_projection);
        auto* gated = take(sequence_projection);
        auto* output = take(sequence_hidden);
        auto* state_cursor =
            static_cast<std::byte*>(state_buffer->get());
        auto* conv_cursor = reinterpret_cast<std::uint16_t*>(state_cursor);
        auto* conv_q = conv_cursor;
        auto* conv_k = conv_q + history_count;
        auto* conv_v = conv_k + history_count;
        auto* recurrent =
            reinterpret_cast<float*>(state_cursor + recurrent_offset);
        const auto state_bytes =
            conv_total_bytes + recurrent_count * sizeof(float);
        const auto state_h2d_bytes = host_state_input ? state_bytes : 0;
        if (official_kda_event_start_.ensure() != cudaSuccess ||
            official_kda_event_end_.ensure() != cudaSuccess) {
            return Result<OfficialKdaCudaResult>::failure(
                ErrorCode::backend_unavailable);
        }
        if (!device_state_mode && device_state_slot &&
            device_state_slot->active) {
            device_state_slot->active = false;
            ++runtime_stats_.official_kda_device_state_invalidations;
        } else if (device_continuation) {
            device_state_slot->active = false;
        }
        const auto state_failure = [this, device_state_mode,
                                    device_state_slot]() {
            if (device_state_mode) {
                device_state_slot->active = false;
                ++runtime_stats_.official_kda_device_state_invalidations;
            }
            return Result<OfficialKdaCudaResult>::failure(
                ErrorCode::backend_unavailable);
        };
        if (cudaMemcpyAsync(
                device_hidden,
                device_io ? static_cast<const void*>(official_kda_device_input_)
                          : static_cast<const void*>(hidden.data()),
                hidden.size_bytes(),
                device_io ? cudaMemcpyDeviceToDevice : cudaMemcpyHostToDevice,
                stream_) != cudaSuccess ||
            (host_state_input &&
             (cudaMemcpyAsync(conv_q, state.conv_q.data(), conv_state_bytes,
                              cudaMemcpyHostToDevice, stream_) != cudaSuccess ||
              cudaMemcpyAsync(conv_k, state.conv_k.data(), conv_state_bytes,
                              cudaMemcpyHostToDevice, stream_) != cudaSuccess ||
              cudaMemcpyAsync(conv_v, state.conv_v.data(), conv_state_bytes,
                              cudaMemcpyHostToDevice, stream_) != cudaSuccess ||
              cudaMemcpyAsync(recurrent, state.recurrent_v_first.data(),
                              state.recurrent_v_first.size_bytes(),
                              cudaMemcpyHostToDevice, stream_) != cudaSuccess)) ||
            cuda::launch_round_bf16_inplace(device_hidden, sequence_hidden,
                                            stream_) != cudaSuccess ||
            cudaEventRecord(official_kda_event_start_.get(), stream_) != cudaSuccess) {
            return state_failure();
        }
        std::uint64_t launches = 1;
        for (std::size_t token = 0; token < sequence; ++token) {
            const auto* input = device_hidden + token * config.hidden_size;
            if (cuda::launch_bf16_matvec(input, device_bf16[0],
                    projected_q + token * projection, projection,
                    config.hidden_size, stream_) != cudaSuccess ||
                cuda::launch_bf16_matvec(input, device_bf16[1],
                    projected_k + token * projection, projection,
                    config.hidden_size, stream_) != cudaSuccess ||
                cuda::launch_bf16_matvec(input, device_bf16[2],
                    projected_v + token * projection, projection,
                    config.hidden_size, stream_) != cudaSuccess ||
                cuda::launch_bf16_matvec(input, device_bf16[3],
                    forget_low + token * config.head_dim, config.head_dim,
                    config.hidden_size, stream_) != cudaSuccess ||
                cuda::launch_bf16_matvec(input, device_bf16[5],
                    beta_projection + token * config.heads, config.heads,
                    config.hidden_size, stream_) != cudaSuccess ||
                cuda::launch_bf16_matvec(input, device_bf16[6],
                    output_gate + token * projection, projection,
                    config.hidden_size, stream_) != cudaSuccess) {
                return state_failure();
            }
            launches += 6;
        }
        for (std::size_t token = 0; token < sequence; ++token) {
            if (cuda::launch_bf16_matvec(
                    forget_low + token * config.head_dim, device_bf16[4],
                    forget + token * projection, projection,
                    config.head_dim, stream_) != cudaSuccess) {
                return state_failure();
            }
            ++launches;
        }
        if (cuda::launch_official_kda_short_conv(
                projected_q, conv_q, device_f32[0], convolved_q, sequence,
                projection, config.conv_width, stream_) != cudaSuccess ||
            cuda::launch_official_kda_short_conv(
                projected_k, conv_k, device_f32[1], convolved_k, sequence,
                projection, config.conv_width, stream_) != cudaSuccess ||
            cuda::launch_official_kda_short_conv(
                projected_v, conv_v, device_f32[2], convolved_v, sequence,
                projection, config.conv_width, stream_) != cudaSuccess ||
            cuda::launch_official_kda_normalize_qk(
                convolved_q, convolved_k, q, k, sequence, config.heads,
                config.head_dim, stream_) != cudaSuccess ||
            cuda::launch_official_kda_decay_beta(
                forget, beta_projection, device_f32[3], device_f32[4],
                log_decay, beta, sequence, config.heads, config.head_dim,
                config.gate_lower_bound, stream_) != cudaSuccess ||
            cuda::launch_official_kda_recurrence(
                q, k, convolved_v, log_decay, beta, recurrent,
                recurrent_output, sequence, config.heads, config.head_dim,
                stream_) != cudaSuccess ||
            cuda::launch_official_kda_gate_norm(
                recurrent_output, output_gate, device_f32[5], gated,
                sequence, config.heads, config.head_dim,
                config.rms_norm_epsilon, stream_) != cudaSuccess) {
            return state_failure();
        }
        launches += 7;
        for (std::size_t token = 0; token < sequence; ++token) {
            if (cuda::launch_bf16_matvec(
                    gated + token * projection, device_bf16[7],
                    output + token * config.hidden_size, config.hidden_size,
                    projection, stream_) != cudaSuccess) {
                return state_failure();
            }
            ++launches;
        }
        if (cudaEventRecord(official_kda_event_end_.get(), stream_) != cudaSuccess) {
            return state_failure();
        }
        OfficialKdaCudaResult result;
        result.executed = true;
        result.state_published = publish_state;
        if (!device_io) result.output.resize(sequence_hidden);
        if (publish_state) {
            result.conv_q.resize(history_count);
            result.conv_k.resize(history_count);
            result.conv_v.resize(history_count);
            result.recurrent_v_first.resize(recurrent_count);
        }
        if (cudaMemcpyAsync(
                device_io ? static_cast<void*>(official_kda_device_output_)
                          : static_cast<void*>(result.output.data()),
                output, hidden.size_bytes(),
                device_io ? cudaMemcpyDeviceToDevice : cudaMemcpyDeviceToHost,
                stream_) != cudaSuccess ||
            (publish_state &&
             (cudaMemcpyAsync(result.conv_q.data(), conv_q, conv_state_bytes,
                              cudaMemcpyDeviceToHost, stream_) != cudaSuccess ||
              cudaMemcpyAsync(result.conv_k.data(), conv_k, conv_state_bytes,
                              cudaMemcpyDeviceToHost, stream_) != cudaSuccess ||
              cudaMemcpyAsync(result.conv_v.data(), conv_v, conv_state_bytes,
                              cudaMemcpyDeviceToHost, stream_) != cudaSuccess ||
              cudaMemcpyAsync(result.recurrent_v_first.data(), recurrent,
                              recurrent_count * sizeof(float),
                              cudaMemcpyDeviceToHost, stream_) != cudaSuccess)) ||
            cudaStreamSynchronize(stream_) != cudaSuccess) {
            return state_failure();
        }
        float milliseconds{};
        if (cudaEventElapsedTime(&milliseconds, official_kda_event_start_.get(),
                                 official_kda_event_end_.get()) != cudaSuccess) {
            return state_failure();
        }
        const auto state_d2h_bytes = publish_state ? state_bytes : 0;
        ++runtime_stats_.stream_synchronization_count;
        runtime_stats_.weight_h2d_bytes += uploaded_weight_bytes;
        runtime_stats_.activation_h2d_bytes +=
            (device_io ? 0 : hidden.size_bytes()) + state_h2d_bytes;
        runtime_stats_.device_to_host_bytes +=
            (device_io ? 0 : hidden.size_bytes()) + state_d2h_bytes;
        ++runtime_stats_.official_kda_calls;
        runtime_stats_.official_kda_kernel_launches += launches;
        runtime_stats_.official_kda_state_h2d_bytes += state_h2d_bytes;
        runtime_stats_.official_kda_state_d2h_bytes += state_d2h_bytes;
        runtime_stats_.official_kda_output_d2h_bytes +=
            device_io ? 0 : hidden.size_bytes();
        if (state_mode == OfficialKdaStateMode::device_seed) {
            ++runtime_stats_.official_kda_device_state_seeds;
        } else if (device_continuation) {
            ++runtime_stats_.official_kda_device_state_continuations;
        }
        if (state_mode == OfficialKdaStateMode::device_publish) {
            ++runtime_stats_.official_kda_device_state_publications;
        }
        if (retain_state) {
            ++device_state_generation_;
            if (!device_state_generation_) ++device_state_generation_;
            device_state_slot->generation = device_state_generation_;
            device_state_slot->config = config;
            device_state_slot->active = true;
            result.device_state =
                {device_state_owner_, device_state_generation_};
        }
        record(phase, ProfileOperation::dense_matvec,
               NumericPrecision::bf16_rounded, layer, operation_start,
               total_weight_bytes, uploaded_weight_bytes,
               static_cast<std::uint64_t>(std::llround(milliseconds * 1.0e6)),
               true);
        return Result<OfficialKdaCudaResult>::success(std::move(result));
    }

    Result<bool> discard_official_kda_device_state(
        OfficialKdaDeviceStateToken token) override {
        if (token.owner != device_state_owner_) {
            return Result<bool>::failure(ErrorCode::invalid_state);
        }
        auto slot = std::find_if(
            device_state_slots_.begin(), device_state_slots_.end(),
            [token](const OfficialKdaDeviceStateSlot& candidate) {
                return candidate.active &&
                       candidate.generation == token.generation;
            });
        if (slot == device_state_slots_.end()) {
            return Result<bool>::failure(ErrorCode::invalid_state);
        }
        slot->active = false;
        ++runtime_stats_.official_kda_device_state_invalidations;
        return Result<bool>::success(true);
    }

    Result<OfficialLayerFrontResult> official_layer_front(
        std::span<const float> host_hidden, std::span<const float> host_block,
        OfficialLayerHiddenToken input_token, OfficialLayerFrontWeights weights,
        OfficialKdaCudaStateView state, OfficialKdaCudaConfig config,
        std::uint32_t layer, ProfilePhase phase,
        OfficialKdaStateControl state_control) override {
        const bool host_input = !host_hidden.empty() || !host_block.empty();
        const bool token_input = input_token.owner || input_token.generation ||
                                 input_token.producing_layer ||
                                 input_token.width || input_token.slot;
        const auto valid_bf16 = [](Bf16WeightView view, std::size_t rows,
                                   std::size_t cols) {
            return view.tensor_id && rows && cols && view.rows == rows &&
                   view.cols == cols && view.values.size() == rows * cols &&
                   std::all_of(view.values.begin(), view.values.end(),
                               [](std::uint16_t value) {
                                   return (value & 0x7f80U) != 0x7f80U;
                               });
        };
        if (options_.kind != BackendKind::cuda_custom ||
            options_.cuda_boundary != CudaBoundaryMode::moe_layer ||
            options_.cuda_allocation != CudaAllocationMode::reused ||
            options_.cuda_transfer != CudaTransferMode::synchronous ||
            options_.cuda_batching != CudaBatchingMode::resident_grid ||
            options_.cuda_weights != CudaWeightMode::resident ||
            options_.cuda_weight_validation !=
                CudaWeightValidationMode::admission ||
            !resident_weights_ || layer < 1 || layer > 2 ||
            !config.hidden_size ||
            config.hidden_size >
                std::numeric_limits<std::size_t>::max() /
                    (4 * sizeof(float)) ||
            host_input == token_input ||
            (state_control.mode != OfficialKdaStateMode::device_seed &&
             state_control.mode != OfficialKdaStateMode::device_continue &&
             state_control.mode != OfficialKdaStateMode::device_publish) ||
            (host_input &&
             (layer != 1 || host_hidden.size() != config.hidden_size ||
              host_block.size() != config.hidden_size ||
              !std::all_of(host_hidden.begin(), host_hidden.end(),
                           [](float value) { return std::isfinite(value); }) ||
              !std::all_of(host_block.begin(), host_block.end(),
                           [](float value) { return std::isfinite(value); }))) ||
            !weights.self_residual_norm.tensor_id ||
            weights.self_residual_norm.values.size() != config.hidden_size ||
            !valid_bf16(weights.self_residual_proj, 1, config.hidden_size) ||
            !weights.input_norm.tensor_id ||
            weights.input_norm.values.size() != config.hidden_size) {
            return Result<OfficialLayerFrontResult>::failure(
                ErrorCode::invalid_extent);
        }
        if (moe_prepared_active_) {
            return Result<OfficialLayerFrontResult>::failure(
                ErrorCode::invalid_state);
        }

        std::size_t slot_index = 2;
        if (host_input) {
            if (std::any_of(
                    layer_hidden_slots_.begin(), layer_hidden_slots_.end(),
                    [](const OfficialLayerHiddenSlot& slot) {
                        return slot.state !=
                               OfficialLayerHiddenSlotState::free;
                    })) {
                return Result<OfficialLayerFrontResult>::failure(
                    ErrorCode::invalid_state);
            }
            for (std::size_t index = 0; index < layer_hidden_slots_.size();
                 ++index) {
                if (layer_hidden_slots_[index].state ==
                    OfficialLayerHiddenSlotState::free) {
                    slot_index = index;
                    break;
                }
            }
        } else {
            if (input_token.owner != layer_hidden_owner_ ||
                input_token.generation == 0 || input_token.slot >= 2 ||
                input_token.producing_layer + 1 != layer ||
                input_token.width != config.hidden_size) {
                return Result<OfficialLayerFrontResult>::failure(
                    ErrorCode::invalid_state);
            }
            slot_index = input_token.slot;
            const auto& slot = layer_hidden_slots_[slot_index];
            if (slot.state != OfficialLayerHiddenSlotState::live ||
                slot.generation != input_token.generation ||
                slot.producing_layer != input_token.producing_layer ||
                slot.width != input_token.width ||
                layer_hidden_slots_[1 - slot_index].state !=
                    OfficialLayerHiddenSlotState::free) {
                return Result<OfficialLayerFrontResult>::failure(
                    ErrorCode::invalid_state);
            }
        }
        if (slot_index >= 2) {
            return Result<OfficialLayerFrontResult>::failure(
                ErrorCode::invalid_state);
        }
        auto* activation = slot_index == 0 ? &official_layer_hidden_one_
                                           : &official_layer_hidden_two_;
        const auto hidden_bytes = config.hidden_size * sizeof(float);
        if (activation->reserve(hidden_bytes * 2) != cudaSuccess ||
            official_layer_front_scratch_.reserve(hidden_bytes * 4) !=
                cudaSuccess) {
            return Result<OfficialLayerFrontResult>::failure(
                ErrorCode::backend_unavailable);
        }
        auto* device_hidden = static_cast<float*>(activation->get());
        auto* device_block = device_hidden + config.hidden_size;
        auto* front_scratch =
            static_cast<float*>(official_layer_front_scratch_.get());
        auto* rounded_hidden = front_scratch;
        auto* normalized_input = rounded_hidden + config.hidden_size;
        auto* kda_output = normalized_input + config.hidden_size;
        auto* prefix = kda_output + config.hidden_size;

        const std::array self_views{
            Bf16WeightView{weights.self_residual_norm.values, 1,
                           config.hidden_size,
                           weights.self_residual_norm.tensor_id},
            weights.self_residual_proj,
            Bf16WeightView{weights.input_norm.values, 1, config.hidden_size,
                           weights.input_norm.tensor_id},
        };
        std::array<const std::uint16_t*, 3> device_views{};
        std::unordered_set<std::uint64_t> tensor_ids;
        std::uint64_t uploaded_weight_bytes{};
        for (std::size_t index = 0; index < self_views.size(); ++index) {
            const auto& view = self_views[index];
            if (!tensor_ids.insert(view.tensor_id).second ||
                !valid_bf16(view, view.rows, view.cols)) {
                return Result<OfficialLayerFrontResult>::failure(
                    ErrorCode::invalid_extent);
            }
            const ImmutableWeightIdentity identity{
                view.values.data(), view.values.size_bytes(), view.rows,
                view.cols};
            const auto found = immutable_weights_.find(view.tensor_id);
            if (found != immutable_weights_.end() &&
                found->second != identity) {
                return Result<OfficialLayerFrontResult>::failure(
                    ErrorCode::invalid_extent);
            }
            if (found == immutable_weights_.end()) {
                ++runtime_stats_.immutable_validation_scans;
                runtime_stats_.immutable_validation_bytes +=
                    view.values.size_bytes();
                immutable_weights_.emplace(view.tensor_id, identity);
            } else {
                ++runtime_stats_.immutable_validation_hits;
            }
            auto acquired = resident_weights_->acquire(
                {view.tensor_id, cuda::WeightRepresentation::dense_bf16,
                 view.rows, view.cols, 0},
                std::as_bytes(view.values), {});
            if (!acquired || acquired.value().disposition ==
                                 cuda::ResidentDisposition::bypass) {
                return Result<OfficialLayerFrontResult>::failure(
                    ErrorCode::invalid_extent);
            }
            uploaded_weight_bytes += acquired.value().uploaded_bytes;
            device_views[index] = static_cast<const std::uint16_t*>(
                acquired.value().primary);
        }
        auto& hidden_slot = layer_hidden_slots_[slot_index];
        hidden_slot.state = OfficialLayerHiddenSlotState::prepared;
        if (host_input &&
            (cudaMemcpyAsync(device_hidden, host_hidden.data(), hidden_bytes,
                             cudaMemcpyHostToDevice, stream_) != cudaSuccess ||
             cudaMemcpyAsync(device_block, host_block.data(), hidden_bytes,
                             cudaMemcpyHostToDevice, stream_) != cudaSuccess)) {
            hidden_slot.state = OfficialLayerHiddenSlotState::free;
            return Result<OfficialLayerFrontResult>::failure(
                ErrorCode::backend_unavailable);
        }
        if (cuda::launch_official_moe_prepare(
                device_hidden, device_block, device_views[0], device_views[1],
                device_views[2], rounded_hidden, normalized_input,
                config.hidden_size, config.rms_norm_epsilon, stream_) !=
            cudaSuccess) {
            hidden_slot.state = OfficialLayerHiddenSlotState::free;
            return Result<OfficialLayerFrontResult>::failure(
                ErrorCode::backend_unavailable);
        }

        std::vector<float> shape(config.hidden_size);
        official_kda_device_input_ = normalized_input;
        official_kda_device_output_ = kda_output;
        auto kda = official_kda(shape, weights.kda, state, config, layer,
                                phase, state_control);
        official_kda_device_input_ = nullptr;
        official_kda_device_output_ = nullptr;
        if (!kda) {
            hidden_slot.state = OfficialLayerHiddenSlotState::free;
            return Result<OfficialLayerFrontResult>::failure(
                kda.error(), kda.message());
        }
        const auto discard_kda = [&]() {
            if (kda.value().device_state.owner) {
                discard_official_kda_device_state(
                    kda.value().device_state);
            }
        };
        if (cuda::launch_bf16_vector_add(rounded_hidden, kda_output, prefix,
                                         config.hidden_size, stream_) !=
            cudaSuccess) {
            discard_kda();
            hidden_slot.state = OfficialLayerHiddenSlotState::free;
            return Result<OfficialLayerFrontResult>::failure(
                ErrorCode::backend_unavailable);
        }
        official_moe_route_device_prefix_ = prefix;
        official_moe_route_device_block_ = device_block;
        auto route = prepare_official_moe_route(
            shape, shape, weights.moe, config.rms_norm_epsilon, layer, phase);
        official_moe_route_device_prefix_ = nullptr;
        official_moe_route_device_block_ = nullptr;
        if (!route) {
            discard_kda();
            hidden_slot.state = OfficialLayerHiddenSlotState::free;
            return Result<OfficialLayerFrontResult>::failure(
                route.error(), route.message());
        }
        moe_prepared_hidden_slot_ = static_cast<std::uint32_t>(slot_index);
        runtime_stats_.weight_h2d_bytes += uploaded_weight_bytes;
        runtime_stats_.activation_h2d_bytes +=
            host_input ? hidden_bytes * 2 : 0;
        return Result<OfficialLayerFrontResult>::success(
            {true, std::move(kda.value()), std::move(route.value())});
    }

    Result<OfficialLayerTailResult> official_layer_tail(
        OfficialMoePreparedToken prepared, OfficialMoeFfnView weights,
        std::span<const Mxfp4MlpView> experts,
        std::span<const std::uint32_t> expert_ids,
        std::span<const float> contributions, float epsilon, float situ_beta,
        std::optional<float> situ_linear, std::uint32_t layer,
        ProfilePhase phase, bool retain_hidden) override {
        if (!moe_prepared_active_ || moe_prepared_hidden_slot_ >= 2 ||
            prepared.owner != moe_prepared_owner_ ||
            prepared.generation != moe_prepared_generation_ ||
            moe_prepared_layer_ != layer) {
            return Result<OfficialLayerTailResult>::failure(
                ErrorCode::invalid_state);
        }
        const auto source_slot = moe_prepared_hidden_slot_;
        const auto target_slot = static_cast<std::uint32_t>(1 - source_slot);
        if (layer_hidden_slots_[source_slot].state !=
                OfficialLayerHiddenSlotState::prepared ||
            (retain_hidden &&
             layer_hidden_slots_[target_slot].state !=
                 OfficialLayerHiddenSlotState::free)) {
            return Result<OfficialLayerTailResult>::failure(
                ErrorCode::invalid_state);
        }
        official_layer_tail_retain_ = retain_hidden;
        official_layer_tail_target_slot_ = retain_hidden ? target_slot : 2;
        auto moe = official_mxfp4_moe_ffn_prepared(
            prepared, weights, experts, expert_ids, contributions, epsilon,
            situ_beta, situ_linear, layer, phase);
        official_layer_tail_retain_ = false;
        official_layer_tail_target_slot_ = 2;
        if (!moe) {
            if (moe_prepared_active_ &&
                prepared.owner == moe_prepared_owner_ &&
                prepared.generation == moe_prepared_generation_) {
                const auto discarded =
                    discard_official_moe_prepared(prepared);
                if (!discarded) {
                    return Result<OfficialLayerTailResult>::failure(
                        ErrorCode::invalid_state);
                }
            } else {
                layer_hidden_slots_[source_slot].state =
                    OfficialLayerHiddenSlotState::free;
                moe_prepared_hidden_slot_ = 2;
            }
            return Result<OfficialLayerTailResult>::failure(
                moe.error(), moe.message());
        }
        layer_hidden_slots_[source_slot].state =
            OfficialLayerHiddenSlotState::free;
        OfficialLayerHiddenToken hidden;
        if (retain_hidden) {
            ++layer_hidden_generation_;
            if (!layer_hidden_generation_) ++layer_hidden_generation_;
            auto& target = layer_hidden_slots_[target_slot];
            target.generation = layer_hidden_generation_;
            target.producing_layer = layer;
            target.width = moe_prepared_width_;
            target.state = OfficialLayerHiddenSlotState::live;
            hidden = {layer_hidden_owner_, layer_hidden_generation_, layer,
                      moe_prepared_width_, target_slot};
        }
        moe_prepared_hidden_slot_ = 2;
        return Result<OfficialLayerTailResult>::success(
            {true, hidden, std::move(moe.value().output),
             std::move(moe.value().selected_expert_ids)});
    }

    Result<bool> discard_official_layer_hidden(
        OfficialLayerHiddenToken token) override {
        if (token.owner != layer_hidden_owner_ || token.slot >= 2) {
            return Result<bool>::failure(ErrorCode::invalid_state);
        }
        auto& slot = layer_hidden_slots_[token.slot];
        if (slot.state != OfficialLayerHiddenSlotState::live ||
            slot.generation != token.generation ||
            slot.producing_layer != token.producing_layer ||
            slot.width != token.width) {
            return Result<bool>::failure(ErrorCode::invalid_state);
        }
        slot.state = OfficialLayerHiddenSlotState::free;
        return Result<bool>::success(true);
    }

    Result<OfficialMoeRoutePrepareResult> prepare_official_moe_route(
        std::span<const float> prefix, std::span<const float> block,
        OfficialMoeRoutePrepareView weights, float epsilon,
        std::uint32_t layer, ProfilePhase phase) override {
        const auto operation_start = std::chrono::steady_clock::now();
        const bool device_io = official_moe_route_device_prefix_ != nullptr;
        const auto finite_f32 = [](std::span<const float> values) {
            return std::all_of(values.begin(), values.end(),
                               [](float value) { return std::isfinite(value); });
        };
        const auto valid_bf16 = [](Bf16WeightView view,
                                   std::size_t rows, std::size_t cols) {
            return view.tensor_id && view.rows == rows && view.cols == cols &&
                   rows && cols && rows <=
                       std::numeric_limits<std::size_t>::max() / cols &&
                   view.values.size() == rows * cols;
        };
        if (options_.kind != BackendKind::cuda_custom ||
            options_.cuda_boundary != CudaBoundaryMode::moe_layer ||
            options_.cuda_allocation != CudaAllocationMode::reused ||
            options_.cuda_transfer != CudaTransferMode::synchronous ||
            options_.cuda_batching != CudaBatchingMode::resident_grid ||
            options_.cuda_weights != CudaWeightMode::resident ||
            !resident_weights_ || prefix.empty() || prefix.size() != block.size() ||
            (!device_io && (!finite_f32(prefix) || !finite_f32(block))) ||
            !std::isfinite(epsilon) || epsilon <= 0.0F ||
            !weights.residual_norm.tensor_id ||
            weights.residual_norm.values.size() != prefix.size() ||
            !valid_bf16(weights.residual_proj, 1, prefix.size()) ||
            !weights.post_norm.tensor_id ||
            weights.post_norm.values.size() != prefix.size() ||
            !valid_bf16(weights.router, weights.router.rows, prefix.size())) {
            return Result<OfficialMoeRoutePrepareResult>::failure(
                ErrorCode::invalid_extent);
        }
        const std::array views{
            Bf16WeightView{weights.residual_norm.values, 1, prefix.size(),
                           weights.residual_norm.tensor_id},
            weights.residual_proj,
            Bf16WeightView{weights.post_norm.values, 1, prefix.size(),
                           weights.post_norm.tensor_id},
            weights.router};
        std::unordered_set<std::uint64_t> tensor_ids;
        std::uint64_t total_weight_bytes{};
        std::array<ImmutableWeightIdentity, 4> identities{};
        std::array<bool, 4> validation_hits{};
        std::array<bool, 4> validation_scans{};
        for (std::size_t index = 0; index < views.size(); ++index) {
            const auto& view = views[index];
            if (!tensor_ids.insert(view.tensor_id).second) {
                return Result<OfficialMoeRoutePrepareResult>::failure(
                    ErrorCode::invalid_extent);
            }
            total_weight_bytes += view.values.size_bytes();
            identities[index] = {view.values.data(), view.values.size_bytes(),
                                 view.rows, view.cols};
            if (options_.cuda_weight_validation ==
                CudaWeightValidationMode::per_call) {
                validation_scans[index] = true;
            } else {
                const auto found = immutable_weights_.find(view.tensor_id);
                if (found == immutable_weights_.end()) {
                    validation_scans[index] = true;
                } else if (found->second == identities[index]) {
                    validation_hits[index] = true;
                } else {
                    return Result<OfficialMoeRoutePrepareResult>::failure(
                        ErrorCode::invalid_extent);
                }
            }
        }
        if (total_weight_bytes > options_.cuda_resident_bytes) {
            return Result<OfficialMoeRoutePrepareResult>::failure(
                ErrorCode::invalid_extent);
        }
        const auto validation_start = std::chrono::steady_clock::now();
        bool scanned{};
        for (std::size_t index = 0; index < views.size(); ++index) {
            if (!validation_scans[index]) continue;
            scanned = true;
            ++runtime_stats_.immutable_validation_scans;
            runtime_stats_.immutable_validation_bytes +=
                views[index].values.size_bytes();
            if (!std::all_of(
                    views[index].values.begin(), views[index].values.end(),
                    [](std::uint16_t value) {
                        return (value & 0x7f80U) != 0x7f80U;
                    })) {
                return Result<OfficialMoeRoutePrepareResult>::failure(
                    ErrorCode::invalid_extent);
            }
        }
        if (scanned) {
            runtime_stats_.immutable_validation_nanoseconds +=
                static_cast<std::uint64_t>(std::chrono::duration_cast<
                    std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now() - validation_start)
                                               .count());
        }
        if (options_.cuda_weight_validation ==
            CudaWeightValidationMode::admission) {
            for (std::size_t index = 0; index < views.size(); ++index) {
                if (validation_hits[index]) {
                    ++runtime_stats_.immutable_validation_hits;
                } else {
                    immutable_weights_.emplace(views[index].tensor_id,
                                               identities[index]);
                }
            }
        }
        std::array<const std::uint16_t*, 4> device_views{};
        std::uint64_t uploaded_weight_bytes{};
        for (std::size_t index = 0; index < views.size(); ++index) {
            const auto& view = views[index];
            auto acquired = resident_weights_->acquire(
                {view.tensor_id, cuda::WeightRepresentation::dense_bf16,
                 view.rows, view.cols, 0},
                std::as_bytes(view.values), {});
            if (!acquired || acquired.value().disposition ==
                                 cuda::ResidentDisposition::bypass) {
                return Result<OfficialMoeRoutePrepareResult>::failure(
                    ErrorCode::invalid_extent);
            }
            uploaded_weight_bytes += acquired.value().uploaded_bytes;
            device_views[index] = static_cast<const std::uint16_t*>(
                acquired.value().primary);
        }
        const auto hidden_bytes = prefix.size_bytes();
        const auto logits_bytes = weights.router.rows * sizeof(float);
        const auto scratch_bytes =
            (device_io ? 0 : hidden_bytes * 2) + logits_bytes;
        if (official_moe_prepared_.reserve(hidden_bytes * 2) != cudaSuccess ||
            official_moe_route_scratch_.reserve(scratch_bytes) != cudaSuccess ||
            official_moe_route_event_start_.ensure() != cudaSuccess ||
            official_moe_route_event_end_.ensure() != cudaSuccess) {
            return Result<OfficialMoeRoutePrepareResult>::failure(
                ErrorCode::backend_unavailable);
        }
        if (moe_prepared_active_) {
            if (moe_prepared_hidden_slot_ < 2) {
                layer_hidden_slots_[moe_prepared_hidden_slot_].state =
                    OfficialLayerHiddenSlotState::free;
            }
            moe_prepared_hidden_slot_ = 2;
            moe_prepared_active_ = false;
            ++runtime_stats_.official_moe_prepared_invalidations;
        }
        auto* prepared_prefix =
            static_cast<float*>(official_moe_prepared_.get());
        auto* prepared_hidden = prepared_prefix + prefix.size();
        auto* route_scratch =
            static_cast<float*>(official_moe_route_scratch_.get());
        const auto* input_prefix =
            device_io ? official_moe_route_device_prefix_ : route_scratch;
        const auto* input_block =
            device_io ? official_moe_route_device_block_
                      : route_scratch + prefix.size();
        auto* device_logits = device_io
                                  ? route_scratch
                                  : route_scratch + prefix.size() + block.size();
        OfficialMoeRoutePrepareResult result;
        result.executed = true;
        result.router_logits.resize(weights.router.rows);
        if ((!device_io &&
             (cudaMemcpyAsync(const_cast<float*>(input_prefix), prefix.data(),
                              hidden_bytes, cudaMemcpyHostToDevice, stream_) !=
                  cudaSuccess ||
              cudaMemcpyAsync(const_cast<float*>(input_block), block.data(),
                              hidden_bytes, cudaMemcpyHostToDevice, stream_) !=
                  cudaSuccess)) ||
            cudaEventRecord(official_moe_route_event_start_.get(), stream_) !=
                cudaSuccess ||
            cuda::launch_official_moe_prepare(
                input_prefix, input_block, device_views[0], device_views[1],
                device_views[2], prepared_prefix, prepared_hidden,
                prefix.size(), epsilon, stream_) != cudaSuccess ||
            cuda::launch_official_moe_router_logits(
                prepared_hidden, device_views[3], device_logits,
                weights.router.rows, weights.router.cols, stream_) != cudaSuccess ||
            cudaEventRecord(official_moe_route_event_end_.get(), stream_) !=
                cudaSuccess ||
            cudaMemcpyAsync(result.router_logits.data(), device_logits,
                            logits_bytes, cudaMemcpyDeviceToHost, stream_) !=
                cudaSuccess ||
            cudaStreamSynchronize(stream_) != cudaSuccess ||
            !finite_f32(result.router_logits)) {
            return Result<OfficialMoeRoutePrepareResult>::failure(
                ErrorCode::backend_unavailable);
        }
        float milliseconds{};
        if (cudaEventElapsedTime(&milliseconds,
                                 official_moe_route_event_start_.get(),
                                 official_moe_route_event_end_.get()) !=
            cudaSuccess) {
            return Result<OfficialMoeRoutePrepareResult>::failure(
                ErrorCode::backend_unavailable);
        }
        ++moe_prepared_generation_;
        if (!moe_prepared_generation_) ++moe_prepared_generation_;
        moe_prepared_layer_ = layer;
        moe_prepared_width_ = prefix.size();
        moe_prepared_active_ = true;
        result.prepared = {moe_prepared_owner_, moe_prepared_generation_};
        ++runtime_stats_.stream_synchronization_count;
        runtime_stats_.weight_h2d_bytes += uploaded_weight_bytes;
        runtime_stats_.activation_h2d_bytes +=
            device_io ? 0 : hidden_bytes * 2;
        runtime_stats_.device_to_host_bytes += logits_bytes;
        ++runtime_stats_.official_moe_route_prepare_calls;
        runtime_stats_.official_moe_route_prepare_kernel_launches += 2;
        runtime_stats_.official_moe_router_logit_d2h_bytes += logits_bytes;
        ++runtime_stats_.official_moe_prepared_seeds;
        runtime_stats_.official_moe_prepared_slot_bytes = hidden_bytes * 2;
        record(phase, ProfileOperation::moe_mix,
               NumericPrecision::bf16_rounded, layer, operation_start,
               total_weight_bytes, uploaded_weight_bytes,
               static_cast<std::uint64_t>(
                   std::llround(static_cast<double>(milliseconds) * 1.0e6)),
               true);
        return Result<OfficialMoeRoutePrepareResult>::success(
            std::move(result));
    }

    Result<bool> discard_official_moe_prepared(
        OfficialMoePreparedToken token) override {
        if (!moe_prepared_active_ || token.owner != moe_prepared_owner_ ||
            token.generation != moe_prepared_generation_) {
            return Result<bool>::failure(ErrorCode::invalid_state);
        }
        if (moe_prepared_hidden_slot_ < 2) {
            layer_hidden_slots_[moe_prepared_hidden_slot_].state =
                OfficialLayerHiddenSlotState::free;
        }
        moe_prepared_hidden_slot_ = 2;
        moe_prepared_active_ = false;
        ++runtime_stats_.official_moe_prepared_discards;
        return Result<bool>::success(true);
    }

    Result<OfficialMoeFfnResult> official_mxfp4_moe_ffn(
        std::span<const float> hidden, std::span<const float> prefix,
        OfficialMoeFfnView weights, std::span<const Mxfp4MlpView> experts,
        std::span<const std::uint32_t> expert_ids,
        std::span<const float> contributions, float epsilon, float situ_beta,
        std::optional<float> situ_linear, std::uint32_t layer,
        ProfilePhase phase) override {
        if (moe_prepared_active_) {
            if (moe_prepared_hidden_slot_ < 2) {
                layer_hidden_slots_[moe_prepared_hidden_slot_].state =
                    OfficialLayerHiddenSlotState::free;
            }
            moe_prepared_hidden_slot_ = 2;
            moe_prepared_active_ = false;
            ++runtime_stats_.official_moe_prepared_invalidations;
        }
        return official_mxfp4_moe_ffn_impl(
            hidden, prefix, std::nullopt, weights, experts, expert_ids,
            contributions, epsilon, situ_beta, situ_linear, layer, phase);
    }

    Result<OfficialMoeFfnResult> official_mxfp4_moe_ffn_prepared(
        OfficialMoePreparedToken prepared, OfficialMoeFfnView weights,
        std::span<const Mxfp4MlpView> experts,
        std::span<const std::uint32_t> expert_ids,
        std::span<const float> contributions, float epsilon, float situ_beta,
        std::optional<float> situ_linear, std::uint32_t layer,
        ProfilePhase phase) override {
        return official_mxfp4_moe_ffn_impl(
            {}, {}, prepared, weights, experts, expert_ids, contributions,
            epsilon, situ_beta, situ_linear, layer, phase);
    }

    Result<OfficialMoeFfnResult> official_mxfp4_moe_ffn_impl(
        std::span<const float> hidden, std::span<const float> prefix,
        std::optional<OfficialMoePreparedToken> prepared,
        OfficialMoeFfnView weights, std::span<const Mxfp4MlpView> experts,
        std::span<const std::uint32_t> expert_ids,
        std::span<const float> contributions, float epsilon, float situ_beta,
        std::optional<float> situ_linear, std::uint32_t layer,
        ProfilePhase phase) {
        const auto operation_start = std::chrono::steady_clock::now();
        const auto finite = [](std::span<const float> values) {
            return std::all_of(values.begin(), values.end(),
                               [](float value) { return std::isfinite(value); });
        };
        const auto valid_bf16 = [](Bf16WeightView view) {
            return view.tensor_id && view.rows && view.cols &&
                   view.rows <= std::numeric_limits<std::size_t>::max() /
                                   view.cols &&
                   view.values.size() == view.rows * view.cols;
        };
        if (options_.kind != BackendKind::cuda_custom ||
            options_.cuda_boundary != CudaBoundaryMode::moe_layer ||
            options_.cuda_allocation != CudaAllocationMode::reused ||
            options_.cuda_transfer != CudaTransferMode::synchronous ||
            options_.cuda_batching != CudaBatchingMode::resident_grid ||
            (!prepared && (hidden.empty() || hidden.size() != prefix.size())) ||
            (prepared &&
             (!hidden.empty() || !prefix.empty() || !moe_prepared_active_ ||
              prepared->owner != moe_prepared_owner_ ||
              prepared->generation != moe_prepared_generation_ ||
              moe_prepared_layer_ != layer || !moe_prepared_width_)) ||
            experts.empty() || experts.size() > 65535 ||
            experts.size() != expert_ids.size() ||
            experts.size() != contributions.size() ||
            (!prepared && (!finite(hidden) || !finite(prefix))) ||
            !finite(contributions) ||
            !std::isfinite(epsilon) || epsilon <= 0.0F ||
            !std::isfinite(situ_beta) || situ_beta <= 0.0F ||
            (situ_linear &&
             (!std::isfinite(*situ_linear) || *situ_linear <= 0.0F)) ||
            !valid_bf16(weights.routed_down) ||
            !valid_bf16(weights.routed_up) ||
            !valid_bf16(weights.shared.gate) ||
            !valid_bf16(weights.shared.up) ||
            !valid_bf16(weights.shared.down) ||
            !weights.routed_norm.tensor_id ||
            weights.routed_norm.values.empty()) {
            return Result<OfficialMoeFfnResult>::failure(
                ErrorCode::invalid_mxfp4);
        }
        const auto hidden_width = prepared ? moe_prepared_width_ : hidden.size();
        const auto latent_width = weights.routed_down.rows;
        const auto routed_width = experts.front().down.rows;
        const auto intermediate_width = experts.front().gate.rows;
        const auto shared_width = weights.shared.gate.rows;
        if (weights.routed_down.cols != hidden_width ||
            weights.routed_norm.values.size() != routed_width ||
            weights.routed_up.rows != hidden_width ||
            weights.routed_up.cols != routed_width ||
            weights.shared.gate.cols != hidden_width ||
            weights.shared.up.rows != shared_width ||
            weights.shared.up.cols != hidden_width ||
            weights.shared.down.rows != hidden_width ||
            weights.shared.down.cols != shared_width) {
            return Result<OfficialMoeFfnResult>::failure(
                ErrorCode::invalid_mxfp4);
        }
        std::unordered_set<std::uint32_t> selected_ids;
        std::unordered_set<std::uint64_t> tensor_ids;
        const auto insert_tensor = [&tensor_ids](std::uint64_t id) {
            return id && tensor_ids.insert(id).second;
        };
        if (!insert_tensor(weights.routed_down.tensor_id) ||
            !insert_tensor(weights.routed_norm.tensor_id) ||
            !insert_tensor(weights.routed_up.tensor_id) ||
            !insert_tensor(weights.shared.gate.tensor_id) ||
            !insert_tensor(weights.shared.up.tensor_id) ||
            !insert_tensor(weights.shared.down.tensor_id)) {
            return Result<OfficialMoeFfnResult>::failure(
                ErrorCode::invalid_mxfp4);
        }
        double contribution_sum = 0.0;
        std::uint64_t total_weight_bytes = 0;
        const std::array bf16_views{
            weights.routed_down,
            Bf16WeightView{weights.routed_norm.values, 1, routed_width,
                           weights.routed_norm.tensor_id},
            weights.routed_up, weights.shared.gate, weights.shared.up,
            weights.shared.down};
        for (const auto& view : bf16_views) {
            total_weight_bytes += view.values.size_bytes();
        }
        for (std::size_t index = 0; index < experts.size(); ++index) {
            if (!selected_ids.insert(expert_ids[index]).second ||
                contributions[index] <= 0.0F ||
                experts[index].gate.cols != latent_width ||
                experts[index].gate.rows != intermediate_width ||
                experts[index].up.cols != latent_width ||
                experts[index].up.rows != intermediate_width ||
                experts[index].down.cols != intermediate_width ||
                experts[index].down.rows != routed_width ||
                experts[index].gate.group_size != 32 ||
                experts[index].up.group_size != 32 ||
                experts[index].down.group_size != 32 ||
                !valid_mxfp4_size(latent_width, experts[index].gate) ||
                !valid_mxfp4_size(latent_width, experts[index].up) ||
                !valid_mxfp4_size(intermediate_width, experts[index].down)) {
                return Result<OfficialMoeFfnResult>::failure(
                    ErrorCode::invalid_mxfp4);
            }
            contribution_sum += contributions[index];
            for (const auto& view : std::array{
                     experts[index].gate, experts[index].up,
                     experts[index].down}) {
                if (!insert_tensor(view.tensor_id)) {
                    return Result<OfficialMoeFfnResult>::failure(
                        ErrorCode::invalid_mxfp4);
                }
                total_weight_bytes +=
                    view.packed.size_bytes() + view.scales.size_bytes();
            }
        }
        if (std::abs(contribution_sum - 1.0) > 1.0e-5 ||
            (options_.cuda_weights == CudaWeightMode::resident &&
             (resident_weights_ == nullptr ||
              total_weight_bytes > options_.cuda_resident_bytes))) {
            return Result<OfficialMoeFfnResult>::failure(
                ErrorCode::invalid_mxfp4);
        }
        if (prepared) {
            moe_prepared_active_ = false;
            ++runtime_stats_.official_moe_prepared_consumes;
        }

        std::vector<std::unique_ptr<cuda::DeviceAllocation>> transient;
        transient.reserve(bf16_views.size() + experts.size() * 6);
        std::uint64_t uploaded_weight_bytes = 0;
        const auto acquire = [&](cuda::ResidentWeightKey key,
                                 std::span<const std::byte> primary,
                                 std::span<const std::byte> secondary)
            -> Result<cuda::ResidentAcquisition> {
            if (resident_weights_) {
                auto value = resident_weights_->acquire(key, primary, secondary);
                if (!value || value.value().disposition ==
                                  cuda::ResidentDisposition::bypass) {
                    return Result<cuda::ResidentAcquisition>::failure(
                        ErrorCode::invalid_mxfp4);
                }
                uploaded_weight_bytes += value.value().uploaded_bytes;
                return value;
            }
            auto first = std::make_unique<cuda::DeviceAllocation>(
                &memory_stats_, &runtime_stats_);
            auto second = std::make_unique<cuda::DeviceAllocation>(
                &memory_stats_, &runtime_stats_);
            if (first->allocate(primary.size()) != cudaSuccess ||
                (!secondary.empty() &&
                 second->allocate(secondary.size()) != cudaSuccess) ||
                cudaMemcpyAsync(first->get(), primary.data(), primary.size(),
                                cudaMemcpyHostToDevice, stream_) != cudaSuccess ||
                (!secondary.empty() &&
                 cudaMemcpyAsync(second->get(), secondary.data(),
                                 secondary.size(), cudaMemcpyHostToDevice,
                                 stream_) != cudaSuccess)) {
                return Result<cuda::ResidentAcquisition>::failure(
                    ErrorCode::backend_unavailable);
            }
            const auto* primary_pointer = first->get();
            const auto* secondary_pointer =
                secondary.empty() ? nullptr : second->get();
            uploaded_weight_bytes += primary.size() + secondary.size();
            transient.push_back(std::move(first));
            if (!secondary.empty()) transient.push_back(std::move(second));
            return Result<cuda::ResidentAcquisition>::success(
                {cuda::ResidentDisposition::admitted, primary_pointer,
                 secondary_pointer, primary.size() + secondary.size()});
        };

        std::array<const std::uint16_t*, 6> device_bf16{};
        for (std::size_t index = 0; index < bf16_views.size(); ++index) {
            const auto& view = bf16_views[index];
            auto value = acquire(
                {view.tensor_id, cuda::WeightRepresentation::dense_bf16,
                 view.rows, view.cols, 0}, std::as_bytes(view.values), {});
            if (!value) {
                return Result<OfficialMoeFfnResult>::failure(
                    value.error(), value.message());
            }
            device_bf16[index] = static_cast<const std::uint16_t*>(
                value.value().primary);
        }
        std::vector<cuda::Mxfp4DeviceMatrix> descriptors(experts.size() * 3);
        for (std::size_t expert = 0; expert < experts.size(); ++expert) {
            const std::array views{
                experts[expert].gate, experts[expert].up, experts[expert].down};
            for (std::size_t projection = 0; projection < views.size(); ++projection) {
                const auto& view = views[projection];
                auto value = acquire(
                    {view.tensor_id, cuda::WeightRepresentation::mxfp4,
                     view.rows, view.cols, view.group_size},
                    view.packed, view.scales);
                if (!value) {
                    return Result<OfficialMoeFfnResult>::failure(
                        value.error(), value.message());
                }
                descriptors[projection * experts.size() + expert] = {
                    static_cast<const std::uint8_t*>(value.value().primary),
                    static_cast<const std::uint8_t*>(value.value().secondary)};
            }
        }

        const auto hidden_bytes = hidden_width * sizeof(float);
        const auto latent_bytes = latent_width * sizeof(float);
        const auto routed_bytes = routed_width * sizeof(float);
        const auto shared_bytes = shared_width * sizeof(float);
        const auto expert_intermediate_count =
            experts.size() * intermediate_width;
        const auto expert_intermediate_bytes =
            expert_intermediate_count * sizeof(float);
        const auto expert_output_bytes =
            experts.size() * routed_width * sizeof(float);
        const auto contribution_bytes = contributions.size_bytes();
        const auto descriptor_bytes =
            descriptors.size() * sizeof(cuda::Mxfp4DeviceMatrix);
        if (layer_input_scratch_.reserve(hidden_bytes * 2) != cudaSuccess ||
            layer_routed_latent_scratch_.reserve(latent_bytes) != cudaSuccess ||
            layer_descriptor_scratch_.reserve(descriptor_bytes) != cudaSuccess ||
            layer_expert_gate_scratch_.reserve(expert_intermediate_bytes) != cudaSuccess ||
            layer_expert_up_scratch_.reserve(expert_intermediate_bytes) != cudaSuccess ||
            layer_expert_activation_scratch_.reserve(expert_intermediate_bytes) != cudaSuccess ||
            layer_expert_output_scratch_.reserve(expert_output_bytes) != cudaSuccess ||
            layer_contribution_scratch_.reserve(contribution_bytes) != cudaSuccess ||
            layer_mixed_scratch_.reserve(routed_bytes) != cudaSuccess ||
            layer_normalized_scratch_.reserve(routed_bytes) != cudaSuccess ||
            layer_routed_hidden_scratch_.reserve(hidden_bytes) != cudaSuccess ||
            layer_shared_gate_scratch_.reserve(shared_bytes) != cudaSuccess ||
            layer_shared_up_scratch_.reserve(shared_bytes) != cudaSuccess ||
            layer_shared_activation_scratch_.reserve(shared_bytes) != cudaSuccess ||
            layer_shared_hidden_scratch_.reserve(hidden_bytes) != cudaSuccess ||
            layer_final_hidden_scratch_.reserve(hidden_bytes) != cudaSuccess) {
            return Result<OfficialMoeFfnResult>::failure(
                ErrorCode::backend_unavailable,
                "official MoE FFN scratch allocation failed");
        }
        auto* device_hidden = static_cast<float*>(layer_input_scratch_.get());
        auto* device_prefix = device_hidden + hidden_width;
        auto* device_latent = static_cast<float*>(layer_routed_latent_scratch_.get());
        auto* device_descriptors = static_cast<cuda::Mxfp4DeviceMatrix*>(layer_descriptor_scratch_.get());
        auto* device_gate = static_cast<float*>(layer_expert_gate_scratch_.get());
        auto* device_up = static_cast<float*>(layer_expert_up_scratch_.get());
        auto* device_activation = static_cast<float*>(layer_expert_activation_scratch_.get());
        auto* device_expert_output = static_cast<float*>(layer_expert_output_scratch_.get());
        auto* device_contributions = static_cast<float*>(layer_contribution_scratch_.get());
        auto* device_mixed = static_cast<float*>(layer_mixed_scratch_.get());
        auto* device_normalized = static_cast<float*>(layer_normalized_scratch_.get());
        auto* device_routed = static_cast<float*>(layer_routed_hidden_scratch_.get());
        auto* device_shared_gate = static_cast<float*>(layer_shared_gate_scratch_.get());
        auto* device_shared_up = static_cast<float*>(layer_shared_up_scratch_.get());
        auto* device_shared_activation = static_cast<float*>(layer_shared_activation_scratch_.get());
        auto* device_shared = static_cast<float*>(layer_shared_hidden_scratch_.get());
        auto* device_combined = static_cast<float*>(layer_final_hidden_scratch_.get());
        const auto* prepared_prefix = prepared
            ? static_cast<const float*>(official_moe_prepared_.get())
            : nullptr;
        const auto* prepared_hidden = prepared
            ? prepared_prefix + hidden_width
            : nullptr;
        if ((!prepared &&
             (cudaMemcpyAsync(device_hidden, hidden.data(), hidden_bytes,
                              cudaMemcpyHostToDevice, stream_) != cudaSuccess ||
              cudaMemcpyAsync(device_prefix, prefix.data(), hidden_bytes,
                              cudaMemcpyHostToDevice, stream_) != cudaSuccess)) ||
            (prepared &&
             (cudaMemcpyAsync(device_hidden, prepared_hidden, hidden_bytes,
                              cudaMemcpyDeviceToDevice, stream_) != cudaSuccess ||
              cudaMemcpyAsync(device_prefix, prepared_prefix, hidden_bytes,
                              cudaMemcpyDeviceToDevice, stream_) != cudaSuccess)) ||
            cudaMemcpyAsync(device_contributions, contributions.data(),
                            contribution_bytes, cudaMemcpyHostToDevice,
                            stream_) != cudaSuccess ||
            cudaMemcpyAsync(device_descriptors, descriptors.data(),
                            descriptor_bytes, cudaMemcpyHostToDevice,
                            stream_) != cudaSuccess) {
            return Result<OfficialMoeFfnResult>::failure(
                ErrorCode::backend_unavailable,
                "official MoE FFN activation upload failed");
        }
        if (official_moe_event_start_.ensure() != cudaSuccess ||
            official_moe_event_end_.ensure() != cudaSuccess ||
            cudaEventRecord(official_moe_event_start_.get(), stream_) != cudaSuccess ||
            cuda::launch_bf16_matvec(device_hidden, device_bf16[0],
                                     device_latent, latent_width, hidden_width,
                                     stream_) != cudaSuccess ||
            cuda::launch_round_bf16_inplace(device_prefix, hidden_width,
                                            stream_) != cudaSuccess ||
            cuda::launch_mxfp4_matvec_grid(
                device_latent, device_descriptors, device_gate,
                intermediate_width, latent_width, experts.size(), 1,
                cuda::ExpertGridInputLayout::shared_token_major, stream_) != cudaSuccess ||
            cuda::launch_mxfp4_matvec_grid(
                device_latent, device_descriptors + experts.size(), device_up,
                intermediate_width, latent_width, experts.size(), 1,
                cuda::ExpertGridInputLayout::shared_token_major, stream_) != cudaSuccess ||
            cuda::launch_situ_glu(device_gate, device_up, device_activation,
                                  expert_intermediate_count, situ_beta,
                                  situ_linear.has_value(),
                                  situ_linear.value_or(0.0F), false,
                                  stream_) != cudaSuccess ||
            cuda::launch_mxfp4_matvec_grid(
                device_activation, device_descriptors + experts.size() * 2,
                device_expert_output, routed_width, intermediate_width,
                experts.size(), 1,
                cuda::ExpertGridInputLayout::expert_token_major, stream_) != cudaSuccess ||
            cuda::launch_round_bf16_inplace(
                device_expert_output, experts.size() * routed_width,
                stream_) != cudaSuccess ||
            cuda::launch_ordered_expert_mix_bf16(
                device_expert_output, device_contributions, contributions,
                device_mixed, routed_width, stream_) != cudaSuccess ||
            cuda::launch_bf16_rms_norm(device_mixed, device_bf16[1],
                                       device_normalized, routed_width,
                                       epsilon, stream_) != cudaSuccess ||
            cuda::launch_bf16_matvec(device_normalized, device_bf16[2],
                                     device_routed, hidden_width, routed_width,
                                     stream_) != cudaSuccess ||
            cuda::launch_bf16_matvec(device_hidden, device_bf16[3],
                                     device_shared_gate, shared_width,
                                     hidden_width, stream_) != cudaSuccess ||
            cuda::launch_bf16_matvec(device_hidden, device_bf16[4],
                                     device_shared_up, shared_width,
                                     hidden_width, stream_) != cudaSuccess ||
            cuda::launch_situ_glu(device_shared_gate, device_shared_up,
                                  device_shared_activation, shared_width,
                                  situ_beta, situ_linear.has_value(),
                                  situ_linear.value_or(0.0F), false,
                                  stream_) != cudaSuccess ||
            cuda::launch_bf16_matvec(device_shared_activation,
                                     device_bf16[5], device_shared,
                                     hidden_width, shared_width,
                                     stream_) != cudaSuccess ||
            cuda::launch_bf16_vector_add(device_routed, device_shared,
                                         device_combined, hidden_width,
                                         stream_) != cudaSuccess ||
            cuda::launch_bf16_vector_add(device_prefix, device_combined,
                                         device_routed, hidden_width,
                                         stream_) != cudaSuccess ||
            cudaEventRecord(official_moe_event_end_.get(), stream_) != cudaSuccess) {
            return Result<OfficialMoeFfnResult>::failure(
                ErrorCode::backend_unavailable,
                "official MoE FFN kernel launch failed");
        }
        std::vector<float> output;
        if (!official_layer_tail_retain_) output.resize(hidden_width);
        auto* tail_target = official_layer_tail_target_slot_ == 0
                                ? &official_layer_hidden_one_
                            : official_layer_tail_target_slot_ == 1
                                ? &official_layer_hidden_two_
                                : nullptr;
        auto* tail_source = moe_prepared_hidden_slot_ == 0
                                ? &official_layer_hidden_one_
                            : moe_prepared_hidden_slot_ == 1
                                ? &official_layer_hidden_two_
                                : nullptr;
        if ((official_layer_tail_retain_ &&
             (!tail_target || !tail_source ||
              tail_target->reserve(hidden_bytes * 2) != cudaSuccess)) ||
            (official_layer_tail_retain_
                 ? (cudaMemcpyAsync(tail_target->get(), device_routed,
                                    hidden_bytes, cudaMemcpyDeviceToDevice,
                                    stream_) != cudaSuccess ||
                    cudaMemcpyAsync(
                        static_cast<float*>(tail_target->get()) + hidden_width,
                        static_cast<const float*>(tail_source->get()) +
                            hidden_width,
                        hidden_bytes, cudaMemcpyDeviceToDevice, stream_) !=
                        cudaSuccess)
                 : cudaMemcpyAsync(output.data(), device_routed, hidden_bytes,
                                   cudaMemcpyDeviceToHost, stream_) !=
                       cudaSuccess) ||
            cudaStreamSynchronize(stream_) != cudaSuccess) {
            return Result<OfficialMoeFfnResult>::failure(
                ErrorCode::backend_unavailable,
                "official MoE FFN output transfer failed");
        }
        const auto activation_bytes =
                                      (prepared ? 0 : hidden_bytes * 2) +
                                      contribution_bytes +
                                      descriptor_bytes;
        ++runtime_stats_.stream_synchronization_count;
        runtime_stats_.activation_h2d_bytes += activation_bytes;
        runtime_stats_.weight_h2d_bytes += uploaded_weight_bytes;
        runtime_stats_.device_to_host_bytes +=
            official_layer_tail_retain_ ? 0 : hidden_bytes;
        ++runtime_stats_.resident_moe_layer_calls;
        runtime_stats_.resident_moe_layer_experts += experts.size();
        runtime_stats_.resident_moe_layer_kernel_launches += 17;
        runtime_stats_.resident_moe_layer_contribution_h2d_bytes +=
            contribution_bytes;
        float elapsed_milliseconds = 0.0F;
        if (cudaEventElapsedTime(&elapsed_milliseconds,
                                 official_moe_event_start_.get(),
                                 official_moe_event_end_.get()) != cudaSuccess) {
            return Result<OfficialMoeFfnResult>::failure(
                ErrorCode::backend_unavailable,
                "official MoE FFN event timing failed");
        }
        const auto device_nanoseconds = static_cast<std::uint64_t>(
            std::llround(static_cast<double>(elapsed_milliseconds) * 1.0e6));
        record(phase, ProfileOperation::moe_mix,
               NumericPrecision::bf16_rounded, layer, operation_start, 0, 0,
               device_nanoseconds, true);
        record(phase, ProfileOperation::weight_host_to_device,
               NumericPrecision::bf16_rounded, layer, operation_start, 0,
               uploaded_weight_bytes, 0, true);
        record(phase, ProfileOperation::activation_host_to_device,
               NumericPrecision::bf16_rounded, layer, operation_start, 0,
               activation_bytes, 0, true);
        record(phase, ProfileOperation::device_to_host,
               NumericPrecision::bf16_rounded, layer, operation_start, 0,
               official_layer_tail_retain_ ? 0 : hidden_bytes, 0, true);
        return Result<OfficialMoeFfnResult>::success(
            {true, std::move(output),
             {expert_ids.begin(), expert_ids.end()}});
    }

    Result<std::vector<std::vector<float>>> mxfp4_situ_mlp_batch(
        std::span<const float> inputs, std::size_t batch_size,
        Mxfp4MlpView expert, float situ_beta,
        std::optional<float> situ_linear, std::uint32_t layer,
        ProfilePhase phase) override {
        constexpr auto precision = NumericPrecision::mxfp4_e2m1_e8m0;
        const auto operation_start = std::chrono::steady_clock::now();
        const auto multiply_fits = [](std::size_t left, std::size_t right) {
            return right == 0 ||
                   left <= std::numeric_limits<std::size_t>::max() / right;
        };
        if (options_.kind != BackendKind::cuda_custom ||
            options_.cuda_boundary != CudaBoundaryMode::ffn_block ||
            options_.cuda_allocation != CudaAllocationMode::reused ||
            options_.cuda_weights != CudaWeightMode::transient ||
            options_.cuda_transfer != CudaTransferMode::synchronous ||
            options_.cuda_moe_fusion != CudaMoeFusionMode::none ||
            batch_size == 0 || batch_size > 65535 ||
            expert.gate.group_size != 32 || expert.up.group_size != 32 ||
            expert.down.group_size != 32 ||
            !multiply_fits(batch_size, expert.gate.cols) ||
            !multiply_fits(batch_size, expert.gate.rows) ||
            !multiply_fits(batch_size, expert.down.rows) ||
            inputs.size() != batch_size * expert.gate.cols ||
            !valid_mxfp4_size(expert.gate.cols, expert.gate) ||
            !valid_mxfp4_size(expert.gate.cols, expert.up) ||
            expert.gate.rows != expert.up.rows ||
            !valid_mxfp4_size(expert.gate.rows, expert.down) ||
            !std::isfinite(situ_beta) || situ_beta <= 0.0F ||
            (situ_linear &&
             (!std::isfinite(*situ_linear) || *situ_linear <= 0.0F))) {
            return Result<std::vector<std::vector<float>>>::failure(
                ErrorCode::invalid_mxfp4);
        }

        const auto input_bytes = inputs.size_bytes();
        const auto intermediate_count = batch_size * expert.gate.rows;
        const auto intermediate_bytes = intermediate_count * sizeof(float);
        const auto output_count = batch_size * expert.down.rows;
        const auto output_bytes = output_count * sizeof(float);
        const auto maximum_packed_bytes = std::max({
            expert.gate.packed.size_bytes(), expert.up.packed.size_bytes(),
            expert.down.packed.size_bytes()});
        const auto maximum_scale_bytes = std::max({
            expert.gate.scales.size_bytes(), expert.up.scales.size_bytes(),
            expert.down.scales.size_bytes()});
        const auto total_weight_transfer =
            expert.gate.packed.size_bytes() + expert.gate.scales.size_bytes() +
            expert.up.packed.size_bytes() + expert.up.scales.size_bytes() +
            expert.down.packed.size_bytes() + expert.down.scales.size_bytes();

        std::array<EventOwner, 8> events;
        for (auto& event : events) {
            if (event.ensure() != cudaSuccess) {
                return Result<std::vector<std::vector<float>>>::failure(
                    ErrorCode::backend_unavailable,
                    "CUDA batched expert event creation failed");
            }
        }
        if (ffn_input_scratch_.reserve(input_bytes) != cudaSuccess ||
            mxfp4_packed_scratch_.reserve(maximum_packed_bytes) != cudaSuccess ||
            mxfp4_scales_scratch_.reserve(maximum_scale_bytes) != cudaSuccess ||
            ffn_gate_scratch_.reserve(intermediate_bytes) != cudaSuccess ||
            ffn_up_scratch_.reserve(intermediate_bytes) != cudaSuccess ||
            ffn_activation_scratch_.reserve(intermediate_bytes) != cudaSuccess ||
            ffn_output_scratch_.reserve(output_bytes) != cudaSuccess) {
            return Result<std::vector<std::vector<float>>>::failure(
                ErrorCode::backend_unavailable,
                "CUDA batched expert reusable allocation failed");
        }

        auto* device_input = static_cast<float*>(ffn_input_scratch_.get());
        auto* device_packed = static_cast<std::uint8_t*>(
            mxfp4_packed_scratch_.get());
        auto* device_scales = static_cast<std::uint8_t*>(
            mxfp4_scales_scratch_.get());
        auto* device_gate = static_cast<float*>(ffn_gate_scratch_.get());
        auto* device_up = static_cast<float*>(ffn_up_scratch_.get());
        auto* device_activation = static_cast<float*>(
            ffn_activation_scratch_.get());
        auto* device_output = static_cast<float*>(ffn_output_scratch_.get());
        if (cudaMemcpyAsync(device_input, inputs.data(), input_bytes,
                            cudaMemcpyHostToDevice, stream_) != cudaSuccess) {
            return Result<std::vector<std::vector<float>>>::failure(
                ErrorCode::backend_unavailable,
                "CUDA batched expert activation upload failed");
        }

        const auto launch_weight = [&](Mxfp4WeightView weight,
                                       const float* input, float* output,
                                       EventOwner& start, EventOwner& end) {
            if (cudaMemcpyAsync(device_packed, weight.packed.data(),
                                weight.packed.size_bytes(),
                                cudaMemcpyHostToDevice, stream_) != cudaSuccess ||
                cudaMemcpyAsync(device_scales, weight.scales.data(),
                                weight.scales.size_bytes(),
                                cudaMemcpyHostToDevice, stream_) != cudaSuccess ||
                cudaEventRecord(start.get(), stream_) != cudaSuccess) {
                return false;
            }
            return cuda::launch_mxfp4_matvec_batch(
                       input, device_packed, device_scales, output,
                       weight.rows, weight.cols, batch_size, stream_) ==
                       cudaSuccess &&
                   cudaEventRecord(end.get(), stream_) == cudaSuccess;
        };

        if (!launch_weight(expert.gate, device_input, device_gate,
                           events[0], events[1]) ||
            !launch_weight(expert.up, device_input, device_up,
                           events[2], events[3])) {
            return Result<std::vector<std::vector<float>>>::failure(
                ErrorCode::backend_unavailable,
                "CUDA batched expert gate or up failed");
        }
        if (cudaEventRecord(events[4].get(), stream_) != cudaSuccess ||
            cuda::launch_situ_glu(
                device_gate, device_up, device_activation,
                intermediate_count, situ_beta, situ_linear.has_value(),
                situ_linear.value_or(0.0F), false, stream_) != cudaSuccess ||
            cudaEventRecord(events[5].get(), stream_) != cudaSuccess) {
            return Result<std::vector<std::vector<float>>>::failure(
                ErrorCode::backend_unavailable,
                "CUDA batched expert SiTU failed");
        }
        if (!launch_weight(expert.down, device_activation, device_output,
                           events[6], events[7])) {
            return Result<std::vector<std::vector<float>>>::failure(
                ErrorCode::backend_unavailable,
                "CUDA batched expert down failed");
        }

        std::vector<float> flat_output(output_count);
        const auto d2h_start = std::chrono::steady_clock::now();
        if (cudaMemcpyAsync(flat_output.data(), device_output, output_bytes,
                            cudaMemcpyDeviceToHost, stream_) != cudaSuccess ||
            cudaStreamSynchronize(stream_) != cudaSuccess) {
            return Result<std::vector<std::vector<float>>>::failure(
                ErrorCode::backend_unavailable,
                "CUDA batched expert output copy or synchronization failed");
        }

        std::array<std::uint64_t, 4> durations{};
        for (std::size_t pair = 0; pair < durations.size(); ++pair) {
            float milliseconds = 0.0F;
            if (cudaEventElapsedTime(&milliseconds, events[pair * 2].get(),
                                     events[pair * 2 + 1].get()) !=
                cudaSuccess) {
                return Result<std::vector<std::vector<float>>>::failure(
                    ErrorCode::backend_unavailable,
                    "CUDA batched expert event timing failed");
            }
            durations[pair] = static_cast<std::uint64_t>(std::llround(
                static_cast<double>(milliseconds) * 1.0e6));
        }

        std::vector<std::vector<float>> outputs(batch_size);
        for (std::size_t token = 0; token < batch_size; ++token) {
            outputs[token].assign(
                flat_output.begin() + token * expert.down.rows,
                flat_output.begin() + (token + 1) * expert.down.rows);
        }
        ++runtime_stats_.stream_synchronization_count;
        runtime_stats_.activation_h2d_bytes += input_bytes;
        runtime_stats_.weight_h2d_bytes += total_weight_transfer;
        ++runtime_stats_.ffn_block_calls;
        ++runtime_stats_.ffn_block_experts;
        ++runtime_stats_.batched_expert_ffn_calls;
        runtime_stats_.batched_expert_ffn_tokens += batch_size;
        record(phase, ProfileOperation::activation_host_to_device, precision,
               layer, operation_start, 0, input_bytes, 0, true);
        record(phase, ProfileOperation::weight_host_to_device, precision,
               layer, operation_start, 0, total_weight_transfer, 0, true);
        record(phase, ProfileOperation::device_to_host, precision, layer,
               d2h_start, 0, output_bytes, 0, true);
        record(phase, ProfileOperation::mxfp4_matvec, precision, layer,
               operation_start,
               expert.gate.packed.size_bytes() +
                   expert.gate.scales.size_bytes(),
               0, durations[0], true);
        record(phase, ProfileOperation::mxfp4_matvec, precision, layer,
               operation_start,
               expert.up.packed.size_bytes() + expert.up.scales.size_bytes(),
               0, durations[1], true);
        record(phase, ProfileOperation::mxfp4_matvec, precision, layer,
               operation_start,
               expert.down.packed.size_bytes() +
                   expert.down.scales.size_bytes(),
               0, durations[3], true);
        record(phase, ProfileOperation::situ_glu, NumericPrecision::fp32,
               layer, operation_start, intermediate_bytes, 0, durations[2],
               true);
        return Result<std::vector<std::vector<float>>>::success(
            std::move(outputs));
    }

    Result<std::vector<float>> mxfp4_situ_moe(
        std::span<const float> input, std::span<const Mxfp4MlpView> experts,
        std::span<const float> contributions, float situ_beta,
        std::optional<float> situ_linear, std::uint32_t layer,
        ProfilePhase phase) override {
        if (options_.cuda_moe_fusion !=
                CudaMoeFusionMode::routed_accumulate ||
            experts.empty() || experts.size() != contributions.size()) {
            return Result<std::vector<float>>::failure(
                ErrorCode::invalid_mxfp4);
        }
        auto result = mxfp4_situ_mlp_group_impl(
            input, experts, contributions, situ_beta, situ_linear, layer,
            phase);
        if (!result) {
            return Result<std::vector<float>>::failure(
                result.error(), result.message());
        }
        return Result<std::vector<float>>::success(
            std::move(result.value().front()));
    }

    Result<std::vector<std::vector<float>>> mxfp4_situ_mlp_group_impl(
        std::span<const float> input, std::span<const Mxfp4MlpView> experts,
        std::span<const float> contributions, float situ_beta,
        std::optional<float> situ_linear, std::uint32_t layer,
        ProfilePhase phase) {
        const auto operation_start = std::chrono::steady_clock::now();
        constexpr auto precision = NumericPrecision::mxfp4_e2m1_e8m0;
        const bool fuse_outputs = !contributions.empty();
        if (options_.kind != BackendKind::cuda_custom ||
            (options_.cuda_boundary != CudaBoundaryMode::ffn_block &&
             options_.cuda_boundary != CudaBoundaryMode::moe_layer) ||
            (fuse_outputs && contributions.size() != experts.size()) ||
            !std::isfinite(situ_beta) || situ_beta <= 0.0F ||
            (situ_linear &&
             (!std::isfinite(*situ_linear) || *situ_linear <= 0.0F))) {
            return Result<std::vector<std::vector<float>>>::failure(
                ErrorCode::invalid_mxfp4);
        }
        const auto output_rows = experts.empty() ? 0 : experts.front().down.rows;
        for (std::size_t index = 0; index < experts.size(); ++index) {
            const auto& expert = experts[index];
            if (expert.gate.group_size != 32 || expert.up.group_size != 32 ||
                expert.down.group_size != 32 ||
                !valid_mxfp4_size(input.size(), expert.gate) ||
                !valid_mxfp4_size(input.size(), expert.up) ||
                expert.gate.rows != expert.up.rows ||
                !valid_mxfp4_size(expert.gate.rows, expert.down) ||
                (fuse_outputs && expert.down.rows != output_rows) ||
                (fuse_outputs && !std::isfinite(contributions[index]))) {
                return Result<std::vector<std::vector<float>>>::failure(
                    ErrorCode::invalid_mxfp4);
            }
        }
        if (experts.empty()) {
            return Result<std::vector<std::vector<float>>>::success({});
        }

        struct WeightMember {
            Mxfp4WeightView view;
            void* packed{};
            void* scales{};
            std::uint64_t transfer_bytes{};
        };
        struct ExpertMember {
            std::array<WeightMember, 3> weights;
            std::array<std::unique_ptr<EventOwner>, 8> events;
        };
        std::vector<ExpertMember> members;
        members.reserve(experts.size());
        std::size_t maximum_packed_bytes = 0;
        std::size_t maximum_scale_bytes = 0;
        std::size_t maximum_intermediate = 0;
        std::size_t maximum_output = 0;
        std::uint64_t total_weight_transfer = 0;
        for (const auto& expert : experts) {
            ExpertMember member{{{{expert.gate}, {expert.up}, {expert.down}}}};
            for (auto& weight : member.weights) {
                weight.transfer_bytes =
                    weight.view.packed.size_bytes() + weight.view.scales.size_bytes();
                maximum_packed_bytes = std::max(
                    maximum_packed_bytes, weight.view.packed.size_bytes());
                maximum_scale_bytes = std::max(
                    maximum_scale_bytes, weight.view.scales.size_bytes());
                if (resident_weights_) {
                    const auto acquisition = resident_weights_->acquire(
                        {weight.view.tensor_id,
                         cuda::WeightRepresentation::mxfp4,
                         weight.view.rows, weight.view.cols,
                         weight.view.group_size},
                        weight.view.packed, weight.view.scales);
                    if (!acquisition) {
                        return Result<std::vector<std::vector<float>>>::failure(
                            acquisition.error(), acquisition.message());
                    }
                    if (acquisition.value().disposition !=
                        cuda::ResidentDisposition::bypass) {
                        weight.packed =
                            const_cast<void*>(acquisition.value().primary);
                        weight.scales =
                            const_cast<void*>(acquisition.value().secondary);
                        weight.transfer_bytes =
                            acquisition.value().uploaded_bytes;
                    }
                }
                total_weight_transfer += weight.transfer_bytes;
            }
            maximum_intermediate =
                std::max(maximum_intermediate, expert.gate.rows);
            maximum_output = std::max(maximum_output, expert.down.rows);
            for (auto& event : member.events) {
                event = std::make_unique<EventOwner>();
                if (event->ensure() != cudaSuccess) {
                    return Result<std::vector<std::vector<float>>>::failure(
                        ErrorCode::backend_unavailable,
                        "CUDA expert FFN event creation failed");
                }
            }
            members.push_back(std::move(member));
        }

        const auto input_bytes = input.size_bytes();
        const auto intermediate_bytes = maximum_intermediate * sizeof(float);
        const auto output_bytes = maximum_output * sizeof(float);
        cuda::DeviceAllocation local_input(&memory_stats_, &runtime_stats_);
        cuda::DeviceAllocation local_packed(&memory_stats_, &runtime_stats_);
        cuda::DeviceAllocation local_scales(&memory_stats_, &runtime_stats_);
        cuda::DeviceAllocation local_gate(&memory_stats_, &runtime_stats_);
        cuda::DeviceAllocation local_up(&memory_stats_, &runtime_stats_);
        cuda::DeviceAllocation local_activation(&memory_stats_, &runtime_stats_);
        cuda::DeviceAllocation local_output(&memory_stats_, &runtime_stats_);
        void* device_input = nullptr;
        void* transient_packed = nullptr;
        void* transient_scales = nullptr;
        void* device_gate = nullptr;
        void* device_up = nullptr;
        void* device_activation = nullptr;
        void* device_output = nullptr;
        if (options_.cuda_allocation == CudaAllocationMode::reused) {
            if (ffn_input_scratch_.reserve(input_bytes) != cudaSuccess ||
                mxfp4_packed_scratch_.reserve(maximum_packed_bytes) !=
                    cudaSuccess ||
                mxfp4_scales_scratch_.reserve(maximum_scale_bytes) !=
                    cudaSuccess ||
                ffn_gate_scratch_.reserve(intermediate_bytes) != cudaSuccess ||
                ffn_up_scratch_.reserve(intermediate_bytes) != cudaSuccess ||
                ffn_activation_scratch_.reserve(intermediate_bytes) !=
                    cudaSuccess ||
                ffn_output_scratch_.reserve(output_bytes) != cudaSuccess) {
                return Result<std::vector<std::vector<float>>>::failure(
                    ErrorCode::backend_unavailable,
                    "CUDA expert FFN reusable allocation failed");
            }
            device_input = ffn_input_scratch_.get();
            transient_packed = mxfp4_packed_scratch_.get();
            transient_scales = mxfp4_scales_scratch_.get();
            device_gate = ffn_gate_scratch_.get();
            device_up = ffn_up_scratch_.get();
            device_activation = ffn_activation_scratch_.get();
            device_output = ffn_output_scratch_.get();
        } else if (local_input.allocate(input_bytes) != cudaSuccess ||
                   local_packed.allocate(maximum_packed_bytes) != cudaSuccess ||
                   local_scales.allocate(maximum_scale_bytes) != cudaSuccess ||
                   local_gate.allocate(intermediate_bytes) != cudaSuccess ||
                   local_up.allocate(intermediate_bytes) != cudaSuccess ||
                   local_activation.allocate(intermediate_bytes) != cudaSuccess ||
                   local_output.allocate(output_bytes) != cudaSuccess) {
            return Result<std::vector<std::vector<float>>>::failure(
                ErrorCode::backend_unavailable,
                "CUDA expert FFN allocation failed");
        } else {
            device_input = local_input.get();
            transient_packed = local_packed.get();
            transient_scales = local_scales.get();
            device_gate = local_gate.get();
            device_up = local_up.get();
            device_activation = local_activation.get();
            device_output = local_output.get();
        }
        if (cudaMemcpyAsync(device_input, input.data(), input_bytes,
                            cudaMemcpyHostToDevice, stream_) != cudaSuccess) {
            return Result<std::vector<std::vector<float>>>::failure(
                ErrorCode::backend_unavailable,
                "CUDA expert FFN activation upload failed");
        }

        const auto launch_weight = [&](WeightMember& weight,
                                       const float* matvec_input,
                                       float* output, EventOwner& start,
                                       EventOwner& end, bool scaled = false,
                                       float contribution = 1.0F,
                                       bool accumulate = false) {
            if (!weight.packed) {
                weight.packed = transient_packed;
                weight.scales = transient_scales;
                if (cudaMemcpyAsync(weight.packed, weight.view.packed.data(),
                                    weight.view.packed.size_bytes(),
                                    cudaMemcpyHostToDevice, stream_) !=
                        cudaSuccess ||
                    cudaMemcpyAsync(weight.scales, weight.view.scales.data(),
                                    weight.view.scales.size_bytes(),
                                    cudaMemcpyHostToDevice, stream_) !=
                        cudaSuccess) return false;
            }
            if (cudaEventRecord(start.get(), stream_) != cudaSuccess) {
                return false;
            }
            const auto launched = scaled
                ? cuda::launch_mxfp4_matvec_accumulate(
                      matvec_input,
                      static_cast<const std::uint8_t*>(weight.packed),
                      static_cast<const std::uint8_t*>(weight.scales), output,
                      weight.view.rows, weight.view.cols, contribution,
                      accumulate, stream_)
                : cuda::launch_mxfp4_matvec(
                      matvec_input,
                      static_cast<const std::uint8_t*>(weight.packed),
                      static_cast<const std::uint8_t*>(weight.scales), output,
                      weight.view.rows, weight.view.cols, stream_);
            return launched == cudaSuccess &&
                   cudaEventRecord(end.get(), stream_) == cudaSuccess;
        };

        std::vector<std::vector<float>> outputs(
            fuse_outputs ? 1 : experts.size());
        std::uint64_t total_output_bytes = 0;
        for (std::size_t index = 0; index < experts.size(); ++index) {
            auto& member = members[index];
            const auto& expert = experts[index];
            if (!launch_weight(member.weights[0],
                               static_cast<const float*>(device_input),
                               static_cast<float*>(device_gate),
                               *member.events[0], *member.events[1]) ||
                !launch_weight(member.weights[1],
                               static_cast<const float*>(device_input),
                               static_cast<float*>(device_up),
                               *member.events[2], *member.events[3])) {
                return Result<std::vector<std::vector<float>>>::failure(
                    ErrorCode::backend_unavailable,
                    "CUDA expert FFN gate or up failed");
            }
            if (cudaEventRecord(member.events[4]->get(), stream_) != cudaSuccess ||
                cuda::launch_situ_glu(
                    static_cast<const float*>(device_gate),
                    static_cast<const float*>(device_up), device_activation,
                    expert.gate.rows, situ_beta, situ_linear.has_value(),
                    situ_linear.value_or(0.0F), false, stream_) != cudaSuccess ||
                cudaEventRecord(member.events[5]->get(), stream_) != cudaSuccess) {
                return Result<std::vector<std::vector<float>>>::failure(
                    ErrorCode::backend_unavailable,
                    "CUDA expert FFN SiTU failed");
            }
            if (!launch_weight(member.weights[2],
                               static_cast<const float*>(device_activation),
                               static_cast<float*>(device_output),
                               *member.events[6], *member.events[7],
                               fuse_outputs,
                               fuse_outputs ? contributions[index] : 1.0F,
                               fuse_outputs && index != 0)) {
                return Result<std::vector<std::vector<float>>>::failure(
                    ErrorCode::backend_unavailable,
                    "CUDA expert FFN down failed");
            }
            if (!fuse_outputs) {
                outputs[index].resize(expert.down.rows);
                const auto bytes = expert.down.rows * sizeof(float);
                if (cudaMemcpyAsync(outputs[index].data(), device_output, bytes,
                                    cudaMemcpyDeviceToHost, stream_) !=
                    cudaSuccess) {
                    return Result<std::vector<std::vector<float>>>::failure(
                        ErrorCode::backend_unavailable,
                        "CUDA expert FFN output copy failed");
                }
                total_output_bytes += bytes;
            }
        }
        if (fuse_outputs) {
            outputs.front().resize(output_rows);
            total_output_bytes = output_rows * sizeof(float);
            if (cudaMemcpyAsync(outputs.front().data(), device_output,
                                total_output_bytes, cudaMemcpyDeviceToHost,
                                stream_) != cudaSuccess) {
                return Result<std::vector<std::vector<float>>>::failure(
                    ErrorCode::backend_unavailable,
                    "CUDA fused expert output copy failed");
            }
        }
        const auto d2h_start = std::chrono::steady_clock::now();
        if (cudaStreamSynchronize(stream_) != cudaSuccess) {
            return Result<std::vector<std::vector<float>>>::failure(
                ErrorCode::backend_unavailable,
                "CUDA expert FFN synchronization failed");
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
        for (std::size_t expert_index = 0;
             expert_index < experts.size(); ++expert_index) {
            std::array<std::uint64_t, 4> durations{};
            auto& events = members[expert_index].events;
            for (std::size_t pair = 0; pair < durations.size(); ++pair) {
                if (!elapsed(*events[pair * 2], *events[pair * 2 + 1],
                             durations[pair])) {
                    return Result<std::vector<std::vector<float>>>::failure(
                        ErrorCode::backend_unavailable,
                        "CUDA expert FFN event timing failed");
                }
            }
            const auto& expert = experts[expert_index];
            record(phase, ProfileOperation::mxfp4_matvec, precision, layer,
                   operation_start,
                   expert.gate.packed.size_bytes() +
                       expert.gate.scales.size_bytes(),
                   0, durations[0], true);
            record(phase, ProfileOperation::mxfp4_matvec, precision, layer,
                   operation_start,
                   expert.up.packed.size_bytes() + expert.up.scales.size_bytes(),
                   0, durations[1], true);
            record(phase, ProfileOperation::mxfp4_matvec, precision, layer,
                   operation_start,
                   expert.down.packed.size_bytes() +
                       expert.down.scales.size_bytes(),
                   0, durations[3], true);
            record(phase, ProfileOperation::situ_glu, NumericPrecision::fp32,
                   layer, operation_start,
                   expert.gate.rows * sizeof(float), 0, durations[2], true);
        }
        ++runtime_stats_.stream_synchronization_count;
        runtime_stats_.activation_h2d_bytes += input_bytes;
        runtime_stats_.weight_h2d_bytes += total_weight_transfer;
        runtime_stats_.device_to_host_bytes += total_output_bytes;
        ++runtime_stats_.ffn_block_calls;
        runtime_stats_.ffn_block_experts += experts.size();
        if (fuse_outputs) {
            ++runtime_stats_.fused_moe_calls;
            runtime_stats_.fused_moe_experts += experts.size();
        }
        record(phase, ProfileOperation::activation_host_to_device, precision,
               layer, operation_start, 0, input_bytes, 0, true);
        record(phase, ProfileOperation::weight_host_to_device, precision, layer,
               operation_start, 0, total_weight_transfer, 0, true);
        record(phase, ProfileOperation::device_to_host, precision, layer,
               d2h_start, 0, total_output_bytes, 0, true);
        return Result<std::vector<std::vector<float>>>::success(
            std::move(outputs));
    }

    Result<Mxfp4PrefetchToken> prefetch_mxfp4_situ_mlp_group(
        std::span<const Mxfp4MlpView> experts, std::uint64_t use_sequence,
        std::uint32_t layer, ProfilePhase phase) override {
        if (!async_pipeline_ || experts.empty()) {
            return Result<Mxfp4PrefetchToken>::failure(
                async_pipeline_ ? ErrorCode::invalid_extent
                                : ErrorCode::backend_unavailable);
        }
        const auto input_cols = experts.front().gate.cols;
        const auto output_rows = experts.front().down.rows;
        std::size_t maximum_intermediate = 0;
        std::size_t maximum_output = 0;
        for (const auto& expert : experts) {
            if (expert.gate.cols != input_cols ||
                expert.down.rows != output_rows) {
                return Result<Mxfp4PrefetchToken>::failure(
                    ErrorCode::invalid_mxfp4);
            }
            maximum_intermediate =
                std::max(maximum_intermediate, expert.gate.rows);
            maximum_output = std::max(maximum_output, expert.down.rows);
        }
        auto token = async_pipeline_->prepare(
            experts, use_sequence, layer, phase);
        if (token) {
            prepared_mxfp4_ = PreparedMxfp4Metadata{
                token.value(), layer, phase, input_cols, maximum_intermediate,
                maximum_output, experts.size()};
        }
        return token;
    }

    Result<std::vector<std::vector<float>>>
    mxfp4_situ_mlp_group_prepared(
        std::span<const float> input, Mxfp4PrefetchToken token,
        float situ_beta, std::optional<float> situ_linear,
        std::uint32_t layer, ProfilePhase phase) override {
        return mxfp4_situ_mlp_group_prepared_impl(
            input, token, {}, situ_beta, situ_linear, layer, phase);
    }

    Result<std::vector<float>> mxfp4_situ_moe_prepared(
        std::span<const float> input, Mxfp4PrefetchToken token,
        std::span<const float> contributions, float situ_beta,
        std::optional<float> situ_linear, std::uint32_t layer,
        ProfilePhase phase) override {
        if (options_.cuda_moe_fusion !=
                CudaMoeFusionMode::routed_accumulate ||
            contributions.empty()) {
            return Result<std::vector<float>>::failure(
                ErrorCode::invalid_mxfp4);
        }
        auto result = mxfp4_situ_mlp_group_prepared_impl(
            input, token, contributions, situ_beta, situ_linear, layer,
            phase);
        if (!result) {
            return Result<std::vector<float>>::failure(
                result.error(), result.message());
        }
        return Result<std::vector<float>>::success(
            std::move(result.value().front()));
    }

    Result<std::vector<std::vector<float>>>
    mxfp4_situ_mlp_group_prepared_impl(
        std::span<const float> input, Mxfp4PrefetchToken token,
        std::span<const float> contributions, float situ_beta,
        std::optional<float> situ_linear, std::uint32_t layer,
        ProfilePhase phase) {
        const auto operation_start = std::chrono::steady_clock::now();
        constexpr auto precision = NumericPrecision::mxfp4_e2m1_e8m0;
        const bool fuse_outputs = !contributions.empty();
        if (!async_pipeline_) {
            return Result<std::vector<std::vector<float>>>::failure(
                ErrorCode::backend_unavailable);
        }
        if (!prepared_mxfp4_ || token.value == 0 || token.use_sequence == 0 ||
            token.value != prepared_mxfp4_->token.value ||
            token.use_sequence != prepared_mxfp4_->token.use_sequence ||
            layer != prepared_mxfp4_->layer || phase != prepared_mxfp4_->phase) {
            return Result<std::vector<std::vector<float>>>::failure(
                ErrorCode::invalid_state);
        }
        if ((fuse_outputs &&
             contributions.size() != prepared_mxfp4_->expert_count) ||
            (fuse_outputs &&
             std::any_of(contributions.begin(), contributions.end(),
                         [](float value) { return !std::isfinite(value); })) ||
            input.size() != prepared_mxfp4_->input_cols ||
            !std::isfinite(situ_beta) || situ_beta <= 0.0F ||
            (situ_linear &&
             (!std::isfinite(*situ_linear) || *situ_linear <= 0.0F))) {
            return Result<std::vector<std::vector<float>>>::failure(
                ErrorCode::invalid_mxfp4);
        }

        struct PreparedMember {
            std::array<std::unique_ptr<EventOwner>, 8> events;
        };
        std::vector<PreparedMember> members(prepared_mxfp4_->expert_count);
        for (auto& member : members) {
            for (auto& event : member.events) {
                event = std::make_unique<EventOwner>();
                if (event->ensure() != cudaSuccess) {
                    return Result<std::vector<std::vector<float>>>::failure(
                        ErrorCode::backend_unavailable,
                        "CUDA prepared expert FFN event creation failed");
                }
            }
        }

        const auto input_bytes = input.size_bytes();
        const auto intermediate_bytes =
            prepared_mxfp4_->maximum_intermediate * sizeof(float);
        const auto output_bytes =
            prepared_mxfp4_->maximum_output * sizeof(float);
        if (ffn_input_scratch_.reserve(input_bytes) != cudaSuccess ||
            ffn_gate_scratch_.reserve(intermediate_bytes) != cudaSuccess ||
            ffn_up_scratch_.reserve(intermediate_bytes) != cudaSuccess ||
            ffn_activation_scratch_.reserve(intermediate_bytes) != cudaSuccess ||
            ffn_output_scratch_.reserve(output_bytes) != cudaSuccess) {
            return Result<std::vector<std::vector<float>>>::failure(
                ErrorCode::backend_unavailable,
                "CUDA prepared expert FFN reusable allocation failed");
        }
        auto* device_input = static_cast<float*>(ffn_input_scratch_.get());
        auto* device_gate = static_cast<float*>(ffn_gate_scratch_.get());
        auto* device_up = static_cast<float*>(ffn_up_scratch_.get());
        auto* device_activation =
            static_cast<float*>(ffn_activation_scratch_.get());
        auto* device_output = static_cast<float*>(ffn_output_scratch_.get());
        if (cudaMemcpyAsync(device_input, input.data(), input_bytes,
                            cudaMemcpyHostToDevice, stream_) != cudaSuccess) {
            return Result<std::vector<std::vector<float>>>::failure(
                ErrorCode::backend_unavailable,
                "CUDA prepared expert activation upload failed");
        }

        const auto consumed = async_pipeline_->consume(
            token, layer, phase, stream_);
        if (!consumed ||
            consumed.value().size() != prepared_mxfp4_->expert_count) {
            return Result<std::vector<std::vector<float>>>::failure(
                consumed ? ErrorCode::backend_unavailable : consumed.error(),
                consumed ? "CUDA prepared expert count changed"
                         : consumed.message());
        }
        const auto experts = consumed.value();
        const auto launch_weight = [&](const cuda::DeviceMxfp4WeightView& weight,
                                       const float* matvec_input, float* output,
                                       EventOwner& start, EventOwner& end,
                                       bool scaled = false,
                                       float contribution = 1.0F,
                                       bool accumulate = false) {
            if (cudaEventRecord(start.get(), stream_) != cudaSuccess) {
                return false;
            }
            const auto launched = scaled
                ? cuda::launch_mxfp4_matvec_accumulate(
                      matvec_input, weight.packed, weight.scales, output,
                      weight.rows, weight.cols, contribution, accumulate,
                      stream_)
                : cuda::launch_mxfp4_matvec(
                      matvec_input, weight.packed, weight.scales, output,
                      weight.rows, weight.cols, stream_);
            return launched == cudaSuccess &&
                   cudaEventRecord(end.get(), stream_) == cudaSuccess;
        };

        std::vector<std::vector<float>> outputs(
            fuse_outputs ? 1 : experts.size());
        std::uint64_t total_output_bytes = 0;
        const auto d2h_start = std::chrono::steady_clock::now();
        for (std::size_t index = 0; index < experts.size(); ++index) {
            const auto& expert = experts[index];
            auto& events = members[index].events;
            if (!launch_weight(expert.gate, device_input, device_gate,
                               *events[0], *events[1]) ||
                !launch_weight(expert.up, device_input, device_up,
                               *events[2], *events[3])) {
                return Result<std::vector<std::vector<float>>>::failure(
                    ErrorCode::backend_unavailable,
                    "CUDA prepared expert gate or up failed");
            }
            if (cudaEventRecord(events[4]->get(), stream_) != cudaSuccess ||
                cuda::launch_situ_glu(
                    device_gate, device_up, device_activation,
                    expert.gate.rows, situ_beta, situ_linear.has_value(),
                    situ_linear.value_or(0.0F), false, stream_) != cudaSuccess ||
                cudaEventRecord(events[5]->get(), stream_) != cudaSuccess) {
                return Result<std::vector<std::vector<float>>>::failure(
                    ErrorCode::backend_unavailable,
                    "CUDA prepared expert SiTU failed");
            }
            if (!launch_weight(expert.down, device_activation, device_output,
                               *events[6], *events[7], fuse_outputs,
                               fuse_outputs ? contributions[index] : 1.0F,
                               fuse_outputs && index != 0)) {
                return Result<std::vector<std::vector<float>>>::failure(
                    ErrorCode::backend_unavailable,
                    "CUDA prepared expert down failed");
            }
            if (!fuse_outputs) {
                outputs[index].resize(expert.down.rows);
                const auto bytes = expert.down.rows * sizeof(float);
                if (cudaMemcpyAsync(outputs[index].data(), device_output, bytes,
                                    cudaMemcpyDeviceToHost, stream_) !=
                    cudaSuccess) {
                    return Result<std::vector<std::vector<float>>>::failure(
                        ErrorCode::backend_unavailable,
                        "CUDA prepared expert output copy failed");
                }
                total_output_bytes += bytes;
            }
        }
        if (fuse_outputs) {
            outputs.front().resize(prepared_mxfp4_->maximum_output);
            total_output_bytes =
                prepared_mxfp4_->maximum_output * sizeof(float);
            if (cudaMemcpyAsync(outputs.front().data(), device_output,
                                total_output_bytes, cudaMemcpyDeviceToHost,
                                stream_) != cudaSuccess) {
                return Result<std::vector<std::vector<float>>>::failure(
                    ErrorCode::backend_unavailable,
                    "CUDA prepared fused output copy failed");
            }
        }
        if (cudaStreamSynchronize(stream_) != cudaSuccess) {
            return Result<std::vector<std::vector<float>>>::failure(
                ErrorCode::backend_unavailable,
                "CUDA prepared expert synchronization failed");
        }
        ++runtime_stats_.stream_synchronization_count;
        const auto transfer = async_pipeline_->complete(token);
        if (!transfer) {
            return Result<std::vector<std::vector<float>>>::failure(
                transfer.error(), transfer.message());
        }

        const auto elapsed = [](EventOwner& start, EventOwner& end,
                                std::uint64_t& value) {
            float milliseconds = 0.0F;
            if (cudaEventElapsedTime(&milliseconds, start.get(), end.get()) !=
                cudaSuccess) {
                return false;
            }
            value = static_cast<std::uint64_t>(std::llround(
                static_cast<double>(milliseconds) * 1.0e6));
            return true;
        };
        for (std::size_t index = 0; index < experts.size(); ++index) {
            std::array<std::uint64_t, 4> durations{};
            auto& events = members[index].events;
            for (std::size_t pair = 0; pair < durations.size(); ++pair) {
                if (!elapsed(*events[pair * 2], *events[pair * 2 + 1],
                             durations[pair])) {
                    return Result<std::vector<std::vector<float>>>::failure(
                        ErrorCode::backend_unavailable,
                        "CUDA prepared expert event timing failed");
                }
            }
            const auto& expert = experts[index];
            const auto logical_bytes = [](const auto& weight) {
                return weight.rows * weight.cols / 2 +
                       weight.rows * weight.cols / 32;
            };
            record(phase, ProfileOperation::mxfp4_matvec, precision, layer,
                   operation_start, logical_bytes(expert.gate), 0,
                   durations[0], true);
            record(phase, ProfileOperation::mxfp4_matvec, precision, layer,
                   operation_start, logical_bytes(expert.up), 0,
                   durations[1], true);
            record(phase, ProfileOperation::mxfp4_matvec, precision, layer,
                   operation_start, logical_bytes(expert.down), 0,
                   durations[3], true);
            record(phase, ProfileOperation::situ_glu, NumericPrecision::fp32,
                   layer, operation_start, expert.gate.rows * sizeof(float), 0,
                   durations[2], true);
        }
        runtime_stats_.activation_h2d_bytes += input_bytes;
        ++runtime_stats_.ffn_block_calls;
        runtime_stats_.ffn_block_experts += experts.size();
        if (fuse_outputs) {
            ++runtime_stats_.fused_moe_calls;
            runtime_stats_.fused_moe_experts += experts.size();
        }
        record(phase, ProfileOperation::activation_host_to_device, precision,
               layer, operation_start, 0, input_bytes, 0, true);
        record(phase, ProfileOperation::weight_host_to_device, precision, layer,
               operation_start, 0, transfer.value().bytes, 0, true);
        record(phase, ProfileOperation::device_to_host, precision, layer,
               d2h_start, 0, total_output_bytes, 0, true);
        prepared_mxfp4_.reset();
        return Result<std::vector<std::vector<float>>>::success(
            std::move(outputs));
    }

    BackendMemoryStats memory_stats() const noexcept override { return memory_stats_; }
    std::string_view device_name() const noexcept override { return device_name_; }

private:
    struct PreparedMxfp4Metadata {
        Mxfp4PrefetchToken token;
        std::uint32_t layer{};
        ProfilePhase phase{};
        std::size_t input_cols{};
        std::size_t maximum_intermediate{};
        std::size_t maximum_output{};
        std::size_t expert_count{};
    };

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
        return valid_mxfp4_size(input.size(), weight);
    }

    static bool valid_mxfp4_size(std::size_t input_size,
                                 Mxfp4WeightView weight) {
        if (input_size != weight.cols || !weight.rows || !weight.cols ||
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
    struct ImmutableWeightIdentity {
        const void* data{};
        std::size_t bytes{};
        std::size_t rows{};
        std::size_t cols{};

        bool operator==(const ImmutableWeightIdentity&) const = default;
    };
    std::unordered_map<std::uint64_t, ImmutableWeightIdentity>
        immutable_weights_;
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
    cuda::ScratchBuffer mxfp4_descriptor_scratch_;
    cuda::ScratchBuffer layer_input_scratch_;
    cuda::ScratchBuffer layer_routed_latent_scratch_;
    cuda::ScratchBuffer layer_descriptor_scratch_;
    cuda::ScratchBuffer layer_expert_gate_scratch_;
    cuda::ScratchBuffer layer_expert_up_scratch_;
    cuda::ScratchBuffer layer_expert_activation_scratch_;
    cuda::ScratchBuffer layer_expert_output_scratch_;
    cuda::ScratchBuffer layer_contribution_scratch_;
    cuda::ScratchBuffer layer_mixed_scratch_;
    cuda::ScratchBuffer layer_normalized_scratch_;
    cuda::ScratchBuffer layer_routed_hidden_scratch_;
    cuda::ScratchBuffer layer_shared_gate_scratch_;
    cuda::ScratchBuffer layer_shared_up_scratch_;
    cuda::ScratchBuffer layer_shared_activation_scratch_;
    cuda::ScratchBuffer layer_shared_hidden_scratch_;
    cuda::ScratchBuffer layer_final_hidden_scratch_;
    cuda::ScratchBuffer official_kda_scratch_;
    cuda::ScratchBuffer official_kda_state_;
    cuda::ScratchBuffer official_kda_state_one_;
    cuda::ScratchBuffer official_kda_state_two_;
    cuda::ScratchBuffer official_moe_prepared_;
    cuda::ScratchBuffer official_moe_route_scratch_;
    cuda::ScratchBuffer official_layer_front_scratch_;
    cuda::ScratchBuffer official_layer_hidden_one_;
    cuda::ScratchBuffer official_layer_hidden_two_;
    std::uint64_t device_state_owner_{};
    std::uint64_t device_state_generation_{};
    std::array<OfficialKdaDeviceStateSlot, 2> device_state_slots_{};
    std::uint64_t moe_prepared_owner_{};
    std::uint64_t moe_prepared_generation_{};
    std::uint32_t moe_prepared_layer_{};
    std::size_t moe_prepared_width_{};
    bool moe_prepared_active_{};
    std::uint32_t moe_prepared_hidden_slot_{2};
    const float* official_kda_device_input_{};
    float* official_kda_device_output_{};
    const float* official_moe_route_device_prefix_{};
    const float* official_moe_route_device_block_{};
    std::uint32_t official_layer_tail_target_slot_{2};
    bool official_layer_tail_retain_{};
    std::uint64_t layer_hidden_owner_{};
    std::uint64_t layer_hidden_generation_{};
    std::array<OfficialLayerHiddenSlot, 2> layer_hidden_slots_{};
    EventOwner dense_event_start_;
    EventOwner dense_event_end_;
    EventOwner mxfp4_event_start_;
    EventOwner mxfp4_event_end_;
    EventOwner official_moe_event_start_;
    EventOwner official_moe_event_end_;
    EventOwner official_kda_event_start_;
    EventOwner official_kda_event_end_;
    EventOwner official_moe_route_event_start_;
    EventOwner official_moe_route_event_end_;
    std::vector<std::unique_ptr<EventOwner>> dense_group_event_starts_;
    std::vector<std::unique_ptr<EventOwner>> dense_group_event_ends_;
    std::vector<std::unique_ptr<EventOwner>> mxfp4_group_event_starts_;
    std::vector<std::unique_ptr<EventOwner>> mxfp4_group_event_ends_;
    std::array<EventOwner, 8> resident_grid_events_;
    std::array<EventOwner, 26> resident_moe_layer_events_;
    std::map<DensePlanKey, std::unique_ptr<DensePlan>> dense_plans_;
    std::unique_ptr<cuda::ResidentWeightTable> resident_weights_;
    std::unique_ptr<cuda::AsyncMxfp4Pipeline> async_pipeline_;
    std::optional<PreparedMxfp4Metadata> prepared_mxfp4_;
    std::unique_ptr<BoundedCudaGraphIndex> graph_index_;
    std::map<CudaGraphKey, std::unique_ptr<cuda::GraphEntry>> graph_entries_;
    std::unique_ptr<cuda::GraphEntry> update_graph_entry_;
    std::vector<std::uint64_t> graph_scratch_identity_;
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
    if (options.cuda_transfer == CudaTransferMode::prefetch &&
        (options.kind != BackendKind::cuda_custom ||
         options.cuda_boundary != CudaBoundaryMode::ffn_block ||
         options.cuda_allocation != CudaAllocationMode::reused ||
         options.cuda_weights != CudaWeightMode::transient ||
         options.cuda_pinned_bytes == 0)) {
        return cuda_failure(
            ErrorCode::backend_unavailable,
            "CUDA prefetch options do not satisfy the exact capability contract");
    }
    if (options.cuda_transfer == CudaTransferMode::synchronous &&
        options.cuda_pinned_bytes != 0) {
        return cuda_failure(
            ErrorCode::backend_unavailable,
            "synchronous CUDA transfer cannot allocate pinned staging");
    }
    if ((options.cuda_graph == CudaGraphMode::disabled &&
         options.cuda_graph_entries != 0) ||
        (options.cuda_graph == CudaGraphMode::update &&
         options.cuda_graph_entries != 1) ||
        (options.cuda_graph == CudaGraphMode::cache &&
         options.cuda_graph_entries == 0)) {
        return cuda_failure(ErrorCode::backend_unavailable,
                            "invalid CUDA graph capacity contract");
    }
    if (options.cuda_graph != CudaGraphMode::disabled &&
        (options.kind != BackendKind::cuda_custom ||
         options.dense_precision != DensePrecision::fp32 ||
         options.cuda_allocation != CudaAllocationMode::reused ||
         options.cuda_weights != CudaWeightMode::resident ||
         options.cuda_batching != CudaBatchingMode::resident_grid ||
         options.cuda_boundary != CudaBoundaryMode::moe_layer ||
         options.cuda_transfer != CudaTransferMode::synchronous ||
         options.cuda_moe_fusion != CudaMoeFusionMode::none ||
         options.cuda_weight_validation !=
             CudaWeightValidationMode::admission ||
         options.cuda_resident_bytes == 0)) {
        return cuda_failure(
            ErrorCode::backend_unavailable,
            "CUDA graph execution requires admission-validated resident cuda-custom moe-layer execution");
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
    int device_overlap = 0;
    if (cudaDeviceGetAttribute(&device_overlap, cudaDevAttrGpuOverlap, device) !=
        cudaSuccess) {
        return cuda_failure(ErrorCode::backend_unavailable,
                            "CUDA overlap capability query failed");
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

    auto concrete = std::make_unique<CudaBackend>(
        options, profiler, stream.release(), handle.release(), properties.name,
        static_cast<std::uint64_t>(properties.asyncEngineCount),
        device_overlap != 0);
    if (concrete->initialize_async_pipeline() != cudaSuccess) {
        return cuda_failure(ErrorCode::backend_unavailable,
                            "CUDA async transfer resource initialization failed");
    }
    std::unique_ptr<ComputeBackend> backend = std::move(concrete);
    return Result<std::unique_ptr<ComputeBackend>>::success(std::move(backend));
}

}  // namespace k3x
