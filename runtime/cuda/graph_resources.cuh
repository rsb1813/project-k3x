// CUDA Graph와 실행 객체 및 고정 호스트 staging의 수명을 관리합니다.
#pragma once

#include "pinned_memory.cuh"

#include "k3x/cuda_graph_cache.hpp"

#include <cuda_runtime_api.h>

#include <cstddef>
#include <span>
#include <utility>

namespace k3x::cuda {

class GraphOwner {
public:
    GraphOwner() = default;
    ~GraphOwner();
    GraphOwner(const GraphOwner&) = delete;
    GraphOwner& operator=(const GraphOwner&) = delete;
    GraphOwner(GraphOwner&& other) noexcept;
    GraphOwner& operator=(GraphOwner&& other) noexcept;

    cudaGraph_t get() const noexcept { return graph_; }
    cudaGraph_t* out() noexcept { return &graph_; }
    void reset() noexcept;

private:
    cudaGraph_t graph_{};
};

class GraphExecOwner {
public:
    GraphExecOwner() = default;
    ~GraphExecOwner();
    GraphExecOwner(const GraphExecOwner&) = delete;
    GraphExecOwner& operator=(const GraphExecOwner&) = delete;
    GraphExecOwner(GraphExecOwner&& other) noexcept;
    GraphExecOwner& operator=(GraphExecOwner&& other) noexcept;

    cudaGraphExec_t get() const noexcept { return executable_; }
    cudaGraphExec_t* out() noexcept { return &executable_; }
    void reset() noexcept;

private:
    cudaGraphExec_t executable_{};
};

struct GraphStagingLayout {
    std::size_t input_offset{};
    std::size_t contribution_offset{};
    std::size_t descriptor_offset{};
    std::size_t output_offset{};
    std::size_t total_bytes{};
};

struct GraphEntry {
    GraphEntry(BackendRuntimeStats* runtime, CudaGraphKey entry_key)
        : key(std::move(entry_key)), staging(runtime) {}

    CudaGraphKey key;
    GraphOwner graph;
    GraphExecOwner executable;
    PinnedBuffer staging;
    GraphStagingLayout layout;
};

cudaError_t instrument_linear_graph(
    cudaGraph_t graph, std::span<const cudaEvent_t> timing_events,
    std::size_t prefix_nodes, std::size_t operation_nodes) noexcept;

}  // namespace k3x::cuda
