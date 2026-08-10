// CUDA Graph 관련 RAII 자원의 생성과 해제를 구현합니다.
#include "graph_resources.cuh"

#include <utility>
#include <vector>

namespace k3x::cuda {

GraphOwner::~GraphOwner() { reset(); }

GraphOwner::GraphOwner(GraphOwner&& other) noexcept
    : graph_(std::exchange(other.graph_, nullptr)) {}

GraphOwner& GraphOwner::operator=(GraphOwner&& other) noexcept {
    if (this != &other) {
        reset();
        graph_ = std::exchange(other.graph_, nullptr);
    }
    return *this;
}

void GraphOwner::reset() noexcept {
    if (graph_) cudaGraphDestroy(std::exchange(graph_, nullptr));
}

GraphExecOwner::~GraphExecOwner() { reset(); }

GraphExecOwner::GraphExecOwner(GraphExecOwner&& other) noexcept
    : executable_(std::exchange(other.executable_, nullptr)) {}

GraphExecOwner& GraphExecOwner::operator=(GraphExecOwner&& other) noexcept {
    if (this != &other) {
        reset();
        executable_ = std::exchange(other.executable_, nullptr);
    }
    return *this;
}

void GraphExecOwner::reset() noexcept {
    if (executable_) cudaGraphExecDestroy(
        std::exchange(executable_, nullptr));
}

cudaError_t instrument_linear_graph(
    cudaGraph_t graph, std::span<const cudaEvent_t> timing_events,
    std::size_t prefix_nodes, std::size_t operation_nodes) noexcept {
    if (!graph || timing_events.size() != operation_nodes * 2) {
        return cudaErrorInvalidValue;
    }
    std::size_t node_count = 0;
    std::size_t edge_count = 0;
    if (cudaGraphGetNodes(graph, nullptr, &node_count) != cudaSuccess ||
        cudaGraphGetEdges(graph, nullptr, nullptr, nullptr, &edge_count) !=
            cudaSuccess ||
        node_count != prefix_nodes + operation_nodes + 1 ||
        edge_count + 1 != node_count) {
        return cudaErrorInvalidValue;
    }
    std::vector<cudaGraphNode_t> nodes(node_count);
    std::vector<cudaGraphNode_t> from(edge_count);
    std::vector<cudaGraphNode_t> to(edge_count);
    if (cudaGraphGetNodes(graph, nodes.data(), &node_count) != cudaSuccess ||
        cudaGraphGetEdges(graph, from.data(), to.data(), nullptr,
                          &edge_count) != cudaSuccess) {
        return cudaErrorInvalidValue;
    }
    std::vector<cudaGraphNode_t> ordered;
    ordered.reserve(node_count);
    cudaGraphNode_t current{};
    for (const auto node : nodes) {
        bool has_parent = false;
        for (const auto child : to) {
            if (child == node) {
                has_parent = true;
                break;
            }
        }
        if (!has_parent) {
            if (current) return cudaErrorInvalidValue;
            current = node;
        }
    }
    while (current) {
        ordered.push_back(current);
        cudaGraphNode_t next{};
        for (std::size_t index = 0; index < edge_count; ++index) {
            if (from[index] != current) continue;
            if (next) return cudaErrorInvalidValue;
            next = to[index];
        }
        current = next;
    }
    if (ordered.size() != node_count ||
        cudaGraphRemoveDependencies(graph, from.data(), to.data(), nullptr,
                                    edge_count) != cudaSuccess) {
        return cudaErrorInvalidValue;
    }
    std::vector<cudaGraphNode_t> sequence;
    sequence.reserve(node_count + timing_events.size());
    sequence.insert(sequence.end(), ordered.begin(),
                    ordered.begin() + prefix_nodes);
    for (std::size_t operation = 0; operation < operation_nodes; ++operation) {
        cudaGraphNode_t start{};
        cudaGraphNode_t end{};
        if (cudaGraphAddEventRecordNode(
                &start, graph, nullptr, 0,
                timing_events[operation * 2]) != cudaSuccess ||
            cudaGraphAddEventRecordNode(
                &end, graph, nullptr, 0,
                timing_events[operation * 2 + 1]) != cudaSuccess) {
            return cudaErrorInvalidValue;
        }
        sequence.push_back(start);
        sequence.push_back(ordered[prefix_nodes + operation]);
        sequence.push_back(end);
    }
    sequence.push_back(ordered.back());
    std::vector<cudaGraphNode_t> sequence_from(sequence.size() - 1);
    std::vector<cudaGraphNode_t> sequence_to(sequence.size() - 1);
    for (std::size_t index = 0; index + 1 < sequence.size(); ++index) {
        sequence_from[index] = sequence[index];
        sequence_to[index] = sequence[index + 1];
    }
    return cudaGraphAddDependencies(
        graph, sequence_from.data(), sequence_to.data(), nullptr,
        sequence_from.size());
}

}  // namespace k3x::cuda
