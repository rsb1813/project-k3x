# K3X Milestone 24 Bounded CUDA Graph Cache Design

## Purpose

Milestone 24 answers one narrow systems question left by B-0024: when the exact resident MoE-layer execution repeats an ordered routed-expert set, can CUDA Graph reuse reduce host launch overhead without changing output, routing, traffic, residency, validation, or failure behavior?

This milestone does not move KDA, MLA, routing, logits, argmax, recurrent state, or speculative rollback onto the GPU. It does not select CUDA Graphs as a default. It does not combine graph execution with reduced precision, dynamic L0 eviction, proxy experts, pruning, or a new speculative policy.

## Evidence and source boundary

The design is based on the following primary sources and installed implementation.

- NVIDIA CUDA Programming Guide 13.3, CUDA Graphs: <https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/cuda-graphs.html>.
- NVIDIA CUDA Runtime API 13.3.1 graph management and update contract: <https://docs.nvidia.com/cuda/cuda-runtime-api/index.html>.
- NVIDIA CUDA Runtime API graph thread-safety note: <https://docs.nvidia.com/cuda/cuda-runtime-api/graphs-thread-safety.html>.
- NVIDIA CUDA Samples `simpleCudaGraphs` at commit `b7c5481c556c3fe98db060207ecaa41a4b9a9abc`, which demonstrates both explicit graph construction and stream capture: <https://github.com/NVIDIA/cuda-samples/tree/b7c5481c556c3fe98db060207ecaa41a4b9a9abc/cpp/3_CUDA_Features/simpleCudaGraphs>.
- NVIDIA cuBLAS 13.3 documentation: <https://docs.nvidia.com/cuda/cublas/index.html>.
- Locally installed CUDA compiler `/usr/local/cuda-13.3/bin/nvcc`, version `13.3.73`, and native `sm_120` build on the RTX 5080.

The CUDA guide states that whole-graph update requires topologically identical graphs with consistent dependency and sink ordering. Individual node update avoids topology comparison when only a small number of known nodes changes. Graph objects are not internally synchronized. K3X therefore owns graph objects inside one backend and never exposes them across sessions or threads.

## Alternatives considered

### A. Whole-token CUDA Graph

Capture KDA, MLA, routing, MoE, logits, argmax, and recurrent-state mutation in one device graph.

Benefit: the largest possible reduction in CPU orchestration.

Cost: it expands the correctness boundary far beyond the isolated B-0024 result, requires device-resident recurrent state and routing, and confounds graph launch savings with a new execution architecture.

Decision: rejected for Milestone 24 and retained as a later independent experiment.

### B. One mutable superset graph

Capture the largest expert count once, update or disable nodes for smaller routed sets, and reuse one executable.

Benefit: avoids an ordered-set cache.

Cost: the current resident expert grid already represents a variable number of experts inside fixed kernels. A superset graph would not measure whether ordered-set locality justifies graph specialization and would add node-enable policy without evidence.

Decision: rejected for the first graph attribution.

### C. Reference, whole-update, and bounded ordered-set cache

Keep the current direct launch path, add a whole-update experiment that recaptures the same topology and calls `cudaGraphExecUpdate`, and add a bounded cache keyed by exact ordered execution identity.

Benefit: separately measures direct launches, update overhead, cold instantiate cost, warm cache hits, and cache churn under controlled traces.

Cost: each cache entry owns graph handles and pinned staging, and cache usefulness depends on ordered-set reuse.

Decision: accepted.

## Public runtime contract

Add `CudaGraphMode` with the following values.

| Value | Behavior |
|---|---|
| `disabled` | Existing exact direct launch reference. This remains the default. |
| `update` | Maintain one executable, recapture a topologically identical graph for each call, try whole-graph update, and re-instantiate only when no executable exists or update fails. |
| `cache` | Look up an exact ordered execution key in a bounded backend-local cache. Launch on hit; capture and instantiate on miss; evict deterministically when full. |

Add `cuda_graph_entries` as a hard count limit. The capability contract is strict.

- `disabled` requires `cuda_graph_entries == 0`.
- `update` requires `cuda_graph_entries == 1`.
- `cache` requires `cuda_graph_entries > 0`.
- Non-disabled graph execution requires `cuda-custom`, FP32 dense precision, reused allocation, resident weights, resident-grid batching, MoE-layer boundary, synchronous transfer, fusion none, admission validation, and positive resident capacity.
- CPU, `cuda-dense`, operation, FFN-block, transient, prefetch, fused, BF16-rounded, or per-call validation configurations reject graph options before backend construction.
- CUDA AURORA draft receives graph mode and capacity only when its draft boundary is MoE-layer. Target and draft own independent caches and telemetry.

No failure silently falls back from graph execution to direct execution. Explicit graph modes fail closed with `backend_unavailable` after CUDA capture, update, instantiate, upload, or launch failure. Capacity miss in the resident weight table retains the existing exact `executed=false` fallback before graph lookup or mutation.

## Ordered execution identity

A cache key contains only values that must remain fixed inside a captured executable.

- Dense resident members: six tensor IDs, resident device pointers, rows, and columns.
- Ordered expert set: gate, up, and down tensor IDs plus resident packed/scale device pointers in router order.
- Geometry: hidden, latent, routed, shared, intermediate widths and expert count.
- Scalars: epsilon, SiTU beta, SiTU linear presence/value.
- Dense plan identity: the five selected `DensePlan` object addresses.
- Device scratch generation: stable addresses and byte capacities for every scratch allocation used by the graph.

Dynamic input values and routing contributions are not part of the key. They are copied into entry-owned stable pinned staging before launch. The descriptor staging contains the exact device pointers represented by the key.

The key is not a model routing rule and never changes expert order. A different ordered set is a different key even if it contains the same experts in another order.

## Entry ownership and data flow

Each graph entry owns the following resources.

- One `cudaGraph_t` definition and one `cudaGraphExec_t` executable.
- One combined pinned host staging allocation with aligned slices for input, contributions, descriptors, and output.
- The immutable key and a monotonic last-use sequence.
- The graph node topology produced by stream capture on the backend's existing non-default stream.

The backend continues to own resident weights, device scratch, cuBLASLt plans, the CUDA stream, and timing events. Calls remain serialized by the runtime session. Graph objects never cross backend or session lifetimes.

On a cache hit, the backend performs this sequence.

1. Complete all existing host validation and resident-weight acquisition.
2. Verify the captured scratch addresses and capacities still match the key.
3. Copy input, contributions, and descriptors into the entry's pinned staging.
4. Launch the executable on the backend stream.
5. Synchronize once, copy the stable pinned output into the returned vector, and record exact telemetry.

On a cache miss, the backend allocates staging, captures the same H2D, 13-kernel, event, and D2H topology, instantiates the executable, optionally uploads it, launches it, and inserts it only after successful execution. A failed cold entry is destroyed and never becomes visible in the cache.

Scratch growth invalidates every cached entry before a new capture because captured device addresses cannot be silently reused after reallocation. Resident-weight entries do not move during a backend lifetime; a hard-cap bypass occurs before graph lookup.

## Update mode

`update` uses one entry slot and intentionally measures the documented whole-graph update workflow.

1. Prepare a fresh stable staging allocation and capture a fresh `cudaGraph_t` for the current key.
2. If no executable exists, instantiate it.
3. Otherwise call `cudaGraphExecUpdate` with the new graph.
4. On update success, replace the entry key and staging ownership and launch the updated executable.
5. On update failure, destroy the old executable and instantiate from the fresh graph. Increment both update-failure and instantiation counters.

Update mode is experimental attribution, not an automatic recovery path for cache mode.

## Bounded cache policy

All graph entries have approximately the same captured topology, so a count cap is the first bounded policy. Driver-internal graph memory is not reported as an invented byte count.

Cache hits update a backend-monotonic use sequence. On a miss at capacity, evict the entry with the smallest last-use sequence. Ties use canonical key order. Eviction destroys the executable, definition, and pinned staging before the replacement entry is admitted.

This LRU policy applies only to equal-topology graph executables. It does not replace or weaken the frequency-, task-, transition-, and prediction-aware L0/L1 expert residency policies required by the charter.

## Telemetry

Add independent target and draft counters.

- `cuda_graph_cache_hits`.
- `cuda_graph_cache_misses`.
- `cuda_graph_cache_evictions`.
- `cuda_graph_instantiations`.
- `cuda_graph_update_attempts`.
- `cuda_graph_update_successes`.
- `cuda_graph_update_failures`.
- `cuda_graph_launches`.
- `cuda_graph_invalidations`.
- `cuda_graph_host_nanoseconds` for lookup, capture, update, instantiate, staging copies, and launch submission, excluding stream wait.
- `cuda_graph_resident_entries` and `cuda_graph_peak_entries`.

Existing physical traffic, weight-cache, validation, kernel, synchronization, and MoE-layer counters remain authoritative. Graph execution still represents 13 logical kernels and one host stream synchronization per successful layer call. Kernel launch telemetry is logical work count, while graph launch telemetry records host graph submissions.

Pinned staging contributes to existing `pinned_host_bytes` and `peak_pinned_host_bytes`. No driver-internal graph memory byte figure is claimed.

## Correctness and failure invariants

- `disabled` output and telemetry retain the Milestone 23 reference behavior.
- Graph output must match the CPU oracle and direct CUDA result within the existing tolerance.
- Ordered contributions and expert descriptors must be refreshed on every launch.
- A different input with the same key must produce its own correct output.
- Reordered experts must never hit the previous ordered-set key.
- Cache eviction must not destroy resident weights or dense plans.
- Scratch growth invalidates entries before any stale pointer can launch.
- Non-finite immutable weights and identity conflicts fail before graph capture or launch.
- Resident-capacity bypass returns `executed=false` before graph counters change.
- Failed capture, update, instantiate, or launch never inserts a partially valid cache entry.
- Target and AURORA draft graph caches and counters remain independent.
- Graph mode never changes tokens, routing, recurrent state, or speculative acceptance.

## Test strategy

### Portable tests

- Pure key equality and canonical ordering.
- Deterministic bounded-cache hit, miss, LRU eviction, and tie behavior using a resource-free fake entry.
- Capability validation for every rejected option combination.
- CLI and benchmark schema zero defaults.

### CUDA tests

- Two launches of one ordered set produce one miss, one hit, one instantiation, two graph launches, and exact output.
- Dynamic input and contribution changes on a hit are observed by the output.
- Reordered experts miss and preserve router-order accumulation.
- Capacity-one alternating keys evict; capacity-two alternating keys hit after two cold calls.
- Update mode records one instantiation followed by successful update attempts when CUDA accepts the topology.
- One-byte resident capacity bypass leaves graph counters and pinned staging at zero.
- Admission conflict and non-finite failure remain pre-CUDA and graph-atomic.
- Scratch growth invalidates the old entry and recaptures before launch.
- Compute Sanitizer covers a warm cache hit and an eviction trace.

## B-0025 benchmark matrix

B-0025 remains a direct released-dimension MoE-layer microbenchmark with `routing_semantics=false`.

Use three deterministic traces with four logical expert views.

| Trace | Sequence | Purpose |
|---|---|---|
| `stable-1` | `A A A ...` | Best-case one-key reuse. |
| `alternating-2` | `A B A B ...` | Capacity-one churn versus capacity-two reuse. |
| `rotating-5` | `A B C D E ...` | Over-cap churn for capacities one, two, and four. |

Measure these execution identities with three warmups and 20 timed calls.

- Direct `disabled` reference for each trace.
- `update` with one executable.
- `cache` with capacities 1, 2, and 4.

Every row must preserve maximum absolute error 0, exact output sequence, zero fallback/bypass, zero warm weight H2D, identical activation H2D and D2H, one synchronization per call, 13 logical kernels per call, and exact validation identity. Record median/p10/p90 wall time plus every graph counter.

The report must distinguish cold instantiate, warm hit, update, and eviction calls. It must not emit or infer token throughput, TTFT, routing semantics, coding quality, full-model traffic, or full-checkpoint performance.

## Acceptance and default decision

The implementation is accepted as an experimental capability only if all correctness, traffic, lifecycle, sanitizer, and digest gates pass.

No graph mode becomes default from B-0025 alone. A later default decision requires all of the following.

- Positive warm latency on representative native-Linux execution.
- Sufficient ordered-set or compatible shape reuse under real K3 routing traces.
- A disclosed and bounded graph-memory policy.
- No regression in end-to-end token generation, routing, state, or quality.
- Separate evidence for interaction with dynamic expert residency.

If update or cache is slower, unstable, unsupported by captured cuBLASLt execution, or too churn-heavy, record the measured failure and retain `disabled` without hiding the result.
