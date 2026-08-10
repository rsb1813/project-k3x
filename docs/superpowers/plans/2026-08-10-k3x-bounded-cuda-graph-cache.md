# K3X Bounded CUDA Graph Cache Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add exact, opt-in CUDA Graph whole-update and bounded ordered-set cache experiments to the resident MoE-layer boundary, then measure B-0025 without changing defaults.

**Architecture:** A portable deterministic key/index owns cache policy while the CUDA backend owns graph definitions, executables, stable pinned staging, resident pointers, and capture lifecycle. The existing direct path remains `disabled`; `update` recaptures and calls `cudaGraphExecUpdate`; `cache` stores a hard-capped set of ordered execution identities.

**Tech Stack:** C++20, CUDA Runtime 13.3.73, cuBLASLt 13.3, native `sm_120`, CMake/CTest, Python 3.12/pytest, JSON/CSV evidence.

## Global Constraints

- Correctness remains first; every graph row must match the CPU and direct CUDA oracle.
- `CudaGraphMode::disabled` remains the default and preserves current behavior.
- Non-disabled graph modes are accepted only for exact FP32 resident MoE-layer execution with admission validation.
- Graph failure is fail-closed and never silently becomes a direct launch.
- Target and CUDA AURORA draft graph state and telemetry are independent.
- B-0025 is a direct layer microbenchmark with `routing_semantics=false`; no token/TPS/TTFT or quality claim is allowed.
- No paid cloud resource and no full Kimi K3 checkpoint is used.
- Every new C++ or Python source file starts with a one-line Korean role comment.

---

### Task 1: Portable ordered-key and bounded index

**Files:**
- Create: `runtime/include/k3x/cuda_graph_cache.hpp`
- Create: `runtime/src/cuda_graph_cache.cpp`
- Create: `tests/cpp/test_cuda_graph_cache.cpp`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Produces `k3x::CudaGraphKey` with `std::vector<std::uint64_t> words` and canonical equality/order.
- Produces `k3x::CudaGraphCacheDecision { bool hit; std::optional<CudaGraphKey> evicted; }`.
- Produces `k3x::BoundedCudaGraphIndex::touch(const CudaGraphKey&)`, `erase`, `clear`, `contains`, `size`, and `peak_size`.
- The CUDA backend later owns resources in a separate map keyed by `CudaGraphKey`; this index only chooses hits and victims.

- [x] **Step 1: Write the failing deterministic policy test**

```cpp
// CUDA Graph ordered identity와 bounded LRU index를 검증합니다.
#include "k3x/cuda_graph_cache.hpp"

int main() {
    k3x::BoundedCudaGraphIndex index(2);
    const k3x::CudaGraphKey a{{1, 10}};
    const k3x::CudaGraphKey b{{2, 20}};
    const k3x::CudaGraphKey c{{3, 30}};
    if (index.touch(a).hit || index.touch(b).hit) return 1;
    if (!index.touch(a).hit) return 2;
    const auto miss = index.touch(c);
    if (miss.hit || !miss.evicted || *miss.evicted != b) return 3;
    if (!index.contains(a) || !index.contains(c) || index.contains(b)) return 4;
    if (index.size() != 2 || index.peak_size() != 2) return 5;
    index.clear();
    return index.size() == 0 ? 0 : 6;
}
```

- [x] **Step 2: Configure and run RED**

Run:

```bash
cmake -S . -B build-cpu -DCMAKE_BUILD_TYPE=Release
cmake --build build-cpu -j2 --target test_cuda_graph_cache
```

Expected: configuration or compilation fails because the target and header do not exist.

- [x] **Step 3: Implement the minimum portable index**

Use a monotonic `std::uint64_t sequence_` and `std::map<CudaGraphKey, std::uint64_t> last_use_`. Reject zero capacity in the constructor with `std::invalid_argument`. On a full miss, select the pair with minimum `(last_use, key)` and erase it before inserting the new key.

- [x] **Step 4: Run GREEN and the portable suite**

Run:

```bash
cmake --build build-cpu -j2 --target test_cuda_graph_cache
ctest --test-dir build-cpu --output-on-failure -R 'cuda_graph_cache|backend|ops'
```

Expected: all selected tests pass.

- [x] **Step 5: Commit**

```bash
git add CMakeLists.txt runtime/include/k3x/cuda_graph_cache.hpp runtime/src/cuda_graph_cache.cpp tests/cpp/test_cuda_graph_cache.cpp
git commit -m "runtime: add bounded CUDA Graph index"
```

### Task 2: Strict public options and telemetry schema

**Files:**
- Modify: `runtime/include/k3x/backend.hpp`
- Modify: `runtime/cuda/backend_cuda.cu`
- Modify: `runtime/src/main.cpp`
- Modify: `runtime/src/cuda_moe_layer_bench.cpp`
- Modify: `tools/benchmark_synthetic.py`
- Modify: `tests/cpp/test_backend.cpp`
- Modify: `tests/python/test_cpp_parity.py`
- Modify: `tests/python/test_benchmark_schema.py`

**Interfaces:**
- Add `enum class CudaGraphMode { disabled, update, cache };`.
- Add `BackendOptions::cuda_graph` and `BackendOptions::cuda_graph_entries`.
- Add the twelve graph counters specified by the design to `BackendRuntimeStats`.
- Add CLI `--cuda-graph disabled|update|cache` and `--cuda-graph-entries N`.
- Add target and `draft_` JSON/CSV fields with zero defaults.

- [x] **Step 1: Write RED option and schema tests**

Add tests that invoke `k3x_run` with these cases.

```text
--cuda-graph cache --cuda-graph-entries 0
--cuda-graph update --cuda-graph-entries 2
--cuda-graph cache --cuda-graph-entries 1 --cuda-weight-validation per-call
--cuda-graph cache --cuda-graph-entries 1 --cuda-boundary ffn-block
```

Each must exit 2 with a graph-specific message. Add one `BenchmarkRecord` assertion that every graph field serializes as zero under CPU defaults.

- [x] **Step 2: Run RED**

Run:

```bash
K3X_BUILD_DIR=build-cpu python -m pytest -q \
  tests/python/test_cpp_parity.py -k cuda_graph \
  tests/python/test_benchmark_schema.py -k cuda_graph
```

Expected: tests fail because the arguments and fields are absent.

- [x] **Step 3: Add the strict contract and serialization**

Parse sizes with the existing `from_chars` path. Enforce the exact matrix in both CLI and `make_cuda_backend`; CLI errors remain exit 2, factory errors use `backend_unavailable`. Output mode names exactly `disabled`, `update`, and `cache`.

- [x] **Step 4: Run GREEN and second-order searches**

Run:

```bash
cmake --build build-cpu -j2 --target k3x_run test_backend
K3X_BUILD_DIR=build-cpu python -m pytest -q \
  tests/python/test_cpp_parity.py -k cuda_graph \
  tests/python/test_benchmark_schema.py -k cuda_graph
rg -n "cuda_graph_(cache|update|launch|resident|peak|host)" runtime tools tests
```

Expected: selected tests pass and every public counter appears in backend, runtime JSON, benchmark model, aggregation, and CSV field lists.

- [x] **Step 5: Commit**

```bash
git add runtime/include/k3x/backend.hpp runtime/cuda/backend_cuda.cu runtime/src/main.cpp runtime/src/cuda_moe_layer_bench.cpp tools/benchmark_synthetic.py tests/cpp/test_backend.cpp tests/python/test_cpp_parity.py tests/python/test_benchmark_schema.py
git commit -m "runtime: define strict CUDA Graph modes"
```

### Task 3: Graph resource ownership and warm cache-hit execution

**Files:**
- Create: `runtime/cuda/graph_resources.cuh`
- Create: `runtime/cuda/graph_resources.cu`
- Modify: `runtime/cuda/backend_cuda.cu`
- Modify: `CMakeLists.txt`
- Modify: `tests/cuda/test_cuda_moe_layer.cu`

**Interfaces:**
- `cuda::GraphOwner` owns one `cudaGraph_t`.
- `cuda::GraphExecOwner` owns one `cudaGraphExec_t`.
- `cuda::GraphEntry` owns a key, graph, executable, one `PinnedBuffer`, staging offsets, and last output size.
- `CudaBackend::capture_resident_moe_graph(...)` captures the exact existing H2D, 13 logical operations, timing events, and D2H sequence.

- [x] **Step 1: Write RED warm-hit and dynamic-staging tests**

Extend the CUDA fixture to create `cache` options with capacity 1 and admission validation. Call the same ordered expert set twice with different input and contribution values. Compare both outputs with independent CPU calls and assert this delta after two successful calls.

```cpp
if (stats.cuda_graph_cache_misses != 1 ||
    stats.cuda_graph_cache_hits != 1 ||
    stats.cuda_graph_instantiations != 1 ||
    stats.cuda_graph_launches != 2 ||
    stats.cuda_graph_resident_entries != 1 ||
    stats.cuda_graph_peak_entries != 1) return 40;
```

- [x] **Step 2: Build and run RED**

Run:

```bash
cmake --build build-cuda -j2 --target test_cuda_moe_layer
ctest --test-dir build-cuda --output-on-failure -R '^cuda_moe_layer$'
```

Expected: compile or assertion failure because graph execution does not exist.

- [x] **Step 3: Implement entry resources and capture**

Allocate one combined pinned buffer for input, contribution, descriptor, and output slices with `align_up(offset, alignof(std::max_align_t))`. Copy dynamic host data into staging before capture or launch. Capture only after validation, resident acquisition, dense plan resolution, and scratch reservation succeed. Instantiate and launch; insert the entry only after synchronization and exact output extraction succeed.

- [x] **Step 4: Run GREEN and Compute Sanitizer smoke**

Run:

```bash
cmake --build build-cuda -j2 --target test_cuda_moe_layer
ctest --test-dir build-cuda --output-on-failure -R '^cuda_moe_layer$'
compute-sanitizer --tool memcheck --error-exitcode=99 ./build-cuda/test_cuda_moe_layer
```

Expected: exact output and `ERROR SUMMARY: 0 errors`.

- [x] **Step 5: Commit**

```bash
git add CMakeLists.txt runtime/cuda/graph_resources.cuh runtime/cuda/graph_resources.cu runtime/cuda/backend_cuda.cu tests/cuda/test_cuda_moe_layer.cu
git commit -m "runtime: execute resident MoE CUDA Graph hits"
```

### Task 4: Ordered misses, eviction, update, and invalidation

**Files:**
- Modify: `runtime/cuda/backend_cuda.cu`
- Modify: `tests/cuda/test_cuda_moe_layer.cu`
- Modify: `tests/cpp/test_backend.cpp`

**Interfaces:**
- Cache mode uses `BoundedCudaGraphIndex` and `std::map<CudaGraphKey, std::unique_ptr<cuda::GraphEntry>>`.
- Update mode owns exactly one current entry and calls `cudaGraphExecUpdate` after a fresh capture.
- Scratch pointer/capacity change calls one `invalidate_graph_cache()` before new capture.

- [x] **Step 1: Write RED ordered-set and update tests**

Add four independent cases.

1. Reorder two experts and assert two misses, zero hits, exact CPU output.
2. Alternate A/B/A/B with capacity one and assert four misses and three evictions.
3. Alternate A/B/A/B with capacity two and assert two misses, two hits, zero evictions.
4. Run update mode twice and assert one instantiation, one update attempt, one update success, zero update failure, and two launches.

Also retain the one-byte bypass and invalid admission tests and assert every graph counter stays zero.

- [x] **Step 2: Run RED**

Run:

```bash
cmake --build build-cuda -j2 --target test_cuda_moe_layer
./build-cuda/test_cuda_moe_layer
```

Expected: the new eviction/update assertions fail.

- [x] **Step 3: Implement deterministic eviction and update**

Destroy victim resources before replacement capture. For `cudaGraphExecUpdate`, record attempt first, success only for `cudaGraphExecUpdateSuccess`, and on any other result record failure, destroy the old executable, and instantiate the new graph. Never reuse a failed fresh entry.

- [x] **Step 4: Run GREEN and CUDA CTest**

Run:

```bash
cmake --build build-cuda -j2
ctest --test-dir build-cuda --output-on-failure
```

Expected: all CUDA CTest targets pass.

- [x] **Step 5: Commit**

```bash
git add runtime/cuda/backend_cuda.cu tests/cuda/test_cuda_moe_layer.cu tests/cpp/test_backend.cpp
git commit -m "runtime: bound CUDA Graph update and eviction"
```

### Task 5: CUDA AURORA ownership and end-to-end schema

**Files:**
- Modify: `runtime/src/main.cpp`
- Modify: `tools/benchmark_synthetic.py`
- Modify: `tests/python/test_cuda_aurora_draft.py`
- Modify: `tests/python/test_benchmark_schema.py`

**Interfaces:**
- Add `--aurora-draft-graph disabled|update|cache` and `--aurora-draft-graph-entries N` only for persistent `cuda-custom + moe-layer` draft execution.
- Target options never implicitly enable draft graph mode.
- JSON and CSV expose independent target/draft graph identities and counters.

- [ ] **Step 1: Write RED ownership tests**

Run an exact target with graph disabled and a CUDA AURORA MoE-layer draft with cache capacity one. Assert target graph counters remain zero while draft miss/hit/launch counters are positive. Add the inverse case and invalid draft-boundary rejection.

- [ ] **Step 2: Run RED**

Run:

```bash
K3X_BUILD_DIR=build-cuda python -m pytest -q \
  tests/python/test_cuda_aurora_draft.py -k graph \
  tests/python/test_benchmark_schema.py -k graph
```

Expected: missing CLI fields or ownership assertions fail.

- [ ] **Step 3: Implement explicit draft propagation and aggregation**

Parse draft graph settings separately, validate them with the existing AURORA ownership gates, assign them only to `draft_backend_options`, and serialize both option identities and runtime counters. Preserve zero defaults for ordinary greedy and CPU draft paths.

- [ ] **Step 4: Run GREEN**

Run the same pytest command and `ctest --test-dir build-cuda --output-on-failure`.

- [ ] **Step 5: Commit**

```bash
git add runtime/src/main.cpp tools/benchmark_synthetic.py tests/python/test_cuda_aurora_draft.py tests/python/test_benchmark_schema.py
git commit -m "runtime: separate target and draft graph caches"
```

### Task 6: Canonical B-0025 runner and committed-evidence verifier

**Files:**
- Modify: `runtime/src/cuda_moe_layer_bench.cpp`
- Create: `tools/ablate_cuda_graph_cache.py`
- Create: `tests/python/test_cuda_graph_cache_ablation.py`

**Interfaces:**
- Benchmark CLI adds `--graph`, `--graph-entries`, and `--trace stable-1|alternating-2|rotating-5`.
- Runner emits canonical raw JSON, `summary.json`, and LF-only `summary.csv`.
- Summary records artifact, runner, aggregate, raw, JSON, and CSV SHA-256 digests.

- [ ] **Step 1: Write RED canonical-matrix tests**

The fake runner test must assert case order, command arguments, formulas, and failures for one mutated field each: output error, graph hit/miss count, eviction count, warm weight H2D, synchronization, logical kernel count, activation H2D, and fallback.

- [ ] **Step 2: Run RED**

Run:

```bash
python -m pytest -q tests/python/test_cuda_graph_cache_ablation.py
```

Expected: import failure because the runner does not exist.

- [ ] **Step 3: Implement the 15-row matrix**

For each of the three traces emit direct disabled, update-1, cache-1, cache-2, and cache-4. Require 3 warmups and 20 measured calls for formal evidence. Recompute all graph expectations from the trace and capacity rather than hardcoding summary claims.

- [ ] **Step 4: Run GREEN and live one-iteration smoke**

Run:

```bash
python -m pytest -q tests/python/test_cuda_graph_cache_ablation.py
python -m tools.ablate_cuda_graph_cache \
  --artifact build-fixtures/released-expert.k3x \
  --runner build-cuda/k3x_cuda_moe_layer_bench \
  --output-dir build-results/b0025-smoke \
  --warmup 0 --iterations 1
```

Expected: the verifier passes all 15 rows or records a capability failure without publishing evidence.

- [ ] **Step 5: Commit**

```bash
git add runtime/src/cuda_moe_layer_bench.cpp tools/ablate_cuda_graph_cache.py tests/python/test_cuda_graph_cache_ablation.py
git commit -m "bench: add bounded CUDA Graph ablation"
```

### Task 7: Formal measurement, full verification, and TITAN Ledger

**Files:**
- Create: `results/b0025-cuda-graph-cache-wsl/*`
- Modify: `README.md`
- Modify: `ARCHITECTURE.md`
- Modify: `PERFORMANCE_MODEL.md`
- Modify: `DECISIONS.md`
- Modify: `BENCHMARKS.md`
- Modify: `PLAN.md`
- Modify: `checklist.md`
- Modify: `context-notes.md`
- Modify last: `PROJECT_STATE.md`

**Interfaces:**
- B-0025 records only measured layer-boundary fields and explicit unavailable fields.
- D-050 accepts or rejects graph modes from measured evidence without changing the default absent all design acceptance gates.

- [ ] **Step 1: Run formal B-0025**

Run the canonical runner with 3 warmups and 20 iterations into `results/b0025-cuda-graph-cache-wsl/`. Do not reuse smoke artifacts.

- [ ] **Step 2: Cross-check committed evidence**

Run the evidence verifier after `git add` so it reads canonical Git blobs, not a CRLF-materialized worktree view. Recompute every raw digest, aggregate, formula, and summary digest.

- [ ] **Step 3: Run full verification**

Run CPU CTest/pytest, liburing/direct CTest/pytest, ASan/UBSan CTest, CUDA CTest/pytest, focused evidence tests, and Compute Sanitizer for warm-hit plus eviction traces. Record exact pass/skip counts and sanitizer summaries.

- [ ] **Step 4: Update documentation and state last**

Document measured values only. Mark graph modes experimental and opt-in unless representative native-Linux end-to-end evidence satisfies the design acceptance gate. Update `PROJECT_STATE.md` after every other ledger document.

- [ ] **Step 5: Self-review and commit evidence**

Check the complete `origin/main...HEAD` diff for default-path changes, graph lifecycle leaks, stale pointer paths, target/draft ownership, raw/summary parity, and overclaimed performance. Commit measurement and documentation in separate semantic commits if both are substantial.

- [ ] **Step 6: Final review and public integration**

Request one Critical/Important-only read-only review, apply at most one correction batch, rerun affected gates, push a public PR, require correctness and CodeQL, rebase-merge, and verify post-merge `main` correctness and CodeQL.
