# K3X Official KDA Transformer-Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute and measure one bounded official Kimi K3 layer-1 KDA transformer boundary with exact full/incremental state parity, natural Top-16 native-MXFP4 experts, and independent CPU/CUDA correctness.

**Architecture:** Extend the existing official M28 manufacturing and MoE boundary with one fail-closed 17-tensor KDA plan. An independent PyTorch scalar recurrence derives two-token state and routes before selected-expert materialization; portable C++ and native `sm_120` CUDA then execute the same complete layer behind a dedicated non-production harness.

**Tech Stack:** Python 3.12, PyTorch, safetensors-compatible bounded ranges, K3X v1, C++20, CUDA 13.3 `sm_120`, cuBLASLt, pytest, CMake/CTest, Compute Sanitizer, JSON/CSV, SHA-256, and CRC32C.

## Global Constraints

- Bind `moonshotai/Kimi-K3` to revision `9f62e4e9fffbd0a83ddd60e1c209d828994b3569` and pinned source Git blob `b8c41e8bfce768d74d8da3a37e693f5ee43876a0`.
- Require layer 1, KDA heads 96, head dimension 128, short-convolution width 4, full-rank output gate, gate lower bound `-5.0`, Attention Residual block size 12, and RMS epsilon `1e-5`.
- Require F32 `self_attn.A_log` shape `[128]`; reject `[96]` before payload use.
- Keep K3X v1 fixed records unchanged and keep the final artifact rejected by `k3x_run` through the existing storage-fixture guard.
- Preserve BF16, F32, and native-MXFP4 source bytes exactly; never requantize the official experts.
- Cap every tensor payload request and local copy at 8 MiB; never download a complete shard or checkpoint.
- Never provision paid cloud resources in this plan.
- Keep actual tensor objects and generated K3X files ignored under `artifacts/`; commit only bounded metadata and evidence.
- Witness RED before implementation, run focused GREEN plus regressions, self-review every diff, and make one semantic commit per task.
- Every new Python, C++, or CUDA source file starts with a one-line Korean role comment.
- B-0030 is an isolated layer benchmark. It must not emit or imply decode tok/s, prefill tok/s, TTFT, coding quality, or physical NVMe/PCIe traffic.

---

### Task 1: Pinned official KDA metadata and tensor plan

**Files:**
- Create: `converter/k3x_converter/official_layer.py`
- Modify: `converter/k3x_converter/official_source.py`
- Modify: `tests/python/test_official_source.py`
- Create: `tests/python/test_official_layer.py`

**Interfaces:**
- Extends: `OfficialConfig` with `num_hidden_layers`, `kda_layers`, `kda_heads`, `kda_head_dim`, `short_conv_kernel_size`, `kda_gate_lower_bound`, `kda_use_full_rank_gate`, and `attn_res_block_size`.
- Produces: `OfficialLayerPlan(layer_id, shard_path, index_sha256, source_blob_id, kda_tensors, kda_payload_bytes, moe_plan, base_payload_bytes, maximum_two_token_bytes)`.
- Produces: `plan_official_kda_layer(index, header, config, *, source_blob_id, layer_id=1) -> OfficialLayerPlan`.
- Consumes: `OfficialIndex`, `OfficialShardHeader`, `OfficialConfig`, `PlannedTensor`, and `plan_official_moe_slice`.

- [x] **Step 1: Write config and exact-plan RED tests**

Add tests that load the pinned config fixture and require the exact KDA fields. Build a synthetic header with all 17 accepted tensors and require sorted `PlannedTensor` records, 887,843,840 bytes, source blob binding, and maximum 1,829,256,704 bytes. Mutate `A_log` to `[96]`, each dtype/shape, source blob, shard mapping, KDA membership, gate mode, and byte total independently and require `K3XError`.

```python
plan = plan_official_kda_layer(
    index,
    header,
    config,
    source_blob_id="b8c41e8bfce768d74d8da3a37e693f5ee43876a0",
    layer_id=1,
)
assert plan.kda_payload_bytes == 887_843_840
assert plan.base_payload_bytes == 1_267_744_256
assert plan.maximum_two_token_bytes == 1_829_256_704
assert plan.kda_tensors[0].official_name.endswith("self_attention_res_norm.weight")
assert next(x for x in plan.kda_tensors if x.role == "kda_a_log").shape == (128,)
```

- [x] **Step 2: Run RED**

```bash
PYTHONPATH=converter:reference /home/jolib/.venvs/k3x-m1/bin/python -m pytest \
  tests/python/test_official_source.py \
  tests/python/test_official_layer.py -q
```

Expected: import failure for `k3x_converter.official_layer` or missing KDA config fields.

- [x] **Step 3: Implement strict config parsing and the pure plan**

Parse the existing pinned `linear_attn_config` without a second network request. In `official_layer.py`, define the exact 17 suffix/dtype/shape/role records from the design, resolve each through the index and inspected header, require one shard, and reuse the already validated M28 `OfficialMoePlan`.

```python
@dataclass(frozen=True)
class OfficialLayerPlan:
    layer_id: int
    shard_path: str
    index_sha256: str
    source_blob_id: str
    kda_tensors: tuple[PlannedTensor, ...]
    kda_payload_bytes: int
    moe_plan: OfficialMoePlan
    base_payload_bytes: int
    maximum_two_token_bytes: int
```

- [x] **Step 4: Run focused GREEN and official-source regressions**

```bash
PYTHONPATH=converter:reference /home/jolib/.venvs/k3x-m1/bin/python -m pytest \
  tests/python/test_official_source.py \
  tests/python/test_official_moe.py \
  tests/python/test_official_layer.py -q
```

Expected: all runnable tests pass without network or payload access.

- [x] **Step 5: Self-review and commit**

```bash
git diff --check
git add converter/k3x_converter/official_source.py converter/k3x_converter/official_layer.py \
  tests/python/test_official_source.py tests/python/test_official_layer.py
git commit -m "feat: plan official KDA layer tensors"
```

### Task 2: Independent PyTorch KDA recurrence and state contract

**Files:**
- Create: `reference/k3x_ref/official_kda.py`
- Create: `tests/python/test_official_kda.py`

**Interfaces:**
- Produces: `OfficialKdaConfig`, `OfficialKdaWeights`, `OfficialKdaState`, and `OfficialKdaResult` frozen dataclasses.
- Produces: `zero_official_kda_state(config, batch_size, device) -> OfficialKdaState`.
- Produces: `official_kda(hidden, weights, state, config) -> OfficialKdaResult` for `[batch, sequence, hidden]`.
- State identity: BF16 convolution histories `[B,3,12288]` and FP32 V-first recurrence `[B,96,128V,128K]` at official dimensions.

- [x] **Step 1: Write tiny recurrence RED tests**

Use a tiny `hidden=4, heads=2, head_dim=2, conv_width=3` fixture with literal tensors. Independently compute one step from a nonzero state using the paper recurrence `(I - beta*k*k^T) @ (Diag(alpha) @ S) + beta*k*v^T`. Require exact boundary shapes, finite output, V-first serialization, and unchanged input state.

```python
full = official_kda(torch.stack((token_a, token_b), dim=1), weights, zero, cfg)
first = official_kda(token_a[:, None], weights, zero, cfg)
second = official_kda(token_b[:, None], weights, first.state, cfg)
torch.testing.assert_close(full.output, torch.cat((first.output, second.output), dim=1))
torch.testing.assert_close(full.state.recurrent_v_first, second.state.recurrent_v_first)
```

Add independent negatives for F32/BF16 dtype drift, `[heads]` A-log, wrong convolution history width, K-first state labeling, non-finite weights, and state alias mutation.

- [x] **Step 2: Run RED**

```bash
PYTHONPATH=converter:reference /home/jolib/.venvs/k3x-m1/bin/python -m pytest \
  tests/python/test_official_kda.py -q
```

Expected: import failure because `official_kda.py` does not exist.

- [x] **Step 3: Implement the scalar semantic oracle**

Implement explicit BF16 operation boundaries, F32 convolution weights, SiLU short convolution, Q/K L2 normalization, channel-wise decay, scalar-per-head beta, FP32 recurrence, sigmoid-gated head RMSNorm, and BF16 output projection. Convert V-first storage to mathematical K-by-V only around the recurrence and convert back before publishing state.

```python
decayed = alpha.unsqueeze(-1) * state_kv
prediction = torch.einsum("...k,...kv->...v", k.float(), decayed)
delta = (v.float() - prediction) * beta.float().unsqueeze(-1)
updated = decayed + k.float().unsqueeze(-1) * delta.unsqueeze(-2)
output = torch.einsum("...k,...kv->...v", q.float(), updated)
```

- [x] **Step 4: Run GREEN and existing synthetic KDA regressions**

```bash
PYTHONPATH=converter:reference /home/jolib/.venvs/k3x-m1/bin/python -m pytest \
  tests/python/test_official_kda.py tests/python/test_kda.py tests/python/test_model.py -q
```

Expected: all runnable tests pass and no existing synthetic semantics change.

- [x] **Step 5: Self-review and commit**

```bash
git diff --check
git add reference/k3x_ref/official_kda.py tests/python/test_official_kda.py
git commit -m "feat: add official KDA scalar oracle"
```

### Task 3: Bounded complete-layer manufacturing and route/state manifest

**Files:**
- Modify: `converter/k3x_converter/official_layer.py`
- Modify: `converter/k3x_converter/official_moe.py`
- Modify: `tools/discover_official_kimi_k3.py`
- Modify: `tests/python/test_official_layer.py`
- Modify: `tests/python/test_official_discovery_cli.py`

**Interfaces:**
- Produces: `OfficialLayerInput`, `OfficialLayerRouteStep`, `OfficialLayerRoutes`, and `OfficialLayerMaterializationReport`.
- Produces: `derive_official_layer_routes(plan, objects, inputs) -> OfficialLayerRoutes` using the Task 2 oracle and existing natural router.
- Produces: `materialize_official_kda_layer(snapshot, index, config, header, plan, transport, output_dir, *, chunk_bytes) -> OfficialLayerMaterializationReport`.
- CLI: adds `--scope kda-layer`; default remains dry-run and payload requires the existing explicit `--materialize --output-dir` pair.

- [ ] **Step 1: Write source-order, fail-atomic, and CLI RED tests**

Require the 17 KDA objects to be materialized before route publication, the route/state manifest before selected experts, and final packing in exact execution order. Verify completed-object rehash/reuse, verified-partial resume, corrupt-partial restart, no unselected expert request, 8 MiB cap, response-byte accounting, final root binding, and `NON_EXECUTABLE_ARTIFACT`.

```python
assert report.requested_payload_bytes == 887_843_840 + 379_900_416 + 17_547_264 * len(report.selected_experts)
assert report.maximum_response_bytes <= 8 * 1024 * 1024
assert manifest["state_layout"] == "v-first-fp32"
assert manifest["steps"][0]["name"] == "a"
assert manifest["steps"][1]["consumes_state_sha256"] == manifest["steps"][0]["state_sha256"]
```

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=converter:reference /home/jolib/.venvs/k3x-m1/bin/python -m pytest \
  tests/python/test_official_layer.py \
  tests/python/test_official_discovery_cli.py -q
```

Expected: missing complete-layer materializer and unsupported `kda-layer` scope.

- [ ] **Step 3: Implement orchestration by reusing verified M28 primitives**

Reuse `materialize_official_range_object`, source assembly, expert planning, and conversion rather than duplicating transport or ledger logic. Publish route/state JSON atomically only after both full and incremental PyTorch paths agree. Bind source blob, config/index/header identities, input hashes, initial/final state hashes, routes, tensor digests, source digest, and final K3X root.

- [ ] **Step 4: Run GREEN and converter recovery regressions**

```bash
PYTHONPATH=converter:reference /home/jolib/.venvs/k3x-m1/bin/python -m pytest \
  tests/python/test_official_layer.py \
  tests/python/test_official_moe.py \
  tests/python/test_official_discovery_cli.py \
  tests/python/test_official_source.py \
  tests/python/test_converter_resume.py \
  tests/python/test_source_manifest_integrity.py -q
```

- [ ] **Step 5: Self-review and commit**

```bash
git diff --check
git add converter/k3x_converter/official_layer.py converter/k3x_converter/official_moe.py \
  tools/discover_official_kimi_k3.py tests/python/test_official_layer.py \
  tests/python/test_official_discovery_cli.py
git commit -m "feat: manufacture bounded official KDA layer"
```

### Task 4: Portable C++ KDA recurrence

**Files:**
- Create: `runtime/include/k3x/official_kda.hpp`
- Create: `runtime/src/official_kda.cpp`
- Create: `tests/cpp/test_official_kda.cpp`
- Modify: `tests/python/test_cpp_parity.py`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Produces: `OfficialKdaWeightsView`, `OfficialKdaState`, `OfficialKdaBoundaries`, and `OfficialKdaResult`.
- Produces: `Result<OfficialKdaResult> official_kda_cpu(span<float> hidden, OfficialKdaWeightsView, const OfficialKdaState&, const OfficialKdaConfig&)`.
- Consumes: native BF16/F32 byte views and no Reader, filesystem, network, CUDA, or global state.

- [ ] **Step 1: Add C++ tiny RED and Python dump parity**

Register `test_official_kda`. Reuse the Task 2 tiny literals, require full/incremental state parity, and expose `--dump` JSON containing projection, convolution, decay, beta, recurrent, gated, projected, and final-state values. Python independently recomputes every field.

- [ ] **Step 2: Configure/build and witness RED**

```bash
cmake -S . -B build -DK3X_ENABLE_CUDA=OFF
cmake --build build --target test_official_kda -j2
```

Expected: CMake fails because `runtime/src/official_kda.cpp` is absent.

- [ ] **Step 3: Implement the minimum portable recurrence**

Decode BF16 words explicitly, preserve F32 convolution/A-log/bias/norm, use checked dimension products before allocation, and publish a result only after every shape, dtype, layout, and finiteness check passes. Keep state conversion local and deterministic.

- [ ] **Step 4: Run focused GREEN and Python parity**

```bash
cmake --build build --target test_official_kda -j2
ctest --test-dir build -R 'official_kda|official_moe' --output-on-failure
PYTHONPATH=converter:reference /home/jolib/.venvs/k3x-m1/bin/python -m pytest \
  tests/python/test_cpp_parity.py -q
```

- [ ] **Step 5: Self-review and commit**

```bash
git diff --check
git add CMakeLists.txt runtime/include/k3x/official_kda.hpp runtime/src/official_kda.cpp \
  tests/cpp/test_official_kda.cpp tests/python/test_cpp_parity.py
git commit -m "feat: execute portable official KDA"
```

### Task 5: Portable complete-layer composition and pinned harness contract

**Files:**
- Create: `runtime/include/k3x/official_layer.hpp`
- Create: `runtime/src/official_layer.cpp`
- Create: `tests/cpp/test_official_layer.cpp`
- Create: `runtime/src/cuda_official_layer_bench.cpp`
- Create: `tests/python/test_cuda_official_layer.py`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Produces: `OfficialLayerWeights`, `OfficialLayerInput`, `OfficialLayerState`, `OfficialLayerStepResult`, and `official_layer_cpu(...)`.
- Reuses: `official_kda_cpu`, `prepare_official_moe_input`, `route_official_moe`, and `official_moe_cpu`.
- Produces benchmark preflight that validates manifest, Reader root, exact tensor order/types/shapes, inputs, state chain, routes, and CPU oracle before any CUDA backend is constructed.

- [ ] **Step 1: Write composition and preflight RED tests**

Require the exact graph order, token A state feeding token B, full/incremental equality, 16 unique natural IDs per official route, and immutable input/state. Mutate every pinned digest, tensor order, route, contribution, state hash/layout, optional feature, and root binding independently; require failure before backend construction.

- [ ] **Step 2: Build and witness RED**

```bash
cmake --build build --target test_official_layer -j2
```

Expected: missing target/source failure.

- [ ] **Step 3: Implement pure composition and benchmark preflight**

The portable result owns self-Attention-Residual output, input norm, KDA boundaries/state, post-KDA prefix, MLP Attention Residual output, normalized MoE input, route, MoE boundaries, and final output. The benchmark executable accepts only `--artifact`, `--manifest`, `--case a|ab-full|ab-incremental`, `--weight-mode transient|resident`, `--warmups`, and `--iterations`.

- [ ] **Step 4: Run GREEN and guard regressions**

```bash
cmake --build build --target test_official_layer k3x_cuda_official_layer_bench -j2
ctest --test-dir build -R 'official_layer|official_kda|official_moe' --output-on-failure
PYTHONPATH=converter:reference /home/jolib/.venvs/k3x-m1/bin/python -m pytest \
  tests/python/test_cuda_official_layer.py tests/python/test_cpp_reader.py -q
```

- [ ] **Step 5: Self-review and commit**

```bash
git diff --check
git add CMakeLists.txt runtime/include/k3x/official_layer.hpp runtime/src/official_layer.cpp \
  runtime/src/cuda_official_layer_bench.cpp tests/cpp/test_official_layer.cpp \
  tests/python/test_cuda_official_layer.py
git commit -m "feat: compose portable official KDA layer"
```

### Task 6: Native CUDA KDA and complete-layer boundary

**Files:**
- Modify: `runtime/include/k3x/backend.hpp`
- Modify: `runtime/cuda/backend_cuda.cu`
- Create: `tests/cuda/test_cuda_official_kda.cu`
- Create: `tests/cuda/test_cuda_official_layer.cu`
- Modify: `runtime/src/cuda_official_layer_bench.cpp`
- Modify: `tests/python/test_cuda_official_layer.py`
- Modify: `CMakeLists.txt`

**Interfaces:**
- Produces: `OfficialKdaView`, `OfficialKdaStateView`, `OfficialKdaCudaResult`, and virtual `official_kda(...)` with a default `backend_unavailable` implementation.
- Produces: one complete-layer CUDA dispatch that reuses `official_mxfp4_moe_ffn` after KDA and routing, without changing natural IDs or contributions.
- Telemetry separates BF16/F32/MXFP4 weight H2D, activation H2D, state H2D/D2H, final D2H, kernel time, allocations, synchronizations, and exact resident hits/misses/bypasses.

- [ ] **Step 1: Write CUDA tiny RED tests**

Require transient and resident KDA parity against portable C++, full/incremental state parity, zero second-call resident weight H2D, exact state traffic, one final output/state publication, and launch-free failure for malformed shape, `[96]` A-log, non-finite F32 tensor, alias conflict, insufficient capacity, and wrong state layout.

- [ ] **Step 2: Configure/build and witness RED**

```bash
cmake -S . -B build-cuda -DK3X_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=120-real
cmake --build build-cuda --target test_cuda_official_kda test_cuda_official_layer -j2
```

Expected: missing targets and backend API.

- [ ] **Step 3: Implement exact CUDA execution**

Use byte-native BF16/F32 residency, cuBLASLt or existing BF16 matvec paths for large projections, dedicated depthwise convolution and recurrent kernels, FP32 state, and existing official MoE execution. Perform all acquisition/identity checks before scratch allocation or launch. CUDA errors are fatal and never converted into CPU fallback.

- [ ] **Step 4: Run CUDA GREEN and sanitizers**

```bash
cmake --build build-cuda --target test_cuda_official_kda test_cuda_official_layer \
  k3x_cuda_official_layer_bench -j2
ctest --test-dir build-cuda -R 'cuda_official_kda|cuda_official_layer|cuda_official_moe' --output-on-failure
compute-sanitizer --tool memcheck --error-exitcode=99 build-cuda/test_cuda_official_kda
compute-sanitizer --tool memcheck --error-exitcode=99 build-cuda/test_cuda_official_layer
```

- [ ] **Step 5: Self-review and commit**

```bash
git diff --check
git add CMakeLists.txt runtime/include/k3x/backend.hpp runtime/cuda/backend_cuda.cu \
  runtime/src/cuda_official_layer_bench.cpp tests/cuda/test_cuda_official_kda.cu \
  tests/cuda/test_cuda_official_layer.cu tests/python/test_cuda_official_layer.py
git commit -m "feat: execute official KDA layer on CUDA"
```

### Task 7: Strict B-0030 runner and evidence verifier

**Files:**
- Create: `tools/ablate_official_layer.py`
- Create: `tests/python/test_official_layer_ablation.py`
- Modify: `runtime/src/cuda_official_layer_bench.cpp`
- Modify: `tests/python/test_cuda_official_layer.py`

**Interfaces:**
- Fixed rows: `a-transient`, `ab-incremental-resident`, and `ab-full-resident` in that order.
- Produces: canonical raw JSON/CSV, summary JSON/CSV, aggregate digest, artifact/manifest/runner digests, and strict `--verify-existing` mode.
- Strict measurement identity: exactly 3 warmups and 20 measured iterations.

- [ ] **Step 1: Write runner/verifier RED tests**

Use a controlled subprocess fixture. Require exact row order, schema, finite timing, full/incremental route/state/output parity, traffic formulas, resident warm-zero weight H2D, digest binding, LF-only CSV, and atomic no-output failure. Reject any token/TPS/TTFT/quality/physical-traffic field and mutations of raw rows, aggregate, artifact, manifest, runner, or CSV.

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=converter:reference /home/jolib/.venvs/k3x-m1/bin/python -m pytest \
  tests/python/test_official_layer_ablation.py \
  tests/python/test_cuda_official_layer.py -q
```

Expected: missing `tools.ablate_official_layer`.

- [ ] **Step 3: Implement fixed non-ranking evidence tooling**

Write all outputs to siblings with `.partial` suffix, fsync, verify every row and cross-row invariant, then rename only after the complete matrix passes. Do not rerun, rank, or select timing samples.

- [ ] **Step 4: Run GREEN and compile validation**

```bash
PYTHONPATH=converter:reference /home/jolib/.venvs/k3x-m1/bin/python -m pytest \
  tests/python/test_official_layer_ablation.py \
  tests/python/test_cuda_official_layer.py -q
PYTHONPATH=converter:reference /home/jolib/.venvs/k3x-m1/bin/python -m py_compile \
  tools/ablate_official_layer.py
```

- [ ] **Step 5: Self-review and commit**

```bash
git diff --check
git add tools/ablate_official_layer.py tests/python/test_official_layer_ablation.py \
  runtime/src/cuda_official_layer_bench.cpp tests/python/test_cuda_official_layer.py
git commit -m "test: add strict B-0030 evidence pipeline"
```

### Task 8: Bounded materialization, formal B-0030, full verification, and publication

**Files:**
- Generate ignored: `artifacts/m29-official-layer/live/**`
- Generate: `results/b0030-official-layer-wsl/**`
- Modify: `README.md`
- Modify: `ARCHITECTURE.md`
- Modify: `PERFORMANCE_MODEL.md`
- Modify: `DECISIONS.md`
- Modify: `BENCHMARKS.md`
- Modify: `PROJECT_STATE.md`
- Modify: `checklist.md`
- Modify: `context-notes.md`

**Interfaces:**
- Consumes: Tasks 1 through 7 at one reviewed commit.
- Produces: one ignored bounded official layer artifact, one fixed B-0030 matrix, synchronized TITAN Ledger, and public GitHub integration.

- [ ] **Step 1: Run zero-payload dry-run and inspect the exact plan**

```bash
K3X_TEST_OFFICIAL_DISCOVERY=1 PYTHONPATH=converter:reference \
  /home/jolib/.venvs/k3x-m1/bin/python tools/discover_official_kimi_k3.py \
  --scope kda-layer --dry-run
```

Require exact pinned identities, 17 KDA tensors, 887,843,840 KDA bytes, and 1,829,256,704 maximum unaligned bytes before payload authorization already granted by the user is exercised.

- [ ] **Step 2: Materialize the bounded layer fixture once**

```bash
K3X_TEST_OFFICIAL_DISCOVERY=1 PYTHONPATH=converter:reference \
  /home/jolib/.venvs/k3x-m1/bin/python tools/discover_official_kimi_k3.py \
  --scope kda-layer --materialize \
  --output-dir artifacts/m29-official-layer/live
```

Require no complete-shard request, exact natural route/state publication, Reader validity, source/root binding, and `k3x_run` exit 4 with `NON_EXECUTABLE_ARTIFACT`.

- [ ] **Step 3: Run actual CPU/CUDA smoke and Compute Sanitizer**

Run A transient, AB incremental resident, and AB full resident once. Require CPU/CUDA full output and final-state parity, full/incremental equality, zero warm resident weight H2D, and Compute Sanitizer `ERROR SUMMARY: 0 errors` on the AB incremental resident case.

- [ ] **Step 4: Run B-0030 exactly once**

```bash
PYTHONPATH=converter:reference /home/jolib/.venvs/k3x-m1/bin/python \
  tools/ablate_official_layer.py \
  --artifact artifacts/m29-official-layer/live/model.k3x \
  --manifest artifacts/m29-official-layer/live/route-manifest.json \
  --output-dir results/b0030-official-layer-wsl \
  --warmups 3 --iterations 20
```

If the run fails, fix the proven defect and restart the fixed matrix once; never retain partial evidence or rerun to select favorable timing.

- [ ] **Step 5: Run the complete verification matrix**

Run CPU CTest/Python, liburing/direct CTest/Python, ASan/UBSan CTest, CUDA CTest/Python with the actual artifact, strict committed-evidence verification, and actual AB incremental resident Compute Sanitizer. Record fresh counts and exact environment.

- [ ] **Step 6: Update all durable documentation and evidence boundaries**

Record actual routes, union size, payload/artifact bytes, state bytes, timings, traffic, VRAM/RAM, errors, sanitizer result, digests, and limitations. Keep theoretical bounds visibly separate from measured values. Update `PROJECT_STATE.md` last.

- [ ] **Step 7: Commit evidence and documentation separately**

```bash
git add results/b0030-official-layer-wsl tools/ablate_official_layer.py
git commit -m "bench: record B-0030 official layer results"
git add README.md ARCHITECTURE.md PERFORMANCE_MODEL.md DECISIONS.md BENCHMARKS.md \
  PROJECT_STATE.md checklist.md context-notes.md
git commit -m "docs: complete Milestone 29 ledger"
```

- [ ] **Step 8: Publish and verify public integration**

Push `codex/milestone-twenty-nine-official-layer`, open a public pull request, wait for correctness and CodeQL, rebase-merge only after success, then verify post-merge `main` correctness and CodeQL. Record the public head and run IDs in the ledger without creating a self-referential publication loop.
