# K3X Official Two-Layer Device Closure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Execute the released Kimi K3 decoder layers 1 and 2 as one exact bounded trace while retaining the inter-layer hidden activation, shared Attention Residual block source, and independent per-layer KDA recurrence on the RTX 5080.

**Architecture:** Add a dependency-ordered two-layer manufacturer and portable oracle without changing K3X v1. Establish an interleaved CPU/CUDA host-round-trip baseline. Extend the experimental CUDA backend from one KDA state slot to exactly two layer-keyed slots, then add a device front/tail API with an opaque ping-pong hidden token. Keep canonical natural Top-16 and dynamic expert resolution on the host. Measure the complete boundary only after exact actual-artifact and sanitizer gates pass.

**Tech Stack:** Python 3.12, PyTorch reference operators, C++20, CUDA 13.3 `sm_120`, CMake/CTest, pytest, Compute Sanitizer, K3X v1, canonical JSON/LF CSV, SHA-256, crash-safe atomic publication, and the pinned Kimi K3 revision.

## Global constraints

- Keep host routing, host KDA state, per-call validation, direct execution, and every production path source-compatible by default.
- Use real official layers 1 and 2. Layer-1 replay may appear in tiny unit tests but cannot satisfy an actual-artifact or benchmark gate.
- Preserve one canonical host implementation of sigmoid, correction, natural Top-16 order, expert-ID tie breaking, selected mass, and contribution normalization.
- Expose no device pointer. Every state, prepared activation, and inter-layer hidden handoff uses backend-owner/generation identity and single-use validation.
- Bound the KDA registry to two active layer slots and the hidden lifetime to two ping-pong activation slots. Do not add arbitrary-session or arbitrary-layer state management.
- Keep K3X v1 and the existing official-fixture optional bits. Do not weaken `NON_EXECUTABLE_ARTIFACT` production rejection.
- Manufacture only pinned ranges required by the dependent A/B trace. Rehash reusable objects and publish dry-run byte bounds before downloading new tensor payload.
- Do not download the full checkpoint or a complete source shard payload. Do not provision Cloud Run or any paid resource.
- Witness RED before implementation, run focused GREEN plus historical regressions, self-review the diff, and create one semantic commit per task.
- Every new Python, C++, or CUDA source file starts with a one-line Korean role comment.
- B-0034 must not emit token throughput, TTFT, quality, physical PCIe/NVMe traffic, utilization, bandwidth, or native-Linux claims.

---

### Task 1: Bounded layer-2 planning authority

**Files:**
- Modify: `converter/k3x_converter/official_moe.py`
- Modify: `converter/k3x_converter/official_layer.py`
- Modify: `tests/python/test_official_moe.py`
- Modify: `tests/python/test_official_layer.py`

**Contract:**
- `plan_official_moe_slice` and `plan_official_kda_layer` accept only released bounded layer IDs 1 or 2.
- Every canonical name, shard binding, range, shape, dtype, payload byte count, source blob, and KDA-layer identity remains strict.
- Existing single-layer materialization remains fixed to layer 1 and produces byte-compatible historical manifests.

- [x] **Step 1: Write layer-2 planner RED tests.**

Clone the pure metadata fixture with layer-2 canonical names and a distinct shard/header identity. Require exact tensor order and byte counts for layer 2, and rejection of layer 0, layer 3, mixed-layer metadata, wrong shard bindings, and non-KDA configuration. Require the historical layer-1 plan to remain equal to its current value.

- [x] **Step 2: Witness focused RED.**

Run:

```powershell
rtk wsl -d Ubuntu-24.04 -- bash -lc 'cd /mnt/c/Users/jolib/Documents/project-k3x/.worktrees/milestone-twenty-four-cuda-graph-cache && source /home/jolib/.venvs/k3x-m1/bin/activate && pytest -q tests/python/test_official_moe.py tests/python/test_official_layer.py'
```

Expected result: only the new layer-2 cases fail with `INVALID_OFFICIAL_MOE_CONFIG` or `INVALID_OFFICIAL_LAYER`.

- [x] **Step 3: Implement the minimum bounded generalization.**

Replace the literal layer-1 planner guard with a shared exact `{1, 2}` identity check. Do not alter fixed inputs, single-layer route-manifest format, artifact name, or `materialize_official_kda_layer`'s layer-1-only transaction.

- [x] **Step 4: Run GREEN and source-integrity regressions.**

Run the focused suite plus `tests/python/test_official_source.py`, `test_source_manifest_integrity.py`, and `test_converter_resume.py`. Run Python compile validation and `git diff --check`.

- [x] **Step 5: Self-review and commit.**

Commit message: `feat: plan bounded official layer two`.

### Task 2: Exact dependency-ordered two-layer manufacturer

**Files:**
- Create: `converter/k3x_converter/official_two_layer.py`
- Modify: `converter/k3x_converter/cli.py`
- Modify: `converter/k3x_converter/official_moe.py`
- Modify: `converter/k3x_converter/__init__.py`
- Create: `tests/python/test_official_two_layer.py`
- Modify: `tests/python/test_k3x_format.py`

**Contract:**
- Produce `OfficialTwoLayerPlan`, `OfficialTwoLayerTrace`, and `OfficialTwoLayerMaterializationReport` for exactly `(1, 2)`.
- Interleave A/layer-1, A/layer-2, B/layer-1, B/layer-2 with independent zero-seeded KDA states.
- Use exact source-byte BF16 boundaries and native MXFP4 decode to compute layer-1 final hidden before deriving layer-2 routes.
- Assemble one execution-ordered `k3-official-moe-slice-v1` source and K3X v1 artifact with both layer directories and one two-layer metadata envelope.

- [ ] **Step 1: Write pure trace and transaction RED tests.**

Use tiny monkeypatched KDA/MoE tensors to require exact interleaving, unchanged block-source propagation, independent state hashes, layer-1-output-to-layer-2-input digests, per-layer routes/expert unions, and a two-layer oracle payload. Require plan/header drift, duplicate canonical names, missing expert bytes, route mismatch, non-atomic publication, and resume-object corruption to fail before finalization.

- [x] **Step 2: Witness import-level RED.**

Run `pytest -q tests/python/test_official_two_layer.py`. Expected result: import fails because `official_two_layer` does not exist.

- [x] **Step 3: Implement the pure two-layer trace.**

Reuse the official KDA reference, Attention Residual/RMSNorm rules, canonical routing helper, and `k3x_ref.mxfp4` primitives. Decode and execute one selected expert at a time so peak temporary memory stays bounded. Persist per-step input, output, route, state-consumption, state-publication, and contribution digests.

- [ ] **Step 4: Implement dependency-ordered materialization and CLI.**

Add explicit `official-two-layer --dry-run|--materialize` handling. Fetch and verify both layers' trunks first, publish a partial trace manifest, fetch layer-1 selected experts, derive layer-2 inputs/routes, fetch only layer-2 selected experts, then assemble and convert. Accept distinct source shards per layer. Reuse content-addressed objects only after digest verification. Emit the exact requested/downloaded/reused byte accounting and maximum response size.

- [ ] **Step 5: Round-trip the tiny composite artifact.**

Require tensor physical order to match the manifest, layer directory IDs to equal `(1, 2)`, all tensor/expert records to resolve, source/tensor/root hashes to verify, interrupted conversion to resume, and production `k3x_run` to retain the non-executable guard.

- [ ] **Step 6: Run GREEN and historical converter regressions.**

Run the new suite, official layer/MoE/source suites, K3X format, source integrity, resume, CLI, Python compile validation, and `git diff --check`.

- [ ] **Step 7: Self-review and commit.**

Commit message: `feat: manufacture bounded official two-layer traces`.

### Task 3: Portable interleaved two-layer runtime oracle

**Files:**
- Create: `runtime/include/k3x/official_two_layer.hpp`
- Create: `runtime/src/official_two_layer.cpp`
- Modify: `CMakeLists.txt`
- Create: `tests/cpp/test_official_two_layer.cpp`
- Modify: `tests/python/test_cpp_parity.py`

**Contract:**
- Compose the existing exact official layer primitives in model order for A/B across layers 1 and 2.
- Own two independent `OfficialKdaState` values and publish both final states.
- Return each layer's natural route, contribution, hidden-output, and state digests for parity attribution.

- [ ] **Step 1: Write the portable RED.**

Build a tiny two-layer fixture with distinct weights and assert that A/layer-2 consumes A/layer-1 output, B/layer-1 consumes only layer-1 state, and B/layer-2 consumes only layer-2 state. Add swapped-layer, shared-state, missing-expert, wrong block, and route-drift failures.

- [ ] **Step 2: Witness compile RED.**

Build `test_official_two_layer`. Expected result: the new header and target are missing.

- [ ] **Step 3: Implement the minimum portable orchestrator.**

Call one official layer step at a time, retain two host states, and avoid duplicating KDA, MoE, or routing math. Keep the old one-layer interfaces unchanged.

- [ ] **Step 4: Run portable GREEN and parity.**

Run `ctest -R "official_(moe|kda|layer|two_layer)"`, focused C++ parity, strict warning compilation, and ASan/UBSan for the new target.

- [ ] **Step 5: Self-review and commit.**

Commit message: `feat: add exact official two-layer oracle`.

### Task 4: Capacity-two layer-keyed CUDA KDA state registry

**Files:**
- Modify: `runtime/include/k3x/backend.hpp`
- Modify: `runtime/cuda/backend_cuda.cu`
- Modify: `runtime/src/backend_cuda_stub.cpp`
- Modify: `tests/cuda/test_cuda_official_kda.cu`
- Modify: `tests/cuda/test_cuda_official_layer.cu`

**Contract:**
- Retain independent device KDA states for exactly layers 1 and 2.
- Preserve existing single-layer seed/continue/publish behavior and telemetry.
- Consume a state token before mutation and issue an independently generated successor for the same layer.

- [ ] **Step 1: Write two-slot RED ownership tests.**

Require simultaneous layer-1/layer-2 seeds, alternating continuations, independent publications, exact host parity, and unchanged state H2D/D2H accounting. Reject a third layer, stale generation, cross-backend token, wrong-layer continuation, overwrite, double publish, and partial-failure reuse.

- [ ] **Step 2: Witness CUDA RED.**

Build and run `test_cuda_official_kda` and `test_cuda_official_layer`. Expected result: the second seed invalidates or overwrites the first state under the current global slot.

- [ ] **Step 3: Implement the fixed registry.**

Replace the single state scratch/metadata with two fixed slots keyed to layers 1 and 2. Keep owner identity backend-wide and make generations unique. Validate every token/config/layer before state upload or launch. Do not allocate a map, eviction policy, or session registry.

- [ ] **Step 4: Run GREEN, historical state tests, and sanitizer.**

Run CUDA KDA/layer tests, non-CUDA unavailable tests, M31 device-state Python regressions, and Compute Sanitizer with `--launch-timeout 0`.

- [ ] **Step 5: Self-review and commit.**

Commit message: `feat: retain two official KDA layer states`.

### Task 5: Opaque inter-layer hidden token and CUDA front/tail bridge

**Files:**
- Modify: `runtime/include/k3x/backend.hpp`
- Modify: `runtime/cuda/official_moe_route.cuh`
- Modify: `runtime/cuda/official_moe_route.cu`
- Modify: `runtime/cuda/backend_cuda.cu`
- Modify: `runtime/src/backend_cuda_stub.cpp`
- Modify: `tests/cuda/test_cuda_official_layer.cu`
- Modify: `tests/cuda/test_cuda_official_moe.cu`

**Contract:**
- Add `OfficialLayerHiddenToken` with backend owner, generation, producing layer, width, and bounded slot identity.
- A device front consumes host hidden/block or a valid preceding-layer token and returns raw logits plus an opaque prepared token.
- A device tail consumes the prepared token and either retains final hidden/block for the next layer or publishes final hidden to the host.
- No inter-layer hidden D2H/H2D occurs on the retained path.

- [ ] **Step 1: Write front/tail RED parity and lifetime tests.**

Require tiny exact parity at self residual, normalized KDA input, KDA output/state, prefix, prepared hidden, raw logits, selected route, FFN output, and final hidden. Require one layer-1 retained token to feed layer 2 without hidden transfer counters. Reject zero, stale, cross-backend, wrong-producer, wrong-consumer, wrong-width, wrong-slot, double-consumed, and unexpectedly live tokens.

- [ ] **Step 2: Witness CUDA compile RED.**

Build `test_cuda_official_layer`. Expected result: hidden token and front/tail APIs are undeclared.

- [ ] **Step 3: Implement bounded activation ownership.**

Add two grow-only ping-pong slots for hidden plus block source. Upload host input only at the first front. Consume and invalidate a prior hidden generation before launching the next front. Ensure prepared-route scratch and final-hidden scratch cannot alias a live slot.

- [ ] **Step 4: Implement the complete device front.**

Reuse existing exact residual, RMSNorm, KDA, prefix, route-preparation, and resident-weight admission logic without a host KDA-output copy. Return only 896 raw logits and opaque ownership tokens. Keep one canonical host route decision per layer.

- [ ] **Step 5: Implement retain-or-publish tail cleanup.**

Reuse the exact prepared MXFP4/shared FFN. On retain, write the final hidden to the next ping-pong slot and preserve the block source. On publish, copy only the final requested host vector. Discard all outstanding prepared/hidden/state generations on downstream failure according to the wrapper's transaction boundary.

- [ ] **Step 6: Run focused GREEN and sanitizer.**

Run CUDA official MoE/KDA/layer tests, non-CUDA stubs, strict warning compilation, and Compute Sanitizer on both official MoE and layer targets.

- [ ] **Step 7: Self-review and commit.**

Commit message: `feat: retain official inter-layer activations`.

### Task 6: Exact two-layer CUDA wrapper and dedicated harness

**Files:**
- Modify: `runtime/include/k3x/official_two_layer.hpp`
- Modify: `runtime/src/official_two_layer.cpp`
- Create: `runtime/src/cuda_official_two_layer_bench.cpp`
- Modify: `CMakeLists.txt`
- Create: `tests/cuda/test_cuda_official_two_layer.cu`
- Create: `tests/python/test_cuda_official_two_layer.py`

**Contract:**
- Add explicit host-round-trip and device-closure modes over the same interleaved trace.
- Both modes use device KDA states, resident admission-validated weights, canonical host natural Top-16, and exact selected experts.
- The dedicated harness validates the two-layer artifact, trace manifest, oracle, tensor order, and closed telemetry schema before execution.

- [ ] **Step 1: Write wrapper and CLI RED tests.**

Require host/device exact route IDs, contribution tolerance, per-layer state/output digests, final output parity, zero warm weight H2D, and explicit activation/state/route counters. Require the device row to report zero inter-layer hidden transfer bytes. Add invalid artifact, swapped layer, missing tensor/expert, wrong trace, invalid mode, insufficient residency, and production-guard cases.

- [ ] **Step 2: Witness target and parsing RED.**

Build `test_cuda_official_two_layer` and run `pytest -q tests/python/test_cuda_official_two_layer.py`. Expected result: target/module/executable paths do not exist.

- [ ] **Step 3: Implement the interleaved wrapper and dedicated harness.**

Keep historical one-layer CLI schemas untouched. Execute A1→A2→B1→B2, resolve experts only after each canonical route, publish both final KDA states, and make failure cleanup explicit. Emit no token/TPS fields.

- [ ] **Step 4: Run tiny GREEN and production guard.**

Run the new CUDA/pytest tests, historical official-layer suites, one tiny host/device parity pair, strict warning compilation, and `k3x_run` non-executable rejection.

- [ ] **Step 5: Self-review and commit.**

Commit message: `feat: execute bounded official two-layer closure`.

### Task 7: Strict B-0034 evidence transaction

**Files:**
- Create: `tools/ablate_official_two_layer_closure.py`
- Create: `tests/python/test_official_two_layer_ablation.py`

**Contract:**
- Produce exactly two rows in host-round-trip then device-closure order.
- Fix three warmups, twenty measured A/B traces, resident admission, device KDA state, canonical host routing, and one immutable two-layer artifact/manifest/oracle identity.

- [ ] **Step 1: Write controlled verifier RED.**

Require exact row order/configuration, artifact/manifest/oracle/runner identity, route/contribution/output/final-state parity, zero warm weight H2D, transfer formulas, state/hidden/prepared lifetime counters, forbidden metric absence, LF CSV, fsync, atomic directory replacement, and rejection of every one-field mutation.

- [ ] **Step 2: Witness import RED.**

Run `pytest -q tests/python/test_official_two_layer_ablation.py`. Expected result: import fails because the tool does not exist.

- [ ] **Step 3: Implement the minimum publisher/verifier.**

Reuse B-0033 canonical serialization and staging authorities without changing historical schemas. Add no ranking or timing interpretation to the verifier.

- [ ] **Step 4: Run GREEN and B-0030 through B-0033 regressions.**

Run the new evidence suite, all four historical official-layer evidence suites, Python compile validation, and `git diff --check`.

- [ ] **Step 5: Self-review and commit.**

Commit message: `test: add strict B-0034 two-layer evidence`.

### Task 8: Bounded official materialization, B-0034, Ledger, and publication

**Files:**
- Generate: `artifacts/m33-official-two-layer/**`
- Generate: `results/b0034-official-two-layer-closure-wsl/**`
- Modify: `README.md`
- Modify: `ARCHITECTURE.md`
- Modify: `PERFORMANCE_MODEL.md`
- Modify: `DECISIONS.md`
- Modify: `BENCHMARKS.md`
- Modify: `PROJECT_STATE.md`
- Modify: `checklist.md`
- Modify: `context-notes.md`

- [ ] **Step 1: Run metadata-only live dry-run.**

Verify the pinned snapshot, both shard headers, layer IDs, every accepted range, maximum requested bytes, object-reuse candidates, and disk/VRAM capacity. Commit no new payload and stop if the bound exceeds the declared bounded fixture contract.

- [ ] **Step 2: Materialize exactly one bounded two-layer fixture.**

Reuse rehashed layer-1 objects, download only missing layer-2 trunk and selected expert ranges, seal the trace/oracle/source/K3X identities, and verify resume behavior. Do not fetch a complete shard or checkpoint.

- [ ] **Step 3: Pass pre-measurement gates.**

Run portable source-byte parity, actual-artifact host/device correctness, zero inter-layer hidden-transfer assertions, Compute Sanitizer with `--launch-timeout 0`, production guard, strict verifier dry-run, and final Critical/Important self-review. Fix findings once and rerun affected gates before measurement.

- [ ] **Step 4: Run B-0034 exactly once.**

Execute the fixed two-row transaction with exactly three warmups and twenty measured traces. Do not rerun or select samples after successful publication.

- [ ] **Step 5: Seal and independently verify evidence.**

Rehash artifact, manifests, oracle, runner, every raw JSON/CSV, summary JSON/CSV, aggregate digest, LF endings, row order, formulas, and forbidden-field absence. Commit evidence separately as `bench: record B-0034 two-layer closure results`.

- [ ] **Step 6: Run the complete verification matrix.**

Run CPU CTest/Python, capability-aware liburing/direct, ASan/UBSan, CUDA CTest/live Python, B-0034 through B-0030 evidence regressions, actual-artifact sanitizer, and production guard. Record exact pass/skip counts.

- [ ] **Step 7: Synchronize the TITAN Ledger last.**

Record only measured values and limitations in README, architecture, performance model, decisions, benchmarks, checklist, and context notes. Update `PROJECT_STATE.md` last. Commit as `docs: record Milestone 33 results`.

- [ ] **Step 8: Publish and verify.**

Push a public ready PR, require branch/PR correctness and C++/Python CodeQL, rebase-merge, update README/public state if the merge SHA changes the ledger, and verify post-merge `main` correctness and CodeQL. Do not create paid cloud resources.
