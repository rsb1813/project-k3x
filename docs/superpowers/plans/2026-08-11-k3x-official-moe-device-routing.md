# K3X Official MoE Device Routing Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Move exact official MLP residual preparation and router matvec to CUDA while preserving canonical natural Top-16 selection and handing backend-owned prepared activations to the resident MXFP4 FFN.

**Architecture:** Add a canonical raw-logit routing helper shared by host and device paths. CUDA stage one retains prefix and normalized hidden vectors in one backend-owned slot and returns raw logits plus an opaque token; host code selects experts; CUDA stage two consumes the token in the existing exact FFN. Keep the current host route path and all production defaults unchanged.

**Tech Stack:** C++20, CUDA 13.3 `sm_120`, existing resident-weight/admission infrastructure, Python 3.12, pytest, CMake/CTest, Compute Sanitizer, canonical JSON/CSV, SHA-256, and atomic publication.

## Global Constraints

- Keep host routing, host KDA state, per-call validation, and direct execution as source-compatible defaults.
- Preserve sigmoid, correction, natural Top-16 ordering, expert-ID tie breaking, and contribution normalization in one canonical host helper.
- Expose no CUDA pointer; accept only an opaque backend-owner/generation token.
- Support one prepared-activation slot per backend and reject stale, consumed, cross-backend, wrong-layer, wrong-width, and mismatched-weight tokens before upload or launch.
- Invalidate an outstanding token on a new preparation, a host/default FFN call, successful consumption, explicit discard, or downstream wrapper failure.
- Reuse exact resident/admission identity and the existing MXFP4 FFN implementation; do not add reduced precision, adaptive Top-K, proxy, pruning, or eviction.
- Keep the bounded official artifact non-executable through `k3x_run`.
- Do not download another official range, a complete shard/checkpoint, or provision paid cloud resources.
- Witness RED before implementation, run focused GREEN plus historical regressions, and create one semantic commit per task.
- Every new Python, C++, or CUDA source file starts with a one-line Korean role comment.
- B-0033 must not emit token throughput, TTFT, quality, physical traffic, utilization, bandwidth, or native-Linux claims.

---

### Task 1: Canonical raw-logit routing authority

**Files:**
- Modify: `runtime/include/k3x/official_moe.hpp`
- Modify: `runtime/src/official_moe.cpp`
- Modify: `tests/cpp/test_official_moe.cpp`

**Interfaces:**
- Produces: `Result<OfficialRoute> route_official_moe_logits(std::span<const float> logits, std::span<const float> correction, std::size_t top_k)`.
- Preserves: `route_official_moe(hidden, router, correction, top_k)` computes BF16 router logits and delegates policy to the new helper.

- [x] **Step 1: Write the failing canonical-helper tests**

Add a fixed logit/correction case that asserts sigmoid scores, expert-ID tie breaking, selected IDs, normalized contributions, and invalid length/non-finite/top-k rejection. Add a parity assertion that `route_official_moe` and `route_official_moe_logits` return identical route fields for the existing tiny fixture.

- [x] **Step 2: Run the CPU RED**

Run:

```powershell
rtk wsl -d Ubuntu-24.04 -- bash -lc 'cd /mnt/c/Users/jolib/Documents/project-k3x/.worktrees/milestone-twenty-four-cuda-graph-cache && cmake --build build -j2 --target test_official_moe && ctest --test-dir build -R "^official_moe$" --output-on-failure'
```

Expected result: compilation fails because `route_official_moe_logits` is undeclared.

- [x] **Step 3: Implement the minimum shared policy helper**

Move only sigmoid, correction, partial sort with ID tie break, selected-score sum, and contribution normalization into the new function. Leave BF16 matvec validation and accumulation in `route_official_moe`.

- [x] **Step 4: Run GREEN and nearby portable regressions**

Run the focused target above, then `ctest --test-dir build -R "official_(moe|layer)" --output-on-failure`.

- [x] **Step 5: Self-review and commit**

Commit message: `refactor: share canonical official MoE routing`.

### Task 2: CUDA route preparation and opaque token lifetime

**Files:**
- Modify: `runtime/include/k3x/backend.hpp`
- Modify: `runtime/src/backend_cuda_stub.cpp`
- Create: `runtime/cuda/official_moe_route.cuh`
- Create: `runtime/cuda/official_moe_route.cu`
- Modify: `runtime/cuda/backend_cuda.cu`
- Modify: `CMakeLists.txt`
- Modify: `tests/cuda/test_cuda_official_moe.cu`

**Interfaces:**
- Produces: `OfficialMoeRoutePrepareView`, `OfficialMoePreparedToken`, and `OfficialMoeRoutePrepareResult`.
- Produces backend operations: `prepare_official_moe_route`, `official_mxfp4_moe_ffn_prepared`, and `discard_official_moe_prepared`.
- Consumes: Task 1 canonical helper only after raw logits return to the host caller.

- [x] **Step 1: Write CUDA RED lifetime and parity tests**

Require exact tiny CPU-oracle prepared activation, bounded parity for every raw logit, a nonzero opaque token, empty prepared activation in the public result, one successful prepared FFN consumption, and exact final output. Require rejection of zero, stale, consumed, cross-backend, wrong-layer, wrong-width, mismatched-view, and duplicate-consumption tokens before upload or launch. Require a host/default FFN call and a second preparation to invalidate the outstanding generation.

- [x] **Step 2: Build and witness CUDA RED**

Run:

```powershell
rtk wsl -d Ubuntu-24.04 -- bash -lc 'cd /mnt/c/Users/jolib/Documents/project-k3x/.worktrees/milestone-twenty-four-cuda-graph-cache && cmake --build build-cuda -j2 --target test_cuda_official_moe'
```

Expected result: compilation fails because the prepare/token/consume API does not exist.

- [x] **Step 3: Add the minimal public surface and CUDA kernels**

Implement deterministic BF16 prefix/block rounding, two residual projection scores, stable two-way softmax, BF16 mixed residual, BF16 post RMSNorm, and 896 deterministic router dot products. Store prefix, prepared hidden, raw logits, immutable identity, layer, width, owner, and generation in a dedicated grow-only backend slot. Return only logits and token.

- [x] **Step 4: Implement fail-closed prepared FFN consumption**

Validate the complete token and immutable-view identity before mutation, consume the generation before upload/launch, reuse the existing resident exact FFN kernels with slot-owned prefix/hidden pointers, and make explicit discard idempotently succeed only for the current valid generation. Add exact preparation/consume/discard/invalidation/slot-byte telemetry.

- [x] **Step 5: Run focused GREEN, stubs, and sanitizer**

Run CUDA `cuda_official_moe`, `cuda_moe_layer`, and `cuda_official_kda`; run the non-CUDA `backend_unavailable` build; run Compute Sanitizer on `test_cuda_official_moe` with `--launch-timeout 0`.

- [x] **Step 6: Self-review and commit**

Commit message: `feat: retain official MoE routing activations on device`.

### Task 3: Official-layer wrapper and explicit harness mode

**Files:**
- Modify: `runtime/include/k3x/official_layer.hpp`
- Modify: `runtime/src/official_layer.cpp`
- Modify: `runtime/src/cuda_official_layer_bench.cpp`
- Modify: `tests/cuda/test_cuda_official_layer.cu`
- Modify: `tests/python/test_cuda_official_layer.py`

**Interfaces:**
- Produces: explicit `OfficialMoeRoutePreparationMode::{host,device}` control at the wrapper boundary.
- Preserves: omitted control follows the exact historical host path and emits no new JSON keys.
- Consumes: Task 1 canonical raw-logit helper and Task 2 prepare/consume/discard operations.

- [x] **Step 1: Write wrapper and CLI RED tests**

Require exact host/device route IDs, contribution tolerance, final output/state parity, one prepare and consume per step, zero invalidations/discards on success, and cleanup after route, missing-expert, or FFN failure. Add `--route-preparation host|device` parsing, explicit-field schema, implicit historical-schema absence, and invalid-combination exits. Restrict device route preparation to `ab-incremental + state-transfer=device + resident + admission`.

- [x] **Step 2: Witness CUDA and Python RED**

Build `test_cuda_official_layer` and run the focused `tests/python/test_cuda_official_layer.py` cases. Expected result: compilation and parsing assertions fail because the control does not exist.

- [x] **Step 3: Implement the minimum wrapper and harness path**

In device mode call prepare, run canonical host selection over returned logits, resolve exact expert views, consume the token in prepared FFN, and discard on every failure after preparation. Keep the existing host path byte-for-byte in control flow when the option is absent. Emit new telemetry only when `--route-preparation` is explicit.

- [x] **Step 4: Run tiny GREEN and one actual-artifact smoke pair**

Require matched route/output/state identities, zero warm weight H2D, nonzero prepared telemetry only in the device row, and unchanged production `NON_EXECUTABLE_ARTIFACT` behavior.

- [x] **Step 5: Self-review and commit**

Commit message: `feat: expose official MoE device routing`.

### Task 4: Strict B-0033 evidence transaction

**Files:**
- Create: `tools/ablate_official_moe_device_routing.py`
- Create: `tests/python/test_official_moe_device_routing_ablation.py`

**Interfaces:**
- Produces: fixed two-row B-0033 raw JSON/CSV, summary JSON/CSV, manifest, hashes, and aggregate digest.
- Consumes: explicit host/device route-preparation harness schemas from Task 3 and immutable B-0032 artifact/manifest identities.

- [x] **Step 1: Write controlled RED verifier tests**

Fix row order to host then device routing, three warmups, twenty measured sequences, resident admission, and device KDA state. Require exact artifact/manifest/runner identity, route/output/final-state parity, zero warm weight H2D, prepared counter formulas, forbidden metric absence, LF CSV, fsync, atomic publication, and rejection of every one-field mutation.

- [x] **Step 2: Run RED**

Run `pytest -q tests/python/test_official_moe_device_routing_ablation.py`. Expected result: import fails because the tool does not exist.

- [x] **Step 3: Implement the minimum non-ranking publisher and verifier**

Reuse B-0032 canonical JSON, checksum, staging-blob, and atomic-directory authorities without modifying historical tools or schemas. Add no timing interpretation to the verifier.

- [x] **Step 4: Run GREEN and historical evidence regressions**

Run the B-0033 test plus B-0032/B-0031/B-0030 evidence tests and Python compile validation.

- [x] **Step 5: Self-review and commit**

Commit message: `test: add strict B-0033 device-routing evidence`.

### Task 5: Formal B-0033, verification, Ledger, and publication

**Files:**
- Generate: `results/b0033-official-moe-device-routing-wsl/**`
- Modify: `README.md`
- Modify: `ARCHITECTURE.md`
- Modify: `PERFORMANCE_MODEL.md`
- Modify: `DECISIONS.md`
- Modify: `BENCHMARKS.md`
- Modify: `PROJECT_STATE.md`
- Modify: `checklist.md`
- Modify: `context-notes.md`

**Interfaces:**
- Consumes: all Task 1 through Task 4 code, telemetry, tests, and strict evidence gates.
- Produces: one sealed B-0033 measurement and synchronized public M32 state.

- [ ] **Step 1: Pass pre-measurement gates**

Run actual-artifact host/device parity, prepared-path Compute Sanitizer with `--launch-timeout 0`, production guard, strict verifier dry-run, and a final Critical/Important self-review. Correct findings once and rerun affected gates before measurement.

- [ ] **Step 2: Run B-0033 exactly once**

Execute the fixed two-row transaction with exactly three warmups and twenty measured sequences. Do not rerun or select samples after successful publication.

- [ ] **Step 3: Seal and independently verify evidence**

Rehash artifact, manifest, runner, every raw JSON/CSV, summary JSON/CSV, aggregate digest, LF endings, row order, formulas, and forbidden-field absence. Commit evidence separately as `bench: record B-0033 device-routing results`.

- [ ] **Step 4: Run the complete verification matrix**

Run CPU CTest/Python, capability-aware liburing/direct, ASan/UBSan, CUDA CTest/live Python, B-0033 through B-0030 evidence regressions, actual-artifact sanitizer, and production guard. Record exact pass/skip counts without converting theoretical values into measurements.

- [ ] **Step 5: Synchronize the TITAN Ledger last**

Record measured values and limitations in README, architecture, performance model, decisions, benchmarks, checklist, and context notes. Update `PROJECT_STATE.md` last. Commit as `docs: record Milestone 32 results`.

- [ ] **Step 6: Publish and verify**

Push a public ready PR, require branch/PR correctness and C++/Python CodeQL, rebase-merge, and verify post-merge `main` correctness and CodeQL. Do not create paid cloud resources.
