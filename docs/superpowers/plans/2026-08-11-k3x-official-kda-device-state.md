# K3X Official KDA Device-State Handoff Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Remove the intermediate official KDA state round trip between two exact incremental calls while retaining host-state execution as the default and oracle.

**Architecture:** Move KDA state into a dedicated backend-owned CUDA allocation. Add a single opaque owner/generation token, explicit seed/continue/publish control, fail-closed lifetime validation, and cumulative transfer/state telemetry. Extend the dedicated official-layer harness without changing production artifact capability or historical evidence schemas.

**Tech Stack:** C++20, CUDA 13.3 `sm_120`, existing K3X backend and profiler, Python 3.12, pytest, CMake/CTest, Compute Sanitizer, canonical JSON/CSV, SHA-256, and atomic publication.

## Global constraints

- Keep host round-trip state as the source-compatible default.
- Expose no raw CUDA pointer and accept only a backend-owned opaque token.
- Support one active device state per backend; reject stale, consumed, cross-backend, wrong-layer, and wrong-config tokens before upload or launch.
- Store KDA state in a dedicated allocation independent of sequence-sized scratch.
- Preserve dynamic input and immutable-weight validation semantics from M30.
- Preserve outputs, routes, contributions, final host state, resident weights, and the non-executable artifact guard.
- Do not add concurrency or multi-session state policy in this milestone.
- Do not download a complete shard/checkpoint or provision paid cloud resources.
- Witness RED before implementation, run focused GREEN and regressions, and create semantic commits.
- Every new Python, C++, or CUDA source file starts with a one-line Korean role comment.
- B-0032 must not emit token throughput, TTFT, quality, physical traffic, utilization, or bandwidth fields.

---

### Task 1: Opaque device-state lifetime and transfer semantics

**Files:**
- Modify: `runtime/include/k3x/backend.hpp`
- Modify: `runtime/cuda/backend_cuda.cu`
- Modify: `tests/cuda/test_cuda_official_kda.cu`

- [x] Write RED tests for host default parity, seed/continue/publish, exact transfer counters, empty unpublished host vectors, single-use generation, wrong owner/layer/config, and invalidation after mutation begins.
- [x] Build and witness focused RED because no device-state API exists.
- [x] Add the minimum control/token/result surface and dedicated reusable state allocation.
- [x] Validate all structure, weights, dynamic hidden, and token identity before mutation; invalidate a consumed token before upload/launch and issue a new generation only after success.
- [x] Run focused GREEN plus official KDA/MoE CUDA regressions.
- [x] Self-review and commit as `feat: retain official KDA state on device`.

### Task 2: Official-layer wrapper and explicit harness telemetry

**Files:**
- Modify: `runtime/include/k3x/official_layer.hpp`
- Modify: `runtime/src/official_layer.cpp`
- Modify: `runtime/src/cuda_official_layer_bench.cpp`
- Modify: `tests/cuda/test_cuda_official_layer.cu`
- Modify: `tests/python/test_cuda_official_layer.py`

- [x] Write RED tests for exact A-to-B wrapper parity and `--state-transfer host|device` parsing, default schema preservation, invalid combinations, and closed telemetry.
- [x] Add explicit begin/continue publication at the layer boundary while leaving the existing host call unchanged.
- [x] Restrict harness device state to `ab-incremental + resident + admission`; emit new fields only when the option is explicit.
- [x] Run focused GREEN and one actual-artifact host/device parity smoke.
- [x] Self-review and commit as `feat: expose official KDA device-state handoff`.

### Task 3: Strict B-0032 evidence transaction

**Files:**
- Create: `tools/ablate_official_kda_device_state.py`
- Create: `tests/python/test_official_kda_device_state_ablation.py`

- [ ] Write controlled RED verifier tests for the fixed host-incremental/device-incremental/full-host row order, exact formulas, parity, digests, LF CSV, fsync, and atomic publication.
- [ ] Implement the minimum non-ranking transaction by reusing B-0031 canonical and manifest authorities without changing historical evidence.
- [ ] Run focused GREEN, compile validation, and mutation tests.
- [ ] Self-review and commit as `test: add strict B-0032 state-handoff evidence`.

### Task 4: Formal B-0032, verification, Ledger, and publication

**Files:**
- Generate: `results/b0032-official-kda-device-state-wsl/**`
- Modify: `README.md`
- Modify: `ARCHITECTURE.md`
- Modify: `PERFORMANCE_MODEL.md`
- Modify: `DECISIONS.md`
- Modify: `BENCHMARKS.md`
- Modify: `PROJECT_STATE.md`
- Modify: `checklist.md`
- Modify: `context-notes.md`

- [ ] Review the implementation and pass actual-artifact host/device parity, Compute Sanitizer, and production guard gates.
- [ ] Run the fixed three-row B-0032 transaction exactly once with three warmups and twenty measured sequences.
- [ ] Strictly verify and independently rehash evidence, then commit it separately.
- [ ] Run CPU, liburing/direct, ASan/UBSan, CUDA, actual-artifact, evidence, sanitizer, and production-guard verification.
- [ ] Record measured results and limitations, update `PROJECT_STATE.md` last, and commit the Ledger.
- [ ] Publish a public PR, require correctness and CodeQL, rebase-merge, and verify post-merge `main` gates.
