# K3X Native MXFP4 Python Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Subagents are disabled by user instruction.

**Goal:** Execute and retain official native MXFP4 expert matrices directly from packed CUDA tensors in the Python compatibility runtime.

**Architecture:** Reuse the verified `runtime/cuda/mxfp4.cu` arithmetic through a small PyTorch extension, dispatch CUDA `K3XTensorStore.mxfp4_matvec` calls to it, and share one explicit-budget stable-admission cache through `OfficialRuntimeContext`.

**Tech Stack:** Python 3.12, PyTorch 2.13 CUDA extensions, CUDA 13.3, native `sm_120`, pytest.

**Spec:** `docs/superpowers/specs/2026-08-14-k3x-native-mxfp4-python-design.md`

## Global Constraints

- Preserve the portable CPU/reference path and zero-byte default.
- Preserve natural routing and exact native checkpoint bytes.
- Do not claim token TPS from component or layer measurements.
- Do not use subagents.

---

### Task 1: Native MXFP4 PyTorch extension

**Files:**
- Create: `runtime/cuda/mxfp4_torch_extension.cu`
- Create: `converter/k3x_converter/mxfp4_cuda.py`
- Test: `tests/python/test_mxfp4_cuda_extension.py`

**Interfaces:**
- Produces: `mxfp4_matvec(value, packed, scales, rows, columns) -> torch.Tensor`.

- [ ] Write a failing literal packed-byte CUDA parity test.
- [ ] Run it and confirm failure because the extension module is absent.
- [ ] Implement the minimum native `sm_120` kernel wrapper using current-stream launch semantics.
- [ ] Run the focused test and compile the new source.
- [ ] Commit the extension and test.

### Task 2: K3X store dispatch and residency

**Files:**
- Modify: `converter/k3x_converter/fragment_tensor_store.py`
- Modify: `tools/official_runtime_context.py`
- Modify: `tools/run_official_token.py`
- Test: `tests/python/test_local_shard.py`
- Test: `tests/python/test_official_persistent_runner.py`

**Interfaces:**
- Produces: `PackedMxfp4Cache(host_budget_bytes, device_budget_bytes)` and CUDA dispatch in `K3XTensorStore.mxfp4_matvec`.

- [ ] Add RED coverage proving CUDA bypasses the portable decoder, an L0 hit avoids a second read, and zero budgets preserve misses.
- [ ] Implement validated packed/scales acquisition and native CUDA dispatch.
- [ ] Add explicit context/CLI budgets and timing telemetry.
- [ ] Run the focused store/runtime tests.
- [ ] Commit the runtime bridge.

### Task 3: Released expert measurement and ledger

**Files:**
- Create: `tools/benchmark_official_mxfp4_residency.py`
- Create: `results/b0056-native-mxfp4-residency/*`
- Modify: `ARCHITECTURE.md`, `DECISIONS.md`, `BENCHMARKS.md`, `PROJECT_STATE.md`, `README.md`, `checklist.md`, `context-notes.md`

**Interfaces:**
- Consumes: one released expert triplet from the sealed K3X set.
- Produces: cold and at least five warm timings, cache bytes/hits, portable-oracle error, and checksummed JSON evidence.

- [ ] Run one cold plus five warm expert-triplet executions on RTX 5080.
- [ ] Verify all output digests, cache formulas, and internal JSON digests.
- [ ] Run focused regressions, Python compilation, and `git diff --check`.
- [ ] Update the TITAN Ledger with measured values only and `PROJECT_STATE.md` last.
- [ ] Commit and push the evidence.
