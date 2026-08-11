# K3X Official KDA Admission-Validation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Attribute repeated official KDA immutable-weight validation cost by extending the existing exact resident admission policy and measuring a fixed B-0031 matrix without changing correctness or defaults.

**Architecture:** Keep structural and dynamic-state validation on every official KDA call. Classify the fourteen immutable BF16/F32 views against the existing backend identity registry, scan every new view, commit identities only after all scans pass, and expose cold/measured counter deltas through the dedicated official-layer harness. Reuse the ignored M29 artifact for one fixed four-row experiment.

**Tech Stack:** C++20, CUDA 13.3 `sm_120`, existing K3X backend telemetry, Python 3.12, pytest, CMake/CTest, Compute Sanitizer, canonical JSON/CSV, SHA-256, and fsynced atomic publication.

## Global constraints

- Keep `CudaWeightValidationMode::per_call` as the global and harness default.
- Support official-KDA admission only with exact resident weights; reject admission plus transient execution before backend construction in the harness.
- Validate hidden input, convolution histories, recurrent state, dimensions, unique tensor IDs, shapes, and byte lengths on every call.
- Admit only the exact tuple `(tensor_id, host_pointer, byte_length, rows, columns)`.
- Publish no partial registry entries, uploads, scratch allocations, or launches after a failed first admission.
- Reuse existing runtime counters and the existing immutable identity registry; do not add a KDA-specific cache.
- Preserve the M29 artifact, graph, natural Top-16 routes, contribution vectors, output digest, final-state digest, resident bytes, and production non-executable guard.
- Do not download new model payload, a complete shard, or the full checkpoint. Do not provision paid cloud resources.
- Witness RED before implementation, run focused GREEN and regressions, review the diff, and create one semantic commit per logical task.
- Every new Python, C++, or CUDA source file starts with a one-line Korean role comment.
- B-0031 must not emit token throughput, TTFT, quality, physical NVMe/PCIe traffic, GPU utilization, or memory-bandwidth fields.

---

### Task 1: Official KDA immutable admission semantics

**Files:**
- Modify: `tests/cuda/test_cuda_official_kda.cu`
- Modify: `runtime/cuda/backend_cuda.cu`

**Interfaces:**
- Reuses: `CudaWeightValidationMode`, `ImmutableWeightIdentity`, `immutable_weights_`, and the four `immutable_validation_*` counters.
- Validates: eight `Bf16WeightView` values and six F32 `DenseWeightView` values.

- [ ] **Step 1: Write CUDA RED coverage**

Extend the tiny KDA fixture to run resident per-call and admission modes. Require first admission to scan 14 views and 384 bytes, the second identical call to add 14 hits and no bytes, and per-call mode to scan all 14 views on every call. Add exact output/state parity.

Add fail-closed tests for a different allocation under the same tensor IDs, non-finite BF16 and F32 weights, and atomic recovery. After a failed first admission, a valid call on the same backend must scan all fourteen views with zero prior hits and execute successfully.

- [ ] **Step 2: Build and witness RED**

```bash
cmake --build build-cuda --target test_cuda_official_kda -j2
ctest --test-dir build-cuda -R '^cuda_official_kda$' --output-on-failure
```

Expected: admission scan/hit assertions fail because official KDA still scans every immutable payload on every call and never uses the registry.

- [ ] **Step 3: Implement atomic admission**

Split metadata checks from payload finiteness checks. Form canonical BF16 and F32 view descriptors, classify every view before scanning, reject any identity conflict, scan only required views, and insert new identities only after all scans pass. Record scan count, bytes, time, and hits with the existing counters. Keep all dynamic validation outside the admission cache.

- [ ] **Step 4: Run focused GREEN and CUDA regressions**

```bash
cmake --build build-cuda --target test_cuda_official_kda test_cuda_official_layer test_cuda_moe_layer -j2
ctest --test-dir build-cuda -R 'cuda_official_kda|cuda_official_layer|cuda_moe_layer' --output-on-failure
```

- [ ] **Step 5: Self-review and commit**

```bash
git diff --check
git add runtime/cuda/backend_cuda.cu tests/cuda/test_cuda_official_kda.cu
git commit -m "feat: admit official KDA immutable weights"
```

### Task 2: Official-layer validation CLI and telemetry

**Files:**
- Modify: `runtime/src/cuda_official_layer_bench.cpp`
- Modify: `tests/python/test_cuda_official_layer.py`

**Interfaces:**
- CLI: `--validation per-call|admission`, default `per-call`.
- JSON: adds `validation`, cold `immutable_validation_*` deltas, and measured `immutable_validation_*` deltas.

- [ ] **Step 1: Write CLI and schema RED tests**

Require default per-call parsing, explicit admission parsing, rejection of unknown validation values, and rejection of admission with transient weights. Extend the controlled schema expectations to cover all cold and measured validation fields.

- [ ] **Step 2: Run RED**

```bash
K3X_BUILD_DIR=build-cuda PYTHONPATH=converter:reference \
  /home/jolib/.venvs/k3x-m1/bin/python -m pytest \
  tests/python/test_cuda_official_layer.py -q
```

Expected: the new option is rejected and the JSON fields are absent.

- [ ] **Step 3: Implement the minimum harness extension**

Parse one validation enum, propagate it through `backend_options`, and emit counter deltas around the already existing cold and measured snapshots. Do not alter the M29 execution cases or historical B-0030 runner.

- [ ] **Step 4: Run GREEN and actual-artifact smoke**

Run the focused Python test plus one `ab-incremental` resident admission call against the ignored artifact. Require exact output/state identities, fourteen first-call scans, fourteen second-call cold hits, and zero measured scan bytes after admission.

- [ ] **Step 5: Self-review and commit**

```bash
git diff --check
git add runtime/src/cuda_official_layer_bench.cpp tests/python/test_cuda_official_layer.py
git commit -m "feat: expose official KDA validation telemetry"
```

### Task 3: Strict B-0031 evidence transaction

**Files:**
- Create: `tools/ablate_official_kda_validation.py`
- Create: `tests/python/test_official_kda_validation_ablation.py`

**Interfaces:**
- Fixed rows: incremental/full resident crossed with per-call/admission in the design order.
- Produces: canonical raw JSON, canonical summary JSON, LF-only summary CSV, aggregate digest, and strict `--verify-existing` mode.
- Strict identity: exactly 3 warmups and 20 measured sequences.

- [ ] **Step 1: Write runner/verifier RED tests**

Use controlled subprocess records. Require the fixed row order, exact schema, validation formulas, resident warm-zero weight H2D, output/state/route/contribution parity, artifact/manifest/runner/raw/aggregate/CSV digests, atomic publication, and LF-only CSV. Mutate every new counter and cross-row identity independently and require verification failure.

- [ ] **Step 2: Run RED**

```bash
PYTHONPATH=converter:reference /home/jolib/.venvs/k3x-m1/bin/python -m pytest \
  tests/python/test_official_kda_validation_ablation.py -q
```

Expected: import failure because the B-0031 runner does not exist.

- [ ] **Step 3: Implement the non-ranking transaction**

Reuse the B-0030 canonical parsing, hashing, fsync, and atomic-directory patterns without changing B-0030. Validate the closed formulas below for twenty measured sequences.

| Case | Validation | KDA calls | Scans | Hits | Scanned bytes |
|---|---|---:|---:|---:|---:|
| incremental | per-call | 40 | 560 | 0 | 35,512,033,280 |
| incremental | admission | 40 | 0 | 560 | 0 |
| full | per-call | 20 | 280 | 0 | 17,756,016,640 |
| full | admission | 20 | 0 | 280 | 0 |

- [ ] **Step 4: Run GREEN and compile validation**

```bash
PYTHONPATH=converter:reference /home/jolib/.venvs/k3x-m1/bin/python -m pytest \
  tests/python/test_official_kda_validation_ablation.py \
  tests/python/test_cuda_official_layer.py -q
PYTHONPATH=converter:reference /home/jolib/.venvs/k3x-m1/bin/python -m py_compile \
  tools/ablate_official_kda_validation.py
```

- [ ] **Step 5: Self-review and commit**

```bash
git diff --check
git add tools/ablate_official_kda_validation.py \
  tests/python/test_official_kda_validation_ablation.py
git commit -m "test: add strict B-0031 evidence pipeline"
```

### Task 4: Formal B-0031, full verification, Ledger, and publication

**Files:**
- Generate: `results/b0031-official-kda-validation-wsl/**`
- Modify: `README.md`
- Modify: `ARCHITECTURE.md`
- Modify: `PERFORMANCE_MODEL.md`
- Modify: `DECISIONS.md`
- Modify: `BENCHMARKS.md`
- Modify: `PROJECT_STATE.md`
- Modify: `checklist.md`
- Modify: `context-notes.md`

- [ ] **Step 1: Review the implementation and run capability gates**

Run focused CPU/CUDA tests, one actual-artifact per-call/admission parity smoke, strict production guard, and admission-mode resident incremental Compute Sanitizer with `--launch-timeout 0`. Fix proven defects before formal measurement.

- [ ] **Step 2: Run B-0031 exactly once**

```bash
PYTHONPATH=converter:reference /home/jolib/.venvs/k3x-m1/bin/python \
  tools/ablate_official_kda_validation.py \
  --artifact artifacts/m29-official-layer/official-kda-layer-l1.k3x \
  --manifest artifacts/m29-official-layer/route-state-manifest.json \
  --runner build-cuda/k3x_cuda_official_layer_bench \
  --output-dir results/b0031-official-kda-validation-wsl \
  --warmups 3 --iterations 20
```

If the transaction fails, retain no partial evidence. Fix only the proven defect and restart the complete fixed matrix; never rerun to select timing.

- [ ] **Step 3: Verify and commit evidence**

Run `--verify-existing`, independently rehash all files, cross-check summary/raw/CSV values, and commit the evidence separately.

```bash
git add results/b0031-official-kda-validation-wsl
git commit -m "bench: record B-0031 KDA validation results"
```

- [ ] **Step 4: Run the complete local verification matrix**

Run CPU CTest/Python, liburing/direct CTest/Python, ASan/UBSan CTest, CUDA CTest/Python with the actual artifact, the committed B-0030 and B-0031 verifiers, actual admission Compute Sanitizer, and `k3x_run` non-executable guard. Record fresh counts and the exact environment.

- [ ] **Step 5: Synchronize durable documentation**

Record measured validation deltas, wall/kernel/orchestration results, correctness identities, limitations, and the next measured bottleneck. Update `PROJECT_STATE.md` last and keep projections separate from measurements.

- [ ] **Step 6: Commit documentation and publish**

```bash
git add README.md ARCHITECTURE.md PERFORMANCE_MODEL.md DECISIONS.md BENCHMARKS.md \
  PROJECT_STATE.md checklist.md context-notes.md
git commit -m "docs: complete Milestone 30 ledger"
```

Push `codex/milestone-thirty-kda-admission`, open a public pull request, wait for correctness and CodeQL, rebase-merge only after success, and verify post-merge `main` correctness and CodeQL. Record publication once without creating a self-referential documentation loop.
