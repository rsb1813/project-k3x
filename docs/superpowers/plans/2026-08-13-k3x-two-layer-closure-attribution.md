# K3X Milestone 34 Two-Layer Closure Attribution Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Attribute the measured official two-layer device-closure regression to front, host route, tail, and wrapper remainder without changing execution.

**Architecture:** Reuse the existing CUDA backend profiler and snapshot newly emitted events around the existing front and tail calls. Attribution is caller-owned and opt-in; the historical benchmark schema and B-0034 evidence remain unchanged when disabled.

**Tech Stack:** C++20, CUDA 13.3, existing K3X `Profiler`, pytest, CTest, Compute Sanitizer.

**Spec:** `docs/superpowers/specs/2026-08-13-k3x-two-layer-closure-attribution-design.md`

## Global Constraints

- Preserve exact natural Top-16 routing and the portable reference path.
- Do not add CUDA synchronization or events solely for attribution.
- Default CLI output must preserve the B-0034 schema.
- No kernel fusion, full checkpoint, complete shard, paid cloud resource, or token-throughput claim.
- Every implementation task follows RED, GREEN, focused verification, and a semantic commit.

---

### Task 1: Attribution accumulator and exact wrapper regions

**Files:**
- Modify: `runtime/include/k3x/official_two_layer.hpp`
- Modify: `runtime/src/official_two_layer.cpp`
- Modify: `tests/cuda/test_cuda_official_two_layer.cu`

**Interfaces:**
- Produces: `OfficialTwoLayerAttribution` with front/route/tail/remainder wall fields and front/tail device fields.
- Produces: optional final parameter `OfficialTwoLayerAttribution* attribution = nullptr` on `official_two_layer_cuda`.

- [ ] **Step 1: Write the CUDA RED**

Add a profiler-backed backend call that passes an accumulator and asserts four front/tail regions, positive front/tail device time, nonzero host route time, nonnegative closed remainder, and unchanged route/output/state parity. Add a disabled call asserting historical behavior.

- [ ] **Step 2: Run the RED**

Run `ctest --test-dir build-cuda -R cuda_official_two_layer --output-on-failure` and require compilation failure on the missing type/signature.

- [ ] **Step 3: Implement the minimum accumulator**

Snapshot `Profiler::events().size()` before and after each backend call, sum only successful newly appended `device_nanoseconds`, measure wall regions with `steady_clock`, measure canonical route plus expert resolution, and compute a checked wrapper remainder.

- [ ] **Step 4: Run focused GREEN and sanitizer**

Run the CUDA test, portable two-layer CTest, and Compute Sanitizer on `test_cuda_official_two_layer`. Require exact prior parity and zero sanitizer errors.

- [ ] **Step 5: Commit**

Commit as `feat: attribute two-layer closure regions`.

### Task 2: Opt-in harness schema

**Files:**
- Modify: `runtime/src/cuda_official_two_layer_bench.cpp`
- Modify: `tests/python/test_cuda_official_two_layer.py`
- Modify: `tests/python/test_official_two_layer_ablation.py`

**Interfaces:**
- Consumes: `OfficialTwoLayerAttribution` from Task 1.
- Produces: `--attribution true|false`, default false.
- Produces: `k3x-official-two-layer-attribution-v1` JSON only when enabled.

- [ ] **Step 1: Write parser and schema RED tests**

Assert invalid booleans fail before file access, default output contains no attribution fields and retains the B-0034 schema, and enabled output contains exactly the six timing fields with closed formulas.

- [ ] **Step 2: Run the RED**

Run the two focused Python files and require failure because the option and fields do not exist.

- [ ] **Step 3: Implement minimal CLI wiring**

Create a `Profiler` only when attribution is enabled, pass it to `make_cuda_backend`, pass the accumulator to the wrapper, aggregate each measured call, and serialize the explicit schema. Do not alter the default branch.

- [ ] **Step 4: Run focused GREEN and B-0034 reverify**

Run both Python files, CUDA two-layer CTest, and strict `tools/ablate_official_two_layer_closure.py --verify-existing` against committed B-0034.

- [ ] **Step 5: Commit**

Commit as `feat: expose opt-in closure attribution`.

### Task 3: B-0035 fail-closed evidence and measurement

**Files:**
- Create: `tools/ablate_official_two_layer_attribution.py`
- Create: `tests/python/test_official_two_layer_attribution.py`
- Generate: `results/b0035-official-two-layer-attribution-wsl/`

**Interfaces:**
- Consumes: attribution harness schema from Task 2.
- Produces: canonical raw JSON, summary JSON/CSV, aggregate SHA-256, and strict `--verify-existing`.

- [ ] **Step 1: Write evidence RED tests**

Fix host/device row order, 3/20 warmup/iteration identity, artifact/manifest/oracle/runner hashes, exact correctness and traffic gates, six timing fields, closed nonnegative formulas, LF CSV, and forbidden token/quality/physical-traffic fields.

- [ ] **Step 2: Run the RED**

Run the new pytest file and require import failure for the missing tool.

- [ ] **Step 3: Implement the minimum publisher/verifier**

Follow the B-0034 atomic publication pattern while accepting only the attribution schema and preserving raw measured values.

- [ ] **Step 4: Run dry-run verification and actual gates**

Run focused evidence tests, CUDA CTest, actual harness correctness, production guard, and Compute Sanitizer before measurement.

- [ ] **Step 5: Run and seal B-0035**

Execute one successful 3-warmup/20-iteration transaction, run strict verification and an independent hash/formula cross-check, then commit the evidence as `bench: seal two-layer closure attribution`.

### Task 4: Decision and TITAN Ledger synchronization

**Files:**
- Modify: `README.md`
- Modify: `ARCHITECTURE.md`
- Modify: `DECISIONS.md`
- Modify: `BENCHMARKS.md`
- Modify: `PROJECT_STATE.md` last
- Modify: `checklist.md`
- Modify: `context-notes.md`

**Interfaces:**
- Consumes: sealed B-0035 evidence.
- Produces: D-072 and the next measured bottleneck decision.

- [ ] **Step 1: Record measured evidence without projection**

Document exact hashes, hardware, medians, attribution shares, correctness, logical traffic, and all unmeasured benchmark fields.

- [ ] **Step 2: Decide the next optimization boundary**

Accept only a region supported by B-0035. Record fusion, wider closure, and no-change alternatives with evidence.

- [ ] **Step 3: Run final verification**

Run portable/CUDA CTest, focused and broad Python suites within explicit limits, evidence regressions, sanitizer, production guard, `git diff --check`, and self-review.

- [ ] **Step 4: Commit and publish**

Commit as `docs: record two-layer closure attribution`, push a public PR, require correctness and CodeQL, merge, verify post-merge CI, and update publication state separately if needed.
