# K3X Milestone 5 Persistent L1 Expert Cache Implementation Plan

> Execute with test-driven development in the existing isolated worktree. Commit each numbered task as one reversible unit.

**Goal:** Add a bounded immutable system-RAM expert cache that avoids repeated K3X extent reads, feeds both exact synchronous and prepared CUDA paths, and is measured independently without introducing eviction policy.

**Architecture:** A model-adjacent `HostExpertStore` owns complete native MXFP4 gate/up/down payloads under `(layer, expert)` keys. Disabled mode preserves the current transient loader. Static admission is whole-expert atomic, no-eviction, capacity-bounded, and exact-bypass on no room.

**Toolchain:** C++20, CMake/Ninja, Python 3.12/pytest, optional CUDA 13.3 `sm_120`, Compute Sanitizer, JSON/CSV benchmark tooling.

**Normative design:** `docs/superpowers/specs/2026-08-09-k3x-persistent-l1-expert-cache-design.md`.

---

## Task 1: Define runtime options, statistics, and CLI validation

**Files:**
- Modify `runtime/include/k3x/model.hpp`.
- Modify `runtime/src/model.cpp`.
- Modify `runtime/src/main.cpp`.
- Modify `tests/cpp/test_backend.cpp` only if shared defaults belong there; otherwise create `tests/cpp/test_model_options.cpp` and add its CMake target.
- Modify `tests/python/test_cpp_parity.py`.

- [ ] Write RED tests for `disabled|static`, zero/positive capacity pairing, unknown mode, CPU acceptance, and backward-compatible disabled defaults.
- [ ] Add `L1ExpertCacheMode`, `RuntimeOptions`, `L1ExpertCacheStats`, result fields, and an options-based generation overload.
- [ ] Parse and serialize `--l1-expert-cache` and `--l1-expert-cache-bytes` without coupling them to CUDA switches.
- [ ] Run CPU and CUDA targeted tests.
- [ ] Commit as `feat: define bounded L1 expert cache contract`.

## Task 2: Implement the immutable bounded host expert store

**Files:**
- Create `runtime/include/k3x/host_expert_store.hpp`.
- Create `runtime/src/host_expert_store.cpp`.
- Create `tests/cpp/test_host_expert_store.cpp`.
- Modify `CMakeLists.txt`.

- [ ] Write RED tests for first admission, stable hit identity, exact capacity, oversized bypass, no-room bypass, duplicate key, disabled behavior if represented by the class, and loader failure atomicity.
- [ ] Define exact projection/payload/key types and immutable shared handles.
- [ ] Implement loader-on-miss, whole-expert charged-byte calculation with overflow guards, hard capacity, no eviction, and zero-copy hits.
- [ ] Ensure a failed loader and invalid payload cannot change success counters or residency.
- [ ] Run the native CPU test in Debug/Release-compatible assertions.
- [ ] Commit as `feat: add immutable bounded host expert store`.

## Task 3: Integrate the store into every expert execution path

**Files:**
- Modify `runtime/src/model.cpp`.
- Modify `tests/python/test_cpp_parity.py`.

- [ ] Write RED graph tests for disabled/static parity across operation and FFN-block paths and verify the same token/routing trace.
- [ ] Replace transient value payloads with shared handles whose lifetime covers view construction and backend execution.
- [ ] Keep disabled behavior byte/read identical and make static hits skip all six reader calls.
- [ ] Verify static capacity bypass returns the exact transient result and never changes routing.
- [ ] Run CPU full-prefix/incremental and CUDA synchronous graph parity.
- [ ] Commit as `feat: cache exact expert payloads in system RAM`.

## Task 4: Feed Milestone 4 prepared transfer from L1 safely

**Files:**
- Modify `runtime/src/model.cpp` if needed.
- Modify `tests/python/test_cpp_parity.py`.
- Modify `tests/cuda/test_cuda_async_ffn.cu` only if a lower-level lifetime regression is required.

- [ ] Write RED tests requiring static-cache hits with prefetch while preserving exact tokens, routing, H2D, prepared-call counts, and single-use identity.
- [ ] Hold cache handles through prepare and consume; do not expose device pointers or weaken the owned pinned-copy contract.
- [ ] Test no-room exact bypass and admitted-hit paths under FP32/BF16 and scalar/grouped where applicable.
- [ ] Run CUDA graph parity and affected Compute Sanitizer targets.
- [ ] Commit as `feat: feed exact prefetch from persistent L1`.

## Task 5: Expose L1 and reader accounting in benchmark records

**Files:**
- Modify `runtime/src/main.cpp`.
- Modify `tools/benchmark_synthetic.py`.
- Modify `tests/python/test_benchmark_schema.py`.

- [ ] Write RED schema tests for configured mode/capacity, hits, misses, bypasses, current/peak resident bytes, reader calls, requested bytes, and completed bytes.
- [ ] Serialize runtime JSON and preserve literal integer fields in JSON/CSV.
- [ ] Require deterministic cache counters across measured samples; continue using medians only for timing/RSS fields.
- [ ] Verify disabled zero counters and static positive bounded counters on CPU and CUDA.
- [ ] Commit as `feat: report persistent L1 cache accounting`.

## Task 6: Add the B-0006 matched ablation runner

**Files:**
- Create `tools/ablate_l1_expert_cache.py`.
- Create `tests/python/test_l1_expert_cache_ablation.py`.

- [ ] Write RED fake-record tests for the exact four-case order from the design.
- [ ] Reject token/routing/provenance mismatch, nonzero disabled counters, zero static hits/misses, capacity overflow, unexpected bypass, unchanged-or-higher static reader traffic, matched transfer-counter changes, and missing raw files.
- [ ] Write one JSON/CSV per row and one measured-delta-only summary.
- [ ] Run unit tests and FP32/BF16 one-sample real smokes.
- [ ] Commit as `feat: add persistent L1 cache ablation`.

## Task 7: Full verification and B-0006 measurement

**Files:**
- Create `results/b0006-l1-expert-cache.json`.
- Create raw FP32/BF16 result directories.

- [ ] Run complete CPU/CUDA builds, CTest, and pytest suites.
- [ ] Run every CUDA Compute Sanitizer target.
- [ ] Regenerate the seeded synthetic source and a chunk-257 K3X artifact; record maximum source read and artifact SHA-256.
- [ ] Run FP32 and BF16 B-0006 with three warmups and 20 samples per row.
- [ ] Programmatically cross-check compact values, exact tokens/routing, provenance, reader traffic, cache capacity, and transfer accounting against all raw records.

## Task 8: Update TITAN Ledger, review, and publish

**Files:**
- Modify `ARCHITECTURE.md`, `DECISIONS.md`, `BENCHMARKS.md`, `README.md`, `checklist.md`, `context-notes.md`.
- Modify `PROJECT_STATE.md` last.

- [ ] Decide the default only from B-0006 correctness, reader traffic, memory cost, and end-to-end timing; never call logical reads physical NVMe traffic.
- [ ] Update durable documents in TITAN protocol order and keep eviction/policies/L2 explicitly unimplemented.
- [ ] Request one final Terra high read-only review limited to Critical/Important correctness, lifetime, capacity, accounting, and documentation issues.
- [ ] Apply evidence-backed findings once and rerun affected verification.
- [ ] Commit results/ledger, push `codex/milestone-five-l1-cache`, create a draft PR, wait for Linux CI, mark ready, ancestry-check, fast-forward public `main`, and confirm post-merge CI.
- [ ] Preserve the worktree for the next milestone.
