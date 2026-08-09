# K3X Current Project State

## Current milestone

Milestone 10 is published and verified. Milestone 11 Adaptive Top-K and exact cold-expert rescue design and TDD plan are complete; implementation has not started.

State recorded on 2026-08-09 at verified Milestone 10 publication head `a82d733`. Publication-record correctness runs `31316068815` and `31316069868` succeeded, and the active branch is `codex/milestone-eleven-adaptive-topk-rescue`.

## Completed work

- Milestone 0 deterministic synthetic K3-compatible PyTorch graph, K3X v1 streaming converter, strict Python/C++ readers, and independent portable C++20 runtime.
- KDA, Gated MLA, Attention Residual, Stable LatentMoE, router, native MXFP4, full/incremental state, layer/logit/state parity, and exact greedy token tests.
- Public Milestone 1 at `254a9ac` with explicit CPU, `cuda-dense`, and `cuda-custom` identities; CUDA 13.3 native `sm_120`; cuBLASLt dense FP32/BF16; exact native-byte MXFP4 CUDA; deterministic JSON/CSV profiling; and B-0002.
- Milestone 2 runtime switches for `per-operation|reused`, `transient|resident`, and `scalar|grouped`, with reference defaults preserved.
- Tracked CUDA allocations, grow-only scratch buffers, reusable CUDA events and cuBLASLt plans, and exact live/peak VRAM counters.
- Stable tensor identities and bounded static FP32, BF16, and native MXFP4 L0 weight residency. Capacity misses bypass to the exact transient path; this Milestone 2 table remains distinct from Milestone 9 L1 expert eviction.
- Same-input grouped CUDA projections for KDA Q/K/V, dense and shared gate/up, and routed-expert native MXFP4 gate/up. Expert down remains scalar after CPU SiTU-GLU.
- Split immutable-weight and activation H2D profiler aggregation, stable runtime-counter export, and a deterministic four-stage sequential ablation runner.
- B-0003 measurement with three warmups and 20 samples for both FP32 CUDA identities across reference, reuse, residency, and grouped stages, plus fully enabled BF16 measurements.
- Milestone 3 `operation|ffn-block` boundary switch with `cuda-custom`-only capability validation and operation reference default.
- Dependency-closed dense/shared and ordered exact native MXFP4 expert FFN blocks that retain gate/up, strict SiTU-GLU, and down projection work on one CUDA stream through the final result transfer.
- Device SiTU timing, FFN block call/expert counters, exact generated-token capture, prefill routing trace diagnostics, and a provenance-checked four-case FFN boundary ablation runner.
- B-0004 FP32/BF16 measurement with three warmups and 20 samples across operation/FFN-block and scalar/grouped paths.
- Explicit `synchronous|prefetch` transfer identity with strict capability validation and unchanged synchronous default.
- Fixed-capacity page-locked host staging, matching device staging, a nonblocking transfer stream, reusable readiness/timing events, and process-global single-use prepared tokens.
- Exact native MXFP4 triplet preflight and staged router-order device views, with foreign, stale, duplicate, layer, phase, extent, scale, and capacity rejection before invalid consumption.
- Two-phase routed MoE scheduling that overlaps exact expert H2D with the routed-down projection while preserving natural routing, CPU score mixing, recurrent state, and greedy tokens.
- Runtime and benchmark accounting for pinned memory, prefetch calls/bytes, ready/late use, stream waits, staging/device/stall time, async-engine count, and device-overlap capability.
- B-0005 FP32/BF16 measurement with three warmups and 20 samples across synchronous/prefetch and scalar/grouped paths, including raw JSON/CSV and a cross-checked compact manifest.
- Runtime `disabled|static` L1 expert-cache identity, hard capacity pairing, explicit hit/miss/bypass/current/peak counters, and backward-compatible disabled default.
- Immutable whole-expert native MXFP4 handles keyed by layer/expert, complete-expert atomic admission, stable zero-copy hits, overflow guards, loader failure atomicity, no-eviction capacity, and exact transient bypass.
- Shared payload lifetime across CPU/operation, synchronous CUDA FFN-block, and asynchronous prepared-transfer execution without changing router selection or device payloads.
- Reader calls/requested/completed byte accounting in runtime and benchmark JSON/CSV, explicitly labeled as logical file reads rather than physical NVMe traffic.
- Four-case B-0006 runner crossing disabled/static L1 with synchronous/prefetch transfer, including strict option/provenance, token/routing, traffic, capacity, cache-counter, and raw-artifact checks.
- B-0006 FP32/BF16 measurement with three warmups and 20 samples per row, raw JSON/CSV, compact manifest, and programmatic compact/raw cross-check.
- Independent `pread|io_uring` and `buffered|direct` Reader axes, persistent Linux descriptor, ordered extent batches, and one six-extent native MXFP4 expert request.
- Optional liburing engine with explicit offsets, bounded queue depth, stable completion identity, partial-submit draining, and fail-closed unavailable capability behavior.
- Explicit `O_DIRECT` path gated by `STATX_DIOALIGN`, owned aligned bounce buffers, exact logical slice reconstruction, and separate logical versus aligned-storage accounting.
- Runtime/benchmark L2 identity, alignment, batch/completion/error, Reader storage time, and Linux process-I/O fields plus a capability-aware four-case B-0007 runner.
- B-0007 WSL2 ext4 measurement with three warmups and 20 samples per row, raw JSON/CSV, exact token/routing/logical-byte parity, and an explicit non-authoritative environment label.
- Deterministic streaming `k3-storage-slice-v1` source writer that physically materializes one released-dimension expert in bounded chunks without sparse holes.
- K3X optional `STORAGE_FIXTURE` identity, strict source kind/config/shape/length validation, resumable conversion, and gate/up/down physical execution order.
- Python/C++ Reader support for storage-fixture identity plus an explicit `NON_EXECUTABLE_ARTIFACT` generation guard before graph tensor lookup.
- Dedicated exact six-extent `k3x_storage_bench` with ordered digest, latency percentiles, logical/storage/process-I/O accounting, and no token schema.
- Four-case B-0008 runner with capability-only skips, compact/raw cross-checks, and WSL2 ext4 3-warmup/20-sample measurements.
- Content-addressed storage-fixture shard publication, bounded shard/tensor SHA-256 validation, manifest-scoped source identity, and canonical source-matched resume extent validation.
- Bounded single-worker expert-load scheduler ordered by absolute latest-start time, with stable ties, inline resident completion, capacity failure, error propagation, idle draining, and structured telemetry.
- Opt-in `blocking|deadline` L2 expert scheduling with unchanged blocking default and exact current-layer natural Top-K submission after routing.
- Same-layer overlap of non-resident expert loading with routed-down and shared-expert computation, followed by exact wait-before-use semantics.
- Mutex-protected Reader and L1 telemetry snapshots, one pre-submit per-layer fetch estimate, and success/error idle barriers that keep captured Reader/store lifetimes inside generation.
- Eight-case B-0009 runner crossing schedule, Reader engine, and cache mode with capability-only skips, exact token/routing/logical-I/O parity, raw JSON/CSV, and WSL2 ext4 3-warmup/20-sample measurements.
- Runtime-switchable exact `lru`, `lfu`, and `least-stale` L1 policies with unchanged `disabled` default and `static` no-eviction reference.
- Session-monotonic token-forward cycles, complete natural Top-K selected-set protection, deterministic victim ordering, exact capacity bypass, eviction counters, and same-forward collision-miss telemetry.
- SpecMD Least-Stale paper reproduction with stale-before-current priority, processed-left-layer priority, upcoming-layer protection, and deterministic LRU collision 1 versus Least-Stale 0 trace coverage.
- Session-wide generation serialization that prevents concurrent forwards from overwriting the store's active policy context while leaving independent sessions independent.
- Thirteen-case B-0010 runner crossing disabled plus four policies at 2-, 8-, and 16-expert capacities with exact token/routing/numerical-error identity, raw JSON/CSV, and WSL2 ext4 3-warmup/20-sample measurements.
- Bounded canonical `.k3xp` runtime profiles with validated metadata, prior/live expert frequency, adjacent-layer transitions, deterministic hot bank, CRC32C, and sibling temporary publication.
- Opt-in exact `profiled` eviction with explicit prior strength and live-observation crossover; runtime metadata remains outside prompt IDs and default sessions do not observe profile state.
- CLI profile load/save and metadata controls plus bytes/time/prior telemetry, exact resume parity, malformed-input rejection, bounded record behavior, and full-generation artifact-size evidence.
- Five-case B-0011 runner comparing LFU, Least-Stale, cold profile, matching prior, and minimum-overlap alternate prior with exact token/routing/logit/state identity and WSL2 ext4 3-warmup/20-sample measurements.

## Work in progress

- Milestone 11 primary-source and current-code boundary review, alternatives comparison, accepted design, implementation plan, checklist, and context notes are complete. No Milestone 11 source implementation is claimed yet.
- Milestone 9 Terra high final review found one valid Important shared-session policy-context issue. Commit `fd05d95` serializes complete generation calls; re-review found no remaining Critical or Important issue and withdrew an initial collision interpretation concern after deterministic future-layer trace review.
- Static, LRU, LFU, Least-Stale, profiled eviction, and all non-default L2 modes remain experimental and opt-in. Transition prediction and cross-layer asynchronous L2 scheduling remain unimplemented.
- The L2 batch API submits concurrent operations for one batch but waits before returning. It is not the chartered N/N+1/N+2 deadline pipeline yet.
- The deadline worker schedules only the current routed layer and remains slower than blocking in all B-0009 rows. ORBIT, multiple L2 workers, eviction-aware priority, and future-layer recall are not implemented.
- `pread + buffered`, blocking scheduling, and disabled L1 remain defaults because B-0007 through B-0011 are WSL2 ext4 evidence, not native P44 Pro or full-model evidence.
- Worktree: `C:\Users\jolib\Documents\project-k3x\.worktrees\milestone-one-runtime`.
- Linux Python environment: `/home/jolib/.venvs/k3x-m1`; builds: `build-cpu`, `build-uring`, and `build-cuda`.

## Known failures and blockers

- Windows Smart App Control still blocks unsigned `k3x_run.exe`; WSL2 is the verified local CUDA path and native Linux remains the final performance authority.
- The executable checkpoint is synthetic and tiny. The full-dimension artifact contains only one non-executable expert. No full Kimi K3 weights have been downloaded, and B-0008 is neither token throughput nor a full-model claim.
- The graph remains CPU-driven outside FFN blocks. KDA, MLA, routing, score mixing, residual work, state management, and non-FFN boundaries remain on the host.
- L1 misses use ordered Reader batches into pageable host vectors. Deadline mode may run one such blocking batch on a worker, but there is no cross-layer asynchronous L2 pipeline, transition predictor, or N/N+1/N+2 triple buffering.
- Exact prefetch is single-flight and limited to `cuda-custom + ffn-block + reused + transient`; it is not combined with static residency.
- `cuda-dense` intentionally keeps native MXFP4 on the CPU as its documented comparison identity. `cuda-custom` is the exact GPU MXFP4 path.
- GPU utilization, GPU memory bandwidth, and physical NVMe GB/token remain unmeasured. Reader storage elapsed time, logical/aligned bytes, process `rchar/read_bytes`, and L1-to-L0 timing are measured under their stated scopes.
- WSL2 `/mnt/c` is 9p/DrvFS and rejects direct alignment. liburing 2.5 and direct I/O were validated on WSL2 ext4 `/tmp`, which remains a correctness/capability environment rather than native P44 Pro performance authority.
- ThreadSanitizer builds successfully but its runtime exits under WSL2 with `unexpected memory mapping`; no TSan execution result is claimed. ASan/UBSan and explicit concurrency regressions pass.
- Full-model quality, coding/agentic quality, adaptive Top-K, cold rescue, speculation, proxy, and pruning remain unimplemented or unmeasured.

## Next concrete tasks

1. Add RED tests for natural/fixed/adaptive selection, stable ties, invalid configuration, cumulative mass, entropy effective support, boundary confidence, and quality floors.
2. Implement the smallest pure routing-policy boundary while leaving all existing natural paths unchanged.
3. Add an explicit 16-of-24 synthetic fixture and PyTorch oracle before runtime/CLI integration.
4. Add a bounded multi-expert or full-layer slice for representative cache pressure and locality evidence.
5. Run native-Linux P44 Pro warm/cold B-0008 through B-0011 only when that environment exists.

## Hardware assumptions

| Component | Current assumption or observation |
|---|---|
| CPU | AMD Ryzen 7 9800X3D target and local host |
| GPU | NVIDIA GeForce RTX 5080, 16,303 MiB, compute capability 12.0 |
| Driver | 591.86 |
| CUDA | Toolkit 13.3.1, nvcc 13.3.73, native `sm_120` |
| RAM | 96 GB DDR5-4200 target; WSL2 exposed 49,251,213,312 bytes during earlier validation |
| NVMe | Solidigm P44 Pro 2 TB target; not measured by B-0007 or B-0008 |
| Final runtime OS | Linux native |
| Current development OS | WSL2 Ubuntu 24.04.4 on Windows 11 |

## Latest measured bottleneck

B-0011 shows that a matching task/session prior can equal Least-Stale traffic at the 8-expert synthetic capacity but does not improve timing. Matching prior and Least-Stale both record 23 hits, 31 misses, and 628,080 logical Reader bytes, while matching-prior decode is 5,006.701 tok/s versus 5,900.245 tok/s. The minimum-overlap alternate prior records 22/32/629,712 and 4,868.597 tok/s.

The measured bottleneck remains expert miss traffic under constrained residency, with profile bookkeeping and load/save overhead visible on the tiny graph. Helpful prior information did not reduce traffic below Least-Stale, and stale or mismatched priors can regress it. Separate-process WSL2 timing, uncontrolled cold-cache state, and logical Reader bytes cannot establish a native default.

The immediate result is to retain `disabled + blocking + pread + buffered` as defaults and keep `profiled` opt-in. The next implementation gap is adaptive Top-K with exact rescue, followed by representative full-size multi-expert pressure and a future-layer predictor with high recall. Native-Linux physical NVMe traffic, controlled cold/warm behavior, and full-model locality remain unknown. CPU-driven KDA/MLA, routing, residual/state, and non-FFN orchestration also remain visible bottlenecks.

The derived uncached full-model expert traffic remains 25.83 GB/token, but it is not a measured full-model value. Native-Linux NVMe traffic, cache reuse, and full Kimi K3 throughput remain unknown.

## Last known-good state

- Public Milestone 10 publication head: `a82d733102a9e719271af53f64228aed9837925e`; publication branch/main correctness runs `31316068815` and `31316069868` succeeded. PR #10 integration head remains `3d8c042`.
- Latest verified Milestone 10 code head: `5430074` (`fix: preserve bounded profile evidence`).
- Latest verified B-0011 result head: `308b0db`; measurement code is `5430074`.
- B-0011 synthetic K3X artifact SHA-256: `0dfe0fe7c64b364fc745fff5b6c9a1f06d1faf1dc140630a9240591540dd684d`; summary SHA-256: `029004e02a8484f281a332c09d49e7adc8eb1ed343ec692b030760288adbd94f`.
- CPU verification: CTest 10/10; pytest 211 passed and 41 skipped.
- Liburing/direct verification: CTest 11/11; pytest 213 passed and 39 skipped.
- ASan/UBSan liburing verification: CTest 11/11; targeted pytest 82 passed and 33 skipped.
- CUDA verification: CTest 19/19; pytest 244 passed and 8 skipped.
- Compute Sanitizer: `test_cuda_device`, `test_cuda_dense`, `test_cuda_mxfp4`, `test_cuda_memory`, `test_cuda_pinned_memory`, `test_cuda_async_pipeline`, `test_cuda_residency`, `test_cuda_situ`, `test_cuda_ffn`, and `test_cuda_async_ffn` each report `ERROR SUMMARY: 0 errors`.
- Existing executable-model exact generated tokens remain `[43, 32, 28, 49, 9, 28]` in all B-0011 rows.
- B-0011 raw JSON/CSV, profiles, and cross-checked summary: `results/b0011-task-session-profiles-wsl/`.
- Pre-publication ledger check: `git diff --check` passed, the dedicated B-0011 Python regression passed 10/10 with `K3X_BUILD_DIR=build-cpu`, and the recorded summary SHA-256 was reproduced.

## Proposed component status

APOLLO, TITAN COUNCIL, AURORA, PROMETHEUS-X, MERCURY, ORBIT, HELIOS, SHADOW, PHOENIX, VAULT, VEILBREAK, AUTO, and SKYFORGE remain proposed only. ATLAS, CHRONOS, and BLACKSTAR remain reserved without accepted definitions. None is claimed as implemented or benchmarked.
