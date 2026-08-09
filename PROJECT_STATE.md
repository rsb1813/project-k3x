# K3X Current Project State

## Current milestone

Milestone 7 full-dimension bounded expert slice implementation, non-authoritative WSL2 B-0008 measurement, final review, review fixes, and public integration are complete.

State recorded on 2026-08-09 at verified Milestone 7 integration head `e9a5d4d`. Public PR #7 is merged by fast-forward, post-merge main correctness run `31308147456` succeeded, and the preserved active branch is `codex/milestone-seven-bounded-slice`.

## Completed work

- Milestone 0 deterministic synthetic K3-compatible PyTorch graph, K3X v1 streaming converter, strict Python/C++ readers, and independent portable C++20 runtime.
- KDA, Gated MLA, Attention Residual, Stable LatentMoE, router, native MXFP4, full/incremental state, layer/logit/state parity, and exact greedy token tests.
- Public Milestone 1 at `254a9ac` with explicit CPU, `cuda-dense`, and `cuda-custom` identities; CUDA 13.3 native `sm_120`; cuBLASLt dense FP32/BF16; exact native-byte MXFP4 CUDA; deterministic JSON/CSV profiling; and B-0002.
- Milestone 2 runtime switches for `per-operation|reused`, `transient|resident`, and `scalar|grouped`, with reference defaults preserved.
- Tracked CUDA allocations, grow-only scratch buffers, reusable CUDA events and cuBLASLt plans, and exact live/peak VRAM counters.
- Stable tensor identities and bounded static FP32, BF16, and native MXFP4 weight residency. Capacity misses bypass to the exact transient path; no eviction policy exists yet.
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

## Work in progress

- Milestone 7 Terra high final review found no Critical issue and two Important source/resume integrity gaps. Commit `d929f57` closes both with nine added regressions; no Critical or Important finding remains unaddressed.
- Static L1 admission and all non-default L2 modes remain experimental and opt-in. LRU, LFU, Least-Stale, eviction, task/session priors, prediction, and cross-layer asynchronous L2 scheduling remain unimplemented.
- The L2 batch API submits concurrent operations for one batch but waits before returning. It is not the chartered N/N+1/N+2 deadline pipeline yet.
- `pread + buffered` remains the default because B-0007/B-0008 are WSL2 ext4 evidence, not native P44 Pro evidence.
- Worktree: `C:\Users\jolib\Documents\project-k3x\.worktrees\milestone-one-runtime`.
- Linux Python environment: `/home/jolib/.venvs/k3x-m1`; builds: `build-cpu`, `build-uring`, and `build-cuda`.

## Known failures and blockers

- Windows Smart App Control still blocks unsigned `k3x_run.exe`; WSL2 is the verified local CUDA path and native Linux remains the final performance authority.
- The executable checkpoint is synthetic and tiny. The full-dimension artifact contains only one non-executable expert. No full Kimi K3 weights have been downloaded, and B-0008 is neither token throughput nor a full-model claim.
- The graph remains CPU-driven outside FFN blocks. KDA, MLA, routing, score mixing, residual work, state management, and non-FFN boundaries remain on the host.
- L1 misses use ordered Reader batches into pageable host vectors and wait at the batch boundary. There is no cross-layer asynchronous L2 pipeline, eviction policy, predictor, or deadline scheduler.
- Exact prefetch is single-flight and limited to `cuda-custom + ffn-block + reused + transient`; it is not combined with static residency.
- `cuda-dense` intentionally keeps native MXFP4 on the CPU as its documented comparison identity. `cuda-custom` is the exact GPU MXFP4 path.
- GPU utilization, GPU memory bandwidth, and physical NVMe GB/token remain unmeasured. Reader storage elapsed time, logical/aligned bytes, process `rchar/read_bytes`, and L1-to-L0 timing are measured under their stated scopes.
- WSL2 `/mnt/c` is 9p/DrvFS and rejects direct alignment. liburing 2.5 and direct I/O were validated on WSL2 ext4 `/tmp`, which remains a correctness/capability environment rather than native P44 Pro performance authority.
- Full-model quality, coding/agentic quality, adaptive Top-K, cold rescue, speculation, proxy, and pruning remain unimplemented or unmeasured.

## Next concrete tasks

1. Design deadline-aware cross-layer L2-to-L1 scheduling while preserving the exact blocking Reader mode.
2. Implement the smallest synthetic N/N+1/N+2 overlap boundary with an exact synchronous reference switch.
3. Add a bounded multi-expert or full-layer slice when cache-pressure and physical-locality experiments require it.
4. Run native-Linux P44 Pro warm/cold B-0008 only when that environment exists.
5. Continue with Least-Stale, task/session profiles, adaptive Top-K, and exact rescue in charter order.

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

B-0008 physically confirms 17,547,264 bytes per released-dimension native MXFP4 expert. All four L2 combinations preserve the same ordered digest, 120 completions, and 350,945,280 logical/submitted bytes across 20 loads with zero failures. Median wall/Reader latency is 50.685/4.661 ms for buffered pread, 51.592/4.579 ms for buffered io_uring, 60.402/14.832 ms for direct pread, and 56.426/11.633 ms for direct io_uring.

Actual expert data and scale lengths are divisible by the WSL2 ext4 512-byte direct alignment, eliminating B-0007's tiny-extent byte amplification for this shape. Buffered pread has the best wall median, while buffered io_uring has the lowest Reader-only time and direct io_uring beats direct pread. Allocation and SHA-256 dominate wall latency, and no controlled cold-cache preparation was used. The measurement cannot select a P44 Pro native-Linux default, so `pread + buffered` remains unchanged.

The next bottleneck boundary is deadline-aware cross-layer overlap plus representative multi-expert cache pressure. Native-Linux physical NVMe traffic, controlled cold/warm behavior, locality, and eviction remain unknown. CPU-driven KDA/MLA, routing, residual/state, and non-FFN orchestration also remain visible bottlenecks.

The derived uncached full-model expert traffic remains 25.83 GB/token, but it is not a measured full-model value. Native-Linux NVMe traffic, cache reuse, and full Kimi K3 throughput remain unknown.

## Last known-good state

- Public `main` Milestone 7 integration head: `e9a5d4da8f0e7e25a8d47e5d60585b30baa7b792`; PR #7 branch/PR runs `31308034386` and `31308046877` succeeded, and post-merge main run `31308147456` succeeded.
- Latest verified Milestone 7 review-fix commit: `d929f5711f21b853cbfd8c0cf031e0d2425d085c` (`fix: verify storage source and resume extents`).
- Latest verified Milestone 7 measurement-code commit: `9198ed2` (`feat: ablate bounded expert storage reads`).
- B-0008 artifact SHA-256: `b14610fd2b405dd97c09004fb29157f5b318522591546337bce89e7e8a6a2b65`; ordered payload digest: `e5fb7939474a57ab9263a791999d76ba078bd767cc3f155f3522b1bec576c7e4`.
- CPU verification: CTest 8/8; pytest 161 passed and 40 skipped.
- Liburing/direct verification: CTest 9/9; pytest 162 passed and 39 skipped; prior ASan/UBSan CTest 9/9 plus four storage-path targeted pytest passes remain applicable because no C++ path changed.
- CUDA verification: CTest 17/17; pytest 194 passed and 7 skipped.
- Compute Sanitizer: `test_cuda_device`, `test_cuda_dense`, `test_cuda_mxfp4`, `test_cuda_memory`, `test_cuda_pinned_memory`, `test_cuda_async_pipeline`, `test_cuda_residency`, `test_cuda_situ`, `test_cuda_ffn`, and `test_cuda_async_ffn` each report `ERROR SUMMARY: 0 errors`.
- Existing executable-model exact generated tokens remain `[43, 32, 28, 49, 9, 28]`; B-0008 executes no tokens.
- B-0008 raw JSON/CSV and cross-checked manifest: `results/b0008-bounded-slice-wsl/`.

## Proposed component status

APOLLO, TITAN COUNCIL, AURORA, PROMETHEUS-X, MERCURY, ORBIT, HELIOS, SHADOW, PHOENIX, VAULT, VEILBREAK, AUTO, and SKYFORGE remain proposed only. ATLAS, CHRONOS, and BLACKSTAR remain reserved without accepted definitions. None is claimed as implemented or benchmarked.
