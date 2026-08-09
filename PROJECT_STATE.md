# K3X Current Project State

## Current milestone

Milestone 5 public integration is complete. Milestone 6 independent exact L2 reader implementation, local verification, and non-authoritative WSL2 B-0007 measurement are complete; final review and public integration are in progress.

State recorded on 2026-08-09 at verified Milestone 6 code commit `5049f26`. Public `main` remains at Milestone 5 ledger closure `a7e8acf`; active branch is `codex/milestone-six-l2-reader`.

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

## Work in progress

- Milestone 6 code and measurement ledger await one final Terra high read-only review and public PR/CI integration.
- Static L1 admission and all non-default L2 modes remain experimental and opt-in. LRU, LFU, Least-Stale, eviction, task/session priors, prediction, and cross-layer asynchronous L2 scheduling remain unimplemented.
- The L2 batch API submits concurrent operations for one batch but waits before returning. It is not the chartered N/N+1/N+2 deadline pipeline yet.
- `pread + buffered` remains the default because B-0007 is WSL2 ext4 evidence, not native P44 Pro evidence.
- Worktree: `C:\Users\jolib\Documents\project-k3x\.worktrees\milestone-one-runtime`.
- Linux Python environment: `/home/jolib/.venvs/k3x-m1`; builds: `build-cpu`, `build-uring`, and `build-cuda`.

## Known failures and blockers

- Windows Smart App Control still blocks unsigned `k3x_run.exe`; WSL2 is the verified local CUDA path and native Linux remains the final performance authority.
- The executable checkpoint is synthetic and tiny. No full Kimi K3 weights have been downloaded, and B-0007 is not a full-model throughput claim.
- The graph remains CPU-driven outside FFN blocks. KDA, MLA, routing, score mixing, residual work, state management, and non-FFN boundaries remain on the host.
- L1 misses use ordered Reader batches into pageable host vectors and wait at the batch boundary. There is no cross-layer asynchronous L2 pipeline, eviction policy, predictor, or deadline scheduler.
- Exact prefetch is single-flight and limited to `cuda-custom + ffn-block + reused + transient`; it is not combined with static residency.
- `cuda-dense` intentionally keeps native MXFP4 on the CPU as its documented comparison identity. `cuda-custom` is the exact GPU MXFP4 path.
- GPU utilization, GPU memory bandwidth, and physical NVMe GB/token remain unmeasured. Reader storage elapsed time, logical/aligned bytes, process `rchar/read_bytes`, and L1-to-L0 timing are measured under their stated scopes.
- WSL2 `/mnt/c` is 9p/DrvFS and rejects direct alignment. liburing 2.5 and direct I/O were validated on WSL2 ext4 `/tmp`, which remains a correctness/capability environment rather than native P44 Pro performance authority.
- Full-model quality, coding/agentic quality, adaptive Top-K, cold rescue, speculation, proxy, and pruning remain unimplemented or unmeasured.

## Next concrete tasks

1. Complete one final Critical/Important read-only review, apply evidence-backed fixes once, and rerun affected verification.
2. Publish Milestone 6 through a public PR, require Linux CI, ancestry-check the merge, and confirm post-merge CI.
3. Add a full-dimension bounded checkpoint slice so storage request sizes and alignment amplification become representative without downloading the full model.
4. Implement deadline-aware cross-layer L2-to-L1 scheduling while preserving the exact blocking Reader mode.
5. Run native-Linux P44 Pro warm/cold B-0007 only when that environment exists, then continue with Least-Stale, task/session profiles, adaptive Top-K, and exact rescue.

## Hardware assumptions

| Component | Current assumption or observation |
|---|---|
| CPU | AMD Ryzen 7 9800X3D target and local host |
| GPU | NVIDIA GeForce RTX 5080, 16,303 MiB, compute capability 12.0 |
| Driver | 591.86 |
| CUDA | Toolkit 13.3.1, nvcc 13.3.73, native `sm_120` |
| RAM | 96 GB DDR5-4200 target; WSL2 exposed 49,251,213,312 bytes during earlier validation |
| NVMe | Solidigm P44 Pro 2 TB target; not measured by B-0007 |
| Final runtime OS | Linux native |
| Current development OS | WSL2 Ubuntu 24.04.4 on Windows 11 |

## Latest measured bottleneck

B-0007 preserves exact tokens, the 24-entry routing trace, 428 logical calls, 158 batches, and 665,616 logical bytes across all four L2 combinations. Buffered `pread`/`io_uring` measure 5,870.8082/5,616.1034 decode tok/s. Direct `pread`/`io_uring` measure 163.3491/428.8471 and submit 756,736 aligned bytes, 13.69% above logical bytes.

The direct rows expose the current tiny-read problem: hundreds of small aligned reads dominate the synthetic CPU graph. io_uring reduces that direct-mode cost relative to serial pread, but buffered pread remains fastest in this warm WSL2 ext4 measurement. The measurement cannot select a P44 Pro native-Linux default, so `pread + buffered` remains unchanged.

The next bottleneck boundary is a full-dimension bounded slice plus deadline-aware cross-layer overlap. Native-Linux physical NVMe traffic, cold/warm behavior, realistic expert sizes, and cache pressure remain unknown. CPU-driven KDA/MLA, routing, residual/state, and non-FFN orchestration also remain visible bottlenecks.

The derived uncached full-model expert traffic remains 25.83 GB/token, but it is not a measured full-model value. Native-Linux NVMe traffic, cache reuse, and full Kimi K3 throughput remain unknown.

## Last known-good state

- Public `main`: Milestone 5 ledger closure `a7e8acf7011e1583a0fb561e4bf93b093302e796`; correctness run `31302070711` succeeded.
- Latest verified Milestone 6 code and B-0007 measurement commit: `5049f26` (`feat: measure and ablate L2 reader modes`).
- CPU verification: CTest 8/8; pytest 135 passed and 40 skipped.
- Liburing/direct verification: CTest 8/8; pytest 136 passed and 39 skipped.
- CUDA verification: CTest 17/17; pytest 168 passed and 7 skipped.
- Compute Sanitizer: `test_cuda_device`, `test_cuda_dense`, `test_cuda_mxfp4`, `test_cuda_memory`, `test_cuda_pinned_memory`, `test_cuda_async_pipeline`, `test_cuda_residency`, `test_cuda_situ`, `test_cuda_ffn`, and `test_cuda_async_ffn` each report `ERROR SUMMARY: 0 errors`.
- Exact generated tokens: `[43, 32, 28, 49, 9, 28]` and the same 24-entry routing trace across all B-0007 rows.
- B-0007 artifact SHA-256: `039d61ee9c2e13e27c9a2514bb476f8b122b8b37be0b7f85baf26c1a6611a2e9`.
- B-0007 raw JSON/CSV and cross-checked manifest: `results/b0007-l2-reader-wsl/`.

## Proposed component status

APOLLO, TITAN COUNCIL, AURORA, PROMETHEUS-X, MERCURY, ORBIT, HELIOS, SHADOW, PHOENIX, VAULT, VEILBREAK, AUTO, and SKYFORGE remain proposed only. ATLAS, CHRONOS, and BLACKSTAR remain reserved without accepted definitions. None is claimed as implemented or benchmarked.
