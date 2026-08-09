# K3X Current Project State

## Current milestone

Milestone 5 bounded persistent L1 expert-cache implementation and public integration are complete. Milestone 6 independent L2 reader is in the accepted-design and implementation-planning stage.

State recorded on 2026-08-09 after public Milestone 5 ledger closure `a7e8acf`; final main correctness run `31302070711` and branch run `31302069225` passed. Active branch is `codex/milestone-six-l2-reader`.

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

## Work in progress

- Milestone 5 code and durable measurement documents are public on `main`.
- Milestone 6 Reader hot-path investigation, primary Linux/liburing reference review, accepted design, B-0007 boundary, checklist, and implementation plan are complete.
- Static L1 admission is experimental and opt-in. LRU, LFU, Least-Stale, eviction, task/session priors, prediction, and L2 async I/O remain unimplemented.
- The proposed L2 reader has independent `pread|io_uring` and `buffered|direct` axes plus an ordered batch contract; no new data-plane implementation is claimed yet.
- The TITAN Ledger, README, checklist, context notes, and compact/raw B-0005 artifacts are synchronized with the measured Milestone 4 implementation.
- Final Terra high review found two Important Milestone 5 gaps: generation-local cache lifetime and insufficient native payload admission validation. Commit `2a0cb27` adds explicit `RuntimeSession` ownership and strict native group-32/triplet validation. No Critical or Important finding remains unaddressed.
- Public PR #4 merged by ancestry-verified fast-forward at `c961026`; post-merge correctness run `31298966035` succeeded.
- Public PR #5 merged by ancestry-verified fast-forward at `dac4ed0`; both pre-merge correctness runs and post-merge correctness run `31301960103` succeeded.
- Worktree: `C:\Users\jolib\Documents\project-k3x\.worktrees\milestone-one-runtime`.
- Linux Python environment: `/home/jolib/.venvs/k3x-m1`; builds: `build-cpu` and `build-cuda`.

## Known failures and blockers

- Windows Smart App Control still blocks unsigned `k3x_run.exe`; WSL2 is the verified local CUDA path and native Linux remains the final performance authority.
- The executable checkpoint is synthetic and tiny. No full Kimi K3 weights have been downloaded, and B-0005 is not a full-model throughput claim.
- The graph remains CPU-driven outside FFN blocks. KDA, MLA, routing, score mixing, residual work, state management, and non-FFN boundaries remain on the host.
- L1 misses still perform synchronous K3X extent reads into pageable host vectors. There is no asynchronous L2 NVMe path, eviction policy, predictor, or deadline scheduler.
- Exact prefetch is single-flight and limited to `cuda-custom + ffn-block + reused + transient`; it is not combined with static residency.
- `cuda-dense` intentionally keeps native MXFP4 on the CPU as its documented comparison identity. `cuda-custom` is the exact GPU MXFP4 path.
- GPU utilization, GPU memory bandwidth, NVMe GB/token, and storage I/O stall time remain unmeasured. L1-to-L0 staging, device-copy, readiness, wait, and exposed-stall counters are measured.
- WSL2 exposes the repository through 9p/DrvFS and has no liburing development package installed. It is a correctness environment, not native P44 Pro performance authority.
- Full-model quality, coding/agentic quality, adaptive Top-K, cold rescue, speculation, proxy, and pruning remain unimplemented or unmeasured.

## Next concrete tasks

1. Implement the ordered batch contract and persistent buffered `pread` baseline with red-first Reader/model tests.
2. Batch the six native expert extents without weakening atomic L1 admission or exact transient bypass.
3. Add optional liburing and explicit `O_DIRECT` capability paths, then connect counters and B-0007.
4. Run native-Linux P44 Pro measurements only when that environment is actually available; keep WSL2 evidence non-authoritative.
5. Add full-dimension bounded checkpoint slices, then continue with policy/Least-Stale, task/session profiles, adaptive Top-K, and exact rescue in charter order.

## Hardware assumptions

| Component | Current assumption or observation |
|---|---|
| CPU | AMD Ryzen 7 9800X3D target and local host |
| GPU | NVIDIA GeForce RTX 5080, 16,303 MiB, compute capability 12.0 |
| Driver | 591.86 |
| CUDA | Toolkit 13.3.1, nvcc 13.3.73, native `sm_120` |
| RAM | 96 GB DDR5-4200 target; WSL2 exposed 49,251,213,312 bytes during earlier validation |
| NVMe | Solidigm P44 Pro 2 TB target; not measured in Milestone 4 |
| Final runtime OS | Linux native |
| Current development OS | WSL2 Ubuntu 24.04.4 on Windows 11 |

## Latest measured bottleneck

B-0006 measures FP32 disabled/static synchronous at 16.5587/47.6845 decode tok/s and prefetch at 16.7636/50.6235. BF16 disabled/static synchronous measures 16.4052/47.7956 and prefetch 16.5073/47.6198. Static admission remains opt-in because this roughly 2.88–3.02x difference is a tiny repeated-route WSL2 graph result, not a full-model projection.

Static rows admit 18 complete experts into 29,376 bytes, record 36 hits and zero bypasses, and reduce logical Reader calls/bytes from 428/665,616 to 212/606,864. Exact tokens, routing, H2D, D2H, FFN work, and synchronization remain unchanged. These counters do not measure physical NVMe traffic.

The next measured boundary is representative L2 behavior and admission under real capacity pressure. The synthetic cache fits every observed expert, so it cannot select eviction or Least-Stale policy. Native-Linux physical NVMe traffic, storage I/O stall, buffered versus `io_uring`/`O_DIRECT`, and full-dimension expert sizes remain unknown. CPU-driven KDA/MLA, routing, residual/state, and non-FFN orchestration also remain visible bottlenecks.

The derived uncached full-model expert traffic remains 25.83 GB/token, but it is not a measured full-model value. Native-Linux NVMe traffic, cache reuse, and full Kimi K3 throughput remain unknown.

## Last known-good state

- Public Milestone 5 ledger closure `main`: `a7e8acf7011e1583a0fb561e4bf93b093302e796`; correctness run `31302070711` succeeded.
- B-0006 measurement code commit: `2a0cb27` (`fix: persist and validate L1 experts`).
- B-0006 raw/compact result commit: `fe328e4` (`bench: refresh B-0006 after review fixes`).
- Latest local validation commit: `fe328e4`.
- CPU verification: CTest 6/6; pytest 117 passed and 34 CUDA-only skipped.
- CUDA verification: CTest 15/15; pytest 150 passed and one CPU-build-only skipped.
- Compute Sanitizer: `test_cuda_device`, `test_cuda_dense`, `test_cuda_mxfp4`, `test_cuda_memory`, `test_cuda_pinned_memory`, `test_cuda_async_pipeline`, `test_cuda_residency`, `test_cuda_situ`, `test_cuda_ffn`, and `test_cuda_async_ffn` each report `ERROR SUMMARY: 0 errors`.
- Exact generated tokens: `[43, 32, 28, 49, 9, 28]` and the same 24-entry routing trace across all B-0006 rows.
- B-0006 artifact SHA-256: `077e10a3ba478e83ac8dfd2509ea51a6ea2bfdfe670b60fcadc7f74b97ff810c`; converter maximum source read: 257 bytes.
- B-0006 compact manifest: `results/b0006-l1-cache.json`; raw JSON/CSV: `results/b0006-l1-cache-fp32/` and `results/b0006-l1-cache-bf16/`.

## Proposed component status

APOLLO, TITAN COUNCIL, AURORA, PROMETHEUS-X, MERCURY, ORBIT, HELIOS, SHADOW, PHOENIX, VAULT, VEILBREAK, AUTO, and SKYFORGE remain proposed only. ATLAS, CHRONOS, and BLACKSTAR remain reserved without accepted definitions. None is claimed as implemented or benchmarked.
