# K3X Current Project State

## Current milestone

Milestone 2 implementation and local measurement are complete. Publication review, the public branch, and CI remain in progress.

State recorded on 2026-08-09 after B-0003 measurement on branch `codex/milestone-two-residency`.

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

## Work in progress

- The TITAN Ledger and README are synchronized with B-0003.
- Final read-only review found no Critical or Important issue. Commit the measurement artifacts and documents, push the public branch, open a PR, and verify Linux CI.
- Worktree: `C:\Users\jolib\Documents\project-k3x\.worktrees\milestone-one-runtime`.
- Linux Python environment: `/home/jolib/.venvs/k3x-m1`; builds: `build-linux` and `build-cuda`.

## Known failures and blockers

- Windows Smart App Control still blocks unsigned `k3x_run.exe`; WSL2 is the verified local CUDA path and native Linux remains the final performance authority.
- The executable checkpoint is synthetic and tiny. No full Kimi K3 weights have been downloaded, and B-0003 is not a full-model throughput claim.
- The graph remains CPU-driven. Activation/result transfers, host activation and routing work, and frequent operation boundaries remain.
- Static residency has no eviction, L1 RAM tier, L2 NVMe tier, prefetch, pinned-memory overlap, or deadline scheduler.
- `cuda-dense` intentionally keeps native MXFP4 on the CPU as its documented comparison identity. `cuda-custom` is the exact GPU MXFP4 path.
- GPU utilization, GPU memory bandwidth, NVMe GB/token, I/O stall time, and system-wide transfer overlap remain unmeasured.
- Full-model quality, coding/agentic quality, adaptive Top-K, cold rescue, speculation, proxy, and pruning remain unimplemented or unmeasured.

## Next concrete tasks

1. Finish public Milestone 2 review, PR, and Linux CI.
2. Design and measure a wider layer/block GPU execution boundary that keeps intermediate activations on device.
3. Build the first asynchronous L0/L1 transfer pipeline only after the wider boundary exposes representative transfer deadlines.
4. Add full-dimension bounded checkpoint slices before any full Kimi K3 throughput claim.
5. Continue with expert cache policies, task/session profiles, adaptive Top-K, and exact rescue in charter order.

## Hardware assumptions

| Component | Current assumption or observation |
|---|---|
| CPU | AMD Ryzen 7 9800X3D target and local host |
| GPU | NVIDIA GeForce RTX 5080, 16,303 MiB, compute capability 12.0 |
| Driver | 591.86 |
| CUDA | Toolkit 13.3.1, nvcc 13.3.73, native `sm_120` |
| RAM | 96 GB DDR5-4200 target; WSL2 exposed 49,251,213,312 bytes during earlier validation |
| NVMe | Solidigm P44 Pro 2 TB target; not measured in Milestone 2 |
| Final runtime OS | Linux native |
| Current development OS | WSL2 Ubuntu 24.04.4 on Windows 11 |

## Latest measured bottleneck

B-0003 measures `cuda-dense` FP32 reference, reuse, residency, and grouped decode at 12.1261, 17.4560, 18.0041, and 17.9018 tok/s. `cuda-custom` measures 12.2647, 17.1425, 17.2723, and 16.8348 tok/s. Reusable allocation removes most allocation churn, and static residency reduces weight H2D by about 88.5–88.9%.

Grouping reduces activation H2D by 21.86–23.74% and synchronization by 19.23–22.86%, but it is 0.57–2.53% slower than scalar residency. CUDA-event kernel time remains only 16.02–19.01 ms for the fastest scalar-residency runs while end-to-end decode spans hundreds of milliseconds. The next measured bottleneck is the narrow CPU/GPU operation boundary and CPU-resident graph, not redundant weight upload or allocation alone.

The derived uncached full-model expert traffic remains 25.83 GB/token, but it is not a measured full-model value. Native-Linux NVMe traffic, cache reuse, and full Kimi K3 throughput remain unknown.

## Last known-good state

- Public `main`: `254a9acf8d62682693e2ce0bde37008ee69e8caf`; prior Linux workflow `31259325702` succeeded.
- Latest measured code commit: `a468db8` (`feat: add CUDA residency ablation reporting`).
- CPU verification: CTest 5/5; pytest 65 passed and 23 CUDA-only skipped.
- CUDA verification: CTest 9/9; pytest 87 passed and one CPU-build-only skipped.
- Compute Sanitizer: `test_cuda_memory`, `test_cuda_residency`, `test_cuda_dense`, and `test_cuda_mxfp4` each report zero errors.
- Exact generated tokens: `[43, 32, 28, 49, 9, 28]` across the complete CUDA option matrix and BF16 fully enabled modes.
- B-0003 raw JSON/CSV: `results/m2-cuda-dense/` and `results/m2-cuda-custom/`.

## Proposed component status

APOLLO, TITAN COUNCIL, AURORA, PROMETHEUS-X, MERCURY, ORBIT, HELIOS, SHADOW, PHOENIX, VAULT, VEILBREAK, AUTO, and SKYFORGE remain proposed only. ATLAS, CHRONOS, and BLACKSTAR remain reserved without accepted definitions. None is claimed as implemented or benchmarked.
