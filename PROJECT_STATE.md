# K3X Current Project State

## Current milestone

Milestone 2 implementation, measurement, public merge, and Linux CI are complete. Milestone 3 implementation, full verification, Compute Sanitizer validation, B-0004 measurement, and final read-only review are complete; public integration is in progress.

State recorded on 2026-08-09 on branch `codex/milestone-three-ffn-blocks` after selecting the dependency-closed CUDA FFN boundary.

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

## Work in progress

- The TITAN Ledger, README, checklist, context notes, and compact/raw B-0004 artifacts are synchronized with the measured Milestone 3 implementation.
- Final Terra high read-only review found one Important fixed-group validation gap. Commit `3df8d3f` rejects non-32 MXFP4 FFN views before side effects and adds the group-size 16/64 regression. Documentation commit, public PR, Linux CI, and merge remain in the publication gate.
- Worktree: `C:\Users\jolib\Documents\project-k3x\.worktrees\milestone-one-runtime`.
- Linux Python environment: `/home/jolib/.venvs/k3x-m1`; builds: `build-linux` and `build-cuda`.

## Known failures and blockers

- Windows Smart App Control still blocks unsigned `k3x_run.exe`; WSL2 is the verified local CUDA path and native Linux remains the final performance authority.
- The executable checkpoint is synthetic and tiny. No full Kimi K3 weights have been downloaded, and B-0004 is not a full-model throughput claim.
- The graph remains CPU-driven outside FFN blocks. KDA, MLA, routing, score mixing, residual work, state management, and non-FFN boundaries remain on the host.
- Static residency has no eviction, L1 RAM tier, L2 NVMe tier, prefetch, pinned-memory overlap, or deadline scheduler.
- `cuda-dense` intentionally keeps native MXFP4 on the CPU as its documented comparison identity. `cuda-custom` is the exact GPU MXFP4 path.
- GPU utilization, GPU memory bandwidth, NVMe GB/token, I/O stall time, and system-wide transfer overlap remain unmeasured.
- Full-model quality, coding/agentic quality, adaptive Top-K, cold rescue, speculation, proxy, and pruning remain unimplemented or unmeasured.

## Next concrete tasks

1. Complete the Milestone 3 read-only review, public PR, Linux CI, and verified merge.
2. Design and implement the first exact asynchronous L0/L1 transfer pipeline against the now-measured FFN block deadline boundary while preserving a synchronous reference mode.
3. Add pinned host staging, explicit transfer deadlines, overlap instrumentation, and deterministic no-output-change tests before selecting an asynchronous default.
4. Add full-dimension bounded checkpoint slices before any full Kimi K3 throughput claim.
5. Continue with L2 NVMe integration, expert cache policies, task/session profiles, adaptive Top-K, and exact rescue in charter order.

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

B-0004 measures the FP32 operation-scalar reference at 16.3576 tok/s and the FP32 FFN-block scalar path at 17.0713 tok/s, a 4.36% decode increase. The block path reduces activation H2D from 126,144 to 92,736 bytes, D2H from 111,600 to 83,952 bytes, and synchronization from 630 to 423 while preserving exact generated tokens and routing trace.

The FP32 FFN-block grouped path reaches 17.0270 tok/s, so scalar is the synthetic experimental recommendation and operation remains the correctness default. BF16 block scalar reaches 16.9847 tok/s with maximum absolute error 0.00402409 and does not displace FP32. CUDA-event kernel time rises from 20.35 ms to 22.24 ms in the FP32 scalar comparison, so the remaining measured bottleneck is CPU-driven KDA/MLA, routing, residual and state work, non-FFN transfer boundaries, and launch orchestration rather than FFN activation traffic alone.

The derived uncached full-model expert traffic remains 25.83 GB/token, but it is not a measured full-model value. Native-Linux NVMe traffic, cache reuse, and full Kimi K3 throughput remain unknown.

## Last known-good state

- Public `main`: `cd72613d0da1645e407980758646d4332a9f3225`; Milestone 2 Linux CI succeeded.
- Latest measured code commit: `0f6bbdd` (`feat: add FFN block ablation reporting`).
- Latest validated code commit: `3df8d3f` (`fix: enforce native MXFP4 FFN groups`). The valid group-32 B-0004 path is unchanged.
- CPU verification: CTest 5/5; pytest 70 passed and 26 CUDA-only skipped.
- CUDA verification: CTest 11/11; pytest 95 passed and one CPU-build-only skipped.
- Compute Sanitizer: `test_cuda_dense`, `test_cuda_mxfp4`, `test_cuda_memory`, `test_cuda_residency`, `test_cuda_situ`, and `test_cuda_ffn` each report zero errors.
- Exact generated tokens: `[43, 32, 28, 49, 9, 28]` across the complete CUDA option matrix and BF16 fully enabled modes.
- B-0004 artifact SHA-256: `59c1f83f571fb59dcdad27ef80da8d42b03176dfb5fa63ae5195717c141775ed`.
- B-0004 compact manifest: `results/b0004-ffn-blocks.json`; raw JSON/CSV: `results/b0004-ffn-blocks-fp32/` and `results/b0004-ffn-blocks-bf16/`.

## Proposed component status

APOLLO, TITAN COUNCIL, AURORA, PROMETHEUS-X, MERCURY, ORBIT, HELIOS, SHADOW, PHOENIX, VAULT, VEILBREAK, AUTO, and SKYFORGE remain proposed only. ATLAS, CHRONOS, and BLACKSTAR remain reserved without accepted definitions. None is claimed as implemented or benchmarked.
