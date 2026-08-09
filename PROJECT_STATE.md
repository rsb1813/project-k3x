# K3X Current Project State

## Current milestone

Milestone 4 exact asynchronous L1-to-L0 transfer implementation, B-0005 measurement, and final read-only review are complete. Public PR/CI integration is in progress.

State recorded on 2026-08-09 after review fix `190459b`, complete post-fix CPU/CUDA suites, affected CUDA memcheck targets, and FP32/BF16 post-fix smokes passed. B-0005 remains the 3-warmup/20-sample measurement at `99cf1e4` because the valid execution order is unchanged.

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

## Work in progress

- Milestone 4 code, review fixes, and measurement artifacts are locally complete. Public PR/CI integration remains.
- The TITAN Ledger, README, checklist, context notes, and compact/raw B-0005 artifacts are synchronized with the measured Milestone 4 implementation.
- Final Terra high review found three Important contract/test gaps. Commit `190459b` enforces use-sequence identity before side effects, strengthens failure-atomicity coverage, and requires matched H2D/synchronization equality. No Critical or Important finding remains unaddressed.
- Public `main` remains at the completed Milestone 3 ancestry until the Milestone 4 PR passes review and CI.
- Worktree: `C:\Users\jolib\Documents\project-k3x\.worktrees\milestone-one-runtime`.
- Linux Python environment: `/home/jolib/.venvs/k3x-m1`; builds: `build-cpu` and `build-cuda`.

## Known failures and blockers

- Windows Smart App Control still blocks unsigned `k3x_run.exe`; WSL2 is the verified local CUDA path and native Linux remains the final performance authority.
- The executable checkpoint is synthetic and tiny. No full Kimi K3 weights have been downloaded, and B-0005 is not a full-model throughput claim.
- The graph remains CPU-driven outside FFN blocks. KDA, MLA, routing, score mixing, residual work, state management, and non-FFN boundaries remain on the host.
- The implemented prefetch starts after synchronous K3X extent reads into pageable host vectors. There is no persistent L1 expert cache, asynchronous L2 NVMe path, eviction policy, predictor, or deadline scheduler.
- Exact prefetch is single-flight and limited to `cuda-custom + ffn-block + reused + transient`; it is not combined with static residency.
- `cuda-dense` intentionally keeps native MXFP4 on the CPU as its documented comparison identity. `cuda-custom` is the exact GPU MXFP4 path.
- GPU utilization, GPU memory bandwidth, NVMe GB/token, and storage I/O stall time remain unmeasured. L1-to-L0 staging, device-copy, readiness, wait, and exposed-stall counters are measured.
- Full-model quality, coding/agentic quality, adaptive Top-K, cold rescue, speculation, proxy, and pruning remain unimplemented or unmeasured.

## Next concrete tasks

1. Commit the synchronized Milestone 4 ledger/results, publish the branch through PR/CI, verify ancestry, fast-forward public `main`, and confirm post-merge CI.
2. Design a bounded persistent L1 expert cache that owns exact source bytes independently of the current temporary reader vectors and can feed the existing prepared-transfer boundary.
3. Add an independently switchable L2 NVMe reader and benchmark buffered I/O, `io_uring`, and `O_DIRECT` on native Linux before choosing a default.
4. Add full-dimension bounded checkpoint slices before any full Kimi K3 throughput claim, then implement deadline-aware L2-to-L1-to-L0 scheduling.
5. Continue with Least-Stale/cache policies, task/session profiles, adaptive Top-K, and exact rescue in charter order.

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

B-0005 measures FP32 synchronous/prefetch scalar at 16.9701/16.7947 decode tok/s and grouped at 16.7055/16.7914. BF16 scalar measures 16.6366/16.5735 and grouped 16.5529/16.7021. Matched prefetch changes range from -1.03% to +0.90%, so synchronous transfer remains the default.

Every prefetch row performs 27 exact prepares and waits, with all 27 transfers ready before use, unchanged H2D bytes, and no additional host synchronization. The fixed pipeline adds 1 MiB of pinned host staging and 1,048,032 bytes of peak device staging, while exposed transfer stall is 0.198--0.312 ms per run. The narrow overlap mechanism is working, but the synthetic graph does not provide enough expert transfer work for a stable end-to-end gain.

The latest measured bottleneck is now the boundary before prefetch: synchronous K3X file-to-pageable-host materialization and the lack of persistent L1 residency, together with the still CPU-driven KDA/MLA, routing, residual/state, and non-FFN orchestration. Native-Linux L2 NVMe traffic and storage I/O stall remain unknown.

The derived uncached full-model expert traffic remains 25.83 GB/token, but it is not a measured full-model value. Native-Linux NVMe traffic, cache reuse, and full Kimi K3 throughput remain unknown.

## Last known-good state

- Public Milestone 3 `main`: `b6c3d473ffe99d009d75a3e909d8681bc110bca3`; implementation merge `5de2514` passed post-merge correctness run `31295154288`, and `b6c3d47` then closed the ledger.
- B-0005 measurement commit: `99cf1e4164510824ee67755c410b74887793fa8a` (`feat: add asynchronous transfer ablation`).
- Latest validated code commit: `190459b` (`fix: enforce async transfer identity invariants`); the valid B-0005 execution path and ordering are unchanged.
- CPU verification: CTest 5/5; pytest 98 passed and 27 CUDA-only skipped.
- CUDA verification: CTest 14/14; pytest 124 passed and one CPU-build-only skipped.
- Compute Sanitizer: `test_cuda_device`, `test_cuda_dense`, `test_cuda_mxfp4`, `test_cuda_memory`, `test_cuda_pinned_memory`, `test_cuda_async_pipeline`, `test_cuda_residency`, `test_cuda_situ`, `test_cuda_ffn`, and `test_cuda_async_ffn` each report `ERROR SUMMARY: 0 errors`.
- Post-review Compute Sanitizer repetition: affected `test_cuda_async_pipeline` and `test_cuda_async_ffn` each report `ERROR SUMMARY: 0 errors`.
- Exact generated tokens: `[43, 32, 28, 49, 9, 28]` and the same 24-entry routing trace across all B-0005 rows.
- B-0005 artifact SHA-256: `e245c52759dffcfaccfe182bbba56fa069288d99f0d70a1cd779169bb51e6993`; converter maximum source read: 257 bytes.
- B-0005 compact manifest: `results/b0005-async-transfer.json`; raw JSON/CSV: `results/b0005-async-transfer-fp32/` and `results/b0005-async-transfer-bf16/`.

## Proposed component status

APOLLO, TITAN COUNCIL, AURORA, PROMETHEUS-X, MERCURY, ORBIT, HELIOS, SHADOW, PHOENIX, VAULT, VEILBREAK, AUTO, and SKYFORGE remain proposed only. ATLAS, CHRONOS, and BLACKSTAR remain reserved without accepted definitions. None is claimed as implemented or benchmarked.
