# K3X Current Project State

## Current milestone

Milestone 2 — reusable CUDA allocation, bounded exact weight residency, and same-input projection batching.

Milestone 1 implementation, measurement, public merge, and Linux CI are complete. The user approved the staged Milestone 2 design; its written specification is under review and implementation has not started.

State recorded after the 2026-08-08 Milestone 2 design approval on branch `codex/milestone-two-residency`.

## Completed work

- Milestone 0 deterministic synthetic K3-compatible PyTorch graph.
- KDA, Gated MLA, Attention Residual, Stable LatentMoE, router, native MXFP4, incremental state, and greedy generation tests.
- K3X v1 streaming converter, strict Python/C++ reader, aligned extents, CRC32C, root SHA-256, crash-safe resume, and stale-ledger recovery.
- Independent portable C++20 CPU synthetic runtime.
- Python/C++ layer, logits, state, and exact-token parity.
- Reproducible synthetic benchmark driver with JSON/CSV.
- Public repository and Linux correctness workflow.
- Approved Milestone 1 design and detailed TDD implementation plan.
- TITAN LEDGER charter, architecture registry, decision ledger, benchmark ledger, and state protocol.
- WSL 2.7.11.0 with Ubuntu 24.04.4 LTS, Linux 6.18.33.2, CUDA Toolkit 13.3.1, and RTX 5080 passthrough.
- Linux Release baseline: CTest 2/2 and pytest 39/39 before cross-language path changes.
- Cross-language native binary resolution through optional `K3X_BUILD_DIR`, verified by the complete Linux pytest suite.
- Deterministic runtime `ProfileEvent` storage and successful-work aggregation, verified without clock, thread, JSON, or CUDA dependencies.
- Explicit `ComputeBackend` boundary for exact CPU dense and native MXFP4 matrix operations, with compatibility and backend-selected generation overloads.
- Optional CUDA 13.3 build with native `sm_120`, RTX 5080 capability validation, nonblocking stream and cuBLASLt RAII ownership, plus a CUDA-free OFF stub.
- Release native tests use explicit return codes so `NDEBUG` cannot remove their behavior checks.
- Row-major cuBLASLt dense matvec for FP32 and BF16-rounded operands with FP32 accumulation/output, zero-workspace heuristic selection, exact directional transfer-byte accounting, CUDA-event timing, and per-call device-memory accounting.
- Exact native-byte K3 MXFP4 CUDA matvec with low-nibble-first E2M1 decode, E8M0/32 scaling, FP32 accumulation, stride coverage, typed validation, transfer accounting, and CUDA-event timing.
- Explicit `cpu`, `cuda-dense`, and `cuda-custom` runtime selection plus FP32/BF16 dense precision selection without silent fallback.
- End-to-end CPU/CUDA layer, logits, recurrent state, and exact-token parity on the deterministic K3X artifact.
- JSON/CSV export of backend, device, precision, kernel time, directional transfer bytes, peak backend-owned VRAM, logical reads, layer timing, and numerical error.
- B-0002 synthetic comparison with three warmups and 20 samples for CPU and both CUDA backends in FP32, plus both CUDA backends in BF16.
- Milestone 1 fast-forward merge to public `main` at `254a9ac` and successful post-merge Linux correctness run `31259325702`.
- Approved Milestone 2 design covering three independently switchable allocation, residency, and batching axes with B-0002 reference preservation.

## Work in progress

- Branch: `codex/milestone-two-residency`.
- Worktree: `C:\Users\jolib\Documents\project-k3x\.worktrees\milestone-one-runtime`.
- The written Milestone 2 design is complete and awaiting the required user review before a TDD implementation plan is written.
- Linux development runs as the unprivileged `jolib` user. The isolated Python environment is `/home/jolib/.venvs/k3x-m1`; repository build output is `build-linux`.

## Known failures and blockers

- Windows Smart App Control blocks the newly linked unsigned `build/k3x_run.exe` before process creation.
- Code Integrity events 3033 and 3077 cite policy `{0283ac0f-fff1-49ae-ada1-8a933130cad6}` and an unmet Enterprise signing level.
- Fresh Windows CTest binaries run, but five Python cross-language cases that launch `k3x_run.exe` remain blocked on Windows. This does not block the verified WSL Linux path.
- CUDA calls allocate and transfer per operation in the current correctness baseline; persistent residency and asynchronous overlap remain unimplemented.
- Milestone 2 implementation is intentionally gated on review of the accepted written specification; no production code is claimed yet.
- `cuda_dense` intentionally uses the CPU MXFP4 oracle with zero device traffic; this is its documented comparison contract, not an automatic fallback after a CUDA failure.
- Direct cuBLASLt FP4 is rejected for exact K3 MXFP4 because NVIDIA requires UE4M3 scales per 16 FP4 values while K3 uses E8M0 scales per 32 values.

## Next concrete tasks

1. Complete user review of the Milestone 2 written specification.
2. Write and commit the detailed TDD implementation plan.
3. Implement and independently ablate reusable allocation, exact static residency, and grouped projections.
4. Remeasure CPU, CUDA dense, and CUDA custom before selecting any default.
5. Use the measured remaining boundary cost to choose a wider GPU executor or the first L0/L1 asynchronous transfer pipeline.

## Hardware assumptions

| Component | Current assumption or observation |
|---|---|
| CPU | AMD Ryzen 7 9800X3D target |
| GPU | Locally observed NVIDIA GeForce RTX 5080, 16,303 MiB, compute capability 12.0 |
| Driver | Locally observed 591.86 |
| CUDA | WSL-installed toolkit 13.3.1 and nvcc 13.3.73 with `sm_120` support |
| RAM | 96 GB DDR5-4200 target |
| NVMe | Solidigm P44 Pro 2 TB target |
| Final runtime OS | Linux native |
| Current development OS | WSL2 Ubuntu 24.04.4 on Windows 11; Windows Smart App Control remains enforced |

## Latest measured bottleneck

B-0002 measures the tiny synthetic graph at 19.4858 CPU, 11.6682 FP32 `cuda-dense`, and 10.1118 FP32 `cuda-custom` decode tok/s. BF16 halves H2D bytes and backend-owned peak VRAM but does not improve decode throughput materially. CUDA-event work totals only 11.56--14.52 ms per run while end-to-end graph work takes hundreds of milliseconds. The latest measured bottleneck is per-operation allocation, host staging, synchronous copy/synchronization, and CPU-resident graph execution, not the CUDA kernel arithmetic.

This does not replace the derived full-model traffic model. Uncached natural Top-16 expert reads across 92 MoE layers still imply 25.83 GB/token, but native-Linux NVMe traffic, cache reuse, I/O stalls, GPU utilization, and full-model throughput remain unmeasured.

## Last known-good state

- Public `main`: `254a9acf8d62682693e2ce0bde37008ee69e8caf`.
- GitHub Actions correctness run: `31259325702`, success on Linux after the Milestone 1 merge.
- Public-main local tests: CPU CTest 5/5 and pytest 53 passed; CUDA CTest 7/7 and pytest 59 passed.
- Milestone 1 design commit: `84edc6e`.
- Linux environment: WSL 2.7.11.0, Ubuntu 24.04.4, kernel 6.18.33.2, CUDA Toolkit 13.3.1, nvcc 13.3.73, RTX 5080 compute capability 12.0.
- Current worktree code commit: `c92f498` (`feat: profile backend-selected synthetic inference`).
- CPU-only CTest: 5/5 pass; `k3x_run` has no CUDA or cuBLAS dynamic dependency.
- CUDA CTest: 7/7 pass on RTX 5080; cuBLASLt FP32/BF16-rounded and exact native-byte MXFP4 literal checks pass, and MXFP4 `compute-sanitizer --tool memcheck` reports 0 errors.
- CPU-only pytest: 53 passed and 7 CUDA-only cases skipped with `K3X_BUILD_DIR=build-cpu`.
- CUDA-enabled pytest: 59 passed and 1 CPU-build-only case skipped with `K3X_BUILD_DIR=build-cuda`; FP32/BF16 CUDA graph parity and benchmark schema tests are included.
- B-0002 raw JSON/CSV records are generated for CPU FP32 and both CUDA identities in FP32/BF16.

## Proposed component status

APOLLO, TITAN COUNCIL, AURORA, PROMETHEUS-X, MERCURY, ORBIT, HELIOS, SHADOW, PHOENIX, VAULT, VEILBREAK, AUTO, and SKYFORGE are proposed only. ATLAS, CHRONOS, and BLACKSTAR are reserved names without accepted definitions. None is claimed as implemented or benchmarked.
