# K3X Current Project State

## Current milestone

Milestone 1 — exact runtime backend boundary, structured profiler, cuBLASLt dense baseline, and custom K3 MXFP4 CUDA baseline.

The design is approved, and the Linux portability, deterministic profiler, exact CPU backend, and optional CUDA resource-shell tasks are complete. The cuBLASLt dense baseline is next.

State recorded after the 2026-08-08 optional CUDA shell commit and dual-build correctness verification.

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

## Work in progress

- Branch: `feat/milestone-one-runtime`.
- Worktree: `C:\Users\jolib\Documents\project-k3x\.worktrees\milestone-one-runtime`.
- The next code task is literal cuBLASLt FP32 and BF16-rounded dense matvec with exact transfer and device-time profiling.
- Linux development runs as the unprivileged `jolib` user. The isolated Python environment is `/home/jolib/.venvs/k3x-m1`; repository build output is `build-linux`.

## Known failures and blockers

- Windows Smart App Control blocks the newly linked unsigned `build/k3x_run.exe` before process creation.
- Code Integrity events 3033 and 3077 cite policy `{0283ac0f-fff1-49ae-ada1-8a933130cad6}` and an unmet Enterprise signing level.
- Fresh Windows CTest binaries run, but five Python cross-language cases that launch `k3x_run.exe` remain blocked on Windows. This does not block the verified WSL Linux path.
- No CUDA matrix kernel or end-to-end CUDA runtime exists yet, so the resource-shell result is not a CUDA correctness or performance claim for model compute.
- Direct cuBLASLt FP4 is rejected for exact K3 MXFP4 because NVIDIA requires UE4M3 scales per 16 FP4 values while K3 uses E8M0 scales per 32 values.

## Next concrete tasks

1. Implement and measure the cuBLASLt FP32/BF16-rounded dense baseline.
2. Implement the custom native-byte MXFP4 kernel and verify it against the CPU oracle.
3. Instrument the graph with explicit profile events and export JSON/CSV summaries.
4. Run end-to-end synthetic CPU versus CUDA profiling before accepting a default path.

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

Milestone 0 did not measure full-model or NVMe performance. Its only measured run is the tiny CPU synthetic B-0001 entry in `BENCHMARKS.md`.

The next expected bottleneck is expert traffic, based on a derived model rather than measurement: uncached natural Top-16 expert reads across 92 MoE layers total 25.83 GB/token. Linux NVMe, RAM-to-GPU, kernel, and stall counters remain not measured until the relevant runtime exists.

## Last known-good state

- Public `main`: `b86280ed5eefc41992b1ea02e20204edea6b61cf`.
- GitHub Actions correctness run: `31249173770`, success on Linux.
- Public-main tests: Python 46/46 and CTest 2/2.
- Milestone 1 design commit: `84edc6e`.
- Linux environment: WSL 2.7.11.0, Ubuntu 24.04.4, kernel 6.18.33.2, CUDA Toolkit 13.3.1, nvcc 13.3.73, RTX 5080 compute capability 12.0.
- Current worktree code commit: `5b6d1e7`.
- CPU-only CTest: 5/5 pass; `k3x_run` has no CUDA or cuBLAS dynamic dependency.
- CUDA CTest: 5/5 pass on RTX 5080; runtime archive and device test contain native `sm_120` cubins.
- CPU-only pytest: 47/47 pass with `K3X_BUILD_DIR=build-cpu`.
- CUDA-enabled cross-language parity: 5/5 pass with `K3X_BUILD_DIR=build-cuda` while the CLI remains on the exact CPU backend.

## Proposed component status

APOLLO, TITAN COUNCIL, AURORA, PROMETHEUS-X, MERCURY, ORBIT, HELIOS, SHADOW, PHOENIX, VAULT, VEILBREAK, AUTO, and SKYFORGE are proposed only. ATLAS, CHRONOS, and BLACKSTAR are reserved names without accepted definitions. None is claimed as implemented or benchmarked.
