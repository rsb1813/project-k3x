# K3X Current Project State

## Current milestone

Milestone 1 — exact runtime backend boundary, structured profiler, cuBLASLt dense baseline, and custom K3 MXFP4 CUDA baseline.

The design is approved, and the Linux portability, deterministic profiler, exact CPU backend, optional CUDA resource shell, cuBLASLt dense baseline, and native-byte K3 MXFP4 CUDA baseline tasks are complete. End-to-end backend selection and profiler export are next.

State recorded after the 2026-08-08 exact custom MXFP4 CUDA implementation commit and fresh dual-build correctness verification.

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

## Work in progress

- Branch: `feat/milestone-one-runtime`.
- Worktree: `C:\Users\jolib\Documents\project-k3x\.worktrees\milestone-one-runtime`.
- The next code task is explicit CLI backend/precision selection, synthetic graph integration, and JSON/CSV profiler export.
- Linux development runs as the unprivileged `jolib` user. The isolated Python environment is `/home/jolib/.venvs/k3x-m1`; repository build output is `build-linux`.

## Known failures and blockers

- Windows Smart App Control blocks the newly linked unsigned `build/k3x_run.exe` before process creation.
- Code Integrity events 3033 and 3077 cite policy `{0283ac0f-fff1-49ae-ada1-8a933130cad6}` and an unmet Enterprise signing level.
- Fresh Windows CTest binaries run, but five Python cross-language cases that launch `k3x_run.exe` remain blocked on Windows. This does not block the verified WSL Linux path.
- cuBLASLt dense and custom MXFP4 operations are not yet connected to the synthetic decoder graph, and no CUDA throughput benchmark has been accepted yet.
- CUDA calls allocate and transfer per operation in the current correctness baseline; persistent residency and asynchronous overlap remain unimplemented.
- `cuda_dense` intentionally uses the CPU MXFP4 oracle with zero device traffic; this is its documented comparison contract, not an automatic fallback after a CUDA failure.
- Direct cuBLASLt FP4 is rejected for exact K3 MXFP4 because NVIDIA requires UE4M3 scales per 16 FP4 values while K3 uses E8M0 scales per 32 values.

## Next concrete tasks

1. Add explicit CLI backend and dense-precision selection without silent CPU fallback.
2. Connect the CUDA backend to the synthetic graph and verify CPU/CUDA layer, state, logits, and token parity.
3. Export backend, device, transfer, kernel, memory, and numerical fields through JSON/CSV benchmark records.
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

The latest implemented CUDA operations are literal dense and native MXFP4 correctness cases, not throughput benchmarks. The next concrete bottleneck is end-to-end graph integration and truthful profiler export, which are required before kernel and transfer costs can be compared. The expected full-model bottleneck remains expert traffic, based on a derived model rather than measurement: uncached natural Top-16 expert reads across 92 MoE layers total 25.83 GB/token. Linux NVMe, RAM-to-GPU, kernel, and stall counters remain unmeasured until the relevant runtime exists.

## Last known-good state

- Public `main`: `b86280ed5eefc41992b1ea02e20204edea6b61cf`.
- GitHub Actions correctness run: `31249173770`, success on Linux.
- Public-main tests: Python 46/46 and CTest 2/2.
- Milestone 1 design commit: `84edc6e`.
- Linux environment: WSL 2.7.11.0, Ubuntu 24.04.4, kernel 6.18.33.2, CUDA Toolkit 13.3.1, nvcc 13.3.73, RTX 5080 compute capability 12.0.
- Current worktree code commit: `7d4ade6` (`ea730c5` custom kernel plus the cuda-dense CPU-oracle contract correction).
- CPU-only CTest: 5/5 pass; `k3x_run` has no CUDA or cuBLAS dynamic dependency.
- CUDA CTest: 7/7 pass on RTX 5080; cuBLASLt FP32/BF16-rounded and exact native-byte MXFP4 literal checks pass, and MXFP4 `compute-sanitizer --tool memcheck` reports 0 errors.
- CPU-only pytest: 47/47 pass with `K3X_BUILD_DIR=build-cpu`.
- CUDA-enabled cross-language parity: 5/5 pass with `K3X_BUILD_DIR=build-cuda` while the CLI remains on the exact CPU backend.

## Proposed component status

APOLLO, TITAN COUNCIL, AURORA, PROMETHEUS-X, MERCURY, ORBIT, HELIOS, SHADOW, PHOENIX, VAULT, VEILBREAK, AUTO, and SKYFORGE are proposed only. ATLAS, CHRONOS, and BLACKSTAR are reserved names without accepted definitions. None is claimed as implemented or benchmarked.
