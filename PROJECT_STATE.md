# K3X Current Project State

## Current milestone

Milestone 1 — exact runtime backend boundary, structured profiler, cuBLASLt dense baseline, and custom K3 MXFP4 CUDA baseline.

The design is approved and the implementation plan is written. Production implementation has not started.

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

## Work in progress

- Branch: `feat/milestone-one-runtime`.
- Worktree: `C:\Users\jolib\Documents\project-k3x\.worktrees\milestone-one-runtime`.
- The next code task is not started; the written plan awaits execution-mode and Linux-environment authorization.
- WSL and Virtual Machine Platform are enabled. The immediate prerequisite is a Windows reboot, followed by Ubuntu 24.04 installation and GPU passthrough verification.

## Known failures and blockers

- Windows Smart App Control blocks the newly linked unsigned `build/k3x_run.exe` before process creation.
- Code Integrity events 3033 and 3077 cite policy `{0283ac0f-fff1-49ae-ada1-8a933130cad6}` and an unmet Enterprise signing level.
- Fresh Windows CTest binaries currently run, but five Python cross-language cases that launch `k3x_run.exe` are blocked. The observed local split is 41 passed and 5 blocked failures.
- WSL and Virtual Machine Platform are enabled with explicit user authorization, but Windows reports pending CBS and file-rename reboot state. Ubuntu is not installed until that reboot completes.
- No Linux GPU execution result exists yet, so no CUDA correctness or performance claim exists.
- Direct cuBLASLt FP4 is rejected for exact K3 MXFP4 because NVIDIA requires UE4M3 scales per 16 FP4 values while K3 uses E8M0 scales per 32 values.

## Next concrete tasks

1. Reboot Windows to finish the already-enabled WSL and Virtual Machine Platform features.
2. Install and initialize Ubuntu 24.04, then verify RTX 5080 GPU passthrough.
3. Reproduce the Milestone 0 Linux CPU baseline before changing production code.
4. Make cross-language build-directory selection explicit through a RED-GREEN test.
5. Implement deterministic profiling primitives through TDD.
6. Extract the exact CPU compute backend without changing numerical output.
7. Add the optional SM 12.0 CUDA backend, cuBLASLt dense baseline, and custom MXFP4 kernel in separate verified commits.

## Hardware assumptions

| Component | Current assumption or observation |
|---|---|
| CPU | AMD Ryzen 7 9800X3D target |
| GPU | Locally observed NVIDIA GeForce RTX 5080, 16,303 MiB, compute capability 12.0 |
| Driver | Locally observed 591.86 |
| CUDA | Locally observed toolkit 13.3 and nvcc 13.3.73 with `sm_120` support |
| RAM | 96 GB DDR5-4200 target |
| NVMe | Solidigm P44 Pro 2 TB target |
| Final runtime OS | Linux native |
| Current development OS | Windows 11 AMD64 with Smart App Control enforcement |

## Latest measured bottleneck

Milestone 0 did not measure full-model or NVMe performance. Its only measured run is the tiny CPU synthetic B-0001 entry in `BENCHMARKS.md`.

The next expected bottleneck is expert traffic, based on a derived model rather than measurement: uncached natural Top-16 expert reads across 92 MoE layers total 25.83 GB/token. Linux NVMe, RAM-to-GPU, kernel, and stall counters remain not measured until the relevant runtime exists.

## Last known-good state

- Public `main`: `b86280ed5eefc41992b1ea02e20204edea6b61cf`.
- GitHub Actions correctness run: `31249173770`, success on Linux.
- Public-main tests: Python 46/46 and CTest 2/2.
- Milestone 1 design commit: `84edc6e`.
- Current worktree CTest baseline: 2/2 pass.
- Current worktree Python baseline: 41 pass, 5 blocked by Windows application control before `k3x_run.exe` starts.

## Proposed component status

APOLLO, TITAN COUNCIL, AURORA, PROMETHEUS-X, MERCURY, ORBIT, HELIOS, SHADOW, PHOENIX, VAULT, VEILBREAK, AUTO, and SKYFORGE are proposed only. ATLAS, CHRONOS, and BLACKSTAR are reserved names without accepted definitions. None is claimed as implemented or benchmarked.
