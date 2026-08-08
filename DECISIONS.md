# K3X Decision Ledger

## Status vocabulary

- `accepted` governs current work.
- `experimental` is implemented only behind a flag and has not earned default status.
- `proposed` has an accepted evaluation slot but no implementation claim.
- `rejected` is not pursued unless new evidence reopens the decision.
- `superseded` is retained for history but no longer governs current work.

## D-001 — Dedicated K3 runtime and format

- Date: 2026-08-08.
- Status: accepted.
- Decision: build a Kimi K3-specific runtime and K3X storage format rather than fork a generic engine and add options.
- Alternatives considered: fork llama.cpp; extend vLLM; implement a dedicated runtime while reusing inspected ideas and kernels.
- Evidence: the required execution-ordered storage, per-expert random access, recurrent state, and three-tier scheduling cross generic runtime boundaries.
- Benchmark result: none at decision time.
- Reason: a dedicated design can make data layout and scheduling first-class while preserving an independent reference path.
- Revisit: only if measured maintenance cost overwhelms a demonstrated end-to-end advantage.

## D-002 — Synthetic correctness before full weights

- Date: 2026-08-08.
- Status: accepted.
- Decision: prove the graph, format, conversion, and independent runtime on a deterministic synthetic K3-compatible checkpoint before downloading full Kimi K3 weights.
- Alternatives considered: begin from checkpoint metadata only; download selected full shards; download the full checkpoint first.
- Evidence: the synthetic model now covers KDA, MLA, Attention Residual, Stable LatentMoE, routing, MXFP4, recurrent state, incremental decode, and greedy tokens.
- Benchmark result: B-0001 in `BENCHMARKS.md`.
- Reason: bounded fixtures make correctness failures reproducible without 1.5 TB storage and transfer requirements.
- Revisit: not applicable; full-weight validation is a later additive milestone.

## D-003 — K3X v1 integrity and resume contract

- Date: 2026-08-08.
- Status: accepted.
- Decision: use a fixed 4 KiB superblock, fixed-width little-endian directories, aligned extents, per-extent CRC32C, root SHA-256, partial artifacts, and atomic resume ledgers.
- Alternatives considered: safetensors as runtime format; GGUF; a K3-specific immutable format.
- Evidence: Python streaming conversion, stale-ledger recovery, corruption rejection, and independent C++ parsing pass the Milestone 0 suite.
- Benchmark result: B-0001 includes strict whole-artifact verification in TTFT.
- Reason: it provides bounded conversion and deterministic validation while permitting execution-ordered packing.
- Revisit: version the format rather than mutating v1 incompatibly.

## D-004 — Hybrid CUDA baseline

- Date: 2026-08-08.
- Status: accepted and implemented as the Milestone 1 comparison baseline.
- Decision: retain the CPU oracle, use cuBLASLt for the independent dense FP32/BF16 path, and compare CPU MXFP4 with a custom CUDA MXFP4 path that changes only expert matrix multiplication.
- Alternatives considered: custom CUDA only; cuBLASLt only; hybrid comparison.
- Evidence: the local CUDA 13.3 toolkit supports `sm_120`. NVIDIA's cuBLAS documentation specifies UE4M3/16 scaling for FP4; K3 uses E8M0/32 MXFP4, so direct native-byte compatibility is disproven. The cuBLASLt FP32/BF16-rounded dense and custom native-byte MXFP4 literal suites pass on the RTX 5080.
- Benchmark result: B-0002 measures 19.49 decode tok/s for CPU, 11.67 for FP32 `cuda-dense`, and 10.11 for FP32 `cuda-custom` on the deterministic synthetic graph. The CUDA paths are therefore not accepted as performance defaults.
- Reason: cuBLASLt remains a strong dense baseline, while the custom path is required to preserve exact K3 MXFP4 bytes and provides a route toward K3-specific fusion.
- Revisit: after persistent residency and a wider layer/block execution boundary remove per-operation allocation, transfer, and synchronization overhead.

## D-005 — TITAN project ledger

- Date: 2026-08-08.
- Status: accepted.
- Decision: maintain stable `PROJECT_CHARTER.md` plus current `ARCHITECTURE.md`, `PROJECT_STATE.md`, `DECISIONS.md`, and `BENCHMARKS.md`.
- Alternatives considered: repeatedly revise the initial prompt; keep status only in commits and ad hoc notes; use a persistent project ledger.
- Evidence: the project spans many independently benchmarked research stages and needs explicit separation between constitution, proposal, implementation, and measurement.
- Benchmark result: not applicable.
- Reason: future sessions can resume from evidence without turning proposals or estimates into implementation claims.
- Revisit: document schema may be extended, but the separation of responsibilities remains stable unless the user changes it explicitly.

## D-006 — Architectural addenda remain proposals

- Date: 2026-08-08.
- Status: accepted.
- Decision: APOLLO, TITAN COUNCIL, AURORA, PROMETHEUS-X, MERCURY, ORBIT, HELIOS, SHADOW, PHOENIX, VAULT, VEILBREAK, and AUTO enter the architecture registry as proposed components. They are not implemented claims.
- Alternatives considered: treat the names as roadmap commitments; omit them until implementation; register them with explicit proposal state.
- Evidence: the user supplied purposes but explicitly stated that they remain proposals until implemented and benchmarked.
- Benchmark result: none.
- Reason: registration preserves continuity without overstating code or evidence.
- Revisit: each component receives a separate decision when its source research, quality contract, interfaces, and benchmark gate are defined.

## D-007 — Undefined reserved names

- Date: 2026-08-08.
- Status: proposed.
- Decision: ATLAS, CHRONOS, and BLACKSTAR are recorded as reserved architectural names with no assigned responsibility.
- Alternatives considered: infer roles from their names; omit them; record them as undefined.
- Evidence: the continuity protocol requires them in `ARCHITECTURE.md` but provides no functional definitions.
- Benchmark result: none.
- Reason: inventing responsibilities would silently alter the charter.
- Revisit: immediately when the user supplies or approves definitions.

## D-008 — Do not weaken Windows application control

- Date: 2026-08-08.
- Status: accepted.
- Decision: do not disable Smart App Control or modify certificate trust automatically to run new K3X binaries.
- Alternatives considered: disable the policy; add a local signing trust chain; use the Linux-native target or separately approved WSL2 GPU environment.
- Evidence: Code Integrity events 3033 and 3077 identify policy `{0283ac0f-fff1-49ae-ada1-8a933130cad6}` as the cause of the local unsigned `k3x_run.exe` block.
- Benchmark result: CTest 2/2 passes; Python 41/46 passes, with five cross-language tests blocked before process creation.
- Reason: changing a host security boundary is not a normal runtime implementation step and the final target is Linux native.
- Revisit: if the user explicitly authorizes a narrowly scoped signing or Linux/WSL setup.

## D-009 — Reject direct cuBLASLt FP4 for exact K3 MXFP4

- Date: 2026-08-08.
- Status: rejected.
- Decision: do not present cuBLASLt native FP4 as a direct executor for K3's native MXFP4 expert payload.
- Alternatives considered: pass K3 bytes directly; repack E8M0/32 to the cuBLASLt UE4M3/16 format; implement exact custom MXFP4 decode and matrix multiplication.
- Evidence: NVIDIA's cuBLAS FP4 contract requires `CUBLASLT_MATMUL_MATRIX_SCALE_VEC16_UE4M3`; its E8M0 vector-32 scaling mode applies to FP8. K3X's verified payload contract is E2M1 with one E8M0 scale per 32 values.
- Primary source: [NVIDIA cuBLAS 13.3 documentation](https://docs.nvidia.com/cuda/archive/13.3.0/cublas/index.html).
- Benchmark result: none; incompatibility is a format-contract result, not a performance result.
- Reason rejected: direct use is invalid, while runtime repacking would add traffic and stop being the exact native MXFP4 path.
- Revisit: only if a future CUDA library exposes E2M1 with E8M0/32 scaling compatible with K3X bytes.

## D-010 — Use WSL2 as the verified local Linux GPU development path

- Date: 2026-08-08.
- Status: accepted.
- Decision: use WSL2 Ubuntu 24.04 for local Milestone 1 correctness and CUDA development while keeping Linux native as the final performance authority.
- Alternatives considered: weaken Windows Smart App Control; require an immediate dual-boot/native Linux installation; use WSL2 without installing a Linux display driver.
- Evidence: WSL 2.7.11.0 exposes the RTX 5080 at compute capability 12.0, CUDA Toolkit 13.3.1 provides nvcc 13.3.73 and `sm_120`, and the existing Release baseline passes CTest 2/2 and pytest 47/47.
- Benchmark result: none; environment validation and correctness tests are not throughput benchmarks.
- Reason: this preserves the host security policy, matches the Linux-first runtime direction, and supplies a reproducible local CUDA compiler and device path.
- Revisit: perform final storage and performance measurements on Linux native because WSL filesystem and I/O behavior are not the final authority.

## D-011 — Keep profiling events explicit and aggregate successful work only

- Date: 2026-08-08.
- Status: accepted.
- Decision: the foundational profiler records caller-supplied events, aggregates successful event time and bytes, separates H2D/D2H traffic, and counts failed events without adding their partial durations or bytes to successful totals.
- Alternatives considered: embed clocks and CUDA events in the profiler; aggregate failed partial work into normal totals; keep raw events without a common summary.
- Evidence: the deterministic aggregation test preserves FP32 and native MXFP4 precision identities, verifies directional transfer bytes, and excludes a failed event from successful totals.
- Benchmark result: B-0002 records nonzero CUDA-event kernel time, exact directional transfer bytes, backend-owned peak VRAM, and numerical error for all measured CUDA modes.
- Reason: explicit events keep measurement source and synchronization policy at the backend boundary while making summaries deterministic and comparable.
- Revisit: add separate attempted-work counters if failed partial work becomes necessary for diagnosis; do not silently mix it into successful throughput totals.

## D-012 — Isolate only matrix operations in the first compute backend boundary

- Date: 2026-08-08.
- Status: accepted.
- Decision: move dense and native MXFP4 matrix-vector operations behind `ComputeBackend` while leaving attention, routing, recurrent state, residual, activation, and token-selection logic in the existing graph.
- Alternatives considered: move the complete decoder layer behind a backend; introduce a general tensor abstraction; isolate only the two operations required by the approved CUDA baseline.
- Evidence: literal CPU dense and MXFP4 tests pass, CTest passes 4/4, and Python/C++ layer, logits, state, and exact-token parity passes 47/47 with the explicit CPU backend.
- Benchmark result: B-0002 shows the narrow per-operation CUDA boundary is slower than CPU on the tiny graph. CUDA-event kernel time is only 11.56--14.52 ms per run while end-to-end execution is hundreds of milliseconds, identifying boundary overhead and CPU-resident graph work rather than kernel arithmetic as the immediate bottleneck.
- Reason: the narrow boundary preserves the numerical oracle and limits CUDA parity work to independently testable operations without creating a generic framework.
- Revisit: now. The measurement justifies persistent buffers and a layer/block boundary while retaining the current operation boundary as the correctness oracle.

## D-013 — Keep native tests active under Release NDEBUG

- Date: 2026-08-08.
- Status: accepted.
- Decision: native C++ tests use explicit nonzero return codes rather than `assert` for behavior checks that must execute in Release builds.
- Alternatives considered: keep `assert`; undefine `NDEBUG` only for tests; use explicit checks consistently with the existing native suite.
- Evidence: the first Release CUDA-unavailable test linked past a nonexistent enum reference because `assert` removed the complete expression. Replacing assertions exposed the intended compile-time RED and the corrected profiler/backend tests pass in Release.
- Benchmark result: none; this is test validity evidence.
- Reason: Release is the performance build and its correctness tests must execute the same checks rather than silently becoming no-ops.
- Revisit: a dedicated test framework may replace return codes later, but it must remain active under `NDEBUG`.

## D-014 — Make CUDA an explicit optional build with no CPU fallback

- Date: 2026-08-08.
- Status: accepted and fully exercised by the Milestone 1 CLI and benchmark.
- Decision: default `K3X_ENABLE_CUDA=OFF` builds a CUDA-free stub, while ON requires CUDA Toolkit 13.3, targets native `sm_120`, validates capability 12.0 or newer, and owns a nonblocking stream plus cuBLASLt handle without global state. An unavailable requested operation never silently changes backend; the documented `cuda_dense` CPU MXFP4 oracle is part of that comparison backend's definition rather than a fallback.
- Alternatives considered: require CUDA for every build; silently fall back to CPU; load CUDA dynamically from one binary; compile mutually exclusive stub and CUDA sources.
- Evidence: CPU CTest passes 5/5 with no CUDA/cuBLAS dynamic dependency; CUDA CTest passes 7/7 on RTX 5080; CUDA artifacts contain `sm_120` cubins; explicit CPU/FP32, CUDA/FP32, and CUDA/BF16 graph tests preserve exact token IDs with bounded numerical error. A CUDA-disabled CLI request returns `BACKEND_UNAVAILABLE`.
- Benchmark result: B-0002 stores the requested backend and observed device in every record; no sample changed identity or fell back.
- Reason: explicit build and runtime identity prevents benchmark mislabeling and preserves portable CPU correctness.
- Revisit: dynamic loading is considered only if separate build artifacts become an operational burden after measured backends exist.

## D-015 — Stage both operands for the BF16-rounded cuBLASLt baseline

- Date: 2026-08-08.
- Status: accepted and implemented.
- Decision: keep the public dense interface in FP32, but round and stage both the input vector and weight matrix as BF16 for `bf16_rounded`; accumulate and return FP32.
- Alternatives considered: stage only BF16 weights with an FP32 input; stage both operands as BF16; round weights to BF16 and expand them back to FP32 before transfer.
- Evidence: NVIDIA cuBLAS 13.3 lists regular BF16 matmul support with a shared `Atype/Btype` of `CUDA_R_16BF`, not a BF16-weight/FP32-input combination. The literal GPU test matches an independently bit-rounded CPU oracle and records 18 H2D bytes instead of the FP32 path's 36 bytes.
- Primary source: [NVIDIA cuBLAS 13.3 `cublasLtMatmul` data type table](https://docs.nvidia.com/cuda/archive/13.3.0/cublas/index.html#cublasltmatmul).
- Benchmark result: B-0002 shows BF16 halves H2D and peak backend-owned VRAM versus FP32 for both CUDA identities, but it does not improve decode throughput on this allocation-bound tiny graph. Maximum absolute diagnostic error is 0.00402409 and all token IDs remain exact; maximum relative error is unstable at 17.50 because reference values near zero are included.
- Reason: using a documented operand combination preserves an honest BF16 baseline and actual transfer accounting; expanding rounded weights to FP32 would erase the intended H2D reduction.
- Revisit: add a separately named weight-only emulation only if a future CUDA release documents mixed BF16/FP32 A/B support or a custom kernel justifies it.

## D-016 — Use a one-block-per-row native MXFP4 correctness kernel

- Date: 2026-08-08.
- Status: accepted and implemented as a baseline.
- Decision: decode K3X low-nibble-first E2M1 plus E8M0/32 bytes directly in a custom CUDA kernel, assign one 256-thread block per output row, stride columns, reduce in FP32 shared memory, and expose it only through `cuda_custom`.
- Alternatives considered: repack to cuBLASLt FP4; dequantize a complete matrix before GEMM; decode and accumulate native bytes in one minimal kernel.
- Evidence: the three-row/two-group literal covers low and high nibbles, signs, E2M1 magnitudes, distinct E8M0 exponents, and a non-warp-multiple row count; a 320-column mutation test proves columns beyond the first 256 are processed. Results match the CPU byte-level oracle within `1e-4`, memcheck reports zero errors, and the archive contains a native `sm_120` cubin for `mxfp4.cu.o`. The `cuda_dense` regression retains the CPU MXFP4 oracle with zero H2D and device time, proving the incompatible cuBLASLt FP4 path is not used.
- Benchmark result: B-0002 measures FP32 `cuda-custom` at 10.11 decode tok/s and 14.52 ms CUDA-event kernel time per run. It trails `cuda-dense` on the tiny graph and is not a default performance path.
- Reason: direct decode preserves exact checkpoint bytes and provides the smallest controlled baseline for later fusion and residency measurements.
- Revisit: defer kernel micro-optimization until persistent buffers and broader batching remove the larger measured boundary overhead. Then compare reduction, vectorized loads, and activation/scaling fusion independently.

## D-017 ??Keep CPU as the Milestone 1 default after end-to-end measurement

- Date: 2026-08-08.
- Status: accepted for the synthetic Milestone 1 runtime.
- Decision: keep `cpu` as the default CLI backend and require explicit selection of `cuda-dense` or `cuda-custom`. Treat both CUDA paths as exact, measured development baselines rather than performance defaults.
- Alternatives considered: default to `cuda-dense`; default to `cuda-custom`; choose the fastest measured exact backend for this milestone.
- Evidence: all backends generate `[43, 32, 28, 49, 9, 28]`; FP32 CUDA maximum absolute diagnostic error is below `1.8e-7`; CPU-only and CUDA-enabled suites pass independently.
- Benchmark result: B-0002 measures CPU at 19.49 decode tok/s, ahead of `cuda-dense` at 11.67 and `cuda-custom` at 10.11 on the tiny synthetic graph. BF16 does not reverse the ordering.
- Reason: selecting a slower GPU path by default would contradict end-to-end evidence. Explicit identities also prevent accidental benchmark mislabeling.
- Revisit: after persistent device residency and batched layer/block execution are implemented and remeasured on the same correctness fixture.
