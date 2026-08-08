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
- Status: accepted design; not implemented.
- Decision: retain the CPU oracle, use cuBLASLt for the independent dense FP32/BF16 path, and compare CPU MXFP4 with a custom CUDA MXFP4 path that changes only expert matrix multiplication.
- Alternatives considered: custom CUDA only; cuBLASLt only; hybrid comparison.
- Evidence: the local CUDA 13.3 toolkit supports `sm_120`. NVIDIA's cuBLAS documentation specifies UE4M3/16 scaling for FP4; K3 uses E8M0/32 MXFP4, so direct native-byte compatibility is disproven.
- Benchmark result: none; CUDA performance is unmeasured.
- Reason: cuBLASLt remains a strong dense baseline, while the custom path is required to preserve exact K3 MXFP4 bytes and provides a route toward K3-specific fusion.
- Revisit: after both paths have end-to-end synthetic measurements.

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
- Benchmark result: none; CTest 3/3 and pytest 47/47 are correctness results.
- Reason: explicit events keep measurement source and synchronization policy at the backend boundary while making summaries deterministic and comparable.
- Revisit: add separate attempted-work counters if failed partial work becomes necessary for diagnosis; do not silently mix it into successful throughput totals.

## D-012 — Isolate only matrix operations in the first compute backend boundary

- Date: 2026-08-08.
- Status: accepted.
- Decision: move dense and native MXFP4 matrix-vector operations behind `ComputeBackend` while leaving attention, routing, recurrent state, residual, activation, and token-selection logic in the existing graph.
- Alternatives considered: move the complete decoder layer behind a backend; introduce a general tensor abstraction; isolate only the two operations required by the approved CUDA baseline.
- Evidence: literal CPU dense and MXFP4 tests pass, CTest passes 4/4, and Python/C++ layer, logits, state, and exact-token parity passes 47/47 with the explicit CPU backend.
- Benchmark result: none; the backend is not yet connected to the benchmark profiler schema, and this refactor does not claim a throughput change.
- Reason: the narrow boundary preserves the numerical oracle and limits CUDA parity work to independently testable operations without creating a generic framework.
- Revisit: widen the boundary only after measured transfer or launch overhead identifies a specific fusion opportunity.
