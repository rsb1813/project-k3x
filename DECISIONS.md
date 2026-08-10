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

## D-018 — Stage CUDA residency before a whole-layer GPU executor

- Date: 2026-08-08.
- Status: implemented and measured as an experimental opt-in primitive.
- Decision: implement independently switchable reusable scratch allocation, bounded tensor-ID-keyed immutable-weight residency, and same-input projection grouping before moving the complete KDA/MLA/MoE layer graph to CUDA.
- Alternatives considered: scratch reuse only; the selected staged design; immediate whole-layer GPU execution.
- Evidence: B-0002 shows only 11.56--14.52 ms of CUDA-event kernel time per run while every matrix call allocates, uploads immutable weights, downloads output, synchronizes, and frees resources. The current `ComputeBackend` also lacks stable tensor identity and grouped operations.
- Benchmark result: B-0002 is the reference. Milestone 2 has no benchmark result until the implementation passes its ablation matrix.
- Reason: the staged design removes measured redundant work while keeping each optimization independently disableable and preserving a narrow numerical oracle. It creates L0 primitives without inventing the later expert eviction policy.
- Revisit: after the four-step allocation/residency/grouping ablation identifies the remaining host/device boundary cost.

## D-019 — Keep CUDA optimizations explicit after Milestone 2 measurement

- Date: 2026-08-09.
- Status: accepted and measured.
- Decision: retain `cpu` plus `per-operation + transient + scalar` as the default reference behavior. Keep reusable allocation, bounded static residency, grouping, and BF16 independently selectable. For synthetic CUDA work, `reused + resident + scalar` is the fastest measured configuration; grouped scheduling and BF16 are not promoted to defaults.
- Alternatives considered: enable every Milestone 2 optimization by default; default to grouped FP32; default to grouped BF16; keep the exact reference defaults and publish the measured switches.
- Evidence: CPU and CUDA graph parity preserve token IDs `[43, 32, 28, 49, 9, 28]`; CPU CTest passes 5/5 and pytest passes 65 with 23 CUDA skips; CUDA CTest passes 9/9 and pytest passes 87 with one CPU-only skip; four CUDA memcheck targets report zero errors.
- Benchmark result: B-0003 measures `cuda-dense` FP32 at 12.1261, 17.4560, 18.0041, and 17.9018 decode tok/s across reference, reuse, residency, and grouped stages. `cuda-custom` measures 12.2647, 17.1425, 17.2723, and 16.8348 tok/s. Grouping reduces activation H2D and synchronization but does not beat scalar residency. Fully enabled BF16 measures 17.6861 and 17.0032 tok/s with maximum absolute error 0.00402409 and exact tokens.
- Reason: allocation reuse and residency have positive end-to-end evidence, while grouping and BF16 do not have a throughput win on this fixture. The synthetic graph is too small and CPU-resident to justify changing the public default or projecting full-model performance.
- Revisit: after a wider layer/block GPU executor removes CPU graph boundaries, and again after native-Linux full-dimension slice measurements establish representative transfer and compute costs.

## D-020 — Keep the FFN block boundary experimental after B-0004

- Date: 2026-08-09.
- Status: accepted and measured.
- Decision: retain `operation` as the default correctness boundary. Expose `ffn-block` only with explicit `cuda-custom` selection and recommend `reused + resident + scalar + FP32` for synthetic CUDA experimentation. Do not promote BF16 or grouped FFN blocks to defaults.
- Alternatives considered: a generic device-tensor boundary; a full decoder-layer CUDA executor; the dependency-closed FFN block; immediately defaulting to the fastest measured block configuration.
- Evidence: dense/shared and exact native MXFP4 block tests cover FP32, BF16-RNE, residency hits, exact capacity bypass, invalid later-expert preflight, ordered outputs, one shared input upload, and one final synchronization. Natural routing traces, tokens, layers, logits, and recurrent state match the operation reference. Fresh verification passes CPU CTest 5/5 and pytest 70 passed/26 skipped, CUDA CTest 11/11 and pytest 95 passed/1 skipped; six CUDA memcheck targets report zero errors.
- Benchmark result: B-0004 measures FP32 operation-scalar at 16.3576 decode tok/s and FFN-block-scalar at 17.0713, a 4.36% gain. D2H falls 24.77%, activation H2D falls 26.48%, and synchronization falls 32.86%. FP32 FFN-block-grouped reaches 17.0270. BF16 block-scalar reaches 16.9847 with exact tokens but maximum absolute error 0.00402409, so it does not beat FP32 block-scalar.
- Reason: the dependency-closed boundary produces a measured end-to-end gain and materially reduces traffic while preserving exact routing, but the evidence is one tiny synthetic WSL2 graph. Kernel time rises and most wall time remains in the CPU-driven graph, so changing the public default would overstate generality.
- Revisit: after KDA/MLA or a larger dependency-closed layer block moves to CUDA, after native-Linux repetition, and after a full-dimension bounded checkpoint slice establishes representative expert sizes and transfer deadlines.

## D-021 — Keep exact L1-to-L0 prefetch opt-in after B-0005

- Date: 2026-08-09.
- Status: accepted and measured.
- Decision: retain `synchronous` as the default transfer mode. Expose the bounded exact `prefetch` path only for `cuda-custom + ffn-block + reused + transient` with an explicit positive pinned capacity. Build the next storage milestone around a persistent bounded L1 expert cache and independently measured L2 read path before considering prefetch as a default.
- Alternatives considered: default the new prefetch mechanism immediately; combine it with static residency or an eviction policy now; implement a general worker/deadline scheduler before proving one exact dependency boundary; retain the narrow two-phase exact prefetch switch.
- Evidence: a fixed pinned/device slab, separate nonblocking transfer stream, reusable CUDA events, single-use prepared tokens, full native MXFP4 preflight, and graph-level routing/token parity are covered by native and Python tests. Complete verification passes CPU CTest 5/5 and pytest 94 passed/27 skipped, CUDA CTest 14/14 and pytest 120 passed/1 skipped. All ten CUDA test binaries report Compute Sanitizer `ERROR SUMMARY: 0 errors`.
- Benchmark result: B-0005 measures FP32 synchronous/prefetch scalar at 16.9701/16.7947 decode tok/s and grouped at 16.7055/16.7914. BF16 scalar measures 16.6366/16.5735 and grouped 16.5529/16.7021. Every row preserves exact tokens and routing; prefetch keeps H2D bytes and host synchronization unchanged, performs 27 prepares and waits with all 27 ready before use, and consumes a 1 MiB pinned/device slab. Exposed stall is 0.198--0.312 ms per run.
- Reason: the mechanism proves correct asynchronous transfer and event ordering, but its matched end-to-end effect on this tiny CPU-driven graph is within -1.03% to +0.90%. That is insufficient evidence to spend pinned memory by default or to couple unmeasured L1/L2 policy into the exact path.
- Revisit: after a persistent bounded L1 cache removes synchronous repeated file reads and pageable staging, after asynchronous L2 I/O is benchmarked on native Linux, and after full-dimension bounded expert slices provide representative transfer deadlines.

## D-022 — Validate prepared identity before work and preserve activation overlap

- Date: 2026-08-09.
- Status: accepted after final review.
- Decision: include the expected use sequence in `Mxfp4PrefetchToken` and validate token ID, sequence, layer, phase, input shape, and activation parameters through backend-owned metadata before scratch growth, activation H2D, or transfer-event submission. Keep activation H2D before the compute-stream readiness wait so it can overlap the expert transfer.
- Alternatives considered: rely on token ID alone; move pipeline `consume` before all allocation and activation upload; add a separate public validation method; extend the existing backend prevalidation with use sequence and stronger side-effect tests.
- Evidence: the backend already rejected token ID, layer, phase, input, and parameter mismatches before work, but the pipeline stored `use_sequence` without checking it. Moving the wait before activation H2D would unnecessarily serialize two independent transfers. New tests require foreign and wrong-sequence failures to preserve device allocation, current/peak device memory, activation H2D, transfer waits, profiler H2D, and the valid pending request.
- Benchmark result: no new performance measurement. The valid transfer and compute order measured in B-0005 is unchanged; FP32/BF16 post-fix one-sample smokes preserve exact tokens, routing, matched H2D, and synchronization.
- Reason: sequence identity closes the stated token contract, while duplicate backend prevalidation provides failure atomicity without sacrificing the intended activation/expert-copy overlap.
- Revisit: when the runtime supports multiple outstanding requests or deadline-aware scheduling; the token may then need a scheduler-owned generation and request identity in addition to use sequence.

## D-023 — Make persistent L1 residency an expert-atomic runtime-session boundary

- Date: 2026-08-09.
- Status: accepted, implemented, and measured.
- Decision: implement the first persistent L1 store between `Model` and `Reader`, owned by `RuntimeSession`, keyed by `(layer, expert)`, and owning immutable complete gate/up/down native MXFP4 payloads. Support disabled and bounded no-eviction static admission with exact transient bypass; keep eviction and policy scoring out of this milestone.
- Alternatives considered: caching individual extents inside `Reader`; owning host residency inside the CUDA backend; a model-adjacent whole-expert store.
- Evidence: current `Model::load_expert` issues six synchronous Reader calls and constructs temporary vectors on every selection before either synchronous compute or Milestone 4 pinned preparation. Reader-level caching loses expert atomicity and mixes trunk/expert decisions. CUDA ownership cannot serve the CPU reference and couples storage to compute.
- Benchmark result: B-0006 admitted 18 synthetic experts into 29,376 bytes, produced 36 hits with zero bypasses, and reduced logical Reader calls from 428 to 212 and completed bytes from 665,616 to 606,864. All FP32/BF16 synchronous/prefetch rows preserved exact tokens, routing, H2D, D2H, FFN counts, and synchronization.
- Reason: whole-expert immutable handles provide the lifetime and accounting unit required by future prediction and policy work without prematurely selecting LRU, LFU, or Least-Stale. Hard-capacity bypass preserves correctness under insufficient RAM.
- Revisit: after B-0006 and full-dimension bounded expert slices establish representative entry sizes; then introduce runtime-switchable admission/eviction policies and reproduce Least-Stale from the original SpecMD work.

## D-024 — Keep static L1 admission opt-in after B-0006

- Date: 2026-08-09.
- Status: accepted.
- Decision: retain `disabled` as the public default and expose no-eviction first-observation `static` admission only through explicit runtime options.
- Alternatives considered: enable static admission by default after the synthetic speedup; remove static until a policy exists; retain it as an exact experimental primitive.
- Evidence: B-0006 measured roughly 2.88–3.02x matched synthetic decode throughput and lower logical Reader traffic with only 29,376 resident bytes. The fixture has only 24 experts, repeated routes, pageable reads, and WSL2 filesystem behavior; it does not represent the 896-expert released model or native-Linux NVMe.
- Benchmark result: FP32 synchronous disabled/static measured 16.5587/47.6845 tok/s and prefetch measured 16.7636/50.6235. BF16 synchronous measured 16.4052/47.7956 and prefetch measured 16.5073/47.6198. Every row retained exact generated tokens and routing.
- Reason: the mechanism and exactness are validated, but first-observation no-eviction is not a production cache policy and the measured gain cannot justify a full-model default.
- Revisit: after native-Linux full-dimension bounded slices establish representative expert sizes, physical L2 traffic, reuse distributions, and policy behavior.

## D-025 — Persist L1 residency in an explicit runtime session and reject malformed native experts before admission

- Date: 2026-08-09.
- Status: accepted and implemented after final review.
- Decision: make `RuntimeSession` the lifetime owner of `HostExpertStore`, while preserving one-shot overload compatibility, and validate native group-32 packed/scales sizes, reserved E8M0 values, and gate/up/down shapes before cache admission.
- Alternatives considered: keep an engine-local cache recreated on every generation call; use process-global cache state; add an explicit session owner with compatibility wrappers.
- Evidence: the engine-local design could not warm across consecutive agent requests. The session regression preserves exact tokens and routing, keeps second-call misses unchanged, increases hits by 54, and reduces second-call Reader traffic. Malformed packed data and reserved scale tests leave misses and residency unchanged.
- Benchmark result: B-0006 was rerun at commit `2a0cb27` after these changes; all eight rows preserve the same token and routing traces and the same matched cache/traffic invariants.
- Reason: explicit session ownership matches the chartered session hot-bank lifetime without hidden global state, while pre-admission validation prevents corrupt exact payloads from becoming durable cache entries.
- Revisit: when VAULT or multi-session serving defines persistence, isolation, and reclamation contracts beyond one in-process runtime session.

## D-026 — Separate L2 I/O engine from page-cache policy and batch exact expert extents

- Date: 2026-08-09.
- Status: accepted and implemented as an experimental, non-default runtime boundary.
- Decision: add an ordered Reader batch API and independent `pread|io_uring` engine and `buffered|direct` cache-mode axes. Keep `pread + buffered` as the default until native-Linux evidence supports a change.
- Alternatives considered: replace every read with immediately awaited io_uring; introduce a general future/executor framework; add one bounded ordered batch and independent axes.
- Evidence: the previous reader opened and sought a new stream for every extent, while one exact native expert requires six reads. The implementation now retains one descriptor and batches those six extents. Linux direct-I/O support remains filesystem-specific: WSL2 `/mnt/c` rejects direct mode while WSL2 ext4 reports explicit alignment and runs it exactly.
- Benchmark result: B-0007 at `5049f26` crossed all four modes for 3 warmups and 20 samples on WSL2 ext4. Every row preserved tokens, the 24-entry routing trace, 428 logical calls, 158 batches, and 665,616 logical bytes. Buffered modes submitted 665,616 bytes; direct modes submitted 756,736 aligned bytes. `io_uring` was slower than `pread` when buffered and faster when direct in this tiny WSL smoke, which is not native P44 Pro evidence.
- Reason: an ordered six-extent batch is the smallest graph-visible boundary that exposes bounded storage concurrency while retaining exact payload reconstruction and independent attribution of engine and cache effects. Native evidence is insufficient to change the default.
- Revisit: after B-0007 native-Linux warm/cold measurements and again when deadline-aware multi-layer prefetch introduces cross-expert request ordering.

Post-review note: final read-only review found that partial-submit or completion-wait errors could return while kernel requests or stale SQEs still referenced batch-owned buffers. The error path now closes the ring before those buffers leave scope and marks it unavailable, while `EINTR` waits retry. A real-ring lifetime-order regression plus the full ASan/UBSan liburing CTest suite validates the fail-closed boundary. The successful B-0007 execution path and measurements are unchanged.

## D-027 — Represent full expert storage with an explicit non-executable slice

- Date: 2026-08-09.
- Status: accepted, implemented, and measured.
- Decision: materialize exactly one released-dimension routed expert as a `STORAGE_FIXTURE`, pack its gate/up/down MXFP4 extents in execution order, expose it to Readers, and reject model execution through K3X optional feature bit 0.
- Alternatives considered: use sparse-file holes to imitate extent lengths; scale the complete executable synthetic graph to released widths; materialize one exact expert and mark it non-executable.
- Evidence: sparse holes would distort checksum, page-cache, and block-I/O behavior. A full-dimension executable graph would require unrelated trunk and recurrent-state storage. The bounded writer and converter instead materialize and read back all 17,547,264 expert bytes with at most a configured 1 MiB source chunk.
- Benchmark result: B-0008 at `9198ed2` ran 3 warmups and 20 measured expert loads for all four Reader combinations on WSL2 ext4. Every row preserved the ordered digest, 120 completions, 350,945,280 logical and submitted bytes, zero failures, and zero direct-I/O byte amplification. Median wall latency was 50.685 ms for buffered pread, 51.592 ms for buffered io_uring, 60.402 ms for direct pread, and 56.426 ms for direct io_uring.
- Reason: the artifact is large enough to expose representative per-expert request sizes while remaining deterministic, streamable, cheap, and incapable of being mistaken for a complete checkpoint. Separate optional identity preserves the executable-model correctness boundary.
- Revisit: when a bounded multi-expert or full layer slice is needed for cache pressure, deadline scheduling, or physical locality experiments, and again on native Linux with the target P44 Pro.

## D-028 — Bind storage-source and resume identities to verified bytes

- Date: 2026-08-09.
- Status: accepted and implemented after final review.
- Decision: publish bounded fixture shards under their SHA-256-derived names before atomically replacing the manifest; verify manifest-declared shard and tensor SHA-256 values; fingerprint only referenced shards; and reuse only a canonical prefix of resume extents whose ID, aligned offset, length, source CRC32C, and partial-file CRC32C all match.
- Alternatives considered: trust generator-authored hashes without checking; overwrite a stable shard name before publishing the manifest; trust any ledger extent whose partial-file CRC matches.
- Evidence: regression tests reproduced fresh conversion of a one-byte-mutated shard, tensor-digest mismatch, old-manifest/new-shard mismatch after a simulated publication failure, duplicate/unknown/zero-length/misaligned ledger entries, and a partial extent whose rewritten CRC no longer matched its source tensor.
- Benchmark result: no B-0008 remeasurement was performed because the payload order, Reader, and benchmark code did not change. Post-fix correctness passed CPU CTest 8/8 and pytest 161/40, liburing CTest 9/9 and pytest 162/39, and CUDA CTest 17/17 and pytest 194/7.
- Reason: source metadata and resume metadata are untrusted recovery inputs. A final root hash cannot recover correctness if corrupted bytes were deliberately accepted while assembling the artifact.
- Revisit: when general checkpoint shard manifests gain signed provenance or when hashing cost becomes measurable on cloud conversion workers.

## D-029 — Keep current-layer deadline loading exact and opt-in

- Date: 2026-08-09.
- Status: accepted, implemented, measured, and retained as experimental.
- Decision: add a bounded single-worker `deadline` schedule after current-layer natural Top-K routing, overlap loads with only routed-down and shared-expert work, and keep `blocking` as the default. Return Reader and L1 telemetry as locked value snapshots and drain all submitted work before every generation return.
- Alternatives considered: replace the blocking Reader API with a general asynchronous API; implement N+1/N+2 prediction and eviction together; use raw telemetry references and rely on ordinary call ordering; introduce the smallest exact current-layer worker boundary first.
- Evidence: scheduler tests cover latest-start priority, stable ties, capacity, failures, resident inline completion, and idle draining. Session and cross-language tests preserve exact tokens, routing, L1 hit/miss counts, and Reader bytes. Final review found three Important race/lifetime gaps; locked snapshots, pre-submit estimate capture, and success/error idle barriers closed them, and one re-review found no remaining Critical or Important issue.
- Benchmark result: replacement B-0009 at `68b3e54` measured 3 warmups and 20 samples for eight WSL2 ext4 rows. Deadline reduced decode by 21.45% for buffered pread, 20.27% for buffered io_uring, 4.91% for direct pread, and 10.15% for direct io_uring. Every row retained exact tokens/routing, 606,864 logical Reader bytes, 212 completions, 36 L1 hits, and 18 misses.
- Reason: the boundary proves exact ownership, scheduling, and telemetry contracts needed by later storage work, but the tiny current-layer overlap cannot amortize thread and synchronization cost. Measured regressions prohibit making it the default.
- Revisit: with representative multi-expert cache pressure on native Linux, then after ORBIT-style future-layer recall and multiple outstanding L2 requests exist. Any future default still requires simultaneous correctness, traffic, and quality evidence.

## D-030 — Keep exact cache policies runtime-switchable and non-default

- Date: 2026-08-09.
- Status: accepted, implemented, and measured.
- Decision: preserve `disabled` as the public default and `static` as the no-eviction reference. Expose exact `lru`, `lfu`, and `least-stale` policies behind explicit runtime options. Reproduce SpecMD Least-Stale with stale-before-current priority, processed-left-layer priority, upcoming-layer protection, and a farthest-future exact fallback when capacity still requires a victim.
- Alternatives considered: make LRU the general default; make Least-Stale the default from the paper result; retain static-only admission; expose all exact policies while withholding a default change.
- Evidence: deterministic equal-size traces select the intended LRU/LFU victims, protect the complete current Top-K set, count a same-forward future-layer collision, and show LRU collision 1 versus Least-Stale 0. CLI, blocking, deadline, CPU, and CUDA parity tests retain exact tokens, routing, logits, and Reader semantics.
- Benchmark result: B-0010 at the 8-expert synthetic capacity records Least-Stale 23 hits/31 misses/0 collisions/628,080 logical bytes, LRU 20/34/1/632,976, and LFU 19/35/7/634,608. At 16 experts LFU records 36 hits and 606,864 bytes, ahead of Least-Stale's 35 hits and 608,496 bytes. Tiny synthetic timing does not establish a universal winner.
- Reason: runtime switches enable controlled evidence without conflating residency with pruning or changing routing. Policy rankings depend on capacity, and WSL2 warm synthetic evidence is insufficient for a production default.
- Revisit: after native-Linux multi-layer or full-model routing traces, representative expert sizes, controlled warm/cold preparation, task/session priors, and physical NVMe/H2D attribution exist.

## D-031 — Serialize generation within one runtime session

- Date: 2026-08-09.
- Status: accepted and implemented after final review.
- Decision: allow only one active generation per `RuntimeSession`; hold a session mutex across the complete generation call. Independent sessions may execute independently.
- Alternatives considered: leave concurrent callers to share one active policy context; attach a separate selected-set context to every forward and every asynchronous load; serialize one session at the generation boundary.
- Evidence: the store owns one active cycle, layer, and protected set. Concurrent forwards could overwrite this context despite data-race-free mutex access. A regression holds the session guard, starts generation on another thread, proves it cannot finish before release, then verifies exact completion.
- Benchmark result: B-0010 was replaced at measurement code `fd05d95` after the guard was added. All 13 rows retain exact tokens, routing, numerical-error identity, and cache/Reader accounting.
- Reason: session serialization is the smallest correct contract for the current single-context cache and matches one agent session's sequential generation use. Per-forward policy contexts would add complexity before multi-request serving is in scope.
- Revisit: when concurrent serving within one logical session becomes a requirement; then move cycle/layer/protection state into explicit forward-owned contexts.

## D-032 — Keep task priors separate from live routing evidence

- Date: 2026-08-09.
- Status: accepted, implemented, and measured.
- Decision: store loaded aggregate frequency as prior evidence and current-process routing as live evidence. Add an explicit opt-in `profiled` eviction policy using normalized frequency and `prior_strength / (prior_strength + live_observations)` rather than seeding LFU counters or preloading a hot bank.
- Alternatives considered: force-load prior hot experts before generation; seed LFU counters from persisted counts; keep prior/live evidence separate behind a new policy.
- Evidence: deterministic traces preserve a matching prior-hot expert, then reverse the victim after four conflicting live observations. Runtime integration keeps exact tokens, routing, logits, and recurrent state. B-0011 matching prior records 23 hits/31 misses/628,080 logical Reader bytes versus cold profile 21/33/631,344.
- Benchmark result: matching-prior traffic equals Least-Stale at the tested 13,056-byte capacity, but matching and alternate profile decode are 5,006.701 and 4,868.597 tok/s versus Least-Stale 5,900.245 tok/s on the tiny warm WSL2 graph. No default changes.
- Reason: separate evidence makes decay auditable and reversible without speculative L2 reads or hidden counter semantics. The measured bookkeeping cost and capacity-specific behavior require the mode to remain experimental.
- Revisit: after native-Linux full-size routing traces, repository-duration sessions, transition prediction, and controlled profile load/save placement are measured.

## D-033 — Observe and persist profiles only by explicit request

- Date: 2026-08-09.
- Status: accepted and implemented after self-review.
- Decision: collect routing maps only for `profiled` mode or explicit metadata/profile I/O. Persist a bounded canonical v1 text profile with CRC32C and temporary-file rename. Claim process-interruption-safe atomic publication, not power-loss durability.
- Alternatives considered: collect profiles for every runtime session; always persist to an implicit repository path; require explicit observation and explicit input/output paths.
- Evidence: the disabled baseline reports zero metadata, live observations, load bytes, and save bytes. Malformed metadata, checksum damage, duplicate file records, count mismatch, and invalid prior strength are rejected. Canonical round-trip is byte-identical.
- Benchmark result: B-0011 full-generation materialized profile sizes exactly match raw telemetry at 1,439, 1,439, and 1,645 bytes. A review fix prevents the final TTFT sample from overwriting those artifacts and prevents record-cap saturation from producing a self-rejected profile.
- Reason: default-path observation would add unrequested maps and transition work to every token. Explicit paths avoid hidden filesystem state and keep evidence attributable.
- Revisit: add file and parent-directory fsync plus an explicit multi-writer ownership protocol before claiming power-loss durability or concurrent profile writers.

## D-034 — Preserve natural Top-K as default and expose reduced K only as a lossy policy

- Date: 2026-08-09.
- Status: accepted, implemented, measured, and retained as experimental.
- Decision: keep checkpoint natural Top-K immutable and default. Compute the full stable router order, then let fixed or adaptive policy select only an order prefix. Renormalize unbiased scores over that prefix, never substitute resident experts, and exact-load every selected cold expert. External failure/critical signals may raise a fixed/adaptive K floor but never lower it; natural mode ignores the floor.
- Alternatives considered: mutate checkpoint Top-K; permanently prune cold experts; replace selected cold experts with lower-ranked resident experts; preserve the full order and expose a reversible execution prefix.
- Evidence: natural and fixed K16 are exact in PyTorch and C++. The 16-of-24 runtime tests prove fixed K4 matches the Python oracle, cache residency does not change selected IDs, failure count 2 raises fixed K4 to K12, and critical raises it to K16.
- Benchmark result: B-0012 records K4/K8/K12 logical Reader reductions of 40.8%/27.2%/13.6% and tiny CPU decode ratios of 3.24x/1.92x/1.34x against natural K16, but all three diverge in tokens, logits, and recurrent state. Fixed K16 and critical escalation are exact. All adaptive rows choose K16 on the nearly uniform synthetic router, while the bounded rescue row performs 108 exact loads with zero hits and no traffic reduction.
- Reason: routing identity and residency must remain separate for correctness. The measured speed/traffic gains do not justify a default because quality diverges and no full-model coding evaluation exists.
- Revisit: after calibrated full-model routing traces, coding/agentic quality suites, native-Linux physical NVMe and H2D attribution, and SHADOW/PHOENIX signal producers are implemented and jointly measured.

## D-035 — Keep routed accumulation fusion opt-in after representative-dimension regression

- Date: 2026-08-10.
- Status: accepted, implemented, measured, and retained as experimental.
- Decision: expose `none|routed-accumulate` only for `cuda-custom + ffn-block`, keep `none` as the default, preserve router order, and fuse only down-projection contribution scaling plus ordered device accumulation before one final D2H.
- Alternatives considered: fuse gate/up into one launch; recompute SiTU-GLU inside every down-output row; fuse down scaling and accumulation without changing the existing gate/up/SiTU kernels.
- Evidence: gate/up launch fusion needs a distinct dual-output kernel and ownership contract. Naive SiTU/down fusion with the current one-block-per-output-row kernel would recompute the hidden activation for every output row. The selected boundary reuses exact native-MXFP4 inputs and reduces expert-result D2H without changing routing or prepared-token semantics.
- Benchmark result: B-0013 synthetic natural Top-16 improved decode by 11.33% synchronous and 8.91% prefetch and reduced D2H by 51,840 bytes per run. The released 3,584-by-3,072 expert repeated over 16 slots reduced D2H by 4,300,800 bytes, or 93.75%, but increased median latency by 630,394 ns, or 8.01%, and aggregate kernel time by 5.88%. All correctness gates passed; the released fixture is kernel/D2H evidence without routing semantics, not full-model TPS.
- Reason: the narrow fusion proves exact ordered accumulation and removes intermediate host traffic, but representative dimensions show that the current sequential expert launches and accumulation dependency cost more than the saved transfer. Measurement therefore rejects a default change.
- Revisit: after native-Linux profiling identifies the accumulation serialization cost, and when expert-major multi-token verification or a kernel shape that shares hidden activations across output work can amortize it.

## D-036 — Establish token-major exact verification before expert-major scheduling

- Date: 2026-08-10.
- Status: accepted, implemented, and measured as an unoptimized reference; retained non-default.
- Decision: first expose an external draft lifecycle whose proposal contains the current accepted anchor and a bounded candidate prefix, then implement strict greedy token-major target verification. Accept only successive target-argmax matches, commit one target bonus token, and report the exact commit back to the draft provider.
- Alternatives considered: implement expert-major unioning immediately; combine a reduced-Top-K AURORA drafter with verification; fix proposal, acceptance, state, and telemetry semantics independently before either optimization.
- Evidence: DSpark paper arXiv `2607.05147` separates proposal and target verification. DeepSpec commit `005e03b81cec38b7da6399833d609ee89a2587f2` uses an anchor-first proposal, commits accepted prefix plus target token, and crops or updates target and draft caches after verification. K3X native and runtime tests at implementation head `2cf50b4` cover perfect, mismatching, empty, invalid, exhausted-provider, and unused-record paths; perfect and mixed blocks preserve greedy tokens, final KDA/MLA state, full routing/K traces, Reader calls/bytes, and L1 counters.
- Benchmark result: B-0014 greedy/perfect/mixed decode is 171.4333/174.0861/173.2344 tok/s with perfect and mixed acceptance rates 1.0 and 0.25. Every row performs five target decode forwards, reads 665,616 bytes, and preserves tokens, final state, full routing/K, Reader, and L1 counters. The +1.55%/+1.05% deltas are not accepted as acceleration because target work and traffic are unchanged on the tiny WSL2 CPU fixture.
- Reason: a deliberately unoptimized exact reference makes later expert-major fetch amortization measurable against stable token, state, routing, and traffic invariants. It also keeps DSpark, AURORA, and cost-aware policies separate from target correctness.
- Revisit: evaluate expert-major scheduling without changing the now-measured proposal or commit semantics; require unique-expert union and physical traffic evidence before any default claim.

## D-037 — Start expert-major verification with an exact CPU layer-major reference

- Date: 2026-08-10.
- Status: accepted, implemented, and measured as an experimental non-default reference.
- Decision: add `token-major|expert-major` target verification while retaining token-major as default. Restrict the first expert-major path to CPU, incremental generation, natural routing, blocking L2, disabled L1, no runtime profile, and the executable synthetic graph. Execute all proposal positions layer by layer, load each natural-route union expert once per layer/block, preserve per-token router accumulation order, and commit only the verifier-selected state snapshot.
- Alternatives considered: implement the full CUDA multi-token path immediately; replay token-major routes without executing true intermediate states; build the smallest exact CPU layer-major reference first.
- Evidence: route replay cannot validate candidate-dependent KDA/MLA state, logits, routing, or Reader traffic. Immediate CUDA work would combine state rollback, payload unioning, H2D ownership, and kernels in one change. Native and cross-language tests cover stable grouping, vector verification, perfect/mixed exact state and routing parity, provider semantics, Reader reduction, evaluated-versus-committed traces, and failure preflight before side effects.
- Benchmark result: B-0015 perfect expert-major block-2 measures 201.5550 tok/s, 655,824 Reader bytes, and 392 calls versus token-major's 160.1659 tok/s, 665,616 bytes, and 428 calls. It loads 24 unique payloads for 30 assignments. The mixed expert-major row evaluates 8 positions, discards 3, measures 122.6010 tok/s, and increases traffic to 680,304 bytes and 482 calls versus token-major's 163.0028 tok/s. Every row preserves greedy tokens, final state, and committed routing.
- Reason: the CPU reference isolates exact scheduling and rollback semantics and proves that reuse depends on acceptance. The mixed-row regression prohibits a default change and supplies a concrete acceptance-aware scheduling target.
- Revisit: after a learned or self-speculative drafter supplies representative acceptance distributions, then for CUDA expert-major grouping, H2D union accounting, and acceptance-aware dynamic block sizing on native Linux.

## D-038 — Extend exact expert-major verification with a single-expert multi-token CUDA batch

- Date: 2026-08-10.
- Status: accepted, implemented, measured, and retained as experimental non-default execution.
- Decision: preserve the stable CPU expert-major plan and add one CUDA backend primitive that accepts a single native MXFP4 expert plus a flat batch of token latents. Restrict runtime use to `cuda-custom + ffn-block + reused + transient + synchronous + fusion none`, disabled L1, blocking L2, natural routing, no runtime profile, and the executable synthetic graph. Keep token-major as default and CPU expert-major as the portable exact oracle.
- Alternatives considered: wrap repeated scalar calls in temporary VRAM residency; implement one persistent multi-expert kernel; generalize the current native MXFP4 row kernel with a token grid dimension and reuse one expert upload.
- Evidence: temporary residency mixes cache ownership with the scheduling experiment and obscures whether one union payload or an incidental cache hit removed H2D. A persistent multi-expert kernel would combine variable assignment plans, routing, state rollback, mixing, and kernel scheduling before the narrow exact boundary was measured. The selected two-dimensional launcher preserves the existing E2M1/E8M0 arithmetic and lets runtime gather/scatter around one expert group without changing routing or accumulation order.
- Benchmark result: B-0016 preserves exact tokens, final KDA/MLA state, and committed routes in all five CUDA graph rows. On the released 17,547,264-byte expert, batch size two reduces 20-iteration weight H2D from 701,890,560 to 350,945,280 bytes and median latency from 3,444,884 to 1,737,798 ns. Batch size four reduces weight H2D from 1,403,781,120 to 350,945,280 bytes and median latency from 6,705,324 to 2,631,900 ns. Activation H2D and D2H are unchanged within each pair, numerical error is zero, and Compute Sanitizer reports zero errors.
- Reason: the primitive proves physical expert-weight reuse at released dimensions and integrates it with exact block state semantics without introducing cache-policy or lossy-routing ambiguity. The mixed graph row still evaluates rejected suffixes and regresses to 40.7627 tok/s, so neither expert-major verification nor a fixed block size becomes default.
- Revisit: after representative learned or self-speculative acceptance traces exist, and after native-Linux full-layer measurements expose multi-expert group sizes, residency, GPU utilization, memory bandwidth, and physical NVMe/H2D costs. Compare dynamic block sizing and multi-expert persistent scheduling separately before considering a default change.

## D-039 — Treat runtime-profile artifacts as canonical bytes in Git

- Date: 2026-08-10.
- Status: accepted and verified.
- Decision: mark `*.k3xp` as `-text` in `.gitattributes` so Git never rewrites canonical runtime-profile bytes during checkout.
- Alternatives considered: replace the recorded SHA-256 values with Windows checkout hashes; normalize line endings inside the hash test; preserve the committed bytes on every platform.
- Evidence: the B-0011 summary hashes match the LF Git blobs, while Windows `core.autocrlf=true` changed those ASCII profile artifacts to CRLF and made the existing integrity test fail. With `-text`, all five profiles materialize as `i/lf w/lf`, and the recorded helpful/conflicting SHA-256 values match the working files.
- Benchmark result: not applicable; this is artifact integrity. CPU CTest 13/13 and Python 262 passed/47 skipped after the correction.
- Reason: `.k3xp` includes canonical checksummed content. Checkout-dependent byte rewriting invalidates evidence and is not a supported text-editing behavior.
- Revisit: only if a future profile format defines a separate normalized textual representation whose checksum explicitly excludes transport line endings.

## D-040 — Establish replay AURORA before persistent self-draft state

- Date: 2026-08-10.
- Status: accepted, implemented, and measured as a non-default experimental reference.
- Decision: produce real self-speculative candidates by replaying the committed prefix through a separate CPU fixed-reduced-Top-K K3X runtime, preserve the existing natural strict target verifier, and adapt proposal length over `{1,2,4}` using observed prefix survival plus measured expert-union cost.
- Alternatives considered: treat scripted traces as representative; integrate an unavailable Kimi K3 DSpark checkpoint; implement persistent KDA/MLA draft state and adaptive scheduling simultaneously; first establish a replay oracle and measured feedback contract.
- Evidence: DeepSpec `005e03b8` truncates DSpark proposals from prefix confidence and updates draft context from verified target state. B-0015/B-0016 show opposite fixed-block outcomes at 1.0 and 0.25 acceptance. K3X has no Kimi K3 DSpark checkpoint or persistent draft-state crop contract. At implementation heads `c20e28c` and `5723f59`, replay candidates equal an independent fixed-K4 greedy oracle, target feedback reaches the provider before its next proposal, ordinary greedy draft counters remain zero, and focused CPU/artifact tests pass.
- Benchmark result: B-0017 measures natural greedy at 1140.3391 tok/s. All six replay rows preserve exact target output but regress decode by 46.35% to 62.52%. Fixed block-2 expert-major is best at 611.7589 tok/s with acceptance 1.0, target Reader 1,102,416 bytes, and additional draft Reader 1,454,112 bytes. Adaptive token/expert rows accept 0.5, replay 2,181,168 draft bytes, and measure 447.3694/427.4438 tok/s.
- Reason: replay is slow but executes the real reduced-routing graph, produces non-scripted acceptance, isolates draft telemetry, and supplies an oracle for later persistent state without combining two correctness boundaries.
- Revisit: after persistent draft-state parity proves identical proposals with less replay work; then evaluate reduced precision, resident experts, and confidence prediction separately. Do not tune the block thresholds around this tiny trace before removing prefix replay.

## D-041 — Persist AURORA with bounded KDA checkpoints and MLA logical crop

- Date: 2026-08-10.
- Status: accepted, implemented, and measured as a non-default experimental path.
- Decision: add an opaque incremental draft cursor that prefills verified context once, snapshots fixed-size KDA state, marks append-only MLA logical sizes, restores the target-accepted prefix, and teacher-forces the target bonus token. Retain complete-prefix replay as the exact oracle and expose persistent execution only through a new non-default mode.
- Alternatives considered: deep-copy complete KDA/MLA state per candidate; replay from periodic context checkpoints; use one mutable state with bounded KDA snapshots and MLA crop; introduce copy-on-write MLA pages immediately.
- Evidence: B-0017 attributes 1,454,112 to 2,181,168 additional logical draft Reader bytes and 13 to 20 replay positions to complete-prefix replay. DeepSpec `005e03b8` crops its speculative draft cache and carries forward only verified target context. Artifact-backed tests prove full accept, rejection rollback, zero proposal, malformed-commit atomicity, fresh-oracle flattened-state equality, replay candidate equality, and token/expert-major target parity. ASan/UBSan covers the CPU cursor and Compute Sanitizer reports zero errors for the combined persistent AURORA plus CUDA expert-major target path.
- Benchmark result: B-0018 measures four exact replay/persistent pairs. Persistent fixed block-2 reduces logical draft Reader bytes by 45.96% and improves paired decode by 14.97% token-major and 14.55% expert-major. Persistent adaptive reduces draft bytes by 63.08% and improves paired decode by 41.75%/27.08%. It replays zero context positions, prefills five once, and preserves proposal counts, acceptance, target tokens, final state, and committed routes.
- Reason: the accepted boundary removes repeated weight execution without replacing it with context-proportional MLA copies, preserves strict target ownership, and keeps paging, precision, residency, and serialization outside one correctness change.
- Revisit: after representative native-Linux/full-layer traces expose whether KDA checkpoint copying or per-token draft weight execution dominates. Consider paged copy-on-write MLA only for multi-branch or VAULT requirements, and evaluate resident, GPU, or reduced-precision drafting as separate axes.
- Publication: PR #23 was rebase-merged at public integration head `30bbf7a8`; branch, pull-request, and post-merge correctness runs `31340338639`, `31340340063`, and `31340476396` passed. This evidence does not promote AURORA from its accepted non-default experimental status.

## D-042 — Reject transient synchronous CUDA drafting as a default

- Date: 2026-08-10.
- Status: accepted implementation boundary, measured, and rejected as a default.
- Decision: implement one exact opt-in CUDA backend for persistent AURORA drafting while keeping CPU drafting as the default and replay CPU-only. Fix the CUDA identity to FP32, reused allocation, transient weights, grouped execution, `ffn-block`, synchronous transfer, fusion `none`, and zero resident/pinned capacity. Keep target and draft telemetry independent.
- Alternatives considered: start with bounded resident draft weights; start with BF16 or mixed precision; combine residency and reduced precision; first isolate exact transient CUDA placement against the existing CPU draft oracle.
- Evidence: provider and CLI tests prove CPU/CUDA proposal, cursor, acceptance, target token, final-state, and committed-route parity; CPU builds fail closed without fallback; separate counters leave the CPU target at zero while CUDA draft kernel/H2D/VRAM counters are positive. Full verification passes CPU CTest 14/14 with pytest 278/50, liburing/direct CTest 15/15 with pytest 284/44, ASan/UBSan CTest 15/15, and CUDA CTest 23/23 with pytest 319/9. Compute Sanitizer reports zero errors.
- Benchmark result: B-0019 measures four exact CPU/CUDA pairs. CUDA draft regresses decode by 96.465% fixed token, 96.810% adaptive token, 97.000% fixed expert, and 96.219% adaptive expert. It adds 5,843,840 to 6,428,224 H2D bytes, 37,471,088 to 54,549,680 ns aggregate draft kernel time, 410 to 451 synchronizations, and 44,448 peak VRAM bytes per run.
- Reason accepted: the exact path establishes capability, ownership, failure, and telemetry contracts needed for later GPU experiments. The measured transient path is not competitive because repeated weight transfer and synchronous fine-grained launches dominate the tiny draft graph.
- Reason rejected as default: all four paired throughput results are strongly negative, the evidence is synthetic and WSL2-bound, and no coding-quality or full-model benefit offsets that cost.
- Revisit: after bounded exact draft-weight residency or persistent multi-token/multi-expert execution removes repeated H2D. Evaluate reduced precision only as a separate quality-measured axis, and do not combine it with the first residency experiment.
- Publication: PR #25 was rebase-merged at public integration head `7899a7ae`; push, pull-request, and post-merge correctness runs `31343260116`, `31343261633`, and `31343401178` passed.
