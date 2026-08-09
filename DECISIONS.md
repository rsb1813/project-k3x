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
