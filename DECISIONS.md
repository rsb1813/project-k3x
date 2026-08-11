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

## D-043 — Retain bounded exact CUDA draft residency as opt-in

- Date: 2026-08-10.
- Status: accepted, implemented, and measured as a non-default experimental path.
- Decision: expose `--aurora-draft-resident-bytes` only for `aurora-persistent + cuda-custom`. Preserve zero as the exact transient identity, map a positive capacity to the existing tensor-ID-keyed no-eviction `ResidentWeightTable`, and use exact transient bypass when the hard capacity cannot admit a weight. Keep CPU drafting as the default, replay CPU-only, and target/draft telemetry independent.
- Alternatives considered: add dynamic L0 eviction and prediction in the same change; build a new persistent multi-token/multi-expert kernel first; combine residency with BF16 or mixed precision; reuse the existing exact static table before changing policy, kernels, or arithmetic.
- Evidence: provider tests cover full-fit admission and one-byte exact bypass with identical proposals and cursor lifecycle. CLI tests prove fail-closed ownership and CPU-build unavailability without fallback. B-0020 preserves proposals, acceptance, target tokens, final KDA/MLA state, and committed routing in all four transient/resident pairs. Full verification passes CPU CTest 14/14 with pytest 284/53, liburing/direct CTest 15/15 with pytest 290/47, ASan/UBSan CTest 15/15, and CUDA CTest 23/23 with pytest 328/9. Compute Sanitizer reports zero errors for both 8 MiB full-fit and one-byte bypass expert-major paths.
- Benchmark result: B-0020 reduces draft weight H2D by 88.809% for fixed rows and 89.775% for adaptive rows. The complete observed working set occupies 644,160 or 647,424 bytes, cache hit rate is 75.682% or 77.273%, and bypasses are zero at 8 MiB. Paired decode changes are +15.582% fixed token, -2.561% adaptive token, +22.673% fixed expert-major, and +5.569% adaptive expert-major.
- Reason accepted: the boundary removes most repeated draft weight transfer without changing routing, arithmetic, proposals, or target authority, and proves a hard-cap exact fallback contract that later cache policy work can reuse.
- Reason not promoted to default: throughput is mixed even on the tiny synthetic graph, 410–451 synchronous waits and fine-grained launches remain, static no-eviction residency does not model full Kimi K3 pressure, and coding quality/native-Linux/full-model traffic remain unmeasured.
- Revisit: after persistent multi-token/multi-expert CUDA execution isolates launch/synchronization amortization. Dynamic eviction/prediction and reduced precision remain separate policy and quality axes.
- Publication: PR #27 was rebase-merged at public integration head `c88456c0`; push, pull-request, and post-merge correctness runs `31346575341`, `31346587586`, and `31346725071` passed.

## D-044 — Implement a resident rectangular expert grid before CUDA Graphs or a whole-device draft graph

- Date: 2026-08-10.
- Status: accepted; portable contract, exact CPU oracle, low-level CUDA primitives, and complete resident backend implemented, runtime integration and benchmark pending.
- Decision: add an opt-in `resident-grid` CUDA batching identity that evaluates equal-shaped native MXFP4 experts and token inputs with four grid-wide launches while returning separate expert/token outputs for the existing stable CPU accumulation order. Require exact resident weights and use whole-request serial fallback on any hard-cap bypass.
- Alternatives considered: cache a CUDA Graph for every ordered routed-expert set; move KDA, MLA, routing, argmax, and draft state to a complete device-resident graph; first implement the smaller dependency-closed resident expert grid.
- Evidence: B-0020 retains 410 to 451 synchronizations after removing most weight H2D. An Nsight Systems diagnostic at public head `01eac162` observed 1,040 kernel launches and 1,346 async copies across ten CUDA draft forwards, while aggregate GPU kernel duration was only about 1.13 ms. This supports reducing operation granularity before adding routing-set graph-cache policy.
- Reason accepted: the grid attacks repeated expert launch overhead without changing KDA/MLA state, routing, arithmetic order, proposals, or target authority, and exposes a multi-token contract that later expert-major consumers can reuse.
- Reason CUDA Graphs deferred: ordered expert-set reuse and a safe bounded graph-cache policy are unmeasured, so graph capture would combine execution and caching policy in one experiment.
- Reason whole-device graph deferred: it would simultaneously replace attention, recurrent state, routing, argmax, and token lifecycle boundaries, making correctness attribution too broad for one milestone.
- Revisit: compare direct-grid launch counts and end-to-end B-0021 results first. If host activation round trips still dominate, move next to a device-resident layer or whole-token graph rather than stacking speculative graph caching.

## D-045 — Retain the resident CUDA expert grid as an opt-in path

- Date: 2026-08-10.
- Status: accepted, implemented, and measured as a non-default experimental path.
- Decision: expose `resident-grid` only for exact `cuda-custom + ffn-block + reused + resident + synchronous + fusion-none` execution. Require the complete requested expert set to be resident, execute gate/up/SiTU/down as four grid-wide launches, preserve separate expert/token outputs and stable CPU accumulation order, and fall back for the whole request to the existing exact serial path on any residency bypass.
- Alternatives considered: promote the grid to the CUDA default after the synthetic result; cache ordered routed-set CUDA Graphs; move the complete draft token graph and recurrent state to the device; retain grouped execution as the only CUDA path.
- Evidence: direct 1x1, 1x4, 2x2, and 4x4 native-MXFP4 cases match the CPU oracle; the 4x4 diagnostic has maximum absolute error `5.96e-08`; capability, full-fit, one-byte bypass, token-major, expert-major, counter, and artifact tests pass. CPU CTest 14/14 with pytest 290 passed/55 skipped, liburing/direct CTest 15/15 with pytest 296 passed/49 skipped, ASan/UBSan CTest 15/15, and CUDA CTest 24/24 with pytest 336 passed/9 skipped all pass. Compute Sanitizer reports zero errors for the direct grid tests.
- Benchmark result: B-0021 reduces MoE launches by 75% in all four matched pairs. Grid decode improves over grouped by 10.794% fixed token, 24.086% adaptive token, 38.005% fixed expert-major, and 21.857% adaptive expert-major, while preserving exact target tokens, final state, routing, acceptance, Reader bytes, and resident weight H2D. Grid fallbacks are zero.
- Reason accepted: the result validates the smallest dependency-closed launch-amortization boundary without changing routing, arithmetic, proposal lifecycle, target authority, or defaults.
- Reason not promoted to default: AURORA currently supplies one token per grid call, total draft H2D increases slightly, host orchestration and activation round trips remain, and the benchmark is a tiny synthetic WSL2 workload without coding-quality or full-model evidence.
- Revisit: repeat on native Linux and representative layer dimensions, then compare a device-resident layer/token graph against routed-set CUDA Graph caching. Keep dynamic eviction and reduced precision as separate policy and quality experiments.
- Publication: PR #29 was rebase-merged at public integration head `90b20c87`; push, pull-request, and post-merge correctness runs `31351465644`, `31351486146`, and `31351649761` passed.

## D-046 — Join the exact resident MoE layer before graph caching

- Date: 2026-08-10.
- Status: accepted, implemented, and measured as a non-default experimental boundary.
- Decision: add an opt-in `moe-layer` CUDA boundary that keeps CPU routing unchanged but executes routed-down, resident expert grid, ordered contribution mixing, RMSNorm, routed-up, shared SiTU MLP, and final addition on one stream with one final result copy and synchronization.
- Alternatives considered: cache CUDA Graphs keyed by ordered routed expert set; move the complete KDA/MLA/router/argmax token graph to the device; first join only the dependency-closed MoE feed-forward layer.
- Evidence: B-0021 fixed grid rows record 30 grid calls, 470 draft stream synchronizations, 108,800 activation-H2D bytes, and 102,880 D2H bytes after exact weight residency. Adaptive rows record 33 calls and 517 synchronizations. The previous model executed routed-down, shared MLP, expert grid, CPU mix/norm, and routed-up as separate subcalls. The implemented runtime now dispatches one complete resident layer before those split computations and reuses the same routing decision, payloads, and contributions on exact hard-cap fallback. Focused CPU ownership/parity tests pass 104/35 and CUDA ownership/parity tests pass 133/6. Independent target/draft runtime and benchmark telemetry passes CPU schema coverage 13/8 and live CUDA coverage 19/2. NVIDIA's CUDA Graph documentation requires explicit definition/instantiation/execution and distinguishes parameter update from topology-changing re-instantiation.
- Reason accepted: the selected boundary directly removes three synchronization and intermediate-copy boundaries per successful MoE call without changing recurrent attention, router selection, proposal lifecycle, or target verification.
- Reason graph cache deferred: ordered expert-set reuse, pointer-update cost, graph count, and bounded eviction are unmeasured. Combining those policies with a new execution boundary would prevent clean attribution.
- Reason whole-token graph deferred: KDA/MLA state, residuals, routing, logits, argmax, and rollback would all change at once.
- Benchmark result: B-0022 records exactly three fewer synchronizations per successful layer call, 14,880/16,368 fewer activation-H2D bytes, 26,880/29,568 fewer D2H bytes, and 14,496/15,984 fewer total-H2D bytes for fixed/adaptive rows. Paired decode changes are +5.619%, -2.753%, -1.216%, and +3.933%.
- Reason not promoted to default: throughput is mixed, the graph is tiny and WSL2-bound, and representative dimensions, native-Linux timing, physical PCIe traffic, and coding quality remain unmeasured.
- Revisit: use B-0022 to decide whether the next boundary should be a CUDA Graph over the stable MoE layer or a larger device-resident token graph. Do not combine reduced precision or dynamic eviction with the first layer measurement.
- Publication: PR #31 was rebase-merged at public integration head `97eb3e4e`; branch, pull-request, and post-merge correctness runs `31355460022`, `31355471896`, and `31355678835` passed. CodeQL run `31355471922` also passed.

## D-047 — Account for the routed norm as a real L0 weight

- Date: 2026-08-10.
- Status: accepted implementation correction and measured.
- Decision: admit the routed RMSNorm vector through the resident weight table and count its uploaded bytes as weight H2D. Compare the layer-minus-split weight-H2D delta with the resident-weight-byte delta, and gate B-0022 on lower total H2D rather than equal weight H2D.
- Alternatives considered: exclude the norm upload from weight telemetry; classify it as activation traffic; force the split path to upload an unused norm vector; record the physical cold admission honestly.
- Evidence: the split path executes routed RMSNorm on CPU, while exact layer execution consumes the norm on GPU. Therefore only the layer path requires the norm in L0, and equal weight H2D cannot be true on a cold process.
- Reason accepted: reclassifying or hiding the upload would corrupt the physical traffic model, and adding an unused split upload would distort the reference merely to satisfy a benchmark gate.
- Benchmark result: B-0022 measures a positive 384-byte layer-minus-split weight-H2D delta and the same 384-byte resident-weight delta in all four pairs. Activation savings exceed this admission, so total H2D still decreases.
- Revisit: B-0022 must confirm the positive weight delta equals the positive resident-byte delta and that activation savings still reduce total H2D. Warm-session measurements may later separate first-admission and steady-state traffic.

## D-048 — Remove repeated immutable-weight validation before graph selection

- Date: 2026-08-10.
- Status: accepted, implemented, and measured as an opt-in experimental mode.
- Decision: keep `moe-layer` experimental and non-default, defer CUDA Graph and larger device-token work, and first move immutable dense/vector finiteness validation out of the per-call hot path into a correctness-preserving admission or construction boundary. Rerun the same released-dimension matrix before selecting another execution boundary.
- Alternatives considered: accept the complete layer because it reduces synchronization and transfer traffic; immediately cache a thirteen-operation CUDA Graph; skip directly to a device-resident KDA/MLA/router token graph; attribute and remove the repeated host validation cost first.
- Evidence: corrected B-0023 uses released hidden 7,168, latent 3,584, intermediate 3,072, and 1/4/16 native expert views under a 1 GiB hard cap, with the split oracle destroyed before selected-backend measurement. All physical gates pass, yet layer median latency is 20.488/20.954/24.422 ms versus split 1.228/2.371/5.681 ms, or +1568.624%/+783.911%/+329.883%. Aggregate kernel time over 20 iterations rises much less, from 15.122/24.507/58.396 ms to 22.971/27.692/61.887 ms. The complete-layer preflight currently scans 469,776,384 immutable dense/vector bytes for finite values on every call before resident lookup and launch.
- Benchmark result: every pair has maximum error 0, zero fallback/bypass, zero warm weight H2D, synchronization 80→20, lower activation H2D and D2H, and exact 14,336-byte cold/resident norm delta. Traffic savings do not compensate for the current host-side path.
- Reason accepted: graph caching would optimize launches while leaving a larger unamortized O(weight-bytes) host operation in place, and a whole-token graph would broaden the correctness surface before the isolated layer boundary is sound.
- Correctness invariant: malformed dimensions, duplicate IDs, non-finite parameters, non-finite immutable tensors, invalid MXFP4 payloads, and hard-cap behavior must still fail or bypass exactly as documented before any CUDA mutation. Validation may be cached only against immutable tensor identity and lifetime.
- Benchmark result: B-0024 records exact per-call warm scan volume of 9,395,527,680 bytes and admission warm scan volume of zero over 20 calls. Profiler-off complete-layer medians fall by 93.629%, 90.643%, and 78.708% at 1, 4, and 16 experts, with maximum error 0, zero warm weight H2D, zero bypass/fallback, and profiler-independent physical counters.
- Revisit: consider CUDA Graphs only after ordered routed-set reuse, graph update/re-instantiation cost, and a bounded graph-cache policy are measured.

## D-049 — Keep admission validation opt-in behind an immutable-host contract

- Date: 2026-08-10.
- Status: accepted.
- Decision: retain `per-call` as the general backend and CLI default. Expose `admission` only for exact resident MoE-layer execution and require admitted host pointer, length, shape, and allocation lifetime to remain immutable for the backend lifetime.
- Alternatives considered: promote admission globally after B-0024; hash all 469,776,384 bytes on every call; add a public prepared-layer token now; keep only per-call validation.
- Evidence: tensor ID plus pointer/shape identity makes repeat validation constant-time and B-0024 removes 78.7% to 93.6% of median boundary latency. It cannot detect in-place mutation behind the same pointer. The current K3X runtime owns immutable checkpoint buffers, but the public backend API also accepts caller-provided spans.
- Reason accepted: opt-in ownership captures the measured speedup without weakening the reference contract or expanding the public API. Transactional six-view preflight and conflict tests preserve failure atomicity before CUDA mutation.
- Publication: PR #38 was rebase-merged at public integration head `e24cac2`; push and pull-request correctness runs `31363433423`/`31363437230`, pull-request CodeQL `31363437226`, and post-merge `main` correctness/CodeQL `31363673811`/`31363673857` all passed.
- Revisit: promote admission only when runtime-owned checkpoint allocations carry an enforceable immutable/prepared identity, or introduce a narrowly scoped prepared-layer handle with lifetime ownership.

## D-050 — Retain bounded CUDA Graph execution as opt-in

- Date: 2026-08-10.
- Status: accepted, implemented, and measured as an experimental non-default path.
- Decision: keep direct `disabled` execution as the default. Expose whole-executable `update` and hard-capped ordered-set `cache` only for exact FP32, reused-allocation, resident, resident-grid, admission-validated MoE-layer execution. Keep target and persistent CUDA AURORA draft ownership independent.
- Alternatives considered: replace direct launches with one mutable graph; enable an unbounded graph per routed set; graph the whole token including KDA/MLA/router; retain only direct execution; measure one-executable update and bounded capacities one/two/four against direct execution.
- Evidence: CUDA 13.3 graph update requires compatible topology, graph objects are not thread-safe, and the existing layer has stable resident weights and grow-only scratch but variable ordered expert identities. The implementation validates the captured 3-H2D + 13-operation + 1-D2H topology, inserts explicit timing nodes, invalidates on scratch identity changes, and owns graph resources with bounded RAII entries.
- Benchmark result: B-0025 preserves maximum error 0, zero warm weight H2D, zero bypass/fallback, one synchronization, and thirteen logical kernels per call across 15 rows. Stable and alternating mode deltas are mixed from -4.407% to +4.466%. Rotating cache capacities with 20 misses and 20 evictions are 6.091%–11.574% slower, and rotating update is 6.794% slower.
- Reason accepted: the bounded implementation supplies exact lifecycle and attribution evidence without weakening the reference path. The mixed result and severe churn penalty do not support a default change.
- Rejected default: neither small mixed stable/alternating deltas nor synthetic AURORA ownership establishes real K3 ordered-set reuse, end-to-end token throughput, physical PCIe/NVMe traffic, or coding quality.
- Publication: PR #40 was rebase-merged at public integration head `13a403f`; branch and pull-request correctness runs `31371133295`/`31371136825`, pull-request CodeQL `31371136804`, and post-merge `main` correctness/CodeQL `31371387067`/`31371387081` passed.
- Revisit: collect real K3 routing traces and native-Linux end-to-end timing, then test interaction with dynamic L0 residency. Consider a larger device-token graph only as a separate correctness and performance decision.

## D-051 — Harden K3X v1 before bounded real-checkpoint discovery

- Date: 2026-08-10.
- Status: accepted, implemented, measured, and publicly integrated.
- Decision: retain K3X v1 and harden its generic external-input boundary before discovering any official real shard. Enforce rooted source containment and exact tensor-to-shard ownership, bounded strict safetensors metadata, canonical resume-ledger schema, full committed-prefix validation, and exact truncation of uncommitted partial suffixes.
- Alternatives considered: repeat the already implemented D-028 bounded-fixture hash checks; introduce signed manifests and a new provenance-bearing format immediately; harden the existing format narrowly and defer publisher authenticity until real-source discovery defines the required identity.
- Evidence: D-028 already proves declared shard/tensor hashes, canonical extent-prefix reuse, source CRC, and partial CRC for the released-size bounded fixture. The remaining generic path accepted traversal and ambiguous ownership, parsed unbounded or structurally loose safetensors metadata, trusted loose ledger JSON, and left crash residue after the committed prefix.
- Boundary evidence: a stopped synthetic conversion commits exactly 20,736 bytes after two extents while the next aligned boundary is 24,576. Treating alignment padding as committed would reject a valid interruption, so the recoverable boundary is the final verified extent's exact `offset + length`.
- Benchmark result: B-0026 completes fresh, clean-resume, and 8,192-byte orphan-resume scenarios with a 257-byte maximum individual source read and a Reader-valid 1,421,568-byte final artifact. Both resume cases reuse two verified extents; corruption tests remain fail-atomic.
- Reason accepted: this is the smallest boundary that makes externally supplied shard discovery safe enough to begin without changing the on-disk format or claiming an unauthenticated publisher identity.
- Rejected claims: the audit does not prove power-loss durability, signed supply-chain provenance, peak RSS, real-checkpoint compatibility, token throughput, GPU behavior, physical NVMe traffic, or model quality.
- Publication: PR #42 was rebase-merged at public integration head `ca8c544e`; push and pull-request correctness runs `31379029215`/`31379074639`, pull-request CodeQL `31379074656`, and post-merge `main` correctness/CodeQL `31379311743`/`31379311695` passed.
- Revisit: Milestone 26 must define content-addressed official-source discovery and publisher provenance before a bounded real shard is downloaded. Signed provenance or a format revision should be reconsidered from that evidence rather than guessed in advance.

## D-052 — Accept pinned exact-range provenance for the first real expert smoke

- Date: 2026-08-10.
- Status: accepted, implemented, and measured for a non-executable bounded storage artifact.
- Decision: resolve the official public snapshot to one fixed commit, verify API/index/config/header identities, and download only the exact contiguous layer-1 expert-0 range. Label the result `transport-pinned-range`; never label it `full-shard-verified`.
- Alternatives considered: API inventory only; download and hash the complete 16.99 GB shard first; pinned index plus exact header/payload ranges.
- Evidence: the official index binds 497,220 tensors to 96 shards. The selected six w1/w2/w3 U8 tensors share shard 2 and form one exact 17,547,264-byte range. B-0027 verifies the index LFS digest, config Git blob, exact 206 responses, tensor hashes, content-addressed microshard, Reader-valid K3X, and runtime rejection. Final review additionally demonstrated that raw `//` and `/./` paths are rejected before normalization and that consistently rehashed repository, snapshot, config, index, expert-layout, and artifact mutations fail strict verification.
- Benchmark result: 59,799,719 metadata bytes, 818,704 header bytes, and 17,547,264 tensor-payload bytes were returned across 11 HTTP requests in 14.972839499 seconds. This is conversion wall time, not token throughput or physical NVMe performance.
- Reason accepted: it reaches real native-MXFP4 bytes with bounded cost while keeping the weaker provenance level explicit and preserving the non-executable storage-fixture boundary.
- Rejected claims: no complete shard/full checkpoint verification, token generation, GPU execution, model quality, NVMe GB/token, PCIe traffic, or publisher signature is established.
- Publication: PR #44 was rebase-merged at public implementation head `5b6345db`; both branch and pull-request correctness passed, all pull-request CodeQL checks passed, and post-merge `main` correctness `31386873905` and CodeQL `31386873928` succeeded.
- Revisit: production conversion must recompute complete source-object identity or adopt an equivalently authenticated chunk scheme. M27 must separately prove real-weight CUDA layer correctness before widening the dependency closure.

## D-053 — Prove one official expert FFN before widening real-weight closure

- Date: 2026-08-10.
- Status: accepted, implemented, measured, and publicly integrated through PR #46 at `ec08b827`.
- Decision: add a dedicated benchmark-only path that hard-binds the B-0027 K3X root and ordered expert digest, uses the portable CPU backend as oracle, and measures transient and resident RTX 5080 execution without changing `k3x_run`.
- Alternatives considered: reuse the released-dimension executable unchanged; add caller-selected official labels to that executable; add a pinned official-expert harness.
- Evidence: B-0028 reports transient/resident medians of 2,508,377/331,868 ns with identical `3.0267983675e-9` maximum absolute error. Both cold calls transfer 17,547,264 weight bytes; twenty transient calls transfer 350,945,280 bytes while resident transfers zero measured weight bytes and records 60 hits. CUDA CTest passes 28/28 and the resident official-expert Compute Sanitizer reports zero errors.
- Reason accepted: the dedicated harness is the smallest boundary that proves actual official MXFP4 CUDA computation while keeping synthetic released evidence, production generation, and official provenance distinct.
- Rejected claims: one expert FFN is not a real full MoE layer, routing result, token throughput, full-checkpoint runtime, physical NVMe measurement, or quality result.
- Publication: branch correctness `31455570571`, pull-request correctness `31455597581`, and pull-request CodeQL `31455597565` passed. PR #46 was rebase-merged at `ec08b827`; post-merge `main` correctness `31455776634` and CodeQL `31455776673` passed.
- Revisit: native Linux and a real changing routed set are required before residency becomes a default or the one-expert latency ratio is used in a model-level projection.

## D-054 — Close a real MoE FFN sublayer before a full transformer layer

- Date: 2026-08-11.
- Status: accepted, implemented, and measured through B-0029; public integration is pending.
- Decision: the next real-weight boundary binds the official router, computes all 896 scores, preserves natural Top-16 selection, materializes and executes the exact selected routed experts, executes the real shared expert, and verifies the complete mixing/residual FFN sublayer against an independent reference.
- Alternatives considered: repeat the single expert with more IDs; benchmark a caller-selected four-expert set; close the real MoE FFN sublayer; download enough attention/KDA/MLA tensors for a complete transformer layer immediately.
- Evidence: B-0028 proves exact single-expert CUDA execution and shows that warm exact residency removes 350,945,280 measured weight-H2D bytes over twenty calls, but it has no routing, shared-expert, changing-set, or output-mixing semantics. A repeated or caller-selected set would not close those correctness gaps, while a complete transformer layer would widen source and graph dependencies before the MoE boundary is independently proven.
- Benchmark result: B-0029 measures 97,095,781 ns transient and 10,153,939 ns resident medians for route A, plus a 20,201,466 ns resident median for an alternating A+B sequence. It implies no token, full-layer, quality, physical NVMe, or native-Linux result.
- Reason accepted: the real FFN sublayer is the smallest boundary that tests natural routing and exact selected-weight residency without conflating it with attention and recurrent-state integration.
- Rejected claims: this decision does not authorize a complete shard/full checkpoint download, paid cloud execution, a residency default, or a full-model TPS projection.
- Publication: PR #48 rebase-merged the complete M28 lineage at `eb2c20860ee9c7c612b9b74984170bd8b4443ba1`. Branch and pull-request correctness, pull-request C++/Python CodeQL, and post-merge `main` correctness/CodeQL all passed.
- Revisit: after the real FFN sublayer passes parity and traffic gates, choose between full transformer-layer closure and multi-layer routing/residency traces from measured evidence.

## D-055 — Persist routes before fetching the selected expert union

- Date: 2026-08-11.
- Status: accepted, implemented, and exercised by the bounded official fixture.
- Decision: manufacture the M28 fixture in two durable phases. First persist all always-active objects and the exact natural A/B route manifest. Then fetch only the first-use selected expert union, assemble gate/up/down physical order, and convert one non-executable K3X fixture.
- Alternatives considered: fetch all possible experts before routing; keep routes only in process memory; split trunk and experts into separate K3X artifacts; persist one dependency-closed fixture.
- Evidence: the official-source and converter recovery matrix passes 149 tests. Focused orchestration and CLI coverage passes 27 tests, including an assertion that the route manifest exists before the first expert plan/fetch. Content-object tests distinguish fresh download, damaged-partial restart, complete-partial finalization, and verified completed-object reuse.
- Benchmark result: the first materialization transferred exactly 941,412,864 tensor-payload bytes; a verified reuse run transferred zero. These are manufacturing counters, not token throughput or physical NVMe measurements.
- Reason accepted: a durable route boundary makes restart reuse deterministic and prevents speculative or preferred-expert substitution, while one fixture preserves a single Reader and identity transaction for the later CPU/CUDA oracle.
- Rejected claims: bounded materialization and sublayer execution do not mean a complete shard/checkpoint, token generation, or coding quality has been validated.
- Revisit: after the one authorized bounded fixture is materialized, compare observed selected-union size and actual downloaded bytes with the 32-expert upper bound before changing packing or concurrency.

## D-056 — Separate natural route derivation from pure portable MoE execution

- Date: 2026-08-11.
- Status: accepted and implemented at `8a13cf5`; the separate CUDA consumer is implemented at `bb634e1`.
- Decision: expose one pure function that derives the canonical natural route from BF16 hidden/router weights and correction bias, and a separate pure CPU oracle that consumes the validated route plus BF16/MXFP4 views. Both remain dimension-driven; official released dimensions belong to the later pinned fixture validator.
- Alternatives considered: embed routing inside the CPU oracle; reuse the production synthetic `ModelSession`; call the existing backend interface at every operation; keep a small standalone pure oracle.
- Evidence: the C++ test fixes BF16 decode patterns and every tiny graph boundary, while Python/PyTorch independently recomputes the same hidden state, route, contributions, two expert outputs, mixed latent, routed/shared outputs, and final vector. CPU CTest passes 17/17 and `test_cpp_parity.py` passes 113 with 32 capability skips.
- Benchmark result: none. The tiny portable graph is a correctness oracle and has no token semantics or performance authority.
- Reason accepted: separating route identity from execution lets CPU and CUDA consume exactly the same selected IDs/contributions and makes malformed route data fail before output without coupling the oracle to production session state.
- Rejected claims: this does not prove CUDA parity, official full-size execution, B-0029, token throughput, cache behavior, or quality.
- Revisit: keep the split unless the pinned harness reveals a route-manifest identity that cannot be validated before execution.

## D-057 — Keep the official MoE CUDA boundary byte-native and opt-in

- Date: 2026-08-11.
- Status: accepted, implemented, and validated on the bounded official 32-expert artifact.
- Decision: add one dedicated CUDA method that consumes Task 3 prepared hidden/prefix data, canonical routes, raw BF16 views, and native MXFP4 experts. Keep both transient and bounded exact resident modes, one final output D2H, and no production dispatch/default change.
- Alternatives considered: convert BF16 tensors to host FP32 and reuse generic dense calls; compose existing public backend calls with intermediate D2H transfers; add a dependency-closed byte-native CUDA boundary.
- Evidence: the focused transient/resident fixture matches the portable oracle within `2e-2`, selected order is exact, caller buffers are unchanged, malformed aliases/routes/capacity fail, and the second resident call adds zero weight H2D while increasing cache hits. CPU CTest passes 17/17, CUDA CTest passes 30/30, and Compute Sanitizer reports `ERROR SUMMARY: 0 errors`.
- Benchmark result: B-0029 records official released-dimension sublayer latency and logical CUDA-copy traffic. It records no token rate, quality, physical PCIe, or NVMe result.
- Reason accepted: this closes the execution contract needed by the pinned fixture without hiding BF16 expansion in host memory or weakening the fail-closed production artifact boundary.
- Rejected claims: this does not prove official full-size parity, changing Top-16 residency pressure, end-to-end token generation, quality, or a CUDA default.
- Revisit: evaluate kernel fusion, residency policy, and a wider device boundary only after the pinned official B-0029 result exposes real dimensions and traffic.

## D-058 — Bind the final artifact root into the durable route manifest

- Date: 2026-08-11.
- Status: accepted and implemented at `a109409` and consumed by `bdab0da`.
- Decision: preserve the pre-fetch route-manifest publication, then atomically add a final artifact record containing filename, K3X root, source digest, and per-source-tensor digests only after conversion and Reader verification succeed.
- Alternatives considered: hard-code a root before materialization; trust a caller-supplied artifact independently from its routes; create a second execution manifest; finalize the existing route manifest after conversion.
- Evidence: the materializer test proves the route manifest exists before expert planning and that the successful final record carries the Reader-observed root. The harness independently rejects malformed/fixed-identity manifests and a generic storage fixture before CUDA construction; 18 synthetic tests and all three actual-fixture smoke cases pass.
- Benchmark result: the B-0029 artifact and route manifest are bound by root `1287d84bbfa02e849ab786808107fbfbfe14459477bf79e3048b2ebb6bdff288` and source digest `d9e4425a11ca71b53abce52b8f120bd257740fc93cbe63df4c1fc3b7465cee35`.
- Reason accepted: the same manifest now binds route derivation and exact artifact bytes without weakening restart durability or inventing a root before the real bytes exist.
- Rejected claims: a self-consistent transport-pinned manifest is not a complete-shard signature, publisher attestation, official full-size execution, or performance result.
- Revisit: consider a separate signed execution manifest only if production manufacturing needs immutable pre- and post-conversion records with distinct retention policies.

## D-059 — Fix B-0029 to three non-ranking evidence rows

- Date: 2026-08-11.
- Status: accepted, implemented, and measured at `bf147fa` after two documented fail-closed corrections.
- Decision: run exactly A transient, A resident, and alternating resident with one process per row, 3 warmups, and 20 measured iterations. Validate each row before writing and never rerun or rank timings.
- Alternatives considered: add B transient; search over warmups/iterations; rank repeated runs; keep the smallest matrix that isolates exact residency and changing-route union behavior.
- Evidence: the first attempted formal run stopped before output because the verifier required exact floating-point contribution equality instead of the accepted `1e-6` tolerance. The second stopped before output because the transient allocation formula omitted 64 per-call temporary allocations. Both defects were corrected before the sole published matrix. Strict verification now rehashes and validates all three raw rows, aggregate, summary JSON, and LF CSV.
- Benchmark result: A transient/resident medians are 97,095,781/10,153,939 ns; alternating resident is 20,201,466 ns per A+B sequence. Both resident rows have zero warm weight H2D. Maximum absolute error is zero for A and `0.00048828125` for alternating.
- Reason accepted: the three rows answer the immediate correctness and residency questions without turning the formal run into a timing search.
- Rejected claims: tool implementation is not measured evidence, official execution, token throughput, quality, or physical traffic.
- Revisit: add B transient only if the first formal matrix exposes a case-specific transient attribution that A cannot represent.

## D-060 — Advance to one complete official transformer-layer boundary

- Date: 2026-08-11.
- Status: accepted as the next milestone; not implemented.
- Decision: after publishing M28, close one layer-1 transformer boundary by adding the smallest dependency-complete KDA/MLA/Attention Residual inputs around the measured MoE FFN. Keep the resulting artifact non-executable in the production token loop until whole-layer parity passes.
- Alternatives considered: optimize the current FFN kernels immediately; collect multi-layer routing traces without attention state; close one complete transformer layer first.
- Evidence: B-0029 establishes exact natural Top-16 routing, shared-expert execution, contribution mixing, residual output, and resident traffic at the FFN boundary. It still begins after attention output and therefore cannot expose KDA/MLA state traffic, full-layer scheduling, or token-loop latency. Optimizing the isolated FFN now risks improving a boundary that may not dominate once recurrent-attention work is included.
- Benchmark result: B-0029 resident A spends about 7.40 ms per call in measured CUDA kernels and about 2.76 ms in remaining boundary orchestration, but there is no full-layer or token timing.
- Reason accepted: one complete layer is the smallest next boundary that can identify whether compute, orchestration, state movement, or weight residency dominates before broader checkpoint manufacturing.
- Rejected claims: this decision does not authorize a full checkpoint download, Cloud Run execution, end-to-end TPS projection, or production default change.
- Revisit: after one complete official layer has independent parity and measured traffic, decide whether to optimize/fuse the layer locally or proceed to multi-layer routing and cache traces.

## D-061 — Close layer 1 at an explicit recurrent-state boundary

- Date: 2026-08-11.
- Status: accepted and implemented through bounded manufacturing, independent Python/C++ KDA oracles, pinned preflight, official portable execution, and native CUDA complete-layer execution; formal B-0030 measurement remains pending.
- Decision: supply deterministic layer-1 hidden/source-bank vectors and explicit zero KDA state, then execute two tokens through the complete layer both together and incrementally. Require checkpoint-authoritative F32 `A_log[128]` and explicit V-first recurrent-state storage. Keep the final artifact non-executable through `k3x_run`.
- Alternatives considered: include embeddings and layer 0; stop at a KDA-only official boundary; close layer 1 around explicit state inputs.
- Evidence: the pinned source blob matches repository metadata, layer 1 is KDA, the exact 17-tensor header payload is 887,843,840 bytes, and the KDA paper defines channel-wise decay. The checkpoint header exposes `A_log[128]` while the source constructor initializes `[96]`, so silent source imitation would be incorrect.
- Benchmark result: none. Header and source identity checks are metadata observations, not a performance benchmark.
- Reason accepted: this is the smallest boundary that exposes recurrent attention, both Attention Residual halves, natural routing, MoE, state movement, and final output without importing a separate layer-0 MLA/dense dependency closure.
- Rejected claims: no B-0030, token rate, quality result, physical traffic, full-shard integrity, or production default exists yet. The native complete-layer result is a bounded benchmark-only capability path.
- Revisit: after B-0030, use measured KDA/MoE/kernel/orchestration and residency data to choose whole-layer fusion or bounded multi-layer tracing.

## D-062 — Extend the M28 manufacturing transaction for the KDA layer fixture

- Date: 2026-08-11.
- Status: accepted and implemented; official M29 payload materialization remains pending.
- Decision: reuse M28 content-addressed range objects, bounded resume, source assembly, expert planning, and the existing non-executable optional-feature bits. Add an `official_layer` source-manifest record and a separate atomic route-state manifest rather than introducing a second transport ledger or K3X v2 before execution requirements are known.
- Alternatives considered: define a new complete-layer storage format and optional bit immediately; build a separate KDA downloader/ledger; extend the verified M28 transaction while preserving exact layer execution order.
- Evidence: RED failed on the missing layer interfaces and unsupported CLI scope. The implementation now fixes 17 KDA objects before route derivation, publishes full/incremental V-first state linkage before selected expert requests, packs KDA then M28 MoE tensors in execution order, revalidates the complete plan before payload, and preserves the production non-executable guard. Converter, recovery, source-integrity, layer, MoE, source, and CLI coverage passes 123 tests.
- Benchmark result: none. A live pinned metadata dry-run completed in 12.867 seconds, returned 17 tensors and the accepted byte bounds, and requested zero tensor-payload bytes. This is metadata latency, not model execution or B-0030.
- Reason accepted: one trust and recovery model minimizes new correctness surface, reuses already measured M28 objects after rehashing, and allows the complete-layer execution manifest to mature before committing a new file-format feature bit.
- Rejected claims: no official M29 payload, natural route union, final artifact root, complete-layer parity, CUDA result, token rate, quality result, full-shard verification, or paid-cloud execution exists yet.
- Revisit: introduce a dedicated file-format identity only if the portable/native harness requires semantics that cannot be bound by the source and route-state manifests plus existing non-executable feature bits.

## D-063 — Bind microshard and K3X converter source identities separately

- Date: 2026-08-11.
- Status: accepted and implemented for the M29 route-state manifest and C++ Reader.
- Decision: retain `artifact.source_sha256` as the assembled safetensors microshard SHA-256 and add `artifact.k3x_source_fingerprint_sha256` for the K3X superblock source fingerprint. Expose the latter through the portable Reader, require exact manifest/superblock parity, and reconstruct the deterministic safetensors header plus payload stream to verify the former before backend construction.
- Alternatives considered: compare the two differently defined hashes; drop source identity from C++ preflight; name and validate both identities independently.
- Evidence: converter inspection shows the superblock fingerprint hashes the source-manifest filename/content plus shard filename/content, while the route manifest previously stored the raw microshard file SHA-256. They are intentionally different values. Focused converter/Reader/preflight coverage passes 36 tests with 4 capability skips.
- Benchmark result: none. This is a correctness and provenance boundary.
- Reason accepted: distinct names prevent a false comparison while preserving both the manufactured microshard identity and the exact converter input identity used to seal the K3X artifact.
- Rejected claims: this does not prove the complete upstream 16.99 GB shard digest, signed publisher provenance, official tensor execution, CUDA parity, or performance.
- Revisit: replace transport-pinned range provenance with authenticated chunk or complete-object verification when full-checkpoint manufacturing begins.

## D-064 — Bind source-byte oracle arrays and compare independent accumulation numerically

- Date: 2026-08-11.
- Status: accepted and implemented for the M29 official-weight portable execution gate.
- Decision: make PyTorch projection tokenwise so full and incremental reference calls use the same sequence-one GEMM boundary. Persist the two reference KDA outputs and final convolution/recurrent state in a 6,541,344-byte content-addressed sidecar, bind its SHA-256 into the route manifest, and compare the independent portable scalar implementation with fixed absolute tolerances while keeping route IDs and within-implementation full/incremental parity strict.
- Alternatives considered: require byte-exact equality with PyTorch oneDNN BF16 GEMM; change C++ from FP64 to FP32 scalar accumulation; accept only manifest hashes without numerical arrays; store the source-byte arrays in a separate ignored oracle sidecar.
- Evidence: official dimensions exposed sequence-length-dependent PyTorch BF16 projection output before the tokenwise reference fix. After that fix, C++ FP64 projection hashes exactly match an independent FP64 reduction but not PyTorch oneDNN or FP32 reduction. The bounded source-byte comparison measures maximum absolute differences of `1.52588e-05` for KDA output, `0.0078125` for Q/K convolution state, `0.00390625` for V convolution state, `4.39133e-05` for recurrent state, and `1.06171e-06` for route contributions; both natural Top-16 ID lists remain exact.
- Benchmark result: none. The 155.389-second verified reuse/materialization wall time and roughly 82-second portable scalar execution are manufacturing/debug observations, not B-0030, layer latency, or token throughput.
- Reason accepted: independent implementations cannot honestly promise byte identity across different GEMM reduction orders. A content-bound numerical oracle preserves source-byte authority, makes divergence measurable, and avoids adding PyTorch or oneDNN as a runtime dependency.
- Rejected claims: these tolerances do not establish CUDA parity, complete-layer output parity after MoE, B-0030, token rate, quality, physical storage traffic, or a production default.
- Revisit: native CUDA must report its own source-byte and portable-oracle error. Tighten or separate tolerances only from measured cross-backend evidence, never by hiding a failing value.

## D-065 — Close the first CUDA layer with host routing and exact GPU KDA/MoE

- Date: 2026-08-11.
- Status: accepted and implemented for the M29 benchmark-only boundary.
- Decision: retain both Attention Residual reductions, RMS normalization, and exact all-896 natural routing on the host while executing all large BF16/F32 KDA projections, convolution, decay, V-first recurrence, output projection, and the exact native-MXFP4 MoE FFN on the CUDA backend. Keep transient and exact-resident modes and publish complete KDA state explicitly.
- Alternatives considered: move routing and residual reductions onto the GPU before any official whole-layer result; stop at KDA-only CUDA; compose the already verified host control flow with the verified GPU KDA and MoE boundaries.
- Evidence: tiny transient/resident and full/incremental parity pass; CPU CTest passes 19/19, CUDA CTest passes 34/34, focused Python passes 175 with 8 capability skips, and both new CUDA binaries report zero Compute Sanitizer errors. The bounded resident A-to-B official smoke preserves both exact routes, full/incremental state, and complete-layer output within `0.00048828125`.
- Benchmark result: no formal benchmark. The cold one-sequence smoke records 381,907,507 ns wall time, 32,897,536 ns profiled kernel time, 1,816,322,048 weight-H2D/resident bytes, 13,025,280 state bytes in each direction, and 1,824,612,416 tracked peak device bytes. It used zero warmups and one iteration and is not B-0030.
- Reason accepted: it is the smallest dependency-closed native boundary that tests official KDA state movement and exact MoE composition without conflating the next optimization step with routing/residual fusion.
- Rejected claims: the cold smoke is not token throughput, quality, physical PCIe/NVMe traffic, utilization, bandwidth, native-Linux evidence, or a production default.
- Revisit: after fixed B-0030 evidence attributes warm kernel and orchestration cost, decide whether host routing/residual fusion or a bounded multi-layer trace is the next measured step.

## D-066 — Publish B-0030 as one fixed atomic three-row transaction

- Date: 2026-08-11.
- Status: accepted, implemented, and measured.
- Decision: fix B-0030 to A transient, A-to-B incremental resident, and A+B full resident in that order, with exactly three warmups and twenty measured samples. Require full/incremental BF16 output and final V-first state digests to match, and atomically publish raw JSON, LF-only CSV, summary, and all hashes only after every invariant passes.
- Alternatives considered: allow arbitrary rows and sample counts; rerun or select favorable samples; adapt the strict M28 evidence transaction with complete-layer state/launch formulas and cross-row parity.
- Evidence: the runner tests pass 8/8 and focused CUDA CTest passes 3/3. Actual one-sample probes independently confirm A transient traffic, resident incremental traffic, full KDA's 24-launch topology, and byte-identical full/incremental output/state digests. The final schema additionally binds BF16/F32/MXFP4 copy categories, process peak RSS, and Reader logical/storage counters.
- Benchmark result: formal evidence commit `bbdccb9` records medians of 262,801,334 ns for A transient, 168,577,563 ns for resident A-to-B incremental, and 114,804,882 ns for resident A+B full. Both resident rows have zero warm weight H2D, identical full/incremental output and final-state digests, and maximum absolute error `0.00048828125`.
- Reason accepted: a closed non-ranking transaction prevents post-hoc row/sample selection and turns routing, state traffic, residency, numerical parity, and evidence identity into executable acceptance gates.
- Rejected claims: implementation and smoke success do not constitute B-0030, token throughput, quality, physical PCIe/NVMe traffic, utilization, bandwidth, native-Linux evidence, or a default policy.
- Revisit: schema expansion requires a new version if a later process-level observer supplies physical counters or token semantics; do not silently add them to v1.

## D-067 — Attribute immutable KDA validation before adding a wider official graph

- Date: 2026-08-11.
- Status: accepted as the next bounded experiment.
- Decision: keep the current exact validation default, then isolate per-call immutable KDA validation behind the already established admission-identity contract and benchmark it independently before materializing another official layer.
- Alternatives considered: immediately add a second KDA/MLA layer; tune CUDA kernels first; infer validation cost from the B-0030 wall/kernel gap without another measurement.
- Evidence: resident incremental and full B-0030 kernel totals differ by only 0.416216% per sequence, while the full-call median is 31.897887% lower and orchestration drops by 53.967941 ms per sequence. The current implementation scans immutable KDA weights on every official KDA call, but B-0030 does not time that scan separately.
- Benchmark result: B-0030 identifies a 53.772681 ms wall delta between two incremental calls and one full call; it does not yet attribute that delta to one component.
- Reason accepted: the experiment is smaller, reversible, and evidence-driven. It can remove or retain a validation boundary before wider official materialization multiplies the same cost.
- Rejected claims: no default change, token-rate projection, or claim that validation alone explains the entire wall delta is accepted without the next ablation.
- Revisit: after isolated validation and orchestration counters exist, choose between admission validation, larger device-resident state, or a wider official multi-layer boundary from measured results.

## D-068 — Retain per-call default after accepting exact KDA admission as an opt-in path

- Date: 2026-08-11.
- Status: accepted, implemented, measured through B-0031, and publicly integrated.
- Decision: reuse the backend-wide immutable identity registry for fourteen official KDA views, require exact resident execution and atomic first admission, expose the mode explicitly in the official-layer harness, and retain `per-call` as the global default.
- Alternatives considered: make admission the default immediately; add a KDA-specific cache; bypass backend validation from the benchmark; proceed directly to a wider official graph.
- Evidence: B-0031 preserves identical routes, contributions, output/final-state digests, resident bytes, zero warm weight H2D, and `0.00048828125` maximum error. Incremental/full medians fall by 59.819421%/44.463194%, measured validation time is 103.874127/55.731721 ms per sequence, and paired kernel totals change by less than 0.4%.
- Benchmark result: incremental per-call/admission medians are 175.667985/70.584413 ms; full per-call/admission medians are 121.067320/67.236923 ms. Admission leaves a 3.347490 ms incremental/full gap.
- Reason accepted: one exact identity authority removes a measured repeated host scan without weakening dynamic input/state checks or duplicating trust logic. Keeping the default unchanged prevents a bounded one-layer WSL2 result from becoming a premature production policy.
- Rejected claims: B-0031 is not token throughput, quality, physical memory/PCIe/NVMe traffic, native-Linux evidence, full-model cache behavior, or proof that admission should be globally default.
- Revisit: after a bounded multi-layer or token-loop path exists, compare admission lifetime/identity ownership under actual runtime residency and measure the remaining 3.347490 ms host/API gap before changing defaults.

## D-069 — Attribute device-resident KDA state handoff before widening the official graph

- Date: 2026-08-11.
- Status: accepted and measured as an explicit non-default experiment.
- Decision: keep exact admission opt-in and isolate KDA state residency across incremental calls so token A's state can feed token B without an intermediate 6,512,640-byte D2H plus H2D round trip. Preserve an explicit final-state publication path and the current host-state reference mode.
- Alternatives considered: materialize a second official layer immediately; fuse host Attention Residual and routing first; make admission the default; start Cloud Run/full-checkpoint work.
- Evidence: B-0031 admission incremental/full medians are 70.584413/67.236923 ms with nearly identical 33.889030/33.958984 ms kernel time. Incremental performs two state transfers in each direction while full performs one, leaving a 3.347490 ms wall gap after validation is removed.
- Benchmark result: B-0032 host/device incremental medians are 73.192169/69.835612 ms, a 4.585951% reduction. Device handoff removes exactly 6,512,640 bytes in each state-transfer direction per sequence. Aggregate kernel time changes +0.339801%, orchestration falls 3.207668 ms per sequence, and the device-incremental/full-host median gap is 1.611085 ms.
- Reason accepted: the opaque single-slot handoff is the smallest exact, reversible boundary that removes the measured transfer without new model payload or routing changes. The result supports retaining the experiment, but not changing the host-round-trip default.
- Rejected claims: B-0032 logical transfer accounting is not physical PCIe measurement, and its 4.585951% bounded reduction does not authorize a default change, token-rate projection, full checkpoint download, or paid cloud resources.
- Revisit: after a multi-layer or token-loop boundary exists, compare state lifetime and concurrency policy with wider device-resident residual/routing orchestration. Do not generalize the single-slot design to VAULT or make it default from B-0032 alone.

## D-070 — Measure exact device route preparation before adding another official layer

- Date: 2026-08-11.
- Status: accepted as the Milestone 32 bounded experiment; not implemented or measured yet.
- Decision: keep host routing as the default and add an explicit two-stage CUDA path that computes exact MLP Attention Residual, post RMSNorm, and 896 raw router logits on device, retains prefix/prepared activation behind a single-use opaque token, runs canonical natural Top-16 selection on the host, and consumes the token in the existing exact resident MXFP4 FFN.
- Alternatives considered: materialize a second official layer immediately; issue one monolithic whole-layer CUDA call including device Top-K; optimize existing KDA or expert kernels without first closing the remaining host routing boundary.
- Evidence: B-0032 device incremental/full-host medians differ by only 1.611085 ms, but both retain roughly 35 ms per sequence outside their roughly 34 ms aggregate CUDA kernel totals. The current wrapper performs residual preparation and the 896 by 7,168 router loop on the CPU and re-enters the backend for the FFN.
- Benchmark result: none yet. B-0033 will compare fixed host-routing and device-routing rows only after parity, actual-artifact, sanitizer, and evidence gates pass.
- Reason accepted: this is the smallest reversible boundary that directly removes the current CPU router loop without new official payload, preserves the dynamic expert scheduling point needed by cache/rescue work, and keeps the natural routing policy in one canonical host implementation.
- Rejected claims: no predicted speedup, token rate, quality result, physical PCIe traffic, native-Linux authority, multi-layer behavior, or default-policy change is accepted before measurement.
- Revisit: after B-0033, proceed to bounded multi-layer closure if route preparation is correct and the remaining orchestration is small; otherwise attribute the surviving synchronization or kernel boundary before widening payload scope.
