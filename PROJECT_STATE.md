# K3X Current Project State

## Current milestone

Milestone 14 exact CPU expert-major speculative verification is implemented, measured, fully verified, final-reviewed, and publicly integrated. Token-major verification remains the default. Milestone 15 CUDA expert-major design preparation has started, but no Milestone 15 runtime implementation or benchmark is claimed yet.

State audited on 2026-08-10 against public documentation baseline `46105f8`. The active development branch is `codex/milestone-fifteen-cuda-expert-major`; this documentation reconciliation is isolated on `codex/docs-sync-m11-m14`. PR #11, PR #12, PR #13, documentation PR #14, and Milestone 14 PR #15 are merged; their implementations, dedicated English README sections, measurements, and ledgers are public.

## Completed work

- Milestone 0 deterministic synthetic K3-compatible PyTorch graph, K3X v1 streaming converter, strict Python/C++ readers, and independent portable C++20 runtime.
- KDA, Gated MLA, Attention Residual, Stable LatentMoE, router, native MXFP4, full/incremental state, layer/logit/state parity, and exact greedy token tests.
- Public Milestone 1 at `254a9ac` with explicit CPU, `cuda-dense`, and `cuda-custom` identities; CUDA 13.3 native `sm_120`; cuBLASLt dense FP32/BF16; exact native-byte MXFP4 CUDA; deterministic JSON/CSV profiling; and B-0002.
- Milestone 2 runtime switches for `per-operation|reused`, `transient|resident`, and `scalar|grouped`, with reference defaults preserved.
- Tracked CUDA allocations, grow-only scratch buffers, reusable CUDA events and cuBLASLt plans, and exact live/peak VRAM counters.
- Stable tensor identities and bounded static FP32, BF16, and native MXFP4 L0 weight residency. Capacity misses bypass to the exact transient path; this Milestone 2 table remains distinct from Milestone 9 L1 expert eviction.
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
- Runtime `disabled|static` L1 expert-cache identity, hard capacity pairing, explicit hit/miss/bypass/current/peak counters, and backward-compatible disabled default.
- Immutable whole-expert native MXFP4 handles keyed by layer/expert, complete-expert atomic admission, stable zero-copy hits, overflow guards, loader failure atomicity, no-eviction capacity, and exact transient bypass.
- Shared payload lifetime across CPU/operation, synchronous CUDA FFN-block, and asynchronous prepared-transfer execution without changing router selection or device payloads.
- Reader calls/requested/completed byte accounting in runtime and benchmark JSON/CSV, explicitly labeled as logical file reads rather than physical NVMe traffic.
- Four-case B-0006 runner crossing disabled/static L1 with synchronous/prefetch transfer, including strict option/provenance, token/routing, traffic, capacity, cache-counter, and raw-artifact checks.
- B-0006 FP32/BF16 measurement with three warmups and 20 samples per row, raw JSON/CSV, compact manifest, and programmatic compact/raw cross-check.
- Independent `pread|io_uring` and `buffered|direct` Reader axes, persistent Linux descriptor, ordered extent batches, and one six-extent native MXFP4 expert request.
- Optional liburing engine with explicit offsets, bounded queue depth, stable completion identity, partial-submit draining, and fail-closed unavailable capability behavior.
- Explicit `O_DIRECT` path gated by `STATX_DIOALIGN`, owned aligned bounce buffers, exact logical slice reconstruction, and separate logical versus aligned-storage accounting.
- Runtime/benchmark L2 identity, alignment, batch/completion/error, Reader storage time, and Linux process-I/O fields plus a capability-aware four-case B-0007 runner.
- B-0007 WSL2 ext4 measurement with three warmups and 20 samples per row, raw JSON/CSV, exact token/routing/logical-byte parity, and an explicit non-authoritative environment label.
- Deterministic streaming `k3-storage-slice-v1` source writer that physically materializes one released-dimension expert in bounded chunks without sparse holes.
- K3X optional `STORAGE_FIXTURE` identity, strict source kind/config/shape/length validation, resumable conversion, and gate/up/down physical execution order.
- Python/C++ Reader support for storage-fixture identity plus an explicit `NON_EXECUTABLE_ARTIFACT` generation guard before graph tensor lookup.
- Dedicated exact six-extent `k3x_storage_bench` with ordered digest, latency percentiles, logical/storage/process-I/O accounting, and no token schema.
- Four-case B-0008 runner with capability-only skips, compact/raw cross-checks, and WSL2 ext4 3-warmup/20-sample measurements.
- Content-addressed storage-fixture shard publication, bounded shard/tensor SHA-256 validation, manifest-scoped source identity, and canonical source-matched resume extent validation.
- Bounded single-worker expert-load scheduler ordered by absolute latest-start time, with stable ties, inline resident completion, capacity failure, error propagation, idle draining, and structured telemetry.
- Opt-in `blocking|deadline` L2 expert scheduling with unchanged blocking default and exact current-layer natural Top-K submission after routing.
- Same-layer overlap of non-resident expert loading with routed-down and shared-expert computation, followed by exact wait-before-use semantics.
- Mutex-protected Reader and L1 telemetry snapshots, one pre-submit per-layer fetch estimate, and success/error idle barriers that keep captured Reader/store lifetimes inside generation.
- Eight-case B-0009 runner crossing schedule, Reader engine, and cache mode with capability-only skips, exact token/routing/logical-I/O parity, raw JSON/CSV, and WSL2 ext4 3-warmup/20-sample measurements.
- Runtime-switchable exact `lru`, `lfu`, and `least-stale` L1 policies with unchanged `disabled` default and `static` no-eviction reference.
- Session-monotonic token-forward cycles, complete natural Top-K selected-set protection, deterministic victim ordering, exact capacity bypass, eviction counters, and same-forward collision-miss telemetry.
- SpecMD Least-Stale paper reproduction with stale-before-current priority, processed-left-layer priority, upcoming-layer protection, and deterministic LRU collision 1 versus Least-Stale 0 trace coverage.
- Session-wide generation serialization that prevents concurrent forwards from overwriting the store's active policy context while leaving independent sessions independent.
- Thirteen-case B-0010 runner crossing disabled plus four policies at 2-, 8-, and 16-expert capacities with exact token/routing/numerical-error identity, raw JSON/CSV, and WSL2 ext4 3-warmup/20-sample measurements.
- Bounded canonical `.k3xp` runtime profiles with validated metadata, prior/live expert frequency, adjacent-layer transitions, deterministic hot bank, CRC32C, and sibling temporary publication.
- Opt-in exact `profiled` eviction with explicit prior strength and live-observation crossover; runtime metadata remains outside prompt IDs and default sessions do not observe profile state.
- CLI profile load/save and metadata controls plus bytes/time/prior telemetry, exact resume parity, malformed-input rejection, bounded record behavior, and full-generation artifact-size evidence.
- Five-case B-0011 runner comparing LFU, Least-Stale, cold profile, matching prior, and minimum-overlap alternate prior with exact token/routing/logit/state identity and WSL2 ext4 3-warmup/20-sample measurements.
- Natural/fixed/adaptive Top-K execution with exact selected cold-expert rescue, external quality floors, 16-of-24 Python/C++ parity, and B-0012 quality/traffic ablation.
- Opt-in `none|routed-accumulate` CUDA MoE identity for `cuda-custom + ffn-block`, with ordered down-projection contribution scaling and device accumulation for synchronous and prepared-prefetch paths.
- Fused-call/expert telemetry, strict contribution and prepared-token validation, FP32/BF16 scalar/grouped graph parity, and a CPU oracle for the routed expert-group contract.
- B-0013 four-case synthetic natural Top-16 end-to-end ablation plus a bounded released-dimension repeated-view kernel/D2H ablation. Synthetic fusion improves decode, while the representative dimension regresses latency and therefore remains non-default.
- Stable first-use expert-major assignment planning and complete-vector strict greedy verification with malformed-route, duplicate, non-finite, size, range, and callback failure coverage.
- Exact CPU layer-major proposal execution with per-position KDA/MLA state snapshots, natural per-layer route unions, one payload load per unique expert, original router-slot accumulation order, and committed-only canonical routing/state adoption.
- `token-major|expert-major` CLI and benchmark identity, evaluated-versus-committed routing telemetry, block/position/union/assignment/payload counters, and fail-closed CPU/L1/L2/routing/profile capability checks.
- B-0015 five-case greedy/token-major/expert-major perfect/mixed ablation with exact token/state/committed-route parity and raw JSON/CSV/summary SHA-256 cross-checks.

## Work in progress

- Milestone 14 library/runtime execution, CLI identity, telemetry, B-0015, the full applicable verification matrix, final evidence review, PR, merge, and post-merge CI gates are complete.
- Milestone 15 is at design preparation only. The intended exact CUDA expert-major boundary, multi-token grouping contract, and H2D union accounting have not yet been accepted, implemented, or measured.
- B-0015 perfect expert-major blocks reduce logical Reader traffic and improve tiny CPU decode, while the mixed row evaluates three rejected positions and regresses both traffic and decode. The evidence therefore supports exact reuse semantics but not a default change.
- The current expert-major path is CPU-only and intentionally excludes L1 caching, deadline scheduling, reduced/adaptive routing, and profile observation. CUDA expert-major scheduling, H2D unioning, acceptance-aware block sizing, and cross-layer prediction remain unimplemented.
- DSpark learned drafting, confidence scheduling, EcoSpec, MoE-Spec, and AcceptMoE remain unimplemented. The accepted interface is lifecycle-compatible, not checkpoint- or tensor-ABI compatible with DeepSpec.
- Milestone 9 Terra high final review found one valid Important shared-session policy-context issue. Commit `fd05d95` serializes complete generation calls; re-review found no remaining Critical or Important issue and withdrew an initial collision interpretation concern after deterministic future-layer trace review.
- Static, LRU, LFU, Least-Stale, profiled eviction, and all non-default L2 modes remain experimental and opt-in. Transition prediction and cross-layer asynchronous L2 scheduling remain unimplemented.
- The L2 batch API submits concurrent operations for one batch but waits before returning. It is not the chartered N/N+1/N+2 deadline pipeline yet.
- The deadline worker schedules only the current routed layer and remains slower than blocking in all B-0009 rows. ORBIT, multiple L2 workers, eviction-aware priority, and future-layer recall are not implemented.
- Natural routing, `pread + buffered`, blocking scheduling, disabled L1, and CUDA MoE fusion `none` remain defaults because B-0007 through B-0013 are WSL2 evidence, not native P44 Pro or full-model evidence.
- Worktree: `C:\Users\jolib\Documents\project-k3x\.worktrees\milestone-one-runtime`.
- Linux Python environment: `/home/jolib/.venvs/k3x-m1`; builds: `build-cpu`, `build-uring`, `build-cuda`, and `build-uring-asan`.

## Known failures and blockers

- Windows Smart App Control still blocks unsigned `k3x_run.exe`; WSL2 is the verified local CUDA path and native Linux remains the final performance authority.
- The executable checkpoint is synthetic and tiny. The full-dimension artifact contains only one non-executable expert. No full Kimi K3 weights have been downloaded, and B-0008 is neither token throughput nor a full-model claim.
- The graph remains CPU-driven outside FFN blocks. KDA, MLA, routing, score mixing, residual work, state management, and non-FFN boundaries remain on the host.
- L1 misses use ordered Reader batches into pageable host vectors. Deadline mode may run one such blocking batch on a worker, but there is no cross-layer asynchronous L2 pipeline, transition predictor, or N/N+1/N+2 triple buffering.
- Exact prefetch is single-flight and limited to `cuda-custom + ffn-block + reused + transient`; it is not combined with static residency.
- `cuda-dense` intentionally keeps native MXFP4 on the CPU as its documented comparison identity. `cuda-custom` is the exact GPU MXFP4 path.
- GPU utilization, GPU memory bandwidth, and physical NVMe GB/token remain unmeasured. Reader storage elapsed time, logical/aligned bytes, process `rchar/read_bytes`, and L1-to-L0 timing are measured under their stated scopes.
- WSL2 `/mnt/c` is 9p/DrvFS and rejects direct alignment. liburing 2.5 and direct I/O were validated on WSL2 ext4 `/tmp`, which remains a correctness/capability environment rather than native P44 Pro performance authority.
- ThreadSanitizer builds successfully but its runtime exits under WSL2 with `unexpected memory mapping`; no TSan execution result is claimed. ASan/UBSan and explicit concurrency regressions pass.
- Full-model quality, coding/agentic quality, full-checkpoint adaptive Top-K and cold-rescue effectiveness, and speculative acceleration remain unmeasured. Exact token-major and CPU expert-major verification are implemented and measured only on the synthetic fixture; learned drafting, CUDA expert-major execution, proxy, and pruning remain unimplemented.

## Next concrete tasks

1. Design the next CUDA expert-major boundary around multi-token expert grouping, H2D union accounting, and acceptance-aware block sizing without changing target semantics.
2. Add representative learned or self-speculative acceptance traces before choosing a dynamic verification block policy.
3. Preserve token-major as the default until native-Linux physical I/O, RTX 5080 execution, and quality evidence justify a change.

## Hardware assumptions

| Component | Current assumption or observation |
|---|---|
| CPU | AMD Ryzen 7 9800X3D target and local host |
| GPU | NVIDIA GeForce RTX 5080, 16,303 MiB, compute capability 12.0 |
| Driver | 591.86 |
| CUDA | Toolkit 13.3.1, nvcc 13.3.73, native `sm_120` |
| RAM | 96 GB DDR5-4200 target; WSL2 exposed 49,251,213,312 bytes during earlier validation |
| NVMe | Solidigm P44 Pro 2 TB target; not measured by B-0007 or B-0008 |
| Final runtime OS | Linux native |
| Current development OS | WSL2 Ubuntu 24.04.4 on Windows 11 |

## Latest measured bottleneck

B-0015 shows that expert-major reuse is acceptance-sensitive. Perfect block-2 loads 24 unique expert payloads for 30 assignments, reduces Reader bytes from 665,616 to 655,824 and calls from 428 to 392, and measures 201.5550 tok/s versus token-major's 160.1659. Mixed block-2 evaluates eight positions but commits five, increasing Reader bytes to 680,304 and calls to 482 while measuring 122.6010 tok/s versus token-major's 163.0028.

B-0013 remains the latest CUDA bottleneck evidence. Eliminating intermediate expert-result D2H improves the tiny natural Top-16 graph but the released 3,584-by-3,072 repeated-expert fixture is 8.01% slower because sequential expert launches and ordered accumulation dominate.

The next speculative bottleneck is avoiding rejected-suffix work through representative drafting and acceptance-aware block sizing before moving exact unioning to CUDA. The immediate CUDA bottleneck remains sequential per-expert gate/up/down launch and ordered accumulation dependency at released dimensions, while KDA, MLA, routing, residual/state, and non-FFN orchestration remain CPU-driven.

The immediate result is to retain CUDA MoE fusion `none` alongside natural routing with `disabled + blocking + pread + buffered` as defaults. Native-Linux physical NVMe traffic, controlled warm/cold behavior, full-model locality, coding quality, GPU utilization, and memory bandwidth remain unknown.

The derived uncached full-model expert traffic remains 25.83 GB/token, but it is not a measured full-model value. Native-Linux NVMe traffic, cache reuse, and full Kimi K3 throughput remain unknown.

## Last known-good state

- Public documentation baseline `46105f8` includes the fully synchronized Milestone 14 README and TITAN Ledger. Its final `main` correctness run `31329200483` succeeded.
- PR #11 merged at `edc6d605` and PR #12 merged at `9e59a9db`; both merge commits are ancestors of the audited public baseline. Their post-merge correctness runs `31318993688` and `31322191670` succeeded.
- M14 exact block runtime head `862d401`, CLI/telemetry head `bdf4a66`, and B-0015 result head `1e73121` form the current development state.
- Pre-publication verification passed CPU CTest 13/13 and Python 253/44, liburing/direct CTest 14/14 and Python 257/42, CUDA CTest 22/22 and Python 291/8, and ASan/UBSan liburing CTest 14/14 plus targeted Python 95/35. CUDA FFN Compute Sanitizer reported zero errors; the CPU-only expert-major CLI has no instrumented CUDA API and is covered by ASan/UBSan rather than a fabricated Compute Sanitizer result.
- B-0015 artifact SHA-256 is `29f3fd10c95dcde9f2b012e10e36962363b5cdd79dfeda5f5e3bbaca0cb89b75`; canonical aggregate-record SHA-256 is `cb95eff274713a21b821695d75ff2655da735513c99215ec5ec14f5ed995b813`.
- Independent evidence validation recomputed all ten raw JSON/CSV hashes, the aggregate hash, exact diagnostic parity, and the six headline percentage deltas. The report is ready to share with the stated synthetic/WSL2/logical-I/O caveats.
- Public Milestone 14 integration head is `012e598`. Branch/PR correctness runs `31328853375`/`31328869071` and post-merge `main` correctness run `31329045623` succeeded. PR #15 is merged.
- M13 implementation head `2cf50b4` provides the pure contract, incremental reference, scripted CLI, telemetry, and B-0014 tooling.
- B-0014 result/ledger commit: `e2e37bf`. Final self-review found no Critical or Important issue; default greedy behavior, proposal prevalidation, exact parity gates, and measured/proposed documentation boundaries remain intact.
- Public M13 integration head `463e9ca`; branch/PR correctness runs `31324378917`/`31324381376` and post-merge main run `31324492327` succeeded. Publication head `f0641de` and publication-main run `31324647692` also succeeded. PR #13 is merged.
- B-0014 summary JSON/CSV SHA-256: `7cd834b1c65d507367320170cdf72ca76aace9f6a743da85a0a9f0cca4a21062` / `9c5fdba84c547f93e2a0a7d4c0b76412181ffb2c635ffd969537a154950ce75b`; raw-summary and exact-parity cross-check passed.
- Post-measurement verification: CPU CTest 12/12 and pytest 245/44; liburing/direct CTest 13/13 and pytest 247/42; CUDA CTest 21/21 and pytest 281/8; ASan/UBSan CTest 13/13 and targeted pytest 26/3 with 104 deselected; perfect and mixed CUDA speculative Compute Sanitizer runs both 0 errors.
- Public documentation synchronization head `4984728` includes dedicated English README sections for merged Milestones 11, 12, and 13, a Milestone 13 badge, and a reconciled current-state ledger. Push/PR/post-merge correctness runs `31325118488`, `31325132631`, and `31325294591` succeeded, and PR #14 is merged.
- Public Milestone 12 publication head: `dc23020706bcca7a68b9a643c055f7627e31698b`; branch/PR/integration-main correctness runs `31322043556`, `31322049903`, and `31322191670` succeeded, and final publication-main run `31322330041` succeeded. PR #12 is merged.
- Latest verified Milestone 12 result head: `0632a0f`; measurement code is `58c36dd`.
- B-0013 synthetic artifact SHA-256: `edeaa4802b4bfac0624fa4d0e73917318076258d95e74e880c97a8b2709dd2d2`; released storage SHA-256: `aab7aea48b03bdcd8e0b4d98c4780128ab689d2bba005089a49970eb0e326890`.
- B-0013 synthetic summary SHA-256: `996dad640c78ea356b1b9d13fb7879e07511cba42e7257a6c43fa95b7f274da7`; released summary SHA-256: `d6f186fb991c67e2c4a1cd4929816ca1cf5567b187a905dd447db99258fd1799`.
- CPU verification: CTest 11/11; pytest 235 passed and 44 skipped.
- Liburing/direct verification: CTest 12/12; pytest 237 passed and 42 skipped.
- ASan/UBSan liburing verification: CTest 12/12; targeted pytest 49 passed and 5 skipped with 57 deselected.
- CUDA verification: CTest 20/20; pytest 271 passed and 8 skipped.
- Compute Sanitizer: ten CUDA test binaries plus the `k3x_cuda_expert_bench` released-dimension routed-accumulate invocation each report `ERROR SUMMARY: 0 errors`.
- Every B-0013 synthetic row generated `[56, 55, 18, 11, 11, 13]` with exact routing/K identity and maximum absolute error `2.4e-7`.
- B-0013 raw JSON/CSV, checksummed manifest, and independently cross-checked summaries: `results/b0013-fused-routed-accumulation/`.

## Proposed component status

APOLLO, TITAN COUNCIL, AURORA, PROMETHEUS-X, MERCURY, ORBIT, HELIOS, SHADOW, PHOENIX, VAULT, VEILBREAK, AUTO, and SKYFORGE remain proposed only. ATLAS, CHRONOS, and BLACKSTAR remain reserved without accepted definitions. None is claimed as implemented or benchmarked.
