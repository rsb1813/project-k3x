# K3X Current Project State

## Current milestone

Milestone 13 exact token-major speculative verification, scripted CLI telemetry, B-0014, full local verification, and final self-review are complete. Public PR integration remains.

State recorded on 2026-08-10 from public documentation head `7748ca3` and M13 implementation head `2cf50b4`. The active branch is `codex/milestone-thirteen-speculative-verification`. PR #11 and PR #12 are merged, their README sections are public, and M13 remains on the development branch until final verification and publication gates complete.

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

## Work in progress

- Milestone 12 implementation and B-0013 measurement are complete. `routed-accumulate` reduces intermediate D2H and improves the tiny synthetic graph, but the released 3,584-by-3,072 repeated-expert fixture is 8.01% slower in median latency. The mode remains experimental and `none` remains the default.
- Milestone 13 design, pure proposal/verification/provider contract, separate incremental runtime entrypoint, scripted CLI, telemetry schema, and B-0014 runner are implemented. Strict greedy target verification accepts only successive argmax matches and commits one target bonus token.
- B-0014 perfect and mixed proposal blocks preserve greedy token, final KDA/MLA state, complete routing/K, Reader, L1, and five target-forward behavior. Measured +1.55%/+1.05% decode deltas are not accepted as acceleration because target work and traffic are identical.
- DSpark learned drafting, confidence scheduling, parallel target execution, expert-major scheduling, EcoSpec, MoE-Spec, and AcceptMoE remain unimplemented. The accepted interface is lifecycle-compatible, not checkpoint- or tensor-ABI compatible with DeepSpec.
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
- Full-model quality, coding/agentic quality, adaptive Top-K and cold-rescue effectiveness, speculation, proxy, and pruning remain unmeasured; speculation, proxy, and pruning remain unimplemented.

## Next concrete tasks

1. Publish Milestone 13 through a green branch/PR/main sequence.
2. Record the public integration commit and CI runs in the ledger.
3. Start expert-major scheduling without changing the measured target routing or commit semantics.

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

B-0014 shows that token-major speculative verification does not reduce target work or weight traffic. Greedy, perfect block-2, and mixed block-2 all perform five target decode forwards and read 665,616 bytes while preserving exact state/routing. The measured +1.55%/+1.05% decode deltas are fixture variation, not an accepted speedup.

B-0013 remains the latest CUDA bottleneck evidence. Eliminating intermediate expert-result D2H improves the tiny natural Top-16 graph but the released 3,584-by-3,072 repeated-expert fixture is 8.01% slower because sequential expert launches and ordered accumulation dominate.

The next speculative bottleneck is the absence of expert-major multi-token reuse. The immediate CUDA bottleneck remains sequential per-expert gate/up/down launch and ordered accumulation dependency at released dimensions, while KDA, MLA, routing, residual/state, and non-FFN orchestration remain CPU-driven.

The immediate result is to retain CUDA MoE fusion `none` alongside natural routing with `disabled + blocking + pread + buffered` as defaults. Native-Linux physical NVMe traffic, controlled warm/cold behavior, full-model locality, coding quality, GPU utilization, and memory bandwidth remain unknown.

The derived uncached full-model expert traffic remains 25.83 GB/token, but it is not a measured full-model value. Native-Linux NVMe traffic, cache reuse, and full Kimi K3 throughput remain unknown.

## Last known-good state

- M13 implementation head `2cf50b4` provides the pure contract, incremental reference, scripted CLI, telemetry, and B-0014 tooling. CPU CTest 12/12 and Python 245 passed/44 skipped succeeded before measurement.
- B-0014 result/ledger commit: `e2e37bf`. Final self-review found no Critical or Important issue; default greedy behavior, proposal prevalidation, exact parity gates, and measured/proposed documentation boundaries remain intact.
- B-0014 summary JSON/CSV SHA-256: `7cd834b1c65d507367320170cdf72ca76aace9f6a743da85a0a9f0cca4a21062` / `9c5fdba84c547f93e2a0a7d4c0b76412181ffb2c635ffd969537a154950ce75b`; raw-summary and exact-parity cross-check passed.
- Post-measurement verification: CPU CTest 12/12 and pytest 245/44; liburing/direct CTest 13/13 and pytest 247/42; CUDA CTest 21/21 and pytest 281/8; ASan/UBSan CTest 13/13 and targeted pytest 26/3 with 104 deselected; perfect and mixed CUDA speculative Compute Sanitizer runs both 0 errors.
- Public documentation head `7748ca3` includes dedicated English README sections for merged Milestones 11 and 12. Public `main` correctness run `31322913019` succeeded.
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
