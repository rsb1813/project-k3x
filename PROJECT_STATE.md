# K3X Current Project State

## Current milestone

Milestone 22 released-dimension resident MoE-layer benchmarking is implemented, measured, documented, reviewed, fully verified, and public through PR #36 at integration head `e4820a18`. B-0023 compares split resident-grid with the complete resident layer at hidden 7,168, latent 3,584, expert-intermediate 3,072, and 1/4/16 repeated-view experts without a full checkpoint. The reviewed benchmark releases the split oracle before selected-backend construction and rejects a default or CUDA Graph decision because the complete boundary is 4.30×–16.69× slower despite lower synchronization and transfer traffic. This follow-up reconciles the README and TITAN Ledger with that public state.

State audited on 2026-08-10 at public Milestone 22 integration head `e4820a18`. Branch/PR correctness runs `31358991710`/`31359003481`, CodeQL `31359003436`, post-merge `main` correctness `31359158926`, and post-merge CodeQL `31359158878` passed. Pre-review Milestone 22 verification passed CPU CTest 14/14 with Python 305 passed/67 skipped, liburing/direct CTest 15/15 with Python 307 passed/65 skipped, ASan/UBSan CTest 15/15, and CUDA CTest 26/26 with Python 362 passed/10 skipped. Post-correction publication gates pass CUDA CTest 26/26 and focused live/evidence Python 22/22. The released one-expert MoE-layer Compute Sanitizer run reports zero errors. No paid cloud resource or full Kimi K3 checkpoint is in use.

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
- Portable exact single-expert multi-token FFN contract with CPU scalar oracle, full prevalidation, and zero-default batch call/token telemetry.
- Native MXFP4 CUDA batch launcher with a two-dimensional row/token grid, scalar batch-size-one parity, multi-token CPU-oracle parity, and unchanged E2M1/E8M0 semantics.
- Synchronous transient CUDA batched expert FFN that uploads one activation batch and one gate/up/down expert payload, executes gate/up/SiTU/down for all grouped tokens, and returns one flat output batch.
- Exact CUDA expert-major runtime gather/batch/scatter integration with strict capability preflight and unchanged token-major default, natural routing, expert union order, router-slot accumulation order, and committed-only state adoption.
- B-0016 five-case CUDA graph ablation plus four-case released-dimension scalar/batch measurement with exact token/state/route parity, raw JSON/CSV digest 9/9, canonical aggregate verification, and released batch Compute Sanitizer coverage.
- AURORA replay provider with a separate fixed-reduced-Top-K CPU session, exact proposal lifecycle, target-corrected commit history, isolated draft Reader/routing/time telemetry, and independent fixed-K4 greedy-oracle parity.
- Pure adaptive `{1,2,4}` proposal scheduler driven by observed prefix survival and target expert-major load/assignment cost, including immediate rejection backoff and bounded smallest-rung recovery.
- End-to-end `aurora-replay` CLI and benchmark schema with fixed K4/6/8/12 drafting, fixed/adaptive blocks, token/expert-major natural target verification, strict capability preflight, and zero draft telemetry for ordinary greedy execution.
- B-0017 seven-case natural/fixed/adaptive replay measurement with exact token/final-state/committed-route parity, 14/14 raw artifact digest validation, canonical aggregate verification, and combined CUDA expert-major AURORA Compute Sanitizer coverage.
- Opaque persistent draft cursor with one-time context prefill, fixed-size KDA checkpoints, append-only MLA logical marks/crop, target-bonus teacher forcing, malformed-commit atomicity, and direct fresh-oracle flattened-state parity.
- `AuroraPersistentDraftProvider` with lazy cursor creation, replay-equivalent lifecycle and adaptive scheduling, separate Reader/routing/time telemetry, and exact fixed/adaptive token-major plus CPU expert-major target parity.
- `aurora-persistent` CLI and JSON/CSV schema with default-zero context-prefill, incremental-forward, rollback, MLA-crop, and KDA-checkpoint counters.
- B-0018 nine-case natural/replay/persistent measurement with four exact matched pairs, 18 raw digest checks, LF-stable summary digest, canonical aggregate, and independently recomputed headline percentages.
- Persistent-only `cpu|cuda-custom` AURORA draft backend selection with CPU default, replay CPU-only validation, fixed exact CUDA identity, backend-unavailable propagation, and no silent fallback.
- Independent target/draft profiler, transfer, memory, allocation, synchronization, and cache telemetry in runtime JSON plus benchmark JSON/CSV.
- B-0019 nine-case natural/CPU/CUDA measurement with four exact matched pairs, 18 raw digest checks, LF-stable summary digest, canonical aggregate, and independently recomputed headline deltas.
- Persistent-only bounded CUDA draft residency with zero-capacity transient identity, positive-capacity tensor-ID residency, hard-cap exact transient bypass, and unchanged CPU/replay defaults.
- Independent draft configured-capacity, current/peak resident-byte, hit/miss/bypass, and weight-H2D telemetry across runtime JSON and benchmark JSON/CSV.
- B-0020 nine-case natural/transient/resident measurement with four exact matched pairs, 18 raw digest checks, LF-stable summary digest, canonical aggregate, independently recomputed pair/H2D/hit-rate metrics, and full-fit plus one-byte sanitizer coverage.
- Portable exact multi-expert/multi-token native-MXFP4 grid contract with CPU oracle, expert-first/token-second output layout, strict shape/identity/overflow validation, and direct 1x1, 1x4, 2x2, and 4x4 CUDA parity coverage.
- Exact resident CUDA expert-grid backend with all-resident preflight, three descriptor tables, four grid-wide gate/up/SiTU/down launches, stable separate outputs, and whole-request exact serial fallback on any residency bypass.
- Closed `resident-grid` target/AURORA capability identity, independent target/draft telemetry, JSON/CSV benchmark schema, and unchanged grouped/CPU defaults.
- B-0021 nine-case natural/grouped/grid measurement with four exact matched pairs, 18 raw digest checks, canonical aggregate verification, 75% MoE launch reduction, zero grid fallbacks, and full target token/state/route/acceptance/Reader parity.
- Portable exact whole-MoE-layer API and CPU oracle, ordered mix/strict RMSNorm/final-add CUDA primitives, and complete thirteen-operation resident CUDA backend with launch-free hard-cap bypass.
- Independent target/draft `moe-layer` CLI ownership, persistent AURORA capability gate, one-decision model dispatch, and exact M20 split-path fallback without rerouting or expert reload.
- Five resident MoE-layer counters independently exported for target and draft through runtime JSON and benchmark JSON/CSV, with zero defaults and first-sample deterministic-counter preservation.
- B-0022 nine-case natural/split-grid/layer measurement with four exact matched pairs, 18 raw digest checks, canonical aggregate verification, exact three-sync-per-call reduction, lower H2D/D2H, and zero layer fallback.
- CUDA-only released-dimension MoE-layer benchmark with separate split oracle, cold admission, warm measurement, 1/4/16 expert views, strict no-token schema, and fail-closed physical gates.
- B-0023 six-row released-dimension split/layer measurement with maximum error 0, zero fallback/bypass, zero warm weight H2D, exact 14,336-byte norm delta, 80→20 synchronization, lower activation/D2H, and digest-backed raw/summary evidence.

## Work in progress

- Milestone 22 design, implementation, corrected B-0023, committed evidence validation, performance model, README, TITAN Ledger synchronization, final review, publication gates, PR #36 integration, and post-merge CI are complete.
- B-0023 uses one existing released expert payload under 1/4/16 unique logical IDs plus deterministic released-size FP32 dense weights. It has `routing_semantics=false`, a 1 GiB hard resident capacity, and no token-throughput claim.
- Source review identifies the per-call 469,776,384-byte immutable dense/vector finiteness scan as the strongest current wall/kernel-gap hypothesis. D-048 requires correctness-preserving admission-time validation and remeasurement before CUDA Graph or whole-token work.
- Milestone 21 design, implementation, telemetry, B-0022 runner, formal evidence, digest checks, full local verification, final review, performance model, README, TITAN Ledger, PR #31 integration, and post-merge CI are complete.
- `moe-layer` now has executable full-fit and one-byte bypass runtime coverage. The exact FP32/native-MXFP4 backend performs all weight acquisition before scratch/events/kernel launch, returns `executed=false` on hard-cap bypass, and never converts CUDA errors into fallback. Focused CPU ownership/parity tests pass 104/35, CUDA ownership/parity tests pass 133/6, CPU schema coverage passes 13/8, and live CUDA schema coverage passes 19/2. CUDA Graph caching and a complete device-resident token graph remain deferred alternatives.
- B-0022 paired decode is mixed at +5.619%, -2.753%, -1.216%, and +3.933%. Fixed/adaptive layer paths reduce synchronizations by 90/99, total H2D by 14,496/15,984 bytes, and D2H by 26,880/29,568 bytes while adding exactly 384 norm-weight/resident bytes. The boundary remains experimental and non-default.
- Milestone 20 design, implementation, B-0021, full verification, sanitizer coverage, evidence cross-checks, final review, public PR integration, post-merge CI, README, and TITAN Ledger synchronization are complete.
- `resident-grid` remains exact, opt-in, and non-default. CUDA Graph caching, a cooperative persistent kernel, device-resident KDA/MLA/router state, reduced precision, and dynamic eviction remain outside this milestone.
- Milestone 19 implementation, B-0020, full verification, sanitizer coverage, evidence cross-checks, final self-review, public PR integration, post-merge CI, and TITAN Ledger synchronization are complete.
- Exact bounded CUDA residency remains opt-in and no-eviction. No default, draft precision, scheduler threshold, routing rule, or target verifier changed in Milestone 19.
- Milestone 18 implementation, B-0019, full verification, sanitizer coverage, evidence cross-checks, TITAN Ledger synchronization, final self-review, public integration, and post-merge CI are complete.
- Exact transient CUDA drafting remains diagnostic-only. No default, draft precision, residency, scheduler threshold, routing rule, or target verifier changed in Milestone 18.
- Complete-prefix `aurora-replay` remains executable as the required candidate/state oracle. Milestone 17 is complete and published through PR #23.
- Milestone 16 implementation, B-0017, full verification, evidence cross-checks, final self-review, public integration, and documentation reconciliation are complete.
- The scheduler explores `{1,2,4}` one rung at a time, uses Laplace-smoothed observed prefix survival and actual expert-major payload-load/assignment ratio, immediately backs off after rejection, and retries the smallest rung after one zero-draft step. These thresholds are an experimental B-0017 identity, not a runtime default.
- Complete-prefix replay intentionally remains the candidate oracle for persistent KDA/MLA draft-state work. Reduced precision, eviction-capable residency, confidence prediction, and learned DSpark integration remain unimplemented.
- Milestone 15 implementation, B-0016, complete CPU/CUDA verification, Compute Sanitizer, final read-only review, public PR integration, and post-merge CI are complete.
- B-0016 perfect CUDA expert-major blocks slightly reduce synthetic Reader/H2D traffic and improve tiny decode, while the mixed row evaluates three rejected positions and regresses traffic and decode. Released single-expert batching proves exact one-payload-per-group H2D reuse at batch sizes two and four. Neither result supports a default change.
- The CUDA expert-major path intentionally excludes L1 caching, deadline scheduling, reduced/adaptive routing, profile observation, asynchronous transfer, and routed-accumulate fusion. Acceptance-aware block sizing, representative drafting, multi-expert persistent scheduling, and cross-layer prediction remain unimplemented.
- DSpark learned drafting, confidence scheduling, EcoSpec, MoE-Spec, and AcceptMoE remain unimplemented. The accepted interface is lifecycle-compatible, not checkpoint- or tensor-ABI compatible with DeepSpec.
- Milestone 9 Terra high final review found one valid Important shared-session policy-context issue. Commit `fd05d95` serializes complete generation calls; re-review found no remaining Critical or Important issue and withdrew an initial collision interpretation concern after deterministic future-layer trace review.
- Static, LRU, LFU, Least-Stale, profiled eviction, and all non-default L2 modes remain experimental and opt-in. Transition prediction and cross-layer asynchronous L2 scheduling remain unimplemented.
- The L2 batch API submits concurrent operations for one batch but waits before returning. It is not the chartered N/N+1/N+2 deadline pipeline yet.
- The deadline worker schedules only the current routed layer and remains slower than blocking in all B-0009 rows. ORBIT, multiple L2 workers, eviction-aware priority, and future-layer recall are not implemented.
- Natural routing, `pread + buffered`, blocking scheduling, disabled L1, and CUDA MoE fusion `none` remain defaults because B-0007 through B-0013 are WSL2 evidence, not native P44 Pro or full-model evidence.
- Worktree: `C:\Users\jolib\Documents\project-k3x` on `codex/milestone-twenty-two-released-moe-layer`.
- Linux Python environment: `/home/jolib/.venvs/k3x-m1`; verified WSL builds: `build-wsl`, `build-uring`, `build-cuda`, and `build-uring-asan`.

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
- Full-model quality, coding/agentic quality, full-checkpoint adaptive Top-K and cold-rescue effectiveness, and speculative acceleration remain unmeasured. Exact token-major plus CPU/CUDA expert-major verification are implemented and measured only on the synthetic graph and bounded released single-expert fixture; learned drafting, proxy, and pruning remain unimplemented.
- B-0020 removes most repeated draft weight H2D but leaves 410 to 451 synchronous waits and fine-grained launches. One adaptive token pair regresses by 2.56% while the other three improve, so bounded static residency is measured capability evidence rather than a default or a full-model speedup claim.
- B-0021 reduces MoE launches by 75% and improves all four paired synthetic decode rows, but total draft H2D rises slightly and AURORA currently supplies one token per grid call. It is not evidence of full-model throughput, coding quality, or true speculative multi-token CUDA concurrency.
- B-0022 removes exactly three synchronizations per successful MoE-layer call and lowers activation/total H2D and D2H, but two of four paired decode rows regress. The latest measured bottleneck is therefore host-driven execution outside the MoE layer plus tiny-kernel/orchestration variance; representative dimensions and native Linux are required before choosing CUDA Graph caching or a larger device-resident token boundary.
- Corrected B-0023 records complete-layer median latency of 20.488/20.954/24.422 ms versus split 1.228/2.371/5.681 ms at 1/4/16 experts. The aggregate kernel-time increase is much smaller, while the complete preflight scans all 469,776,384 immutable dense/vector bytes every call. Causality still requires an attribution benchmark, so CUDA Graph and larger device-token work remain deferred.

## Next concrete tasks

1. Design admission-time immutable-tensor validation with explicit non-finite and lifetime invalidation tests.
2. Add profiler-on/off host attribution and rerun the released-dimension split/layer matrix before selecting CUDA Graphs or a larger device-resident token boundary.
3. Keep dynamic eviction, reduced precision, and graph caching as separately attributable experiments.

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

B-0023 isolates the released-dimension MoE-layer boundary after B-0022 established the desired traffic direction. After removing oracle-backend overlap, at 1/4/16 experts split median latency is 1.228/2.371/5.681 ms while complete-layer latency is 20.488/20.954/24.422 ms. The layer still removes 60 synchronizations over 20 iterations, transfers no warm weights, reduces activation H2D and D2H, and preserves maximum error 0 with no fallback.

Aggregate kernel time rises only from 15.122/24.507/58.396 ms to 22.971/27.692/61.887 ms over 20 iterations, much less than wall time. The complete backend performs an O(weight-bytes) finiteness scan across 469,776,384 immutable dense/vector bytes on every call. This is the strongest code-backed bottleneck hypothesis, but it is not yet a measured decomposition.

The next CUDA systems boundary is therefore admission-time immutable-tensor validation plus profiler-on/off host attribution, followed by the same released-dimension rerun. CUDA Graphs and a larger device-resident token graph are deferred until that evidence exists. Dynamic eviction and prediction remain separate policy axes; reduced precision remains a separate quality axis.

The immediate result is to retain CUDA MoE fusion `none` alongside natural routing with `disabled + blocking + pread + buffered` as defaults. Native-Linux physical NVMe traffic, controlled warm/cold behavior, full-model locality, coding quality, GPU utilization, and memory bandwidth remain unknown.

The derived uncached full-model expert traffic remains 25.83 GB/token, but it is not a measured full-model value. Native-Linux NVMe traffic, cache reuse, and full Kimi K3 throughput remain unknown.

## Last known-good state

- Public Milestone 22 integration head `e4820a18` contains the released benchmark, corrected B-0023 runner, six raw rows, summary CSV/JSON, oracle-lifetime regression coverage, committed-evidence verification, synchronized architecture/performance/decision documents, and the English README. PR #36 is merged.
- Fresh pre-review Milestone 22 verification passes CPU CTest 14/14 and pytest 305/67, liburing/direct CTest 15/15 and pytest 307/65, ASan/UBSan CTest 15/15, and CUDA CTest 26/26 with pytest 362/10. Post-correction publication gates pass CUDA CTest 26/26 and focused live/evidence pytest 22/22; the released one-expert complete-layer Compute Sanitizer reports zero errors.
- Corrected B-0023 artifact/runner/aggregate/summary JSON/CSV SHA-256 are `e087ff78284e99760a7d113cf744562878537a6379e7a63be95585eec8b9f1be`, `3c2695fc31adc01040a992098180a83cb58947d85858412eab62282b66ec6baf`, `88c51b6a58340a4325b2b09faa0fb63d1bc5f4439542261383f6070dbe526ade`, `d67fe356735ddc38e238a9e86e7f46ec3729ef24bc27d1f286aacaaabf0af954`, and `4a95494381c87862aa6933811248f1fd2ff35a28d88e576917da57e50e87d621`.
- Public Milestone 22 post-merge `main` correctness run `31359158926` and CodeQL run `31359158878` passed at `e4820a18`.
- Public documentation reconciliation head `7728acd0` records the Milestone 20 PR #29 publication across README and the TITAN Ledger. PR #30 push/PR correctness runs `31351873585`/`31351882562` and post-merge `main` run `31352046040` succeeded.
- Public Milestone 20 integration head `90b20c87` includes the exact resident grid implementation, B-0021 evidence, English README, and synchronized TITAN Ledger. PR #29 push/PR correctness runs `31351465644`/`31351486146` and post-merge `main` correctness run `31351649761` succeeded.
- Public Milestone 20 evidence head `8e85ff3` contains the B-0021 runner lineage, 9 rows, and 20 result files under `results/b0021-cuda-aurora-grid-wsl/`.
- Fresh Milestone 20 verification passes CPU CTest 14/14 and pytest 290 passed/55 skipped, liburing/direct CTest 15/15 and pytest 296 passed/49 skipped, ASan/UBSan liburing CTest 15/15, and CUDA CTest 24/24 with pytest 336 passed/9 skipped. Direct expert-grid and 4x4 benchmark Compute Sanitizer runs report `ERROR SUMMARY: 0 errors`.
- B-0021 artifact SHA-256 is `7e12595e5e400b4c26946c75927b37f39ed3a0bcb8f90ca72b1e8f7c6cb95cad`; runner SHA-256 is `0497a53a6ba6045d911dbb685e7155ee698a7e83946059e7b611202918bd4aa8`; canonical aggregate SHA-256 is `a628064544cdae0d06af7177539bc253f264946840f59651508121146af2edda`; summary JSON/CSV SHA-256 is `8586f6a1939dfe209813c504727c0952149730a757eb5600b05fb6a02021877f` / `b87b26c1403a2f3d30fa46b5550837f1f72db18a1d162710a907419da8d64401`.
- Public Milestone 19 integration head `c88456c0` includes the complete bounded exact CUDA draft-residency implementation, B-0020 evidence, English README, and synchronized TITAN Ledger. PR #27 push/PR correctness runs `31346575341`/`31346587586` and post-merge `main` correctness run `31346725071` succeeded.
- Local Milestone 19 evidence head `f676957` contains the bounded resident provider/CLI/telemetry lineage, B-0020 runner, 9 rows, and 20 result files under `results/b0020-cuda-aurora-residency-wsl/`.
- Fresh Milestone 19 verification passes CPU CTest 14/14 and pytest 284 passed/53 skipped, liburing/direct CTest 15/15 and pytest 290 passed/47 skipped, ASan/UBSan liburing CTest 15/15, and CUDA CTest 23/23 with pytest 328 passed/9 skipped. Full-fit and one-byte exact-bypass Compute Sanitizer runs both report `ERROR SUMMARY: 0 errors`.
- B-0020 artifact SHA-256 is `47795886397106b3d1a029fefb86e58776be659cb0470ceb7c9998851aedcf26`; runner SHA-256 is `9fd847ff95c0f3b9c3bb3bc90ff568381b3a3d540f80eebcf433551465d79daa`; canonical aggregate SHA-256 is `4bb84fe49cbbc735bc9ef8668ab4d2944fef3d4e3a0f1048a7973410b211df87`; summary JSON/CSV SHA-256 is `32d9795ab3da3107c8f4fe5573be439130795d91b6860bdda91cc7d84635a192` / `059ede44149da8490f8342061b4dc10e623abe08f420266f84d7ca73963e3a62`.
- Public Milestone 18 integration head `7899a7ae` includes the complete implementation, B-0019 evidence, English README, and synchronized TITAN Ledger. PR #25 push/PR correctness runs `31343260116`/`31343261633` and post-merge `main` correctness run `31343401178` succeeded.
- Local Milestone 18 verified head `5058369` contains implementation through B-0019 plus the CUDA live-test capability gate. Evidence head `7257280` contains 9 rows and 20 result files under `results/b0019-cuda-aurora-draft-wsl/`.
- Current full verification passes CPU CTest 14/14 and pytest 278/50, liburing/direct CTest 15/15 and pytest 284/44, ASan/UBSan liburing CTest 15/15, and CUDA CTest 23/23 with pytest 319/9. The exact CUDA draft Compute Sanitizer run reports zero errors.
- B-0019 artifact SHA-256 is `6604d1ec65f8056f6d4f04d09fa357a442c7c2f7a46faf56899caf31671d2ca7`; runner SHA-256 is `fb7bded3cb3edd5b2f626801ec38edd246ba0c19a990e6301955e32d0642d52f`; canonical aggregate SHA-256 is `ce1a599eb04077f3b0c1b8350254b126a58f4dc311421bfc38fc8f7a78478c59`; summary JSON/CSV SHA-256 is `3750254294385cecf503f2efcd69f8d23953a7e982006fe969c4c9ac9ee2913f` / `1b6234889c8997486e5b268d277af3ca892b2b6c152489b4f85cea81717edd1f`.
- Public Milestone 17 integration head `30bbf7a8` includes the complete implementation, B-0018 evidence, English README, and synchronized TITAN Ledger. PR #23 branch/PR correctness runs `31340338639`/`31340340063` and post-merge `main` correctness run `31340476396` succeeded.
- Milestone 17 cursor/transaction head `6b3955a`, provider head `c28a732`, CLI/schema head `3459ca6`, and B-0018 evidence head `de63023` are the current verified lineage.
- Current full verification passes CPU CTest 14/14 and pytest 272 passed/47 skipped, liburing/direct CTest 15/15 and pytest 278 passed/41 skipped, ASan/UBSan liburing CTest 15/15 plus five artifact-backed persistent tests, and CUDA CTest 23/23 with pytest 311 passed/8 skipped. The persistent CUDA expert-major Compute Sanitizer run reports zero errors.
- B-0018 artifact SHA-256 is `81560d6250869426d739040c6e30d9a881b1f37f7a3f639345d27dd69a80ce96`; runner SHA-256 is `0eb212731be6e0a5344048aa6f6d76fb57732423017568112ec9d27f7b74d48d`; canonical aggregate SHA-256 is `abcef1afca7d6208808941323565bce44f25ecc6e9e0d28292ad54bfc7760cd0`; summary JSON/CSV SHA-256 is `a332af2d336cecb3060812a577f16e605bc832f4f21b74f315dfbbf8fd4f6132` / `c65d3bb9d8805f66249d0bb6ba380b8aa2508fd53a073c1bc3dece82e00fe472`.
- Independent B-0018 validation recomputes all 18 raw JSON/CSV digests, the summary CSV digest, canonical aggregate, exact pair invariants, and headline percentage deltas. Results are under `results/b0018-persistent-aurora-wsl/`.
- Milestone 16 runtime head `bc45538` contains the replay provider, adaptive scheduler, target feedback, runtime integration, CLI, and telemetry. Evidence head `51ff8e7` contains B-0017 tooling, raw artifacts, summaries, cross-checks, and verification records.
- Public Milestone 16 integration head `df5c07d` includes the complete implementation, B-0017 evidence, README, and synchronized TITAN Ledger. PR #20 branch/PR correctness runs `31337234073`/`31337240722` and post-merge `main` correctness run `31337365175` succeeded.
- Public Milestone 16 documentation head `0eb0966` records PR #20 and its CI across README, BENCHMARKS, checklist, context notes, plan, and project state. PR #21 push/PR correctness runs `31337548635`/`31337554179` and post-merge `main` correctness run `31337694471` succeeded.
- Historical Milestone 16 verification passed CPU CTest 14/14 and pytest 268 passed/47 skipped, liburing/direct CTest 15/15 and pytest 274 passed/41 skipped, ASan/UBSan liburing CTest 15/15, and CUDA CTest 23/23 with pytest 307 passed/8 skipped. The combined CUDA expert-major AURORA CLI Compute Sanitizer run reported zero errors.
- B-0017 artifact SHA-256 is `c1110ad2a1fe981f92b01e36aaafa216d0d8ea45a6608270f3cf706816c17a7c`; runner SHA-256 is `a20f708073bd27150d27d8eddf5c926072f1b96020257e625ab3caa895a536f7`; canonical aggregate SHA-256 is `fb7febf52c75281417b77c3f7d40787f738dba8a35490cc86d43ac5072cacd23`; summary JSON/CSV SHA-256 is `fdd94c5696d1505e17e0dbc41d465d8edad38b132896f5a3742277c09b852871` / `865d228fb88b1bc22fe147b04e1ce003559f04534052d8ce0180b753832d9551`.
- Independent B-0017 evidence validation recomputed all 14 raw JSON/CSV digests, the canonical aggregate, exact token/state/route parity, and LF-stable summary bytes. Results are under `results/b0017-aurora-replay-wsl/`.
- Public Milestone 15 integration head `c18df33` includes the synchronized Milestone 11–15 implementation lineage, B-0016 evidence, LF-stable artifact digests, and final-review fix. Branch/PR correctness runs `31332732339`/`31332745907` and post-merge `main` run `31332852551` succeeded.
- Public documentation reconciliation head `4052221` records PR #17 publication across README, BENCHMARKS, checklist, context notes, and project state. Its branch/PR correctness runs `31333096506`/`31333098541` and post-merge `main` run `31333233834` succeeded.
- M15 design/plan heads are `b54e4b5`/`f574a36`; portable batch, native launcher, CUDA FFN, and runtime heads are `b9b10dc`, `459303e`, `b0c1a96`, and `e99bbc0`; B-0016 tooling, direct CLI fix, and LF-stable digest fix are `7899603`, `884a74e`, and `5b7b73b`.
- Current full verification passes CPU CTest 13/13 and pytest 262 passed/47 skipped, liburing/direct CTest 14/14 and pytest 264 passed/45 skipped, ASan/UBSan CTest 14/14, and CUDA CTest 22/22 with pytest 301 passed/8 skipped. Native MXFP4, CUDA FFN, released batch-2, perfect expert-major CLI, and mixed expert-major CLI Compute Sanitizer runs each report zero errors.
- B-0016 executable artifact SHA-256 is `039d61ee9c2e13e27c9a2514bb476f8b122b8b37be0b7f85baf26c1a6611a2e9`; released artifact SHA-256 is `aab7aea48b03bdcd8e0b4d98c4780128ab689d2bba005089a49970eb0e326890`; canonical aggregate SHA-256 is `09a2537337df1fd2b8b39439f92ba7306cb09a6ed5e3f8bdc8db7d9d787029aa`.
- Independent evidence validation recomputed all nine raw JSON/CSV digest pairs and the aggregate. Results are under `results/b0016-cuda-expert-major-wsl/` and are included in public integration head `c18df33`.
- PR #11 merged at `edc6d605` and PR #12 merged at `9e59a9db`; both merge commits are ancestors of the audited public baseline. Their post-merge correctness runs `31318993688` and `31322191670` succeeded.
- M14 exact block runtime head `862d401`, CLI/telemetry head `bdf4a66`, and B-0015 result head `1e73121` remain the CPU reference lineage.
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

AURORA's replay reference, persistent reduced-Top-K CPU state, adaptive scheduler, exact transient CUDA draft, bounded exact resident CUDA draft, exact resident expert grid, resident MoE-layer execution, target feedback, CLI, separated telemetry, and B-0017 through B-0023 are implemented and measured as experimental non-default paths. Transient CUDA is rejected as a default; bounded residency, the resident grid, and the resident MoE layer remain opt-in. B-0023 rejects graph/default promotion until repeated immutable-weight validation is removed and remeasured. Reduced precision, eviction-capable residency, a complete device-resident token graph, and learned drafting remain proposed. APOLLO, TITAN COUNCIL, PROMETHEUS-X, MERCURY, ORBIT, HELIOS, SHADOW, PHOENIX, VAULT, VEILBREAK, AUTO, and SKYFORGE remain proposed only. ATLAS, CHRONOS, and BLACKSTAR remain reserved without accepted definitions. None of the proposed-only components is claimed as implemented or benchmarked.
