# K3X 첫 마일스톤 체크리스트

## 설계

- [x] 빈 워크스페이스와 Git 상태 확인.
- [x] 구현 언어와 첫 마일스톤 범위 승인.
- [x] 아키텍처, 데이터 흐름, 정확성 정책, 완료 기준 승인.
- [x] 승인된 설계 명세 작성 및 자체 검토.
- [x] 설계 명세 사용자 검토 승인.
- [x] 세부 구현 계획 작성.

## 구현

- [x] 저장소 도구 구성과 합성 config 계약 테스트.
- [x] PyTorch 기본 연산과 MXFP4 oracle.
- [x] KDA와 recurrent/ShortConv state parity.
- [x] Gated MLA와 incremental KV parity.
- [x] router, Stable LatentMoE, shared expert parity.
- [x] Attention Residual과 decoder graph parity.
- [x] greedy generation full/incremental parity.
- [x] K3X v1 streaming writer, reader 및 validator.
- [x] crash-safe resume와 손상 감지.
- [x] C++20 runtime 연산·layer·end-to-end token parity.

## 검증과 전달

- [x] Python 전체 테스트 통과.
- [x] C++ 전체 테스트 통과.
- [x] 합성 checkpoint round-trip 통과.
- [x] benchmark와 peak RSS/read bytes 측정.
- [x] `ARCHITECTURE.md`, `PERFORMANCE_MODEL.md`, `K3X_FORMAT.md` 완성.
- [x] diff 자체 검토와 semantic commits 완료.
- [x] 측정 결과와 다음 병목 보고.

## Milestone 1 설계와 준비

- [x] RTX 5080, compute capability 12.0, CUDA 13.3 환경 확인.
- [x] custom-only, library-only, hybrid CUDA 접근 비교.
- [x] hybrid backend 설계 사용자 승인.
- [x] exact runtime, profiler, CUDA baseline 설계 명세 작성.
- [x] 격리 worktree와 고정 Python 의존성 구성.
- [x] C++ baseline 2개 테스트 통과.
- [x] 작성된 Milestone 1 설계 명세 사용자 검토 승인.
- [x] TITAN LEDGER 지속성 문서 체계 초기화.
- [x] Milestone 1 세부 구현 계획 작성.
- [x] WSL과 Virtual Machine Platform Windows 기능 활성화.
- [x] Windows 재부팅 후 Ubuntu 24.04 배포판 설치와 GPU passthrough 검증.
- [x] Linux Release baseline CTest 2/2와 pytest 39/39 검증.
- [x] `K3X_BUILD_DIR` cross-language 경로 해석을 TDD로 구현하고 전체 pytest 47/47 검증.

## Milestone 1 구현

- [x] profiler schema와 집계기를 TDD로 구현하고 CTest 3/3 검증.
- [x] CPU compute backend 경계를 도입하고 CTest 4/4 및 pytest 47/47 검증.
- [x] optional CUDA 13.3 및 SM 12.0 resource shell 구성과 CPU-only 격리 검증.
- [x] cuBLASLt FP32/BF16-rounded projection baseline 구현과 memcheck 검증.
- [x] cuBLASLt FP4와 K3 MXFP4 scale 계약 비교 후 direct path 기각.
- [x] exact native-byte custom MXFP4 CUDA baseline 구현과 memcheck 검증.
- [x] CPU, CUDA library, CUDA custom synthetic graph parity 검증.
- [x] JSON/CSV benchmark와 실제 RTX 5080 측정.
- [x] 전체 ablation 결과와 다음 병목 문서화.

## Milestone 2 설계와 구현

- [x] B-0002와 현재 CUDA resource lifetime을 근거로 병목 재검증.
- [x] scratch-only, staged residency, whole-layer GPU executor 대안 비교.
- [x] static weight residency와 grouped projection 설계 승인.
- [x] Milestone 2 설계 명세와 acceptance matrix 작성.
- [x] 사용자 서면 명세 검토.
- [x] TDD 구현 계획 작성.
- [x] CUDA allocation/weight/batching 옵션과 runtime 통계 계약 정의.
- [x] CUDA 실행 CLI 검증과 benchmark JSON/CSV schema 확장.
- [x] stable tensor identity와 CPU grouped projection oracle 구현.
- [x] tracked CUDA allocation과 grow-only scratch primitive 구현.
- [x] allocation reuse 구현과 독립 ablation.
- [x] exact static weight residency 구현과 독립 ablation.
- [x] dense/MXFP4 grouped projection 구현과 독립 ablation.
- [x] weight/activation H2D profiler 분리와 deterministic ablation runner 구현.
- [x] 전체 correctness, memcheck, benchmark, TITAN Ledger 갱신.

## Milestone 3 FFN block executor

- [x] B-0003 병목을 기준으로 operation, generic device handle, full-layer 대안 비교.
- [x] dependency-closed FFN block executor 설계 명세 작성과 self-review.
- [x] TDD 구현 계획 작성과 self-review.
- [x] `cuda-boundary` 옵션과 capability validation 구현.
- [x] strict FP32 SiTU-GLU CUDA kernel과 literal oracle 검증.
- [x] dense/shared FFN block 구현과 graph 연결.
- [x] exact native MXFP4 routed expert block group 구현과 graph 연결.
- [x] profiler/schema/ablation runner 확장.
- [x] 전체 correctness, sanitizer, B-0004, TITAN Ledger 갱신.

## Milestone 4 exact asynchronous L0/L1 transfer

- [x] 현재 resident table, expert load lifetime, CUDA stream, profiler 경계 조사.
- [x] CUDA 13.3 pinned-memory, overlap, event dependency 공식 계약 확인.
- [x] two-phase exact prefetch token 설계와 범위 확정.
- [x] 설계 명세 self-review와 semantic commit.
- [x] 상세 TDD 구현 계획 작성과 self-review.
- [x] bounded pinned staging과 transfer stream primitive 구현.
- [x] prepared exact MXFP4 expert FFN과 graph scheduling 연결.
- [x] profiler/schema/ablation runner 확장.
- [x] 전체 correctness, sanitizer, B-0005, TITAN Ledger 갱신.

## Milestone 5 persistent L1 expert cache

- [x] current expert loader, Reader counters, payload lifetime, prepared-transfer boundary 조사.
- [x] model-adjacent immutable whole-expert store 선택과 대안 기록.
- [x] bounded no-eviction exact-bypass 설계 명세 작성과 self-review.
- [x] 상세 TDD 구현 계획과 B-0006 acceptance matrix 작성.
- [x] runtime options와 hard-capacity host expert store 구현.
- [x] operation/FFN-block/prefetch graph 연결과 exact parity 검증.
- [x] L1/Reader profiler schema와 B-0006 ablation runner 구현.
- [x] 전체 correctness, sanitizer, B-0006 측정.
- [x] TITAN Ledger 최종 검토와 검토 수정 후 전체 검증.
- [x] Public GitHub PR, CI, main 반영과 post-merge CI 확인.

## Milestone 20 — Resident multi-token multi-expert CUDA grid

- [x] Re-read the public charter, architecture, state, D-043, and B-0020 evidence.
- [x] Capture an exact resident AURORA Nsight launch diagnostic.
- [x] Compare resident grid, routed-set CUDA Graph cache, and whole-device draft graph alternatives.
- [x] Write and self-review the accepted resident-grid design.
- [x] Write and self-review the detailed TDD implementation plan.
- [x] Implement the CPU oracle and CUDA rectangular grid through witnessed RED/GREEN cycles.
- [x] Integrate AURORA CLI, exact fallback, telemetry, and B-0021 runner.
- [x] Run B-0021, the complete verification matrix, sanitizer, and evidence cross-checks.
- [x] Synchronize the TITAN Ledger, complete final review, publish, merge, and verify public `main`.

## Milestone 21 — Exact resident MoE-layer CUDA boundary

- [x] Re-read the public charter, architecture, project state, D-045, and B-0021 evidence.
- [x] Inspect the split MoE runtime/backend path and current CUDA synchronization/traffic ownership.
- [x] Compare resident MoE-layer execution, routed-set CUDA Graph caching, and a complete device token graph.
- [x] Write and self-review the accepted dependency-closed design.
- [x] Write and self-review the detailed TDD implementation plan.
- [x] Implement the portable whole-layer API and exact CPU oracle through a witnessed RED/GREEN cycle.
- [x] Implement ordered mix, strict RMSNorm, and final-add CUDA primitives through a witnessed RED/GREEN cycle and Compute Sanitizer.
- [x] Implement the complete resident CUDA MoE-layer backend through a witnessed RED/GREEN cycle and Compute Sanitizer.
- [x] Integrate runtime, AURORA, CLI, and exact split fallback.
- [x] Propagate target/draft MoE-layer telemetry through runtime JSON and benchmark schemas.
- [x] Implement and run B-0022 with committed raw/summary evidence validation.
- [x] Run the complete verification and sanitizer matrices.
- [x] Synchronize the TITAN Ledger, complete final review, publish, merge, and verify public `main`.

## Milestone 22 — Released-dimension resident MoE-layer boundary

- [x] Re-read the public charter, architecture, current state, D-046/D-047, and B-0022 evidence.
- [x] Inspect the released storage fixture and existing CUDA expert benchmark interfaces.
- [x] Compare released repeated-view, scaled synthetic, and executable-checkpoint approaches.
- [x] Write and self-review the accepted released-dimension benchmark design.
- [x] Write and self-review the detailed TDD implementation plan.
- [x] Implement the released-dimension split/layer benchmark through witnessed RED/GREEN cycles.
- [x] Implement and run B-0023 with committed raw/summary evidence validation.
- [x] Run full verification, synchronize the TITAN Ledger and README, review, publish, merge, and verify public `main`.

## Milestone 23 — Admission-time immutable validation attribution

- [x] Re-read the charter, architecture, project state, D-048, and corrected B-0023 evidence.
- [x] Inspect CUDA MoE-layer validation, resident identity, CLI, telemetry, and ownership boundaries.
- [x] Compare resident-table miss validation, prepared-layer tokens, and backend-local admission registry designs.
- [x] Write and self-review the accepted admission validation design and detailed TDD implementation plan.
- [x] Add `per-call|admission` ownership, runtime telemetry, JSON/CSV schema, and reference defaults through witnessed RED/GREEN cycles.
- [x] Prove six-view atomic preflight, repeat identity hits, non-finite last-view failure, and pointer/shape conflict rejection.
- [x] Add profiler-off physical telemetry and the canonical 18-row B-0024 runner.
- [x] Measure B-0024 at released dimensions with 3 warmups and 20 iterations and commit digest-backed evidence.
- [x] Update README, architecture, performance model, decisions, benchmarks, checklist, and context notes.
- [x] Run the complete CPU/liburing/ASan/CUDA/sanitizer verification matrix.
- [x] Complete the final Critical/Important-only read-only review.
- [x] Publish through a public PR, merge, verify post-merge correctness/CodeQL, and update `PROJECT_STATE.md` last.

## Milestone 24 — Bounded CUDA Graph cache attribution

- [x] Re-read the charter, architecture, project state, D-044/D-046/D-048/D-049, and B-0022 through B-0024 evidence.
- [x] Inspect CUDA 13.3 graph update/thread-safety contracts, CUDA Samples, installed CUDA 13.3.73, and the exact resident MoE-layer boundary.
- [x] Compare a whole-token graph, one mutable superset graph, and reference/update/bounded ordered-set cache experiments.
- [x] Write and self-review the accepted bounded CUDA Graph cache design.
- [x] Write and self-review the detailed TDD implementation plan.
- [x] Implement graph identity, bounded cache, strict options, and telemetry through witnessed RED/GREEN cycles.
- [x] Integrate exact CUDA capture/update/cache execution and target/draft ownership.
- [x] Add and run the canonical B-0025 trace ablation with digest-backed evidence.
- [x] Run the full CPU/liburing/ASan/CUDA/sanitizer verification matrix.
- [x] Complete Critical/Important review, synchronize README/TITAN Ledger, publish, merge, and verify public `main`.

## Milestone 25 — Converter trust-boundary integrity

- [x] Re-read the charter, architecture, project state, D-003/D-028, B-0008, current writer, safetensors reader, resume ledger, and integrity regressions.
- [x] Resolve the stale-state contradiction: D-028 already closes bounded fixture hash and canonical extent-prefix findings.
- [x] Compare a D-028 no-op, narrow generic trust-boundary hardening, and premature signed-manifest v2.
- [x] Write and self-review the accepted converter trust-boundary design.
- [x] Write and self-review the detailed TDD implementation plan.
- [x] Harden source manifest containment and exact tensor-to-shard ownership through witnessed RED/GREEN cycles.
- [x] Harden safetensors metadata and resume-ledger schema through witnessed RED/GREEN cycles.
- [x] Recover uncommitted partial suffixes without weakening committed-prefix validation.
- [x] Run the full verification matrix and record measured resource/correctness evidence.
- [x] Complete Critical/Important review, synchronize README/TITAN Ledger, publish, merge, and verify public `main`.

## Milestone 26 — Official bounded range discovery

- [x] Re-read the charter, architecture, project state, D-028/D-051, B-0008/B-0026, current source manifest, safetensors reader, and storage fixture.
- [x] Resolve the official public snapshot and inspect API, index, shard 2 header, exact expert ranges, and w1/w2/w3 roles without tensor-payload download.
- [x] Compare API-only, full-shard-first, and pinned exact-range approaches.
- [x] Write and self-review the accepted official range-discovery design under the standing pre-Cloud-Run approval.
- [x] Write and self-review the detailed TDD implementation plan.
- [x] Implement bounded snapshot/index/config discovery through witnessed RED/GREEN cycles.
- [x] Implement exact expert planning and atomic range materialization through witnessed RED/GREEN cycles.
- [x] Convert and verify one live official expert slice without committing real bytes.
- [x] Record B-0027, run the full verification matrix, close Critical/Important path/evidence findings, and synchronize documentation.
- [x] Publish through PR #44, merge, and verify public `main` correctness and CodeQL.

## Milestone 27 — Official expert CUDA execution

- [x] Re-read the public M26 state and inspect the storage-slice, CPU MXFP4, CUDA FFN, and released benchmark boundaries.
- [x] Execute one read-only exploratory official-artifact CPU/CUDA parity smoke without treating it as benchmark evidence.
- [x] Compare unchanged reuse, mixed-schema extension, and a dedicated pinned official-expert harness.
- [x] Write and self-review the accepted M27 design under the standing pre-Cloud-Run authorization.
- [x] Write and self-review the detailed TDD implementation plan.
- [x] Implement pinned identity validation through a witnessed RED/GREEN cycle.
- [x] Implement the dedicated CUDA harness through witnessed RED/GREEN cycles.
- [x] Implement and verify the B-0028 two-mode runner and strict evidence verifier.
- [x] Run B-0028, the full verification matrix, final review, TITAN Ledger synchronization, and public integration.

## Milestone 12 — Fused routed accumulation CUDA kernel

- [x] TITAN Ledger, 현재 CUDA 경계, 원 구현 근거 재확인.
- [x] SiTU/down fusion, gate/up launch fusion, routed down-accumulation fusion 비교.
- [x] 채택 설계와 TDD 구현 계획 작성 및 self-review.
- [x] Accumulating native MXFP4 kernel을 RED에서 GREEN으로 구현.
- [x] Synchronous/prepared fused expert-group backend 경로 구현.
- [x] 기본값을 보존하는 runtime switch와 telemetry 연결.
- [x] B-0013 synthetic 및 released-dimension bounded 측정 도구 구현.
- [x] 전체 correctness, sanitizer, RTX 5080 ablation, ledger, review, public integration.

## Milestone 13 — Exact speculative block verification

- [x] TITAN Ledger, DSpark paper, DeepSpec implementation, current state boundary review.
- [x] Exact greedy token-major verification design and TDD plan.
- [x] Pure proposal/verification/provider contract with native tests.
- [x] Incremental runtime integration with greedy token/state/routing parity.
- [x] Telemetry and deterministic scripted-draft end-to-end harness.
- [x] B-0014 correctness and overhead measurement.
- [x] Full verification, sanitizer, ledger, review, and public integration.

## Milestone 11–13 documentation synchronization

- [x] Reconfirm that public PR #11, PR #12, and PR #13 are merged.
- [x] Audit README and TITAN Ledger documents for stale Milestone 11–13 status.

## Public documentation reconciliation — Milestones 11–14

- [x] Verify PR #11 and PR #12 are merged and their merge commits are in public `main` history.
- [x] Add the live `main` correctness badge and explicit PR #11/#12 merge provenance to the English README.
- [x] Reconcile `PROJECT_STATE.md` with public baseline `46105f8` and the Milestone 15 design-only boundary.
- [x] Audit ARCHITECTURE, DECISIONS, BENCHMARKS, PERFORMANCE_MODEL, PLAN, K3X_FORMAT, and reference docs; retain accurate historical or constitutional text unchanged.
- [x] Validate the Markdown diff and prepare the documentation change for public merge.

## Milestone 15 — Exact CUDA expert-major execution

- [x] Re-read the M14 CPU expert-major path and current CUDA MXFP4 FFN implementation.
- [x] Inspect pinned MoonshotAI, vLLM, and NVIDIA primary sources relevant to expert-grouped token execution.
- [x] Compare a single-expert multi-token primitive, temporary scalar residency, and a persistent multi-expert kernel.
- [x] Write and self-review the accepted CUDA expert-major design.
- [x] Write and self-review the detailed TDD implementation plan.
- [x] Implement the backend batch contract through witnessed RED and GREEN cycles.
- [x] Integrate the exact CUDA runtime, CLI capability gate, and telemetry.
- [x] Run B-0016, full verification, sanitizer, and ledger synchronization.
- [x] Complete final review, public PR integration, and post-merge CI verification.
- [x] Update the README milestone badge and current implementation summary.
- [x] Correct the stale speculative-implementation statement and publication HEAD in `PROJECT_STATE.md`.
- [x] Run documentation checks, commit, publish, and verify public `main`.

## Milestone 16 — AURORA replay and adaptive block scheduling

- [x] Re-read the charter, architecture, state, speculative decisions, and B-0014 through B-0016 evidence.
- [x] Inspect pinned DSpark/DeepSpec, EcoSpec, and MoE-Spec primary sources and actual DeepSpec scheduler code.
- [x] Compare scripted traces, trained DSpark, replay-based self draft, and persistent AURORA state.
- [x] Repair the Windows canonical `.k3xp` baseline integrity failure and re-run the CPU baseline.
- [x] Write and self-review the replay AURORA and adaptive scheduler design.
- [x] Write and self-review the detailed TDD implementation plan.
- [x] Implement scheduler feedback and provider contracts through witnessed RED/GREEN cycles.
- [x] Integrate AURORA replay CLI, telemetry, and exact target verification.
- [x] Run B-0017, the full verification matrix, sanitizer coverage, and evidence cross-checks.
- [x] Synchronize the TITAN Ledger, complete final review, publish, merge, and verify public `main`.

## Milestone 17 — Persistent AURORA draft state

- [x] Re-read the public charter, architecture, state, D-040, and B-0017 evidence.
- [x] Reinspect DeepSpec `005e03b8` draft-cache crop and verified-context update code.
- [x] Compare complete state copy, partial replay, and transactional KDA snapshot plus MLA crop designs.
- [x] Write, self-review, and commit the persistent draft-state design.
- [x] Write, self-review, and commit the detailed TDD implementation plan.
- [x] Implement the incremental cursor and rollback contract through witnessed RED/GREEN cycles.
- [x] Integrate the persistent provider, CLI, schema, and replay-oracle parity.
- [x] Run B-0018, the full verification matrix, sanitizer coverage, and evidence cross-checks.
- [x] Synchronize the TITAN Ledger, complete final review, publish, merge, and verify public `main`.

## Milestone 18 — Exact CUDA AURORA draft execution

- [x] Re-read the public charter, architecture, current state, D-041, and B-0018 evidence.
- [x] Reproduce and fix fresh-Windows-checkout benchmark digest instability without weakening byte-exact evidence.
- [x] Compare exact transient CUDA draft execution, CUDA residency, and reduced precision as separate experiment axes.
- [x] Write and self-review the exact CUDA draft design.
- [x] Write and self-review the detailed TDD implementation plan.
- [x] Implement draft backend preflight, ownership, and runtime validation through witnessed RED/GREEN cycles.
- [x] Add separated draft CUDA telemetry and benchmark schema coverage.
- [x] Run B-0019, the full verification matrix, sanitizer coverage, and evidence cross-checks.
- [x] Synchronize the TITAN Ledger, complete final review, publish, merge, and verify public `main`.

## Milestone 19 — Bounded exact CUDA AURORA residency

- [x] Re-read the public charter, architecture, current state, D-042, and B-0019 evidence.
- [x] Compare static residency, dynamic L0 eviction, and persistent-kernel alternatives as separate experiment axes.
- [x] Write and self-review the bounded exact draft-residency design.
- [x] Write and self-review the detailed TDD implementation plan.
- [x] Implement provider, CLI, telemetry, and benchmark contracts through witnessed RED/GREEN cycles.
- [x] Run B-0020, the full verification matrix, sanitizer coverage, and evidence cross-checks.
- [x] Synchronize the TITAN Ledger, complete final review, publish, merge, and verify public `main`.

## Milestone 14 — Exact expert-major speculative verification

- [x] Re-read the TITAN Ledger and current token-major/runtime boundaries.
- [x] Inspect official vLLM Kimi K3 multi-token state handling and primary MoE speculative papers.
- [x] Compare full CUDA, route replay, and exact CPU layer-major approaches.
- [x] Write the exact CPU expert-major design with state, routing, traffic, and failure invariants.
- [x] Self-review and commit the design specification.
- [x] Write and commit the detailed TDD implementation plan.
- [x] Implement the pure expert-major plan and block verifier through RED→GREEN.
- [x] Integrate exact block execution, CLI identity, telemetry, and failure atomicity.
- [x] Run B-0015 and independently cross-check raw JSON, CSV, summary, and checksums.
- [x] Synchronize README and every TITAN Ledger document with the verified Milestone 11–14 public and development state.
- [x] Run the full cross-backend and sanitizer verification matrix.
- [x] Complete final diff, evidence, default-path, and failure-boundary review.
- [x] Complete public PR integration and post-merge CI verification.

## Milestone 8 — Deadline-aware exact expert loading

- [x] Deadline scheduler contract와 deterministic unit tests.
- [x] Reader 및 L1 store concurrency safety.
- [x] Blocking reference를 보존하는 exact current-layer runtime integration.
- [x] CLI와 deadline scheduler telemetry.
- [x] B-0009 8-case ablation runner와 correctness 계약.
- [x] B-0009 WSL2 ext4 3-warmup/20-sample 측정과 raw 교차검증.
- [x] 전체 local verification, final review, review fix, TITAN Ledger 갱신.
- [x] Public GitHub PR #8, branch/PR CI, main fast-forward, post-merge CI 확인.

## Milestone 9 — Runtime-switchable expert cache policies

- [x] SpecMD 원문과 공개 구현 상태 재확인.
- [x] LRU/LFU/Least-Stale exact policy 설계와 TDD 계획 작성.
- [x] Deterministic trace oracle과 victim/collision tests.
- [x] Runtime/CLI/telemetry integration.
- [x] B-0010 ablation, full verification, review, ledger, public integration.

## Milestone 10 — Task and session profiles

- [x] Runtime-only metadata, persistent profile, and profiled eviction design.
- [x] TDD implementation and evidence plan.
- [x] Bounded profile data model and atomic canonical publication.
- [x] Opt-in profiled exact eviction with prior-to-live crossover.
- [x] Runtime/CLI integration and prompt/output non-interference proof.
- [x] B-0011 measurement, full verification, self-review fixes, and TITAN Ledger update.
- [x] Public PR, main fast-forward integration, and post-merge CI verification.

## Milestone 11 — Adaptive Top-K and exact rescue

- [x] Inspect the full-score router boundary and primary Kimi K3/vLLM evidence.
- [x] Compare checkpoint mutation, prefix selection, and residency-aware substitution.
- [x] Write the accepted design and TDD implementation plan.
- [x] Implement and verify the pure routing policy.
- [x] Add the 16-of-24 reference fixture and end-to-end parity.
- [x] Integrate CLI, telemetry, escalation, and exact rescue.
- [x] Run B-0012, full verification, ledger, review, and public integration.

## Milestone 6 independent L2 reader

- [x] 기존 Reader hot path와 Linux I/O capability 경계 조사.
- [x] Independent engine/cache axes와 B-0007 설계 명세 작성.
- [x] 상세 TDD 구현 계획 작성.
- [x] Ordered batch contract와 buffered pread 기준선 구현.
- [x] Six-extent expert batch 연결.
- [x] Optional io_uring engine과 explicit O_DIRECT mode 구현.
- [x] Runtime/profiler/B-0007 runner 연결.
- [x] 전체 correctness, sanitizer, 측정, TITAN Ledger, public GitHub 반영.

## Milestone 7 full-dimension bounded expert slice

- [x] 기존 storage boundary, released dimensions, format capability 조사.
- [x] 실제 크기 expert-only storage fixture 설계와 대안 비교.
- [x] 상세 TDD 구현 계획 작성.
- [x] Streaming safetensors source fixture와 manifest 구현.
- [x] K3X optional storage-fixture identity와 converter round-trip 구현.
- [x] 전용 C++ expert-load benchmark와 B-0008 runner 구현.
- [x] 전체 correctness, sanitizer, B-0008 측정, TITAN Ledger 갱신.
- [x] Final review Important 2건의 source integrity 및 resume ledger 검증을 테스트 우선으로 수정.
- [x] Public GitHub PR, CI, main 반영과 post-merge CI 확인.

## Milestone 28 — Official dependency-closed MoE FFN

- [x] Re-read the TITAN Ledger and the public M27 integration state.
- [x] Inspect the pinned official Kimi K3 graph, config, index, and shard header without downloading tensor payloads.
- [x] Compare a dependency-closed artifact, split trunk/expert artifacts, and authentic activation generation.
- [x] Write and self-review the accepted official MoE FFN design.
- [x] Write the detailed TDD implementation plan and continuity notes.
- [x] Add fail-closed native BF16 tensors to K3X v1.
- [x] Implement two-phase bounded official MoE source planning and materialization.
- [x] Implement the portable BF16/MXFP4 official MoE oracle.
- [x] Implement native BF16 CUDA residency and the official MoE boundary.
- [x] Implement the pinned fixture harness.
- [x] Implement strict B-0029 tooling.
- [x] Materialize the bounded real fixture without a full shard or checkpoint.
- [x] Run correctness, sanitizer, B-0029, full verification, and evidence cross-checks.
- [x] Synchronize the TITAN Ledger and complete final self-review.
- [x] Publish through PR #48, rebase-merge, and verify public `main` correctness and CodeQL.

## Milestone 29 — Official layer-1 KDA transformer boundary

- [x] Re-read the TITAN Ledger and public M28 state.
- [x] Verify the pinned official graph, source Git blob, configuration, index, and layer-1 shard header without tensor payload downloads.
- [x] Compare the external state boundary, layer-0 closure, and KDA-only alternatives.
- [x] Write and self-review the accepted complete-layer design.
- [x] Write the detailed TDD implementation plan.
- [x] Implement fail-closed official KDA config/header tensor planning.
- [x] Implement an independent portable KDA recurrence and full/incremental state oracle.
- [x] Implement bounded official KDA tensor materialization.
- [x] Implement portable C++ KDA recurrence and full/incremental boundary parity.
- [x] Compose the tiny complete portable layer with the official M28 MoE boundary.
- [x] Implement pinned artifact/manifest preflight before backend construction.
- [x] Implement the native CUDA complete-layer boundary and telemetry.
- [x] Implement the strict B-0030 runner and committed-evidence verifier.
- [x] Materialize the bounded fixture without a complete shard/checkpoint.
- [x] Run correctness, sanitizer, B-0030, full verification, and evidence cross-checks.
- [x] Synchronize README and the TITAN Ledger locally.
- [x] Publish Milestone 29 and verify pull-request and post-merge CI.

## Milestone 30 — Official KDA admission validation

- [x] Re-read the TITAN Ledger and public M29 state.
- [x] Inspect the existing M23 admission registry and official KDA validation boundary.
- [x] Compare backend-wide admission, KDA-specific caching, and harness-only bypass.
- [x] Write and self-review the accepted M30 design.
- [x] Write the detailed TDD implementation plan.
- [x] Implement and test atomic admission for fourteen official KDA immutable views.
- [x] Expose official-layer validation CLI and cold/measured telemetry.
- [x] Implement and verify the strict B-0031 evidence transaction.
- [x] Run actual-artifact correctness and Compute Sanitizer gates.
- [x] Run B-0031 exactly once and seal the evidence.
- [x] Run the complete verification matrix and synchronize README/TITAN Ledger.
- [x] Publish M30 and verify pull-request and post-merge CI.

## Milestone 31 — Official KDA device-state handoff

- [x] Compare state-residency approaches and accept an opaque single-slot backend token.
- [x] Write the M31 design and implementation plan.
- [x] Write and witness CUDA RED tests for state lifetime and transfer semantics.
- [x] Implement the dedicated device-state allocation and opaque token contract.
- [x] Add exact official-layer wrapper and explicit harness telemetry.
- [x] Build and verify the strict B-0032 evidence transaction.
- [x] Run actual-artifact correctness and Compute Sanitizer gates.
- [x] Run B-0032 exactly once and seal the evidence.
- [x] Run the complete verification matrix and synchronize README/TITAN Ledger.
- [x] Publish M31 and verify pull-request and post-merge CI.

## Milestone 32 — Official MoE device routing

- [x] Re-read the public M31 implementation and measured B-0032 bottleneck.
- [x] Compare device route preparation, a second official layer, and a monolithic whole-layer call.
- [x] Accept D-070 and write the M32 design.
- [x] Write and self-review the detailed TDD implementation plan.
- [x] Write and witness CUDA RED tests for preparation parity and token lifetime.
- [x] Implement exact device residual preparation and raw router logits.
- [x] Implement opaque prepared-activation consumption and wrapper cleanup.
- [x] Add explicit host/device route-preparation harness mode.
- [x] Add strict B-0033 evidence tooling.
- [x] Run actual-artifact correctness, sanitizer, and production-guard gates.
- [x] Run B-0033 exactly once and seal the evidence.
- [x] Run the complete verification matrix and synchronize README/TITAN Ledger.
- [x] Publish M32 and verify pull-request and post-merge CI.

## Milestone 33 — Official two-layer device closure

- [x] Re-read the public M32 implementation and measured B-0033 bottleneck.
- [x] Compare real layers 1 and 2, layer-1 replay, and a monolithic device-Top-K call.
- [x] Accept D-071 and write the M33 design.
- [x] Write and self-review the detailed TDD implementation plan.
- [x] Generalize exact layer-2 MoE/KDA metadata planning without materialization.
- [x] Compose pinned layer-1/2 plans and the pure interleaved trace scheduler without payload.
- [x] Generalize the bounded planner and oracle to an exact layer-1-to-layer-2 trace.
- [x] Build and round-trip one execution-ordered two-layer K3X v1 fixture.
- [x] Add the portable exact A1→A2→B1→B2 runtime oracle with independent KDA state ownership.
- [x] Add the capacity-two layer-keyed KDA state registry with ownership tests.
- [x] Add the opaque inter-layer hidden token and full CUDA front/tail bridge.
- [x] Add the exact host-round-trip/device-closure A1→A2→B1→B2 CUDA wrapper.
- [x] Add explicit two-layer harness mode and closed telemetry.
- [x] Add strict B-0034 evidence tooling.
- [x] Run actual-artifact correctness, sanitizer, production-guard, and full portable regression gates; record the complete CUDA Python timeout without overstating it.
- [x] Run B-0034 exactly once successfully and seal the evidence after two fail-closed unpublished attempts.
- [x] Synchronize README and TITAN Ledger for the measured local milestone.
- [x] Publish M33 through PR #58 and verify pull-request and post-merge correctness and CodeQL.

## Milestone 34 — Two-layer closure attribution

- [x] Read the sealed B-0034 bottleneck and exact front/tail implementation.
- [x] Compare per-kernel events, duplicated harness orchestration, and existing-profiler snapshots.
- [x] Accept an opt-in profiler-snapshot attribution design with no new CUDA synchronization.
- [x] Write and self-review the M34 implementation plan.
- [x] Add the caller-owned attribution accumulator with CUDA ownership/parity tests.
- [x] Add an opt-in harness schema while preserving default B-0034 output.
- [x] Add strict B-0035 evidence tooling.
- [x] Run final actual-artifact correctness, production-guard, and complete regression gates.
- [x] Run B-0035 once successfully, strictly reverify it, and seal the evidence.
- [x] Synchronize the TITAN Ledger, publish M34 through PR #60, and verify pull-request plus post-merge CI.

## Milestone 35 — Two-layer operation attribution

- [x] Read sealed B-0035 and inspect the exact existing profiler event boundary.
- [x] Compare current-event aggregation, new per-kernel CUDA events, and external-only profiling.
- [x] Accept D-073 and write the M35 design plus implementation plan.
- [x] Add RED ownership, classification, closure, and compatibility tests.
- [x] Implement checked existing-event classification without new CUDA synchronization.
- [x] Add the opt-in operation-attribution harness schema and B-0036 evidence transaction.
- [x] Run actual-artifact correctness, sanitizer, production-guard, and full regression gates.
- [x] Run B-0036 once successfully, strictly reverify it, and seal the evidence.
- [ ] Synchronize the TITAN Ledger, publish M35, and verify CI.

## Milestone 36 — Official first token

- [x] Freeze the bounded streaming design and inline execution plan.
- [x] Generate and validate the pinned 93-layer topology manifest without payload download.
- [x] Implement the verified selected-range cache.
- [x] Connect the missing dense, MLA, embedding, output residual, norm, and LM-head nodes.
- [x] Execute actual released layers 0 through 3, including the first KDA/KDA/KDA/MLA hybrid block.
- [ ] Execute and seal one actual released-checkpoint greedy token.
- [ ] Run one focused gate and one integration gate, then publish.

## Milestone 37 — Local Foundry

- [x] Measure authenticated HF Xet throughput on one official 4.7 GB shard.
- [x] Verify the 1.28 TB output, 200 GiB NVMe reserve, and two-slot HDD staging budget.
- [x] Freeze the local Foundry design and inline implementation plan.
- [x] Implement the disk-safe manifest and IMMORTAL ledger.
- [x] Implement and verify the deterministic 3-bit expert codec.
- [x] Pass the portable CPU/reference synthetic local manufacturing integration gate.
- [x] Launch the resumable 96-shard official manufacture.

## Milestone 38 — Quality Local Foundry

- [x] Supersede the 3-bit download lock with explicit user authorization and D-076.
- [x] Implement group-128 8-bit codec, K3X ABI, portable Reader support, and shard converter.
- [x] Download, verify, manufacture, ledger, and source-clean official shards 1 and 2.
- [x] Start the resumable two-slot conductor for shards 3 through 96.
- [x] Decode native BF16 and group-128 Q8 trunk tensors in the portable model loader.
- [x] Batch converter durability checkpoints while preserving resume and final verification gates.
- [x] Open fragments through one checksum-bound sharded K3X index without copying payloads.
- [x] Stream passthrough tensors from the authenticated source shard without a model-sized temporary copy.
- [x] Load BF16/F32/Q8 dense tensors and native MXFP4 expert matvecs by canonical name from K3X fragments.
- [x] Bind every official Python graph entrypoint to a sealed K3XSET1 payload source.
- [x] Avoid rescanning every fragment payload at each sealed compatibility-runner process start.
- [x] Serialize concurrent IMMORTAL ledger publication and parameterize disjoint staging ranges.
- [x] Launch three disjoint workers after shard 6.
- [ ] Measure aggregate three-worker manufacture throughput.
- [x] Remove one duplicate full-source hash while retaining pre-conversion and pre-deletion authentication.
- [x] Arm an idempotent 96-fragment finalizer and K3X first-token runner.
- [x] Bind every K3X prefix state and result to the sealed set digest.
- [ ] Complete and verify all 96 K3X fragments.
- [ ] Complete the resumed original-precision layer-24 reference oracle and record its greedy token.
- [ ] Assemble the final executable checkpoint and generate the first token.
- [ ] Measure and optimize the actual end-to-end bottleneck.

### 3-bit runtime correctness slice

- [x] Re-read the accepted Local Foundry design and implementation plan.
- [x] Add witnessed RED coverage for K3X quantization 2 metadata and CPU decode.
- [x] Implement the minimum Python writer/reader and C++ reader/runtime boundary.
- [x] Convert the synthetic K3-compatible model and prove layer/token parity.
- [x] Update the TITAN Ledger documents and commit the verified boundary.
- [x] Implement scalar direct-packed CUDA execution and synthetic parity.
- [ ] Close the official conversion recipe and quality gates.
  - [x] Seal an exact byte-accounting recipe below 1.28 TB from the pinned official index.
  - [x] Convert one released MXFP4 expert to group-32 3-bit and measure output divergence.
  - [ ] Run the direct-packed CUDA path on the released-size expert fixture.
  - [x] Keep the 96-shard launch disabled because the bounded quality gate did not pass.
