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
- [ ] Write the detailed TDD implementation plan.
- [ ] Implement the backend batch contract through witnessed RED and GREEN cycles.
- [ ] Integrate the exact CUDA runtime, CLI capability gate, and telemetry.
- [ ] Run B-0016, full verification, sanitizer, ledger, review, and public integration.
- [x] Update the README milestone badge and current implementation summary.
- [x] Correct the stale speculative-implementation statement and publication HEAD in `PROJECT_STATE.md`.
- [x] Run documentation checks, commit, publish, and verify public `main`.

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
