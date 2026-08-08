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
