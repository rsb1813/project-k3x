# K3X 첫 마일스톤 체크리스트

## 설계

- [x] 빈 워크스페이스와 Git 상태 확인.
- [x] 구현 언어와 첫 마일스톤 범위 승인.
- [x] 아키텍처, 데이터 흐름, 정확성 정책, 완료 기준 승인.
- [x] 승인된 설계 명세 작성 및 자체 검토.
- [x] 설계 명세 사용자 검토 승인.
- [x] 세부 구현 계획 작성.

## 구현

- [ ] 저장소 도구 구성과 합성 config 계약 테스트.
- [ ] PyTorch 기본 연산과 MXFP4 oracle.
- [ ] KDA와 recurrent/ShortConv state parity.
- [ ] Gated MLA와 incremental KV parity.
- [ ] router, Stable LatentMoE, shared expert parity.
- [ ] Attention Residual과 decoder graph parity.
- [ ] greedy generation full/incremental parity.
- [ ] K3X v1 streaming writer, reader 및 validator.
- [ ] crash-safe resume와 손상 감지.
- [ ] C++20 runtime 연산·layer·end-to-end parity.

## 검증과 전달

- [ ] Python 전체 테스트 통과.
- [ ] C++ 전체 테스트 통과.
- [ ] 합성 checkpoint round-trip 통과.
- [ ] benchmark와 peak RSS/read bytes 측정.
- [ ] `ARCHITECTURE.md`, `PERFORMANCE_MODEL.md`, `K3X_FORMAT.md` 완성.
- [ ] diff 자체 검토와 semantic commits 완료.
- [ ] 측정 결과와 다음 병목 보고.
