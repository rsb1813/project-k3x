# K3X 첫 마일스톤 계획

> 이 문서는 완료된 Milestone 0의 최초 실행 계획을 보존합니다. 현재 마일스톤, 검증 상태, 측정 결과, 다음 작업은 `PROJECT_STATE.md`, `BENCHMARKS.md`, `checklist.md`를 기준으로 합니다.

## 목표

전체 Kimi K3 체크포인트나 유료 클라우드 자원을 사용하지 않고, K3의 텍스트 디코더 그래프를 축소한 합성 모델을 PyTorch 기준 구현과 독립 C++20 runtime에서 동일하게 실행하고 K3X 형식으로 round-trip한다.

## 검증 가능한 단계

1. 출처와 그래프 계약을 고정한다.
   - 검증: 공식 config, 공개 reference 및 serving 구현의 commit과 핵심 연산 순서를 문서에 기록한다.
2. 합성 PyTorch 기준 모델을 구축한다.
   - 검증: 연산별 golden 값, full/incremental parity, greedy token 결과를 테스트한다.
3. K3X v1 writer와 reader를 구축한다.
   - 검증: streaming 변환, byte-exact MXFP4, checksum, 손상 거부, 중단 후 재개를 테스트한다.
4. 독립 C++20 CPU runtime을 구축한다.
   - 검증: PyTorch golden과 연산·layer·state·logit·token 결과를 대조한다.
5. 문서화하고 측정한다.
   - 검증: 전체 테스트와 benchmark를 실행하고 실제 시간, peak RSS 및 read bytes를 기록한다.

## 범위 밖

- 전체 Kimi K3 weight 다운로드.
- CUDA kernel 및 RTX 5080 최적화.
- 비동기 3-tier cache와 speculative decoding.
- Cloud Run Job 배포와 유료 자원 생성.
- MoonViT-V2 합성 실행.

