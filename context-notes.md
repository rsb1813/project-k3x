# K3X 컨텍스트 노트

## 2026-08-08

- 워크스페이스는 시작 시 완전히 비어 있었고 Git 저장소가 아니었다.
- 사용자는 Python/PyTorch reference와 C++20 CPU runtime, 후속 CUDA backend 조합을 승인했다.
- 첫 마일스톤의 실행 범위는 텍스트 디코더이다. MoonViT-V2는 아키텍처 연결 경계만 문서화하고 실행 구현은 연기한다.
- 첫 마일스톤은 KDA, Gated MLA, Attention Residual, Stable LatentMoE, router, native MXFP4 decode, incremental state 및 greedy generation의 correctness를 검증한다.
- 합성 모델은 4개 decoder layer를 `KDA, KDA, KDA, MLA` 순서로 배치한다. 실제 모델의 3 KDA 뒤 1 MLA 패턴을 최소 크기로 재현하기 위한 결정이다.
- 합성 Attention Residual block size는 여러 depth source를 작은 모델에서 실제로 테스트하기 위해 2로 축소한다. 실제 checkpoint 값은 12이며 문서에서 명확히 구분한다.
- Native MXFP4 expert는 dequantize 후 requantize하지 않는다. 합성 converter도 payload와 scale의 byte-exact 보존을 검증한다.
- K3X v1은 4 KiB superblock, little-endian fixed-width records, aligned extents, per-extent CRC32C와 최종 SHA-256, required/optional feature bits를 사용한다.
- 미완성 변환은 `.partial` artifact와 resume manifest에 남기고, 모든 extent 검증 후 원자적으로 최종 이름으로 바꾼다.
- 확인한 공식 Kimi K3 config의 주요 값은 hidden 7168, 93 layers, 69 KDA, 24 MLA, 96 heads, latent MoE 3584, 896 experts, Top-16, 2 shared experts, AttnRes block 12, KDA short convolution 4이다.
- 조사 시점의 source HEAD는 설계 명세의 source ledger에 고정한다. 이후 upstream 변경은 자동으로 현재 설계의 근거가 되지 않는다.

