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
- 사용자가 작성된 첫 마일스톤 설계 명세를 승인했다.
- 구현 계획 작성 시 기본 환경에서 Python 3.14.6만 확인되었고 PyTorch, CMake, C++ compiler 및 WSL은 없었다. Codex 번들 Python 3.12.13은 사용할 수 있지만 PyTorch는 설치되어 있지 않다.
- 2026-08-08 PyPI index에서 확인한 계획상 Python dependency version은 torch 2.13.0, numpy 2.5.1, safetensors 0.8.0, pytest 9.1.1, google-crc32c 1.8.0, psutil 7.2.2이며 build backend는 setuptools 83.0.0이다. 설치 전에는 사용자에게 package-install 권한을 별도로 확인한다.
- 첫 마일스톤 구현은 연결된 correctness graph이므로 기본 실행 방식은 주 agent의 inline execution으로 둔다. 독립 agent 사용은 사용자가 명시적으로 요청할 때만 선택한다.
- 사용자가 로컬 dependency 설치와 전체 구현 진행을 승인했다. 격리 worktree `feat/milestone-zero`에서 작업한다.
- 저장소 전용 `.venv`에 Python 3.12.13, torch 2.13.0, numpy 2.5.1, safetensors 0.8.0, pytest 9.1.1, google-crc32c 1.8.0, psutil 7.2.2, CMake 4.4.2, Ninja 1.13.0을 설치했다. 시스템에는 MSVC 19.51이 이미 있어 LLVM 설치는 진행하지 않았다.
- 합성 config 계약 테스트는 의도한 `ModuleNotFoundError`로 RED를 확인한 뒤 2개가 통과했다. `pip check`도 문제 없음으로 통과했다.
