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
- 고정한 PipeNetwork revision의 vendored text reference와 MLX port를 재검증한 결과 K3 MLA는 `mla_use_nope=true`이며 `qk_rope_head_dim`으로 명명된 64차원 subspace에도 rotary embedding을 적용하지 않는다. 사용자 승인에 따라 설계와 계획에서 이를 shared extra-key NoPE 경로로 정정했다.
- FP32 RMSNorm, SiTU-GLU, native MXFP4 E2M1/E8M0 decode와 reference matmul 테스트 6개가 통과했다. Nibble 순서를 의도적으로 뒤집었을 때 literal test가 실패함을 확인하고 low-nibble-first 구현을 복원했다.
- KDA reference는 causal depthwise ShortConv history, head-wise `A_log`, channel-wise forget logits, sigmoid beta, delta write 후 read, full-rank output gate를 명시적 state로 구현했다. Pre-update state를 읽도록 변이했을 때 literal recurrence test가 실패함을 확인했다.
- Gated MLA reference는 q-LoRA와 normalized KV latent, main key/value, shared extra NoPE key, output gate를 명시적으로 저장한다. 길이 1, 2, 5에서 KDA와 MLA 모두 prefill과 token-by-token decode 출력 및 최종 state가 exact match했다.
- 전체 그래프의 full-prefix와 incremental 경로는 layer 0부터 GEMM batch shape가 달라 최대 `9.31e-9`, 최종 logits에서 최대 `3.58e-7`의 정상적인 FP32 반올림 차이가 측정됐다. 따라서 cross-path 및 cross-language state는 `atol=rtol=1e-6` 수치 비교를 사용하고, byte-exact digest는 같은 runtime의 반복 실행에만 사용한다.
- 공식 vendored `config.json`에서 SiTU beta 4.0, linear beta 25.0, MLA output gate 활성화, routed scaling factor 1.0을 확인해 합성 config와 K3X model metadata에 반영했다.
- Router의 correction bias는 선택에만 사용하고 unbiased sigmoid score를 normalize한다. Stable LatentMoE는 native MXFP4 routed branch와 원래 hidden-space shared branch를 분리하며, AttnRes는 normalized key로 score한 뒤 raw source를 혼합한다.
- Controlled graph는 prompt `[1,7,3,9]`에서 full/incremental 모두 token 5를 6회 생성했다. Seeded graph도 두 mode의 greedy token이 exact match했고, source shard와 golden fixture를 두 번 생성한 SHA-256 manifest가 동일했다.
- K3X v1 writer는 safetensors header만 읽어 tensor extent를 찾고, 설정된 chunk 크기 이하로 payload를 복사한다. native MXFP4 packed code와 scale은 별도 aligned extent로 byte-for-byte 보존한다.
- 각 extent는 flush와 fsync 뒤 partial 파일에서 다시 읽어 CRC32C를 확인한 후에만 원자적 resume ledger에 기록한다. source fingerprint가 달라진 resume는 거부한다.
- 4 KiB superblock literal layout, 128-byte tensor record, extent 정렬·중복·절단 검증, payload corruption, unknown required feature, source mutation, 중단 재개, CLI dry-run 및 native MXFP4 byte parity를 포함해 Python 전체 35개 테스트가 3.49초에 통과했다.
- C++20 runtime은 외부 ML library 없이 portable CRC32C/SHA-256, strict K3X reader, RMSNorm, SiTU, native MXFP4와 합성 KDA/MLA/Attention Residual/Stable LatentMoE graph를 구현한다. Root SHA-256은 1 MiB heap buffer로 증분 검증해 artifact 전체를 RAM에 올리지 않는다.
- C++ reader는 Python writer artifact를 열고 payload CRC corruption, unknown required feature, truncation을 동일한 안정적 오류 코드로 거부했다. C++ full/incremental greedy 결과는 PyTorch의 `[43,32,28,49,9,28]`과 모두 exact match했다.
- 첫 cross-language 실행의 Windows `0xC00000FD`는 reader 함수의 1 MiB stack SHA buffer 때문임을 확인했다. heap buffer로 이동한 뒤 같은 parity test가 통과했다.
- C++ runtime에 prefill, decode, layer별 단조 시계 계측을 추가하고 별도 benchmark driver에서 프로세스 RSS, 논리적 artifact read bytes, KDA/MLA state 크기를 수집한다. 첫 토큰은 prefill logits에서 나오므로 decode tok/s 계산에서는 이후 5개 토큰만 사용한다.
- 최종 리뷰에서 첫 token의 argmax 시계 구간과 실행기 무결성 검증 생략을 발견해 바로잡았다. 2026-08-08 Windows 11 AMD64, MSVC Debug, 3 warmup과 20회 재측정에서 합성 모델 prefill 405.11 tok/s, 실제 5회 incremental forward decode 558.89 tok/s, strict 전체 artifact 검증을 포함한 process-level TTFT 중앙값 86.20 ms, peak child RSS 6,270,976 bytes, logical tensor read 110,936 bytes/generated token을 측정했다. 이는 합성 correctness harness 결과이며 실제 Kimi K3 또는 RTX 5080 성능 근거가 아니다.
- 공개 config 차원으로 계산한 native MXFP4 routed expert는 17,547,264 bytes이다. cache reuse가 전혀 없는 natural Top-16의 92개 MoE layer expert traffic은 25,829,572,608 bytes/token이다. P44 Pro 공개 최대 7.0 GB/s만 적용한 expert-only ceiling은 0.27 tok/s이므로 다음 실제 병목은 NVMe expert traffic 회피율이다.
- 공식 차원으로 계산한 BF16 text trunk 추정치는 약 85.72 GB이며 전체 checkpoint scan 결과가 아니다. 96 GB RAM에서 OS와 expert bank까지 확보하려면 sensitivity-aware trunk quantization이 필요하다는 capacity planning 근거로만 사용한다.
- GitHub Actions는 Linux에서 C++ build/CTest 뒤 Python 및 cross-language suite를 실행한다. 테스트 runner 경로는 Windows `.exe`와 POSIX binary를 모두 찾도록 정정했다.
- 최종 diff 리뷰에서 발견한 실행기 `metadata_only`, 불완전한 C++ layer/expert directory parser, final rename 직후 stale ledger, token-only C++ parity를 모두 회귀 테스트와 함께 수정했다. 실행기는 생성 전에 artifact 전체 무결성을 검증하고, C++ 진단 mode는 prompt logits, 네 layer 출력, canonicalized KDA/MLA state를 PyTorch와 `1e-6` 허용오차로 비교한다.

## 2026-08-08 Milestone 1

- 사용자는 CPU exact runtime과 profiler를 먼저 확장하고, 이어서 RTX 5080용 basic CUDA backend를 만드는 순서를 승인했다.
- custom CUDA only, cuBLASLt only, hybrid baseline을 비교했고 사용자는 cuBLASLt 기준 경로와 custom MXFP4 경로를 함께 두는 hybrid 설계를 승인했다.
- 실제 개발 PC에서 NVIDIA GeForce RTX 5080 16,303 MiB, compute capability 12.0, driver 591.86, CUDA toolkit 13.3과 nvcc 13.3.73을 확인했다.
- 설치된 nvcc는 `compute_120`과 `sm_120`을 지원한다. CUDA 13.3 local header만 보면 E2M1과 UE8M0/32 mode가 함께 보이지만, NVIDIA 공식 cuBLAS 문서는 FP4에 UE4M3/16을 요구하고 UE8M0/32를 FP8 mode로 규정한다. K3의 E2M1+E8M0/32 native MXFP4와 직접 호환되지 않으므로 cuBLASLt FP4 expert path는 기각하고 cuBLASLt는 dense baseline에만 사용한다.
- 격리 worktree `feat/milestone-one-runtime`을 만들고 Python 3.12.13 및 `pyproject.toml`의 고정 dependencies를 새 `.venv`에 설치했다.
- baseline의 첫 Python 실행은 C++ build 전이라 cross-language runner 7개가 없어서 실패했다. MSVC Developer Command 환경과 `.venv`의 Ninja 경로를 명시해 Release build와 CTest 2/2 통과를 확인했다.
- 새 `k3x_run.exe` 실행은 Windows Smart App Control policy `{0283ac0f-fff1-49ae-ada1-8a933130cad6}`에 의해 차단됐다. Code Integrity event 3033과 3077에서 unsigned executable이 Enterprise signing level을 충족하지 못했다는 원인을 확인했다. Python suite는 이 환경 제약으로 cross-language 5개가 차단되고 나머지 41개가 통과했다.
- K3X는 Smart App Control을 자동으로 끄거나 신뢰 저장소를 수정하지 않는다. CUDA runtime 완료 판정은 Linux native 또는 사용자가 별도로 승인한 WSL2 GPU 환경에서 수행한다.
- 사용자는 Milestone 1 hybrid 설계 명세를 승인하고 TITAN LEDGER 지속성 프로토콜을 추가했다. `PROJECT_CHARTER.md`는 안정된 헌장, `ARCHITECTURE.md`는 상태가 표시된 실제 구조, `DECISIONS.md`는 선택 근거, `BENCHMARKS.md`는 측정 원장, `PROJECT_STATE.md`는 마지막에 갱신하는 현재 상태로 분리한다.
- APOLLO, TITAN COUNCIL, AURORA, PROMETHEUS-X, MERCURY, ORBIT, HELIOS, SHADOW, PHOENIX, VAULT, VEILBREAK, AUTO는 구현·benchmark 전까지 proposed이다. ATLAS, CHRONOS, BLACKSTAR는 역할 정의가 제공되지 않아 reserved proposed/undefined로 기록했다.
- Windows에는 WSL distribution이 설치되어 있지 않다. WSL2 설치는 시스템 변경과 재부팅 가능성이 있으므로 명시적 사용자 승인 전에는 실행하지 않는다.
- Milestone 1 구현 계획은 Linux GPU 실행 환경을 선행 관문으로 두고 profiler, CPU backend extraction, optional CUDA shell, cuBLASLt dense, custom MXFP4, end-to-end profiling, 최종 측정 순서로 작성했다.
- 사용자가 inline execution과 WSL2 Ubuntu 24.04 설치를 명시적으로 승인했다. 비관리자 `wsl --install`은 기능 비활성 상태에서 exit 1로 종료됐고, 관리자 DISM으로 `Microsoft-Windows-Subsystem-Linux`와 `VirtualMachinePlatform`만 `/NoRestart` 활성화했다. 두 기능 모두 `InstallState=1`을 확인했으며 CBS/PendingFileRename 재부팅 대기 상태라 자동 재부팅 없이 중단했다.
- 재부팅 후 WSL 2.7.11.0과 Ubuntu 24.04.4 LTS를 설치했다. Linux kernel은 6.18.33.2이며 RTX 5080 16,303 MiB, compute capability 12.0, Windows driver 591.86이 WSL에 정상 노출된다.
- NVIDIA WSL 지침에 따라 Linux display driver는 설치하지 않고 WSL 저장소의 `cuda-toolkit-13-3`만 설치했다. 실제 package version은 13.3.1-1, nvcc는 13.3.73이며 `sm_120`을 제공한다.
- Ubuntu 기본 개발 사용자는 비관리자 `jolib`로 설정했다. Python 환경은 `/home/jolib/.venvs/k3x-m1`, native build는 Git에서 무시되는 `build-linux`를 사용한다.
- 생산 코드 변경 전 Linux Release baseline은 CTest 2/2와 cross-language 제외 pytest 39/39가 통과했다.
- `K3X_BUILD_DIR` resolver의 RED ImportError를 확인한 뒤 공통 native binary resolver를 구현했다. targeted cross-language 8/8, 전체 pytest 47/47, CTest 2/2가 통과했고 커밋은 `b6d900f`다.
- deterministic profiler는 명시적 event만 소유하며 clock, thread, JSON, CUDA 의존성을 갖지 않는다. 실패 event는 실패 횟수만 증가시키고 시간·byte 합계에서는 제외하며 H2D와 D2H bytes를 분리한다. unknown target RED 후 CTest 3/3과 pytest 47/47을 확인했고 커밋은 `f06ce97`이다.
- exact CPU backend는 기존 double 누산 dense와 native MXFP4 matvec만 격리한다. graph는 명시적 backend와 prefill/decode phase를 전달하고 LM head는 global layer로 기록한다. literal backend RED/GREEN, CTest 4/4, cross-language parity 5/5, 전체 pytest 47/47을 확인했고 커밋은 `b439e25`이다.
- Release `NDEBUG`가 최근 `assert` 기반 profiler/backend test 표현식을 제거해 unit checks가 실행되지 않은 문제를 발견했다. 기존 suite 방식인 explicit return codes로 교체해 실제 RED와 GREEN을 재확인했고 커밋은 `d1b52d4`이다.
- optional CUDA shell은 OFF에서 typed unavailable stub만 링크하고 CUDA dependency가 없으며, ON에서 CUDA 13.3, native `sm_120`, capability 12.0 gate, nonblocking stream, cuBLASLt handle을 RAII로 소유한다. CPU/CUDA CTest 각각 5/5, CPU 전체 pytest 47/47, CUDA cross-language parity 5/5를 확인했고 커밋은 `5b6d1e7`이다. CUDA matrix compute는 아직 미구현이다.
- cuBLASLt dense baseline은 row-major FP32 host API를 유지하고 FP32 또는 BF16-rounded GPU operand를 zero-workspace heuristic으로 실행하며 FP32로 누산·반환한다. CUDA 13.3의 정규 행렬 계약은 BF16 `Atype/Btype`을 함께 요구하므로 BF16 mode는 입력과 가중치를 모두 RNE 반올림해 staging한다. literal test에서 FP32 H2D/D2H는 36/8 bytes, BF16-rounded는 18/8 bytes이며 두 mode 모두 독립 oracle과 일치한다. `compute-sanitizer` memcheck 0 errors, CUDA CTest 6/6, CPU CTest 5/5, CPU pytest 47/47, CUDA cross-language parity 5/5를 확인했고 구현 커밋은 `c4b612a`다. 이는 correctness 측정이며 throughput benchmark는 아니다.
- exact custom MXFP4 baseline은 K3X low-nibble-first E2M1과 E8M0/32 payload를 재패킹 없이 읽고, output row당 256-thread block 하나에서 FP32 shared-memory reduction을 수행한다. 3-row/64-column literal은 두 scale group, sign, exponent, high/low nibble을 검증하고 320-column mutation fixture는 두 번째 thread stride를 제거하면 실패한다. invalid extent, non-32 group, reserved `0xFF` scale을 거부한다. 승인된 comparison 계약에 맞춰 `cuda_dense`는 CPU MXFP4 oracle을 유지하며 H2D와 device time이 0이고, custom kernel은 `cuda_custom`에서만 실행된다. `compute-sanitizer` memcheck 0 errors, CUDA CTest 7/7, CPU CTest 5/5, CPU pytest 47/47, CUDA cross-language parity 5/5를 확인했다. 구현 커밋은 `ea730c5`, cuda-dense oracle 정정은 `7d4ade6`이다. archive의 `mxfp4.cu.o`에는 native `sm_120` cubin이 있다. 처리량은 아직 측정하지 않았다.
