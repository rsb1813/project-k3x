# K3X Benchmark Ledger

## Evidence rules

- `measured` means the command ran and its output was recorded.
- `derived` means arithmetic was applied to released dimensions or published hardware specifications.
- `not measured` is never replaced with an estimate in a measured field.
- Every run identifies its commit, hardware, checkpoint scope, execution mode, context, quality contract, and enabled optimizations.

## B-0001 — Milestone 0 synthetic CPU runtime

| Field | Value |
|---|---|
| Evidence | measured |
| Date | 2026-08-08 |
| Result commit | `c8863ba90b0774dde2cfbe7b1f3dc400b822c5b1` |
| Latest portability fix | `b86280ed5eefc41992b1ea02e20204edea6b61cf` |
| Hardware | Windows 11 AMD64 host; CPU execution; RTX 5080 unused |
| Build | MSVC Debug |
| Model/checkpoint | deterministic `synthetic-milestone-zero` K3-compatible K3X artifact |
| Mode | exact C++20 CPU incremental decoding with strict whole-artifact verification |
| Context length | 4 prompt tokens, prompt IDs `[1, 7, 3, 9]` |
| Generated tokens | 6, token IDs `[43, 32, 28, 49, 9, 28]` |
| Warmup / samples | 3 / 20 process runs |
| Decode tok/s | 558.890267, timing exactly five post-prefill forward passes |
| Prefill tok/s | 405.112520 |
| TTFT | 86.199950 ms median, including process startup and strict artifact verification |
| VRAM | not measured; CUDA unused |
| System RAM | 6,270,976 bytes observed peak child RSS |
| NVMe GB/token | not measured |
| H2D GB/token | 0; CUDA unused |
| Logical K3X tensor bytes/generated token | 110,936 bytes; not an OS or NVMe counter |
| Expert-cache hit rate | not applicable; no tiered cache |
| Average Top-K | 2, fixed synthetic router setting |
| Speculative acceptance | not applicable; speculation disabled |
| Unique experts/verification block | not applicable |
| Cold rescue count | not applicable |
| Quality result | CPU and PyTorch layer/logit/state parity within `atol=rtol=1e-6`; greedy tokens exact |
| Enabled optimizations | none; portable reference arithmetic, native MXFP4 payload decode, incremental state |

Median per-layer time was 2.1913, 5.4949, 5.51105, and 5.0384 ms for layers zero through three. These figures apply only to the tiny deterministic model and are not evidence for full Kimi K3 or RTX 5080 throughput.

## B-0002 ??Milestone 1 synthetic CPU/CUDA backend comparison

| Field | Value |
|---|---|
| Evidence | measured; ratios and per-token byte conversions are arithmetic over measured counters |
| Date | 2026-08-08 |
| Code commit | `c92f498` |
| Hardware | AMD Ryzen 7 9800X3D; NVIDIA GeForce RTX 5080 16,303 MiB; driver 591.86; WSL2 exposes 49,251,213,312 bytes RAM |
| Environment | WSL2 Ubuntu 24.04.4, Linux 6.18.33.2, CUDA Toolkit 13.3.1, nvcc 13.3.73, native `sm_120` Release builds |
| Model/checkpoint | deterministic `synthetic-milestone-one` K3-compatible K3X artifact; no full Kimi K3 weights |
| Mode | exact incremental generation; explicit backend; strict artifact verification; no silent fallback |
| Context length | 4 prompt tokens, prompt IDs `[1, 7, 3, 9]` |
| Generated tokens | 6, exact token IDs `[43, 32, 28, 49, 9, 28]` in every mode |
| Warmup / samples | 3 / 20 separate process runs per mode |
| Quality mode | synthetic exact-routing contract; fixed Top-2 router; no pruning, proxy, adaptive K, or speculation |

| Backend / dense precision | Decode tok/s | Prefill tok/s | TTFT ms | Peak RSS MiB | H2D bytes/run | D2H bytes/run | Peak backend VRAM bytes | CUDA kernel ms/run | Max abs. error | Max rel. error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `cpu` / FP32 | 19.4858 | 11.3489 | 797.398 | 225.21 | 0 | 0 | 0 | 0 | 0 | 0 |
| `cuda-dense` / FP32 | 11.6682 | 7.1813 | 1,244.627 | 483.23 | 4,999,104 | 90,864 | 25,216 | 11.561 | 1.640e-7 | 2.485e-4 |
| `cuda-custom` / FP32 | 10.1118 | 6.5095 | 1,296.817 | 483.60 | 5,107,968 | 111,600 | 25,216 | 14.521 | 1.751e-7 | 2.485e-4 |
| `cuda-dense` / BF16 | 11.4957 | 7.2915 | 1,239.816 | 458.96 | 2,499,552 | 90,864 | 12,800 | 11.844 | 0.00402409 | 17.5009 |
| `cuda-custom` / BF16 | 10.1235 | 6.6671 | 1,288.632 | 459.26 | 2,608,416 | 111,600 | 12,800 | 14.314 | 0.00402409 | 17.5009 |

All modes recorded 110,936 logical K3X tensor bytes per generated token. This is not an OS or NVMe counter, so NVMe GB/token and I/O stall time remain not measured. FP32 H2D traffic is 0.000833184 GB/generated token for `cuda-dense` and 0.000851328 GB/generated token for `cuda-custom`; BF16 reduces these to 0.000416592 and 0.000434736 GB/generated token. GPU utilization and memory bandwidth were not sampled. Cache hit rate, cold rescue count, speculative acceptance, and unique experts per verification block are not applicable because those systems do not exist in this milestone.

The maximum relative BF16 error is dominated by reference values close to zero and should not be read as a quality percentage. Maximum absolute error and exact greedy tokens are the useful current checks; broader quality evaluation is still required before BF16 can become a quality mode default.

The CPU backend wins this tiny end-to-end comparison. CUDA-event kernel work occupies only 11.56--14.52 ms per full run, while the process performs hundreds of milliseconds of graph work. The measured immediate bottleneck is per-operation device allocation, host staging, synchronous copies, synchronization, and CPU-resident graph logic. These synthetic numbers do not predict full Kimi K3 throughput, where expert bytes and reuse dominate.

Raw records are stored in `results/m1-*.json` and `results/m1-*.csv`.

## B-0003 — Milestone 2 CUDA allocation, residency, and grouping ablation

| Field | Value |
|---|---|
| Evidence | measured; deltas and per-token conversions are arithmetic over measured counters |
| Date | 2026-08-09 |
| Code commit | `a468db8` |
| Hardware | AMD Ryzen 7 9800X3D; NVIDIA GeForce RTX 5080 16,303 MiB; driver 591.86 |
| Environment | WSL2 Ubuntu 24.04.4, Linux 6.18.33.2, CUDA Toolkit 13.3.1, nvcc 13.3.73, native `sm_120` Release build |
| Model/checkpoint | regenerated deterministic `synthetic-milestone-one` K3-compatible K3X artifact; no full Kimi K3 weights |
| Mode | exact incremental generation; fixed Top-2 synthetic routing; no pruning, proxy, speculation, adaptive K, asynchronous storage, or eviction |
| Context length | 4 prompt tokens, prompt IDs `[1, 7, 3, 9]` |
| Generated tokens | 6; exact `[43, 32, 28, 49, 9, 28]` verified by the benchmark's CPU diagnostic comparison |
| Warmup / samples | 3 / 20 separate process runs per configuration |
| Resident capacity | 8,388,608 bytes; measured resident use never exceeded 573,120 bytes |
| NVMe GB/token | not measured; OS/block-device I/O counters are not implemented |
| GPU utilization / memory bandwidth | not measured |
| I/O stall time | not measured; asynchronous storage is not implemented |
| Average Top-K | 2, fixed synthetic router setting |
| Speculative acceptance / unique experts per verification block | not applicable; speculation is disabled |
| Cold rescue count | not applicable; exact rescue is not implemented |

| Backend / stage | Decode tok/s | Prefill tok/s | TTFT ms | Peak RSS MiB | H2D bytes/run | Weight / activation H2D | D2H bytes/run | Peak backend VRAM bytes | Alloc / free | Sync | Static-cache hit rate | Groups / members | Kernel ms/run |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `cuda-dense` reference | 12.1261 | 7.4290 | 1,206.070 | 483.23 | 4,999,104 | 4,893,696 / 105,408 | 90,864 | 25,216 | 1,404 / 1,404 | 468 | not applicable | 0 / 0 | 12.873 |
| `cuda-dense` reuse | 17.4560 | 9.2222 | 1,097.150 | 483.58 | 4,999,104 | 4,893,696 / 105,408 | 90,864 | 41,472 | 6 / 3 | 468 | not applicable | 0 / 0 | 17.731 |
| `cuda-dense` residency | **18.0041** | 9.2543 | 1,104.549 | 483.64 | 649,152 | 543,744 / 105,408 | 90,864 | 544,512 | 56 / 2 | 468 | 88.89% | 0 / 0 | 16.021 |
| `cuda-dense` grouped | 17.9018 | 9.2907 | 1,100.236 | 483.34 | 626,112 | 543,744 / 82,368 | 90,864 | 569,728 | 59 / 3 | 378 | 88.89% | 63 / 153 | 16.519 |
| `cuda-custom` reference | 12.2647 | 8.1162 | 1,009.112 | 483.79 | 5,107,968 | 4,981,824 / 126,144 | 111,600 | 25,216 | 2,052 / 2,052 | 630 | not applicable | 0 / 0 | 14.046 |
| `cuda-custom` reuse | 17.1425 | 9.1379 | 1,103.961 | 483.88 | 5,107,968 | 4,981,824 / 126,144 | 111,600 | 41,472 | 10 / 3 | 630 | not applicable | 0 / 0 | 19.146 |
| `cuda-custom` residency | **17.2723** | 9.1175 | 1,106.492 | 483.80 | 699,264 | 573,120 / 126,144 | 111,600 | 574,144 | 166 / 2 | 630 | 83.17% | 0 / 0 | 19.012 |
| `cuda-custom` grouped | 16.8348 | 9.0853 | 1,107.039 | 483.88 | 669,312 | 573,120 / 96,192 | 111,600 | 600,160 | 172 / 3 | 486 | 83.17% | 117 / 261 | 22.683 |

The fastest measured FP32 configuration for each CUDA identity is reusable allocation plus static residency with scalar projection calls. Relative to its reference, this improves decode by 48.47% for `cuda-dense` and 40.83% for `cuda-custom`. Residency reduces weight H2D by 88.89% and 88.50%, respectively. Grouping then reduces activation H2D by 21.86% for `cuda-dense` and 23.74% for `cuda-custom`, and reduces synchronization by 19.23% and 22.86%, but decode is 0.57% and 2.53% below scalar residency. Grouping is therefore measured and correct but not selected as a default.

Grouped H2D is 0.000104352 GB/generated token for `cuda-dense` and 0.000111552 GB/generated token for `cuda-custom`, using decimal GB. These are host-to-device counters, not NVMe counters. The static-cache rates describe immutable synthetic tensors, not the future three-tier expert-cache hit rate.

| Fully enabled BF16 backend | Decode tok/s | Prefill tok/s | TTFT ms | H2D bytes/run | Peak backend VRAM bytes | Max abs. error | Max rel. error |
|---|---:|---:|---:|---:|---:|---:|---:|
| `cuda-dense` | 17.6861 | 9.4550 | 1,092.242 | 313,056 | 285,376 | 0.00402409 | 17.5009 |
| `cuda-custom` | 17.0032 | 9.2486 | 1,102.482 | 356,256 | 315,808 | 0.00402409 | 17.5009 |

BF16 halves dense operand traffic and resident bytes but does not beat the corresponding FP32 scalar-residency result. Its large relative error is driven by near-zero reference values; exact tokens and maximum absolute error are the current bounded checks. No general quality benchmark has been run, so BF16 remains explicit rather than default.

Raw records are stored under `results/m2-cuda-dense/` and `results/m2-cuda-custom/`. The first timed `cuda-custom` attempt was interrupted by the shell timeout after writing only partial stage files; the complete command was rerun from the reference stage with a sufficient timeout, and only the final overwritten records plus `summary.json` are retained.

## B-0004 — Milestone 3 CUDA FFN block boundary ablation

| Field | Value |
|---|---|
| Evidence | measured |
| Date | 2026-08-09 |
| Measurement commit | `0f6bbdd` |
| Hardware | AMD Ryzen 7 9800X3D host; NVIDIA GeForce RTX 5080, 16,303 MiB |
| Environment | WSL2 Ubuntu 24.04.4, Linux 6.18.33.2, CUDA Toolkit 13.3.1, native `sm_120` |
| Model/checkpoint | regenerated deterministic 4-layer, 24-expert, 178-tensor synthetic K3X; SHA-256 `59c1f83f571fb59dcdad27ef80da8d42b03176dfb5fa63ae5195717c141775ed` |
| Mode | `cuda-custom`, reused allocations, 8 MiB exact static residency; operation/block crossed with scalar/grouped scheduling |
| Context length | 4 prompt tokens, IDs `[1, 7, 3, 9]` |
| Generated tokens | 6, IDs `[43, 32, 28, 49, 9, 28]` in every row |
| Warmup / samples | 3 / 20 process runs per row |
| NVMe GB/token | not measured; async storage is not implemented |
| Logical K3X reads | 110,936 bytes/generated token in every row |
| Static-cache hit rate | 83.17% in every row; this is synthetic immutable-weight residency, not the future three-tier cache |
| Average Top-K | 2, fixed synthetic router setting |
| Speculative acceptance / unique experts per verification block | not applicable; speculation is disabled |
| Cold rescue count | not applicable; exact rescue is not implemented |
| GPU utilization / memory bandwidth | not measured |
| I/O stall time | not measured; no asynchronous storage path exists |

| Precision / boundary / scheduling | Decode tok/s | Prefill tok/s | TTFT ms | System RSS bytes | H2D bytes/run | Weight / activation H2D | H2D GB/token | D2H bytes/run | Peak backend VRAM bytes | Sync | FFN blocks / experts | Kernel ms/run | Max abs. error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FP32 operation scalar | 16.3576 | 8.5014 | 1,158.914 | 506,961,920 | 699,264 | 573,120 / 126,144 | 0.000116544 | 111,600 | 574,144 | 630 | 0 / 0 | 20.348 | 1.751e-7 |
| FP32 operation grouped | 16.4210 | 8.6722 | 1,150.607 | 507,334,656 | 669,312 | 573,120 / 96,192 | 0.000111552 | 111,600 | 600,160 | 486 | 0 / 0 | 21.981 | 1.751e-7 |
| FP32 FFN block scalar | **17.0713** | **8.9305** | 1,148.841 | 507,174,912 | 665,856 | 573,120 / 92,736 | 0.000110976 | 83,952 | 600,416 | 423 | 63 / 54 | 22.244 | 1.790e-7 |
| FP32 FFN block grouped | 17.0270 | 8.8581 | **1,144.731** | 507,342,848 | 652,032 | 573,120 / 78,912 | 0.000108672 | 83,952 | 617,568 | 369 | 63 / 54 | 24.186 | 1.790e-7 |
| BF16 operation scalar | 16.3874 | 8.8158 | 1,144.064 | 482,033,664 | 374,688 | 301,248 / 73,440 | 0.000062448 | 111,600 | 302,080 | 630 | 0 / 0 | 18.353 | 0.00402409 |
| BF16 operation grouped | 16.1931 | 8.8731 | 1,144.529 | 481,652,736 | 356,256 | 301,248 / 55,008 | 0.000059376 | 111,600 | 315,808 | 486 | 0 / 0 | 22.794 | 0.00402409 |
| BF16 FFN block scalar | **16.9847** | **9.1219** | 1,132.774 | 482,033,664 | 349,344 | 301,248 / 48,096 | 0.000058224 | 83,952 | 315,808 | 423 | 63 / 54 | 21.284 | 0.00402409 |
| BF16 FFN block grouped | 16.9632 | 9.0652 | **1,129.242** | 481,914,880 | 342,432 | 301,248 / 41,184 | 0.000057072 | 83,952 | 324,768 | 369 | 63 / 54 | 24.209 | 0.00402409 |

Against the matching FP32 operation-scalar row, FFN-block-scalar improves decode by 4.36%, reduces activation H2D by 26.48%, D2H by 24.77%, and synchronization by 32.86%. Against operation-grouped, FFN-block-grouped improves decode by 3.69%, reduces activation H2D by 17.96%, D2H by 24.77%, and synchronization by 24.07%.

BF16 block-scalar improves 3.64% over BF16 operation-scalar, while BF16 block-grouped improves 4.76% over its grouped match. BF16 preserves exact tokens but does not beat FP32 block-scalar and retains the previously measured 0.00402409 maximum absolute error. The maximum relative error remains 17.5009 because reference values near zero dominate it.

The traffic reduction produces only a modest end-to-end gain, while CUDA-event kernel time rises in the block rows. Most elapsed time therefore remains in CPU KDA/MLA, routing, residual work, non-FFN boundaries, process startup, and synchronization outside the fused dependency chain. `operation` remains the default reference; FP32 FFN-block-scalar is an experimental synthetic recommendation only.

Raw JSON/CSV records are stored under `results/b0004-ffn-blocks-fp32/` and `results/b0004-ffn-blocks-bf16/`.

Post-measurement validation note: final read-only review found that the new FFN-block preflight accepted non-native MXFP4 group metadata even though the CUDA kernel is fixed to E8M0/32. Commit `3df8d3f` adds a RED/GREEN regression for group sizes 16 and 64 and rejects all non-32 gate/up/down views before side effects. The valid group-32 execution path measured at `0f6bbdd` is unchanged. Post-fix CPU CTest 5/5 and pytest 70/26, CUDA CTest 11/11 and pytest 95/1, `test_cuda_ffn` Compute Sanitizer 0 errors, and one-sample FP32/BF16 four-case smoke all passed. The smoke is validation evidence, not a replacement performance measurement.

## B-0005 — Milestone 4 exact L1-to-L0 transfer ablation

| Field | Value |
|---|---|
| Evidence | measured |
| Date | 2026-08-09 |
| Measurement commit | `99cf1e4` |
| Hardware | AMD Ryzen 7 9800X3D host; NVIDIA GeForce RTX 5080, 16,303 MiB |
| Environment | WSL2 Ubuntu 24.04.4, Linux 6.18.33.2, CUDA Toolkit 13.3.1, native `sm_120` |
| Model/checkpoint | regenerated seeded 4-layer, 24-expert, 178-tensor synthetic K3X; artifact SHA-256 `e245c52759dffcfaccfe182bbba56fa069288d99f0d70a1cd779169bb51e6993`; converter maximum source read 257 bytes |
| Mode | `cuda-custom`, `ffn-block`, reused allocations, transient exact weights; synchronous/prefetch crossed with scalar/grouped scheduling |
| Context length | 4 prompt tokens, IDs `[1, 7, 3, 9]` |
| Generated tokens | 6, IDs `[43, 32, 28, 49, 9, 28]` in every row |
| Routing | exact 24-entry prefill trace identical in every row; fixed synthetic Top-2, average Top-K 2 |
| Warmup / samples | 3 / 20 separate process runs per row |
| Pinned capacity | 0 for synchronous; 1,048,576 bytes for prefetch |
| NVMe GB/token | not measured; logical K3X reads are 110,936 bytes/generated token but no OS/block-device counter exists |
| GPU utilization / memory bandwidth | not measured |
| Storage I/O stall time | not measured; file reads remain synchronous and are outside the new L1-to-L0 transfer timer |
| Expert-cache hit rate / cold rescue | not applicable; weights are transient and no L1/L2 cache or rescue exists |
| Speculative acceptance / unique experts per verification block | not applicable; speculation is disabled |

| Precision / transfer / scheduling | Decode tok/s | Prefill tok/s | TTFT ms | System RSS bytes | H2D bytes/run | H2D GB/token | D2H bytes/run | Peak backend VRAM bytes | Sync | Prefetch calls / ready / late | Transfer wait | Stage / transfer / exposed stall ms | Kernel ms/run | Max abs. error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FP32 synchronous scalar | **16.9701** | 8.7878 | 1,147.210 | 507,334,656 | 5,074,560 | 0.000845760 | 83,952 | 43,680 | 423 | 0 / 0 / 0 | 0 | 0 / 0 / 0 | 20.056 | 1.790e-7 |
| FP32 prefetch scalar | 16.7947 | 8.7133 | **1,144.938** | 507,432,960 | 5,074,560 | 0.000845760 | 83,952 | 1,091,712 | 423 | 27 / 27 / 0 | 27 | 0.009 / 0.155 / 0.198 | 20.577 | 1.790e-7 |
| FP32 synchronous grouped | 16.7055 | 8.8381 | **1,145.039** | 507,125,760 | 5,060,736 | 0.000843456 | 83,952 | 44,448 | 369 | 0 / 0 / 0 | 0 | 0 / 0 / 0 | 22.033 | 1.790e-7 |
| FP32 prefetch grouped | **16.7914** | 8.8773 | 1,153.483 | 507,170,816 | 5,060,736 | 0.000843456 | 83,952 | 1,092,480 | 369 | 27 / 27 / 0 | 27 | 0.009 / 0.211 / 0.312 | 21.357 | 1.790e-7 |
| BF16 synchronous scalar | **16.6366** | 8.8750 | **1,145.690** | 481,959,936 | 2,583,072 | 0.000430512 | 83,952 | 22,752 | 423 | 0 / 0 / 0 | 0 | 0 / 0 / 0 | 20.669 | 0.00402409 |
| BF16 prefetch scalar | 16.5735 | 8.9245 | 1,147.870 | 481,980,416 | 2,583,072 | 0.000430512 | 83,952 | 1,070,784 | 423 | 27 / 27 / 0 | 27 | 0.009 / 0.163 / 0.230 | 18.516 | 0.00402409 |
| BF16 synchronous grouped | 16.5529 | 9.0488 | 1,141.830 | 481,890,304 | 2,576,160 | 0.000429360 | 83,952 | 23,520 | 369 | 0 / 0 / 0 | 0 | 0 / 0 / 0 | 21.075 | 0.00402409 |
| BF16 prefetch grouped | **16.7021** | **9.0539** | **1,136.192** | 481,947,648 | 2,576,160 | 0.000429360 | 83,952 | 1,071,552 | 369 | 27 / 27 / 0 | 27 | 0.009 / 0.198 / 0.288 | 19.864 | 0.00402409 |

Matched prefetch decode changes are -1.03% for FP32 scalar, +0.51% for FP32 grouped, -0.38% for BF16 scalar, and +0.90% for BF16 grouped. Prefetch changes neither total H2D bytes nor synchronization count because it changes transfer ordering rather than payload volume. All 27 prepared blocks in each prefetch row were ready when consumed, yet event-based exposed stall remains 0.198--0.312 ms per run and the fixed slabs add 1,048,032 bytes to peak backend VRAM plus 1 MiB of pinned host memory.

Every row preserves the exact generated tokens and the same natural routing trace. FP32 retains the existing numerical tolerance. BF16 retains maximum absolute error 0.00402409 and is not promoted to a quality default. The small mixed throughput deltas do not justify enabling prefetch by default.

The measured boundary still starts after synchronous K3X extent reads and pageable host allocation. The next bottleneck to isolate is therefore repeated file-to-host materialization and the absent persistent L1 expert cache, followed by native-Linux L2 I/O and deadline scheduling. CPU KDA/MLA, routing, residual/state work, and non-FFN orchestration also continue to dominate this tiny graph.

Raw JSON/CSV records are stored under `results/b0005-async-transfer-fp32/` and `results/b0005-async-transfer-bf16/`; the compact cross-checked manifest is `results/b0005-async-transfer.json`.

Post-measurement validation note: final read-only review found that the prepared token stored but did not enforce its exact use sequence, and that the ablation runner allowed matched-pair H2D or synchronization counters to differ. Commit `190459b` adds sequence identity to the token, rejects mismatches before allocation/H2D/event side effects while preserving the valid pending request, and requires exact matched total/weight/activation H2D plus synchronization. The valid B-0005 execution order is unchanged. Post-fix CPU CTest 5/5 and pytest 98 passed/27 skipped, CUDA CTest 14/14 and pytest 124 passed/1 skipped, both affected CUDA memcheck targets with zero errors, and FP32/BF16 one-sample four-case smokes passed. The smokes are validation evidence, not replacement performance measurements.

## B-0006 — Milestone 5 bounded persistent L1 expert cache

| Field | Value |
|---|---|
| Evidence | measured |
| Date | 2026-08-09 |
| Measurement commit | `2a0cb27` |
| Hardware | AMD Ryzen 7 9800X3D host; NVIDIA GeForce RTX 5080, 16,303 MiB |
| Environment | WSL2 Ubuntu 24.04.4, Linux 6.18.33.2, CUDA Toolkit 13.3.1, native `sm_120` |
| Model/checkpoint | regenerated seeded 4-layer, 24-expert, 178-tensor synthetic K3X; artifact SHA-256 `077e10a3ba478e83ac8dfd2509ea51a6ea2bfdfe670b60fcadc7f74b97ff810c`; converter maximum source read 257 bytes |
| Mode | `cuda-custom`, `ffn-block`, reused allocation, transient GPU weights, scalar scheduling; disabled/static L1 crossed with synchronous/prefetch L1-to-L0 transfer |
| Context length | 4 prompt tokens, IDs `[1, 7, 3, 9]` |
| Generated tokens | 6, IDs `[43, 32, 28, 49, 9, 28]` in every row |
| Routing | exact 24-entry prefill trace identical in every row; fixed synthetic Top-2, average Top-K 2 |
| Warmup / samples | 3 / 20 separate process runs per row |
| L1 capacity | 0 for disabled; 65,536 bytes for static |
| Pinned capacity | 0 for synchronous; 1,048,576 bytes for prefetch |
| NVMe GB/token | not measured; Reader requested/completed bytes are logical file reads, not OS or block-device counters |
| GPU utilization / memory bandwidth | not measured |
| Speculative acceptance / unique experts per verification block | not applicable; speculation is disabled |
| Cold rescue | not implemented |

| Precision / L1 / transfer | Decode tok/s | Prefill tok/s | TTFT ms | RSS bytes | Logical bytes/token | Reader calls / bytes | Hits / misses / bypass | L1 bytes | H2D / D2H bytes | Peak backend VRAM | Kernel ms | Prefetch calls / stall ms | Max abs. error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FP32 disabled synchronous | 16.5587 | 8.8183 | 1,151.464 | 507,240,448 | 110,936 | 428 / 665,616 | 0 / 0 / 0 | 0 | 5,074,560 / 83,952 | 43,680 | 22.846 | 0 / 0 | 1.790e-7 |
| FP32 static synchronous | **47.6845** | 11.1885 | 1,047.856 | 507,400,192 | 101,144 | 212 / 606,864 | 36 / 18 / 0 | 29,376 | 5,074,560 / 83,952 | 43,680 | 26.199 | 0 / 0 | 1.790e-7 |
| FP32 disabled prefetch | 16.7636 | 8.8086 | 1,151.007 | 507,453,440 | 110,936 | 428 / 665,616 | 0 / 0 / 0 | 0 | 5,074,560 / 83,952 | 1,091,712 | 22.509 | 27 / 0.303 | 1.790e-7 |
| FP32 static prefetch | **50.6235** | 11.1989 | 1,060.459 | 507,269,120 | 101,144 | 212 / 606,864 | 36 / 18 / 0 | 29,376 | 5,074,560 / 83,952 | 1,091,712 | 25.261 | 27 / 0.269 | 1.790e-7 |
| BF16 disabled synchronous | 16.4052 | 8.9519 | 1,142.860 | 481,837,056 | 110,936 | 428 / 665,616 | 0 / 0 / 0 | 0 | 2,583,072 / 83,952 | 22,752 | 24.114 | 0 / 0 | 0.00402409 |
| BF16 static synchronous | **47.7956** | 11.4531 | 1,037.164 | 481,951,744 | 101,144 | 212 / 606,864 | 36 / 18 / 0 | 29,376 | 2,583,072 / 83,952 | 22,752 | 25.179 | 0 / 0 | 0.00402409 |
| BF16 disabled prefetch | 16.5073 | 9.0693 | 1,142.449 | 482,025,472 | 110,936 | 428 / 665,616 | 0 / 0 / 0 | 0 | 2,583,072 / 83,952 | 1,070,784 | 21.963 | 27 / 0.250 | 0.00402409 |
| BF16 static prefetch | **47.6198** | 11.5895 | 1,049.233 | 481,955,840 | 101,144 | 212 / 606,864 | 36 / 18 / 0 | 29,376 | 2,583,072 / 83,952 | 1,070,784 | 25.183 | 27 / 0.253 | 0.00402409 |

Static admission reduces matched logical Reader calls by 216 and completed bytes by 58,752 per run. It preserves exact tokens, routing, total/weight/activation H2D, D2H, FFN calls/experts, and synchronization counts. The cache admits 18 complete experts, then serves 36 hits without a Reader call; no selected expert exceeds the 65,536-byte synthetic capacity.

The derived matched decode improvements are +188.0% for FP32 synchronous, +202.0% for FP32 prefetch, +191.3% for BF16 synchronous, and +188.5% for BF16 prefetch. These unusually large synthetic gains isolate repeated WSL2 file-to-pageable-host materialization in a tiny graph with only 24 experts and repeated routes. They are not a full-model speed projection and do not justify enabling first-observation no-eviction admission by default.

Raw JSON/CSV records are stored under `results/b0006-l1-cache-fp32/` and `results/b0006-l1-cache-bf16/`; the compact manifest is `results/b0006-l1-cache.json`. Review added session-lifetime reuse and stricter native payload validation before this replacement measurement. Full post-review verification passed CPU CTest 6/6 and pytest 117 passed/34 skipped, CUDA CTest 15/15 and pytest 150 passed/1 skipped, and all ten CUDA Compute Sanitizer targets with zero errors.

The next bottleneck boundary is representative expert sizing and real L2 behavior. A native-Linux full-dimension bounded slice must measure buffered reads, `io_uring`, and `O_DIRECT`, physical block traffic, and deadlines before selecting an L2 path. Policy work such as Least-Stale also needs representative reuse traces and eviction pressure rather than the current all-fitting synthetic cache.

## B-0007 — Milestone 6 independent L2 reader

| Field | Value |
|---|---|
| Evidence | measured WSL2 capability benchmark; non-authoritative for the target NVMe |
| Date | 2026-08-09 |
| Measurement commit | `5049f26` |
| Hardware | AMD Ryzen 7 9800X3D host; storage device performance intentionally not attributed |
| Environment | WSL2 Ubuntu 24.04.4, Linux 6.18.33.2; artifact copied to WSL ext4 `/tmp`; liburing 2.5 |
| Model/checkpoint | seeded 4-layer, 24-expert, 178-tensor synthetic K3X; artifact SHA-256 `039d61ee9c2e13e27c9a2514bb476f8b122b8b37be0b7f85baf26c1a6611a2e9` |
| Mode | CPU FP32, incremental, L1 disabled; `pread|io_uring` crossed with `buffered|direct`; queue depth 8 |
| Context length | 4 prompt tokens, IDs `[1, 7, 3, 9]` |
| Generated tokens | 6, IDs `[43, 32, 28, 49, 9, 28]` in every row |
| Routing | exact 24-entry prefill trace identical in every row; fixed synthetic Top-2, average Top-K 2 |
| Warmup / samples | 3 / 20 separate process runs per row |
| NVMe GB/token | not measured; process and Reader counters are not attributed to the P44 Pro |
| GPU utilization / bandwidth / VRAM / H2D | not applicable to this CPU storage-boundary isolation |
| Speculative acceptance / unique experts per verification block | not applicable; speculation is disabled |
| Cold rescue | not implemented |

| Engine / cache | Decode tok/s | Prefill tok/s | TTFT ms | RSS bytes | Logical calls / batches | Logical bytes | Submitted bytes | Reader storage ms | Process rchar / read bytes | Direct mem / offset align |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `pread` / buffered | **5,870.8082** | **2,824.4419** | **12.860** | 5,210,112 | 428 / 158 | 665,616 | 665,616 | **0.223** | 665,717 / 0 | 0 / 0 |
| `io_uring` / buffered | 5,616.1034 | 2,681.1601 | 12.866 | 5,353,472 | 428 / 158 | 665,616 | 665,616 | 0.331 | 101 / 0 | 0 / 0 |
| `pread` / direct | 163.3491 | 103.1181 | 50.220 | 5,394,432 | 428 / 158 | 665,616 | 756,736 | 67.173 | 756,837 / 756,736 | 4 / 512 |
| `io_uring` / direct | 428.8471 | 153.7273 | 37.800 | 5,210,112 | 428 / 158 | 665,616 | 756,736 | 35.447 | 101 / 756,736 | 4 / 512 |

All four cases report zero Reader failures and short reads and preserve exact token and routing parity. Direct mode adds 91,120 submitted bytes, a 13.69% alignment amplification on this tiny artifact. Linux `rchar` reflects synchronous read-family accounting and therefore does not mirror io_uring traffic; `read_bytes` is the process block-I/O delta and records zero for warm buffered rows and 756,736 for direct rows. Neither counter establishes physical P44 Pro traffic in WSL2.

The tiny warm buffered case favors `pread`; direct mode is dominated by hundreds of small aligned reads, although batching makes io_uring materially less slow than direct `pread` in this environment. These figures neither select a native-Linux default nor project full-model throughput. `pread + buffered` remains the default until a full-dimension bounded slice is measured on native Linux with the P44 Pro. Raw JSON/CSV and the cross-checked manifest are under `results/b0007-l2-reader-wsl/`.

Final read-only review found an io_uring error-path lifetime gap but no successful-path result defect. The fix closes and invalidates the ring before batch buffers are destroyed on any submit/wait early return and retries `EINTR`; a real-ring guard regression and ASan/UBSan liburing CTest 9/9 pass. The unimplemented multi-megabyte fixture is now explicitly proposed rather than claimed. Because the successful execution order is unchanged, B-0007 raw measurements remain valid; a post-fix four-case smoke preserved exact tokens, routing, bytes, and counters.

Post-review verification passed CPU CTest 8/8 and pytest 136 passed/40 skipped, liburing/direct CTest 9/9 and pytest 137 passed/39 skipped, CUDA CTest 17/17 and pytest 169 passed/7 skipped. All ten CUDA Compute Sanitizer targets from the measurement commit reported zero errors; the review fix does not compile into the non-liburing CUDA build.

## B-0008 — Milestone 7 full-dimension bounded expert storage

| Field | Value |
|---|---|
| Evidence | measured WSL2 expert-load benchmark; non-authoritative for the target NVMe and not token throughput |
| Date | 2026-08-09 |
| Measurement commit | `9198ed2` |
| Hardware | AMD Ryzen 7 9800X3D host; GPU not used by this storage-boundary benchmark |
| Environment | WSL2 Ubuntu 24.04.4, Linux 6.18.33.2; artifact on WSL ext4 `/tmp`; liburing 2.5 |
| Model/checkpoint | one non-executable released-dimension routed expert; source SHA-256 `1e310ebdcdd7a8a7ec124fac1e59ca44bbdf5da1ef279502c81a8b70f06379f5`; K3X SHA-256 `b14610fd2b405dd97c09004fb29157f5b318522591546337bce89e7e8a6a2b65` |
| Payload | gate/up 3,072 x 3,584 and down 3,584 x 3,072; 16,515,072 packed E2M1 bytes plus 1,032,192 E8M0 scale bytes |
| Mode | metadata-only Reader open followed by exact six-extent expert loads; `pread|io_uring` crossed with `buffered|direct`; queue depth 8 |
| Warmup / samples | 3 / 20 loads per row in one process |
| Ordered payload digest | `e5fb7939474a57ab9263a791999d76ba078bd767cc3f155f3522b1bec576c7e4` in every row |
| Tokens / context / Top-K / quality | not applicable; the artifact is explicitly non-executable |
| NVMe GB/token | not applicable and not measured; no token inference occurs and WSL2 storage is not attributed to the P44 Pro |
| GPU utilization / bandwidth / VRAM / H2D | not applicable; GPU is not used |
| Speculation / cold rescue | not applicable; neither path executes |

| Engine / cache | Median ms | p05 / p95 ms | Expert loads/s | Reader storage ms/load | Calls / batches / completions | Logical / submitted bytes per load | Process rchar / read bytes total | Direct mem / offset align |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `pread` / buffered | **50.685** | 48.714 / 52.969 | **19.7434** | 4.661 | 120 / 20 / 120 | 17,547,264 / 17,547,264 | 350,945,381 / 0 | 0 / 0 |
| `io_uring` / buffered | 51.592 | 48.491 / 54.708 | 19.3532 | **4.579** | 120 / 20 / 120 | 17,547,264 / 17,547,264 | 98 / 0 | 0 / 0 |
| `pread` / direct | 60.402 | 57.490 / 65.184 | 16.3861 | 14.832 | 120 / 20 / 120 | 17,547,264 / 17,547,264 | 350,945,385 / 350,945,280 | 4 / 512 |
| `io_uring` / direct | **56.426** | 54.752 / 58.736 | **17.6500** | **11.633** | 120 / 20 / 120 | 17,547,264 / 17,547,264 | 102 / 350,945,280 | 4 / 512 |

Every row reports zero short reads and failures and preserves the same six-extent digest. Each released-dimension data and scale extent is already divisible by the filesystem's 512-byte direct-I/O offset alignment, so direct mode submits exactly the logical 17,547,264 bytes per load. This replaces B-0007's tiny-extent 13.69% amplification observation for this bounded expert shape only; arbitrary trunk or future quantized extents may still amplify.

Wall latency includes allocation of six output vectors and ordered SHA-256 over 17.5 MB. `Reader storage ms/load` is the Reader's timed storage-call portion and excludes hashing. Buffered rows are warm after three explicit loads, while direct rows bypass the page cache. No privileged cache drop was used, so this is not a controlled cold-cache comparison.

Buffered pread has the best wall median in this WSL2 run, while buffered io_uring has a slightly lower Reader-only time. Direct io_uring is faster than direct pread but remains slower than buffered modes. WSL2, hashing/allocation overhead, one expert, and one synchronous batch prevent these results from selecting a native-Linux P44 Pro default. `pread + buffered` remains unchanged.

Raw JSON/CSV and the cross-checked manifest are under `results/b0008-bounded-slice-wsl/`. The compact/raw cross-check succeeded for all four rows. Verification at the measurement code includes CPU CTest 8/8 and pytest 152 passed/40 skipped, liburing/direct CTest 9/9 and pytest 157 passed/35 skipped, CUDA CTest 17/17 and pytest 185 passed/7 skipped, plus ASan/UBSan liburing CTest 9/9 and four storage-path targeted pytest passes.

Final review subsequently strengthened source-manifest and resume-ledger integrity without changing payload order, Reader code, or benchmark code, so B-0008 was not relabeled or remeasured. Post-fix correctness passed CPU CTest 8/8 and pytest 161/40, liburing/direct CTest 9/9 and pytest 162/39, and CUDA CTest 17/17 and pytest 194/7.

## B-0009 — Milestone 8 exact current-layer deadline loading

| Field | Value |
|---|---|
| Evidence | measured WSL2 warm synthetic ablation; non-authoritative for native storage or a full model |
| Date | 2026-08-09 |
| Measurement code commit | `68b3e54` |
| Hardware | AMD Ryzen 7 9800X3D host; CPU execution; RTX 5080 unused |
| Environment | WSL2 Ubuntu 24.04.4, Linux 6.18.33.2; artifact on WSL ext4 `/tmp`; liburing 2.5 |
| Model/checkpoint | deterministic executable `synthetic-milestone-one`; K3X SHA-256 `392b9237274e5580b665cf95afbda9a09e8d01ba7484bed00cf83a4ae99eb4fa`; no full Kimi K3 weights |
| Mode | exact incremental CPU generation; static 65,536-byte L1; `blocking|deadline × pread|io_uring × buffered|direct`; queue depth 8 |
| Context / generated tokens | prompt `[1, 7, 3, 9]`; 6 generated tokens `[43, 32, 28, 49, 9, 28]` |
| Warmup / samples | 3 / 20 separate process runs per row |
| Quality result | exact tokens and 24-entry routing trace in every row; CPU diagnostic maximum absolute and relative error 0 |
| Average Top-K | 2, fixed synthetic natural routing |
| L1 cache | 36 hits, 18 misses, zero bypasses, 29,376 resident bytes in every row; 66.67% hit rate over hit+miss events |
| Logical Reader traffic | 606,864 bytes/run and 212 successful completions in every row; zero failure and short-read counters |
| NVMe GB/token | not measured; logical Reader and aligned submitted bytes are not physical NVMe attribution |
| H2D / VRAM / GPU utilization / memory bandwidth / kernel time | 0 or not applicable; CPU backend used |
| Speculative acceptance / unique verification experts / cold rescue | not applicable; these features are not implemented |
| Enabled optimizations | static exact L1 admission; selected Reader mode; deadline worker only in deadline rows; no adaptive Top-K, proxy, pruning, or speculation |

| Schedule / Reader | Decode tok/s | Prefill tok/s | TTFT ms | Peak RSS MiB | Submitted bytes | Ready / late | Worker / exposed-wait ms |
|---|---:|---:|---:|---:|---:|---:|---:|
| blocking / pread buffered | **6,508.251** | **4,342.914** | 12.936 | 5.055 | 606,864 | 0 / 0 | 0 / 0 |
| deadline / pread buffered | 5,112.555 | 2,882.631 | **12.881** | 5.266 | 606,864 | 48 / 6 | 0.179 / 0.234 |
| blocking / io_uring buffered | **6,234.853** | **3,884.325** | **12.914** | 5.133 | 606,864 | 0 / 0 | 0 / 0 |
| deadline / io_uring buffered | 4,971.308 | 2,698.926 | 12.989 | 5.437 | 606,864 | 47 / 7 | 0.219 / 0.253 |
| blocking / pread direct | **808.171** | **146.302** | 38.791 | 5.195 | 646,144 | 0 / 0 | 0 / 0 |
| deadline / pread direct | 768.502 | 140.055 | **38.069** | 5.137 | 646,144 | 36 / 18 | 17.959 / 16.334 |
| blocking / io_uring direct | **1,966.491** | **206.062** | **31.224** | 5.051 | 646,144 | 0 / 0 | 0 / 0 |
| deadline / io_uring direct | 1,766.937 | 187.898 | 32.361 | 5.336 | 646,144 | 36 / 18 | 6.697 / 4.888 |

Each deadline row records 54 submissions, 54 completions, 36 inline-resident hits, and 18 estimated deadline misses. Exact outputs and logical traffic show that scheduling does not change routing or fetch semantics. It does not improve throughput in this graph: matched decode changes are -21.45%, -20.27%, -4.91%, and -10.15% in table order. The worker remains opt-in; these numbers do not reject later future-layer prefetch on representative native-Linux workloads.

Raw JSON/CSV and the programmatically cross-checked summary are under `results/b0009-deadline-loader-wsl/`. The first recorded B-0009 was replaced after final review moved latency-estimate capture before worker submission; the table is only the post-fix measurement.

Post-review verification passed CPU CTest 9/9 and pytest 175/41, liburing/direct CTest 10/10 and pytest 177/39, CUDA CTest 18/18 and pytest 208/8, plus ASan/UBSan liburing CTest 10/10 and targeted pytest 69/33. All ten CUDA Compute Sanitizer targets reported zero errors. TSan built but could not execute under WSL2 because its runtime terminated with `unexpected memory mapping`; no TSan result is claimed.

## B-0010 — Milestone 9 exact expert cache policies

| Field | Value |
|---|---|
| Evidence | measured WSL2 warm synthetic ablation; non-authoritative for native storage, full-model locality, or policy defaults |
| Date | 2026-08-09 |
| Measurement code commit | `fd05d95` |
| Result commit | `ff65b5b` |
| Hardware | AMD Ryzen 7 9800X3D host; CPU execution; RTX 5080 unused |
| Environment | WSL2 Ubuntu 24.04.4, Linux 6.18.33.2; artifact on WSL ext4 `/tmp`; buffered `pread` |
| Model/checkpoint | deterministic executable `synthetic-milestone-one`; K3X SHA-256 `1d4a197a299493acf6eb39d8374a4f817ee58f6f570ad5f21f29fe9fe298d2de`; no full Kimi K3 weights |
| Mode | exact incremental CPU generation; blocking L2 schedule; `disabled` plus `static|lru|lfu|least-stale` at 3,264, 13,056, and 26,112 bytes |
| Context / generated tokens | prompt `[1, 7, 3, 9]`; 6 generated tokens `[43, 32, 28, 49, 9, 28]` |
| Warmup / samples | 3 / 20 separate process runs per row |
| Quality result | exact tokens, 24-entry natural routing trace, and matched CPU numerical error in every row |
| Average Top-K | 2, fixed synthetic natural routing |
| Peak system RSS | 5,255,168 to 5,427,200 bytes across rows; per-row values are in raw JSON/CSV |
| NVMe GB/token | not measured; logical Reader bytes are not physical NVMe attribution |
| H2D / VRAM / GPU utilization / memory bandwidth / kernel time | 0 or not applicable; CPU backend used |
| Speculative acceptance / unique verification experts / cold rescue | not applicable; these features are not implemented |
| Enabled optimizations | selected exact L1 policy only; no deadline worker, adaptive Top-K, task prior, proxy, pruning, or speculation |

| Policy / capacity bytes | Decode tok/s | Prefill tok/s | TTFT ms | Hits / misses | Evictions / collisions | Logical Reader bytes |
|---|---:|---:|---:|---:|---:|---:|
| disabled / 0 | 5,789.235 | 4,131.860 | 12.979 | 0 / 0 | 0 / 0 | 665,616 |
| static / 3,264 | 5,820.041 | 4,093.596 | 12.887 | 6 / 48 | 0 / 0 | 655,824 |
| LRU / 3,264 | 5,752.148 | 4,015.077 | 13.006 | 0 / 54 | 52 / 9 | 665,616 |
| LFU / 3,264 | 5,800.326 | 4,047.418 | 12.930 | 0 / 54 | 52 / 9 | 665,616 |
| Least-Stale / 3,264 | 5,672.468 | 4,017.756 | 12.917 | 0 / 54 | 52 / 9 | 665,616 |
| static / 13,056 | 6,046.669 | 4,122.494 | 12.908 | 25 / 29 | 0 / 0 | 624,816 |
| LRU / 13,056 | 5,844.559 | 4,138.837 | 12.927 | 20 / 34 | 26 / 1 | 632,976 |
| LFU / 13,056 | 5,986.500 | 4,156.233 | 12.984 | 19 / 35 | 27 / 7 | 634,608 |
| Least-Stale / 13,056 | 6,024.423 | 4,229.175 | 12.895 | 23 / 31 | 23 / 0 | 628,080 |
| static / 26,112 | 6,269.919 | 4,145.694 | 12.970 | 35 / 19 | 0 / 0 | 608,496 |
| LRU / 26,112 | 6,251.985 | 4,190.033 | 12.908 | 35 / 19 | 3 / 0 | 608,496 |
| LFU / 26,112 | 6,439.747 | 4,195.814 | 12.910 | 36 / 18 | 2 / 0 | 606,864 |
| Least-Stale / 26,112 | 6,153.479 | 4,121.591 | 12.992 | 35 / 19 | 3 / 0 | 608,496 |

The 8-expert point distinguishes the policies: Least-Stale eliminates the one LRU and seven LFU collision misses, reduces misses by 3 versus LRU and 4 versus LFU, and reduces logical Reader traffic by 4,896 and 6,528 bytes respectively. The 16-expert point instead gives LFU the best hit and traffic result. Decode differences are small relative to process-level noise on this tiny warm graph, so no timing row selects a default.

A collision is a same-token-forward re-request, including a prior-token future-layer entry evicted by an earlier current-token layer and requested when execution reaches its layer. The deterministic trace regression validates this path independently of B-0010.

Raw JSON/CSV and the programmatically cross-checked summary are under `results/b0010-expert-cache-policies-wsl/`. The final summary SHA-256 is `f2ff111f08fd6e9bc2cee2df426aeef337dd08150663d4e6a71922ebf60c5b8b`.

Post-review verification passed CPU CTest 9/9 and pytest 194/41, liburing/direct CTest 10/10 and pytest 200/35, CUDA CTest 18/18 and pytest 227/8, plus ASan/UBSan liburing CTest 10/10 and targeted pytest 80/33. All ten CUDA Compute Sanitizer targets reported zero errors before the host-only session-serialization fix; their binaries and CUDA sources were unchanged by that fix. TSan rebuilt but its WSL2 runtime again terminated with `unexpected memory mapping`; no TSan execution result is claimed.

## B-0011 — Milestone 10 task and session profiles

| Field | Value |
|---|---|
| Evidence | measured WSL2 warm synthetic ablation; non-authoritative for native storage, full-model locality, repository-duration sessions, or policy defaults |
| Date | 2026-08-09 |
| Measurement code commit | `5430074` |
| Result commit | `308b0db` |
| Hardware | AMD Ryzen 7 9800X3D host; CPU execution; RTX 5080 unused |
| Environment | WSL2 Ubuntu 24.04.4, Linux 6.18.33.2; artifact on WSL ext4 `/tmp`; buffered blocking `pread` |
| Model/checkpoint | deterministic executable `synthetic-milestone-one`; K3X SHA-256 `0dfe0fe7c64b364fc745fff5b6c9a1f06d1faf1dc140630a9240591540dd684d`; no full Kimi K3 weights |
| Mode | exact incremental CPU generation; blocking L2 schedule; LFU, Least-Stale, and opt-in `profiled` at 13,056 bytes |
| Context / generated tokens | target prompt `[1, 7, 3, 9]`; 6 generated tokens `[43, 32, 28, 49, 9, 28]`; alternate seed prompt `[2, 2, 2, 2]` |
| Warmup / samples | 3 / 20 separate process runs per row |
| Quality result | exact tokens, 24-entry natural routing trace, logits, and recurrent-state parity in every row; maximum numerical error 0 |
| Average Top-K | 2, fixed synthetic natural routing |
| Peak system RSS | 5,394,432 to 5,595,136 bytes across rows |
| NVMe GB/token | not measured; logical Reader bytes are not physical NVMe attribution |
| H2D / VRAM / GPU utilization / memory bandwidth / kernel time | 0 or not applicable; CPU backend used |
| Speculative acceptance / unique verification experts / cold rescue | not applicable; these features are not implemented |
| Enabled optimizations | selected exact L1 policy and, for profiled rows, bounded runtime profile observation/persistence; no adaptive Top-K, proxy, pruning, prediction, or speculation |

| Policy / prior | Decode tok/s | Prefill tok/s | TTFT ms | Hits / misses | Evictions / collisions | Logical Reader bytes |
|---|---:|---:|---:|---:|---:|---:|
| LFU / none | 5,757.937 | 4,164.324 | 14.466 | 19 / 35 | 27 / 7 | 634,608 |
| Least-Stale / none | 5,900.245 | 4,192.314 | 14.412 | 23 / 31 | 23 / 0 | 628,080 |
| profiled / cold | 5,924.595 | 4,169.559 | 14.720 | 21 / 33 | 25 / 6 | 631,344 |
| profiled / matching prompt | 5,006.701 | 3,667.612 | 14.884 | 23 / 31 | 23 / 4 | 628,080 |
| profiled / minimum-overlap alternate prompt | 4,868.597 | 3,263.086 | 15.755 | 22 / 32 | 24 / 7 | 629,712 |

| Profiled row | Metadata | Live observations | Prior weight | Load bytes / ns | Save bytes / ns |
|---|---:|---:|---:|---:|---:|
| cold | 0 | 54 | 0 | 0 / 0 | 1,439 / 86,017 |
| matching prompt | 0 | 54 | 0.0689655172 | 863 / 715,193 | 1,439 / 100,993 |
| minimum-overlap alternate prompt | 0 | 54 | 0.0689655172 | 638 / 740,811 | 1,645 / 99,641 |

The matching seed has a 12-expert hot bank. The selected minimum-overlap alternate seed has an 8-expert hot bank with 5 experts overlapping the target seed; it is not described as fully conflicting. The matching prior equals Least-Stale traffic at this capacity but adds enough bookkeeping and profile-I/O overhead to lose tiny-graph timing. The alternate prior is worse on both traffic and timing. No policy default changes.

The materialized full-generation output profiles are exactly 1,439, 1,439, and 1,645 bytes, matching the raw save-byte telemetry. Raw JSON/CSV, seed profiles, materialized profiles, and the programmatically cross-checked summary are under `results/b0011-task-session-profiles-wsl/`. The final summary SHA-256 is `029004e02a8484f281a332c09d49e7adc8eb1ed343ec692b030760288adbd94f`; the synthetic K3X artifact SHA-256 is `0dfe0fe7c64b364fc745fff5b6c9a1f06d1faf1dc140630a9240591540dd684d`. Matching and alternate seed profile SHA-256 values are `897e872dc5d3832f95cbc8b68feb67fd17f706a830a876fad20bdcdb4ad69162` and `1d7987f1462bbb7421ec56896c3588a91622850e06d6d451ade85f14153b2575`.

Post-review verification passed CPU CTest 10/10 and pytest 211/41, liburing/direct CTest 11/11 and pytest 213/39, CUDA CTest 19/19 and pytest 244/8, plus ASan/UBSan liburing CTest 11/11 and targeted pytest 82/33. All ten CUDA Compute Sanitizer targets reported zero errors. TSan was not rerun for this serialized host-only change; the prior WSL2 runtime limitation remains recorded in B-0010.

## B-0012 — Milestone 11 adaptive Top-K and exact cold rescue

| Field | Value |
|---|---|
| Evidence | measured WSL2 warm synthetic ablation; non-authoritative for native storage, full-model quality, coding quality, or defaults |
| Date | 2026-08-09 |
| Measurement code commit | `bf81beb` |
| Result commit | `d58fad7` |
| Hardware | AMD Ryzen 7 9800X3D host; CPU execution; RTX 5080 unused by the measured rows |
| Environment | WSL2 Ubuntu 24.04.4, Linux 6.18.33.2; executable artifact on WSL2 ext4 `/tmp`; buffered blocking `pread` |
| Model/checkpoint | deterministic 24-expert, natural Top-16 `synthetic-milestone-one`; K3X SHA-256 `89ef0b18f1adb55a305d111c6bb67eb8469e6dacc6eeac3363ad74a64ab0e861`; no full Kimi K3 weights |
| Mode | exact natural/fixed K16 references; lossy fixed K4/K8/K12; adaptive mass/boundary/failure variants; fixed-K escalation; fixed K4 plus bounded exact LRU rescue |
| Context / generated tokens | prompt `[1, 7, 3, 9]`; six generated tokens per row; natural tokens `[43, 32, 28, 49, 9, 28]` |
| Warmup / samples | 3 / 20 separate process runs per row |
| Peak system RSS | 5,488,640 to 5,943,296 bytes across rows |
| NVMe GB/token | not measured; logical Reader bytes are not physical NVMe attribution |
| H2D / VRAM / GPU utilization / memory bandwidth / kernel time | 0 or not applicable; CPU backend used |
| Expert-cache hit rate | rescue row 0/108 hits/selected misses; disabled rows do not instantiate L1 caching |
| Speculative acceptance / unique verification experts | not applicable; speculation is not implemented |
| Enabled optimizations | selected routing policy; exact LRU rescue only in the rescue row; no proxy, pruning, prediction, quantization, or speculation |

| Case | Avg K | Decode tok/s | Prefill tok/s | TTFT ms | Logical Reader bytes | Token parity | Prefix rate | Logit / state max abs. error | Cold rescues |
|---|---:|---:|---:|---:|---:|---|---:|---:|---:|
| Natural K16 | 16 | 1,132.006 | 991.844 | 21.218 | 1,294,992 | exact | 1.000 | 0 / 0 | 0 |
| Fixed K4 | 4 | 3,663.871 | 2,484.597 | 18.646 | 766,224 | diverged | 0.917 | 0.050723 / 0.020566 | 0 |
| Fixed K8 | 8 | 2,172.559 | 1,648.859 | 19.802 | 942,480 | diverged | 0.917 | 0.028294 / 0.010886 | 0 |
| Fixed K12 | 12 | 1,512.418 | 1,201.711 | 21.036 | 1,118,736 | diverged | 0.750 | 0.018955 / 0.008148 | 0 |
| Fixed K16 | 16 | 1,169.821 | 981.178 | 20.885 | 1,294,992 | exact | 1.000 | 0 / 0 | 0 |
| Adaptive balanced | 16 | 1,073.366 | 971.098 | 22.111 | 1,294,992 | exact | 1.000 | 0 / 0 | 0 |
| Adaptive mass 0.98 | 16 | 1,137.009 | 994.334 | 21.416 | 1,294,992 | exact | 1.000 | 0 / 0 | 0 |
| Adaptive boundary 0.02 | 16 | 1,147.920 | 999.484 | 20.977 | 1,294,992 | exact | 1.000 | 0 / 0 | 0 |
| Adaptive failure 1 | 16 | 1,156.740 | 998.297 | 21.133 | 1,294,992 | exact | 1.000 | 0 / 0 | 0 |
| Adaptive failure 2 | 16 | 1,167.919 | 1,002.486 | 21.285 | 1,294,992 | exact | 1.000 | 0 / 0 | 0 |
| Adaptive critical | 16 | 1,149.501 | 1,015.294 | 21.093 | 1,294,992 | exact | 1.000 | 0 / 0 | 0 |
| Fixed K4 + failure 1 | 8 | 2,100.988 | 1,670.519 | 19.854 | 942,480 | diverged | 0.917 | 0.028294 / 0.010886 | 0 |
| Fixed K4 + failure 2 | 12 | 1,508.239 | 1,265.230 | 20.284 | 1,118,736 | diverged | 0.750 | 0.018955 / 0.008148 | 0 |
| Fixed K4 + critical | 16 | 1,185.808 | 1,016.261 | 20.909 | 1,294,992 | exact | 1.000 | 0 / 0 | 0 |
| Fixed K4 + LRU rescue | 4 | 3,499.800 | 2,411.754 | 18.752 | 766,224 | diverged | 0.917 | 0.050723 / 0.020566 | 108 |

Fixed K4/K8/K12 reduce logical Reader bytes by 40.8%/27.2%/13.6% and show 3.24x/1.92x/1.34x decode ratios against natural K16 on this tiny CPU graph. They also change greedy tokens, prefill logits, recurrent state, and later routing order, so the throughput ratios are not quality-equivalent speedups. Fixed K16 and fixed K4 plus critical escalation are exact.

The synthetic router has normalized entropy near one, so every tested adaptive threshold conservatively selects K16. This is a measured limitation of the fixture, not evidence that adaptive selection will or will not reduce K on Kimi K3. The 6,528-byte rescue cache performs 108 exact cold loads and preserves the cache-disabled fixed-K4 tokens, route, logits, and state, but records zero hits and identical logical Reader traffic. Residency therefore changes neither routing nor quality, while this capacity provides no reuse benefit.

Raw JSON/CSV, diagnostic JSON, and the programmatically cross-checked summary are under `results/b0012-adaptive-routing-wsl/`. The summary SHA-256 is `72de06f6fe7ff18a82e67b87cb38c0cc7b1c2ee2819ac62fe2a828e69307cac9`.

Verification passed CPU CTest 11/11 and pytest 227/41, liburing/direct CTest 12/12 and pytest 233/35, CUDA CTest 20/20 and pytest 260/8, plus ASan/UBSan liburing CTest 12/12 and targeted pytest 101/34. All ten CUDA Compute Sanitizer targets reported zero errors. A final self-review fix made natural mode ignore an otherwise out-of-range external quality floor; B-0012 uses natural Top-16 and its measured rows are unchanged.

Public branch and PR correctness runs `31318880063` and `31318890885` succeeded at integration head `edc6d60`. PR #11 merged by fast-forward, and post-merge `main` correctness run `31318993688` also succeeded at that head.

## B-0013 — Milestone 12 routed accumulation CUDA fusion

| Field | Value |
|---|---|
| Evidence | measured WSL2 synthetic end-to-end and bounded released-dimension kernel/D2H ablation; non-authoritative for full-model throughput or native-Linux defaults |
| Date | 2026-08-10 |
| Measurement code commit | `58c36dd` |
| Result commit | `0632a0f` |
| Hardware | AMD Ryzen 7 9800X3D; NVIDIA GeForce RTX 5080 16,303 MiB, native `sm_120` |
| Environment | WSL2 Ubuntu 24.04.4; CUDA Toolkit 13.3.1; Linux 6.18.33.2 |
| Model/checkpoint | deterministic natural Top-16 synthetic graph, SHA-256 `edeaa4802b4bfac0624fa4d0e73917318076258d95e74e880c97a8b2709dd2d2`; released expert storage SHA-256 `aab7aea48b03bdcd8e0b4d98c4780128ab689d2bba005089a49970eb0e326890`; no full Kimi K3 weights |
| Mode | `cuda-custom`, FP32, reused allocation, transient weights, scalar FFN block; synchronous/prefetch crossed with `none|routed-accumulate` |
| Context / generated tokens | prompt `[1, 7, 3, 9]`; 6 generated tokens `[56, 55, 18, 11, 11, 13]` in every synthetic row |
| Warmup / samples | 3 / 20 per row |
| Quality result | exact token and routing identity; maximum absolute error `2.4e-7` in every synthetic row; released fixture maximum absolute error 0 |
| Average Top-K | 16, natural synthetic routing |
| VRAM | synthetic peak scratch/VRAM 43,680 bytes synchronous and 1,091,712 bytes prefetch; released fixture peak 23,461,888 bytes |
| System RAM | synthetic peak RSS 507,482,112 to 507,785,216 bytes |
| NVMe GB/token | not measured; synthetic logical Reader bytes are 215,832 bytes/token and are not physical NVMe attribution |
| H2D | synthetic 5,802,048 bytes per run in every row; released fixture timed weight H2D is 0 because the immutable expert is preloaded |
| GPU utilization / memory bandwidth | not measured |
| Cache hit rate | not applicable; L1 disabled and CUDA weights transient |
| Speculative acceptance / unique verification experts | not applicable; speculation is not implemented |
| Cold rescue count | 0 |
| Enabled optimization | routed down-projection contribution scaling and ordered device accumulation only in `routed-accumulate`; all other row identities held fixed |

| Synthetic mode | Decode tok/s | Prefill tok/s | TTFT ms | Kernel ns | D2H bytes | Fused calls / experts |
|---|---:|---:|---:|---:|---:|---:|
| synchronous / none | 13.6984 | 10.6493 | 1,465.538 | 74,281,408 | 134,064 | 0 / 0 |
| synchronous / routed-accumulate | 15.2499 | 11.5772 | 1,431.302 | 68,041,632 | 82,224 | 27 / 432 |
| prefetch / none | 14.7718 | 11.1078 | 1,430.722 | 82,948,992 | 134,064 | 0 / 0 |
| prefetch / routed-accumulate | 16.0877 | 11.9526 | 1,408.182 | 72,680,128 | 82,224 | 27 / 432 |

Synthetic fusion improves decode by 1.5516 tok/s, or 11.33%, with synchronous transfer and by 1.3159 tok/s, or 8.91%, with prefetch. It reduces D2H by 51,840 bytes per complete run and reduces aggregate CUDA-event kernel time in both rows. These are tiny synthetic end-to-end results, not full-model projections.

| Released-dimension mode | Median latency ns | Kernel ns / 20 | D2H bytes / 20 | Peak VRAM bytes | Fused calls / experts |
|---|---:|---:|---:|---:|---:|
| none | 7,867,604 | 76,422,336 | 4,587,520 | 23,461,888 | 0 / 0 |
| routed-accumulate | 8,497,998 | 80,913,440 | 286,720 | 23,461,888 | 20 / 320 |

The released fixture repeats one immutable 17,547,264-byte, 3,584-by-3,072 expert view across 16 slots. It deliberately has `routing_semantics=false`; it isolates kernel and D2H behavior and is neither token throughput nor a full-layer/full-model claim. Fusion reduces D2H by 4,300,800 bytes, or 93.75%, but increases median latency by 630,394 ns, or 8.01%, and aggregate kernel time by 4,491,104 ns, or 5.88%. This representative-dimension regression rejects a default change.

Raw JSON/CSV and checksummed summaries are under `results/b0013-fused-routed-accumulation/`. Synthetic summary SHA-256 is `996dad640c78ea356b1b9d13fb7879e07511cba42e7257a6c43fa95b7f274da7`; released summary SHA-256 is `d6f186fb991c67e2c4a1cd4929816ca1cf5567b187a905dd447db99258fd1799`. The synthetic host command reached its 300-second timeout only after all four raw records and the summary were published; independent raw-summary, schema, and CSV validation passed afterward.

Verification passed CPU CTest 11/11 and pytest 235/44, liburing/direct CTest 12/12 and pytest 237/42, CUDA CTest 20/20 and pytest 271/8, plus ASan/UBSan liburing CTest 12/12 and targeted pytest 49/5 with 57 deselected. Eleven CUDA Compute Sanitizer invocations, including the released-dimension fused benchmark, each reported zero errors. The first liburing pytest invocation omitted the required capability environment and produced one expected-selection failure; the corrected capability-enabled run passed. An initial malformed sanitizer loop ran no valid target; the corrected explicit invocations produced the reported results.

Public branch and PR correctness runs `31322043556` and `31322049903` succeeded at integration head `9e59a9d`. PR #12 merged by fast-forward, and post-merge `main` correctness run `31322191670` also succeeded at that head.

## Derived bottleneck model — not a benchmark

The released dimensions imply 17,547,264 bytes per native MXFP4 routed expert. With no cache reuse, natural Top-16 across 92 MoE layers implies 25,829,572,608 expert bytes/token. Applying the P44 Pro published 7.0 GB/s sequential figure gives a derived expert-only ceiling of about 0.271 tok/s and implies roughly 94.6% expert NVMe-byte avoidance for a 5 tok/s target.

These values are capacity and traffic estimates. They are not inserted into B-0001's NVMe field and must be replaced by Linux block-I/O measurements when the tiered runtime exists.

## B-0014 — Milestone 13 exact token-major speculative verification

- Date: 2026-08-10.
- Commit: implementation `2cf50b4`; result/ledger `e2e37bf`.
- Hardware: AMD Ryzen 7 9800X3D host under WSL2 Linux `6.18.33.2-microsoft-standard-WSL2`; CPU backend only.
- Model/checkpoint: synthetic executable `artifacts/synthetic.k3x`, SHA-256 `039d61ee9c2e13e27c9a2514bb476f8b122b8b37be0b7f85baf26c1a6611a2e9`.
- Mode: incremental natural Top-2, disabled L1, blocking `pread + buffered`, 4 prompt tokens, 6 generated tokens, 3 warmups, 20 measured samples.
- Cases: greedy reference; scripted perfect block-2 with 3/3 accepted draft tokens; scripted mixed block-2 with 1/4 accepted draft tokens including mismatch and empty proposals.

| Case | Decode tok/s | Prefill tok/s | TTFT ms | Acceptance | Blocks | Target forwards | Reader bytes | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| greedy | 171.4333 | 98.5411 | 402.8667 | n/a | 0 | 5 | 665,616 | 5,697,536 |
| perfect-block2 | 174.0861 | 99.0932 | 405.6271 | 1.00 | 2 | 5 | 665,616 | 5,746,688 |
| mixed-block2 | 173.2344 | 98.1697 | 406.1909 | 0.25 | 4 | 5 | 665,616 | 5,545,984 |

Every row generated `[43, 32, 28, 49, 9, 28]` and matched greedy final KDA/MLA state, complete routed expert/K traces, Reader calls/bytes, and L1 hits/misses. H2D, D2H, kernel time, VRAM, and CUDA utilization are zero/not applicable on the CPU backend. The perfect and mixed decode deltas are +1.55% and +1.05%, but identical target-forward and traffic counts mean they are not evidence of speculative acceleration. Expert-major unique-union telemetry is not applicable because this is token-major execution.

Raw and aggregate artifacts are under `results/b0014-speculative-verification-wsl/`. Summary JSON SHA-256 is `7cd834b1c65d507367320170cdf72ca76aace9f6a743da85a0a9f0cca4a21062`; summary CSV SHA-256 is `9c5fdba84c547f93e2a0a7d4c0b76412181ffb2c635ffd969537a154950ce75b`. An independent raw-summary and exact-parity cross-check passed.

Post-measurement verification passed CPU CTest 12/12 and pytest 245/44, liburing/direct CTest 13/13 and pytest 247/42, CUDA CTest 21/21 and pytest 281/8, and ASan/UBSan liburing CTest 13/13 plus targeted pytest 26/3 with 104 deselected. Compute Sanitizer perfect and mixed CUDA speculative CLI runs each reported `ERROR SUMMARY: 0 errors`. The first liburing Python invocation omitted `K3X_TEST_IO_URING=1` and therefore selected the expected-unavailable test incorrectly; the corrected capability-aware run passed.

Public branch and PR correctness runs `31324378917` and `31324381376` succeeded at integration head `463e9ca`. PR #13 merged by fast-forward, and post-merge `main` correctness run `31324492327` also succeeded at that head.

## B-0015 — Milestone 14 exact expert-major speculative verification

- Date: 2026-08-10.
- Commit: implementation `bdf4a66`; measurement tooling and results `1e73121`.
- Hardware: AMD Ryzen 7 9800X3D host under WSL2 Linux; CPU backend only.
- Model/checkpoint: synthetic executable `build-fixtures/synthetic.k3x`, SHA-256 `29f3fd10c95dcde9f2b012e10e36962363b5cdd79dfeda5f5e3bbaca0cb89b75`.
- Mode: incremental natural Top-2, disabled L1, blocking `pread + buffered`, 4 prompt tokens, 6 generated tokens, 3 warmups, 20 measured samples.
- Cases: greedy; token-major and expert-major perfect block-2; token-major and expert-major mixed block-2 with mismatch and empty proposals.

| Case | Decode tok/s | Prefill tok/s | TTFT ms | Peak RSS | Acceptance | Evaluated / discarded | Unique loads / assignments | Reader calls | Reader bytes |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| greedy | 163.1535 | 93.5544 | 329.4235 | 5,656,576 | n/a | 0 / 0 | 0 / 0 | 428 | 665,616 |
| token-major perfect-2 | 160.1659 | 93.5235 | 329.6131 | 5,791,744 | 1.00 | 0 / 0 | 0 / 0 | 428 | 665,616 |
| expert-major perfect-2 | 201.5550 | 94.1396 | 328.2316 | 5,902,336 | 1.00 | 5 / 0 | 24 / 30 | 392 | 655,824 |
| token-major mixed-2 | 163.0028 | 93.8157 | 328.5150 | 5,746,688 | 0.25 | 0 / 0 | 0 / 0 | 428 | 665,616 |
| expert-major mixed-2 | 122.6010 | 93.7455 | 327.9867 | 5,713,920 | 0.25 | 8 / 3 | 39 / 48 | 482 | 680,304 |

Every row generated the greedy sequence and matched final KDA/MLA state plus committed routed expert/K traces. Perfect expert-major execution reused six assignments, reduced Reader bytes by 1.47% and calls by 8.41%, and measured 25.84% higher decode than its token-major pair. Mixed expert-major execution evaluated three rejected positions, increased Reader bytes by 2.21% and calls by 12.62%, and measured 24.79% lower decode. No favorable latency or traffic direction was required for the mixed case.

Average Top-K is 2 and L1 hits are zero in every row. H2D, D2H, kernel time, VRAM, and GPU utilization are zero or not applicable because this is the exact CPU boundary. Median Reader storage time ranges from 64.556 ms for expert-major perfect to 79.983 ms for expert-major mixed; blocking mode reports no separate asynchronous exposed-wait counter. Logical Reader bytes are not physical NVMe bytes. These results are not full-model, native-Linux P44 Pro, coding-quality, or RTX 5080 evidence and do not change the token-major default.

Raw JSON/CSV, diagnostic JSON, and the independently cross-checked aggregate are under `results/b0015-expert-major-verification-wsl/`. The canonical aggregate-record SHA-256 is `cb95eff274713a21b821695d75ff2655da735513c99215ec5ec14f5ed995b813`; `summary.json` also records the SHA-256 of every raw JSON and CSV artifact. The focused ablation/schema cross-check passed 12 tests with 5 capability skips.

Pre-publication verification passed CPU CTest 13/13 and pytest 253/44, liburing/direct CTest 14/14 and pytest 257/42, CUDA CTest 22/22 and pytest 291/8, plus ASan/UBSan liburing CTest 14/14 and targeted pytest 95/35. CUDA FFN Compute Sanitizer reported `ERROR SUMMARY: 0 errors`. Attempting Compute Sanitizer around the CPU-only expert-major CLI correctly produced no instrumented CUDA API call and is not reported as a sanitizer pass; that execution path is covered by ASan/UBSan instead.

Public branch and PR correctness runs `31328853375` and `31328869071` succeeded at integration head `012e598`. PR #15 merged by fast-forward, and post-merge `main` correctness run `31329045623` also succeeded at that head.

## B-0016 — Milestone 15 exact CUDA expert-major execution

- Date: 2026-08-10.
- Commit: batch/runtime implementation `e99bbc0`; measurement tooling `7899603`; direct CLI correction `884a74e`.
- Hardware: AMD Ryzen 7 9800X3D and NVIDIA GeForce RTX 5080 16 GB under WSL2 Ubuntu 24.04.4; CUDA 13.3 native `sm_120`.
- Model/checkpoint: executable synthetic `artifacts/synthetic.k3x`, SHA-256 `039d61ee9c2e13e27c9a2514bb476f8b122b8b37be0b7f85baf26c1a6611a2e9`; non-executable released-dimension storage fixture `artifacts/m12-bounded.k3x`, SHA-256 `aab7aea48b03bdcd8e0b4d98c4780128ab689d2bba005089a49970eb0e326890`.
- Mode: incremental natural Top-2, `cuda-custom + ffn-block + reused + transient + synchronous + fusion none`, disabled L1, blocking `pread + buffered`, 4 prompt tokens, 6 generated tokens, 3 warmups, 20 measured samples.
- Cases: CUDA greedy; token-major and expert-major perfect block-2; token-major and expert-major mixed block-2; released single-expert scalar/batch pairs at batch sizes two and four.

| Graph case | Decode tok/s | Prefill tok/s | TTFT ms | Peak RSS | Peak VRAM | Acceptance | Evaluated / discarded | Batch calls / tokens | Kernel ms | Logical Reader GB/generated token | H2D GB/generated token |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| greedy | 60.9608 | 25.1116 | 886.0143 | 507,486,208 B | 43,680 B | n/a | 0 / 0 | 0 / 0 | 26.0042 | 0.000110936 | 0.000845760 |
| token-major perfect-2 | 60.1815 | 24.9711 | 867.5926 | 507,637,760 B | 43,680 B | 1.00 | 0 / 0 | 0 / 0 | 28.8968 | 0.000110936 | 0.000845760 |
| expert-major perfect-2 | 70.1067 | 25.7569 | 860.8092 | 507,727,872 B | 44,192 B | 1.00 | 5 / 0 | 24 / 30 | 24.0656 | 0.000109304 | 0.000844448 |
| token-major mixed-2 | 57.2332 | 25.9700 | 865.4655 | 507,600,896 B | 43,680 B | 0.25 | 0 / 0 | 0 / 0 | 29.6127 | 0.000110936 | 0.000845760 |
| expert-major mixed-2 | 40.7627 | 25.5761 | 863.9135 | 507,551,744 B | 44,192 B | 0.25 | 8 / 3 | 39 / 48 | 34.7756 | 0.000113384 | 0.001125744 |

All graph rows preserve the greedy token sequence, final KDA/MLA state, and committed expert/K traces. Average Top-K is 2, L1 hits/misses are zero because the cache is disabled, and perfect/mixed speculative acceptance is 1.0/0.25. Logical Reader GB/token is not physical NVMe GB/token. GPU utilization and memory bandwidth were not captured, so no value is inferred for them. The perfect expert-major row is 16.49% faster than its token-major pair and slightly reduces logical Reader and H2D traffic; the mixed row is 28.78% slower and increases both because it evaluates three rejected positions. No favorable direction was required.

| Released case | Median latency ms | Aggregate kernel ms | Weight H2D | Activation H2D | D2H | Peak VRAM | Batch calls / tokens | Max abs. error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| scalar-2 | 3.4449 | 12.1719 | 701,890,560 B | 573,440 B | 573,440 B | 5,914,624 B | 0 / 0 | 0 |
| batch-2 | 1.7378 | 8.0848 | 350,945,280 B | 573,440 B | 573,440 B | 5,980,160 B | 20 / 40 | 0 |
| scalar-4 | 6.7053 | 24.4185 | 1,403,781,120 B | 1,146,880 B | 1,146,880 B | 5,914,624 B | 0 / 0 | 0 |
| batch-4 | 2.6319 | 14.0455 | 350,945,280 B | 1,146,880 B | 1,146,880 B | 6,111,232 B | 20 / 80 | 0 |

The released fixture has `routing_semantics=false`. Each scalar call transfers the 17,547,264-byte expert once per token; each batch call transfers it once per iteration. Batch-2 reduces weight H2D by 50%, median latency by 49.55%, and kernel time by 33.58%. Batch-4 reduces weight H2D by 75%, median latency by 60.75%, and kernel time by 42.48%. Activation H2D and D2H remain identical within each pair. This is a single-expert kernel/traffic result, not full-layer routing, full-model TPS, physical NVMe, or quality evidence.

Raw JSON/CSV and diagnostics are under `results/b0016-cuda-expert-major-wsl/`. Canonical aggregate-record SHA-256 is `09a2537337df1fd2b8b39439f92ba7306cb09a6ed5e3f8bdc8db7d9d787029aa`; summary JSON/CSV SHA-256 is `5c8c32a6fed499a1ff8ddf2d0f2e0fdaa214e6a62715933edea850c5f42812540` / `f100f15803741531d88976a4cc64c0a0975acd94f71be99fce6bb45fe4422f65`. CSV writers explicitly use LF so Git text normalization preserves these digests. Independent validation recomputed all nine raw JSON/CSV digest pairs and the canonical aggregate.

Verification passed CPU CTest 13/13 and pytest 262/47, liburing/direct CTest 14/14 and pytest 264/45, ASan/UBSan CTest 14/14, and CUDA CTest 22/22 with pytest 301/8. Compute Sanitizer reported `ERROR SUMMARY: 0 errors` for native MXFP4, CUDA FFN, released batch-2, perfect expert-major CLI, and mixed expert-major CLI. Branch and pull-request correctness runs `31332732339` and `31332745907` passed, PR #17 was rebase-merged at public integration head `c18df33`, and post-merge `main` run `31332852551` passed.

## B-0017 — Milestone 16 AURORA replay and adaptive scheduling

- Date: 2026-08-10.
- Commit: runtime integration `bc45538`; measurement evidence `51ff8e7`.
- Hardware: AMD Ryzen 7 9800X3D under WSL2 Ubuntu 24.04.4. The CPU graph was measured; the RTX 5080 was used only for the combined-path Compute Sanitizer check.
- Model/checkpoint: the runner-generated temporary synthetic natural Top-16 K3X artifact, SHA-256 `c1110ad2a1fe981f92b01e36aaafa216d0d8ea45a6608270f3cf706816c17a7c`.
- Mode: incremental natural target Top-16, fixed reduced draft Top-4, disabled L1, blocking `pread + buffered`, 4 prompt tokens, 6 generated tokens, 3 warmups, and 20 measured samples.
- Cases: natural greedy; fixed replay blocks 1, 2, and 4 with token-major target verification; adaptive token-major replay; fixed block-2 expert-major replay; adaptive expert-major replay.

| Case | Decode tok/s | Prefill tok/s | TTFT ms | Peak RSS | Acceptance | Blocks | Candidates | Target eval / discard | Target Reader bytes | Draft Reader bytes | Replay positions |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| natural greedy | 1140.3391 | 1032.2611 | 21.3741 | 5,709,824 B | n/a | 0 | 0 | 0 / 0 | 1,294,992 | 0 | 0 |
| fixed-1 token-major | 480.1354 | 1131.4542 | 31.6413 | 6,742,016 B | 0.6667 | 3 | 3 | 0 / 0 | 1,294,992 | 2,161,584 | 20 |
| fixed-2 token-major | 562.7420 | 1145.7370 | 32.9992 | 6,815,744 B | 1.0000 | 2 | 3 | 0 / 0 | 1,294,992 | 1,454,112 | 13 |
| fixed-4 token-major | 514.2685 | 1111.1756 | 32.6129 | 6,553,600 B | 0.6000 | 2 | 5 | 0 / 0 | 1,294,992 | 1,493,280 | 13 |
| adaptive token-major | 447.3694 | 1166.7779 | 31.9693 | 6,590,464 B | 0.5000 | 3 | 4 | 0 / 0 | 1,294,992 | 2,181,168 | 20 |
| fixed-2 expert-major | 611.7589 | 1001.9199 | 32.3485 | 6,483,968 B | 1.0000 | 2 | 3 | 5 / 0 | 1,102,416 | 1,454,112 | 13 |
| adaptive expert-major | 427.4438 | 1010.3564 | 32.1726 | 6,516,736 B | 0.5000 | 3 | 4 | 7 / 2 | 1,197,072 | 2,181,168 | 20 |

Every replay row preserves the natural target token sequence, final KDA/MLA recurrent state, and committed expert/K trace exactly. Target average Top-K is 16 and draft average Top-K is 4. The adaptive token-major and expert-major rows each record one growth and one backoff event. L1 is disabled, so its hit rate is zero. GPU utilization, GPU memory bandwidth, VRAM, H2D, and kernel time are zero or not applicable for this CPU measurement. Logical Reader bytes are not physical NVMe bytes, and physical NVMe GB/token remains unmeasured.

All replay cases regress decode by 46.35% to 62.52% relative to the tiny natural baseline. Fixed block-2 expert-major is the least-slow replay row at 611.7589 tok/s and reduces target Reader traffic through expert-union reuse, but the separate replay drafter adds 1,454,112 logical bytes and repeats 13 committed-prefix positions. Adaptive rows repeat 20 positions and add 2,181,168 draft bytes. The measured bottleneck is complete-prefix replay, so AURORA remains non-default and no scheduler threshold is promoted from this trace.

Raw JSON/CSV, diagnostics, and independently cross-checked summaries are under `results/b0017-aurora-replay-wsl/`. The runner SHA-256 is `a20f708073bd27150d27d8eddf5c926072f1b96020257e625ab3caa895a536f7`; canonical aggregate-record SHA-256 is `fb7febf52c75281417b77c3f7d40787f738dba8a35490cc86d43ac5072cacd23`; summary JSON/CSV SHA-256 is `fdd94c5696d1505e17e0dbc41d465d8edad38b132896f5a3742277c09b852871` / `865d228fb88b1bc22fe147b04e1ce003559f04534052d8ce0180b753832d9551`. Independent validation recomputed all 14 raw JSON/CSV digests, the canonical aggregate, and the LF-stable CSV digest.

Verification passed CPU CTest 14/14 and pytest 268/47, liburing/direct CTest 15/15 and pytest 274/41, ASan/UBSan liburing CTest 15/15, and CUDA CTest 23/23 with pytest 307/8. Compute Sanitizer reported `ERROR SUMMARY: 0 errors` for the combined CUDA expert-major AURORA CLI path. These checks establish synthetic exactness and memory-safety coverage; they do not establish full-model quality or performance.

Public branch and pull-request correctness runs `31337234073` and `31337240722` passed. PR #20 was rebase-merged at public integration head `df5c07d`, and post-merge `main` correctness run `31337365175` passed.

## B-0018 — Milestone 17 persistent AURORA draft state

- Date: 2026-08-10.
- Commit: cursor/provider `c28a732`; CLI/schema `3459ca6`; measurement evidence `de63023`.
- Hardware: AMD Ryzen 7 9800X3D under WSL2 Ubuntu 24.04.4. The RTX 5080 was used for verification, not the CPU timing rows.
- Model/checkpoint: runner-generated temporary synthetic natural Top-16 K3X artifact, SHA-256 `81560d6250869426d739040c6e30d9a881b1f37f7a3f639345d27dd69a80ce96`.
- Mode: incremental natural target Top-16, fixed reduced draft Top-4, disabled L1, blocking `pread + buffered`, 4 prompt tokens, 6 generated tokens, 3 warmups, and 20 measured samples.
- Cases: natural greedy plus matched replay/persistent fixed block-2 and adaptive pairs for token-major and CPU expert-major target verification.

| Case | Decode tok/s | Prefill tok/s | TTFT ms | Peak RSS | Acceptance | Target eval / discard | Target Reader bytes | Draft Reader bytes | Replay / prefill positions | Draft forwards | Rollback / crop | Pair decode delta | Draft-byte reduction |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| natural greedy | 1147.7689 | 988.5636 | 22.1083 | 5,967,872 B | n/a | 0 / 0 | 1,294,992 | 0 | 0 / 0 | 0 | 0 / 0 | n/a | n/a |
| replay fixed-2 token | 588.0806 | 938.8070 | 32.5735 | 6,553,600 B | 1.0000 | 0 / 0 | 1,294,992 | 1,454,112 | 13 / 0 | 0 | 0 / 0 | reference | reference |
| persistent fixed-2 token | 676.0989 | 951.5176 | 32.2148 | 6,418,432 B | 1.0000 | 0 / 0 | 1,294,992 | 785,808 | 0 / 5 | 5 | 0 / 0 | +14.967% | -45.960% |
| replay adaptive token | 474.6266 | 936.8968 | 32.3788 | 6,586,368 B | 0.5000 | 0 / 0 | 1,294,992 | 2,181,168 | 20 / 0 | 0 | 0 / 0 | reference | reference |
| persistent adaptive token | 672.8063 | 938.1414 | 32.7912 | 6,688,768 B | 0.5000 | 0 / 0 | 1,294,992 | 805,392 | 0 / 5 | 6 | 1 / 1 | +41.755% | -63.075% |
| replay fixed-2 expert | 620.4730 | 950.2246 | 32.5430 | 6,529,024 B | 1.0000 | 5 / 0 | 1,102,416 | 1,454,112 | 13 / 0 | 0 | 0 / 0 | reference | reference |
| persistent fixed-2 expert | 710.7307 | 938.8591 | 32.5896 | 6,733,824 B | 1.0000 | 5 / 0 | 1,102,416 | 785,808 | 0 / 5 | 5 | 0 / 0 | +14.547% | -45.960% |
| replay adaptive expert | 431.2070 | 934.4310 | 32.3639 | 6,549,504 B | 0.5000 | 7 / 2 | 1,197,072 | 2,181,168 | 20 / 0 | 0 | 0 / 0 | reference | reference |
| persistent adaptive expert | 547.9973 | 955.9389 | 33.3736 | 6,586,368 B | 0.5000 | 7 / 2 | 1,197,072 | 805,392 | 0 / 5 | 6 | 1 / 1 | +27.085% | -63.075% |

Every matched pair has identical proposed and accepted draft-token counts. All nine rows preserve the natural target token sequence, final KDA/MLA state, and committed expert/K trace exactly. Fixed persistent rows copy 57,600 KDA checkpoint bytes; adaptive persistent rows copy 76,800 bytes and perform one rollback/crop. These are state-copy counters, not Reader or physical storage bytes.

Persistent execution is faster than replay in all four pairs because it eliminates repeated complete-prefix work, but every persistent row remains 38.08% to 52.26% slower than the tiny natural greedy baseline. This benchmark therefore accepts the state architecture while keeping speculation non-default. Reduced precision, draft residency, learned proposals, full-model coding quality, physical NVMe traffic, and native-Linux performance remain unmeasured.

Raw JSON/CSV and independently cross-checked summaries are under `results/b0018-persistent-aurora-wsl/`. Runner SHA-256 is `0eb212731be6e0a5344048aa6f6d76fb57732423017568112ec9d27f7b74d48d`; canonical aggregate-record SHA-256 is `abcef1afca7d6208808941323565bce44f25ecc6e9e0d28292ad54bfc7760cd0`; summary JSON/CSV SHA-256 is `a332af2d336cecb3060812a577f16e605bc832f4f21b74f315dfbbf8fd4f6132` / `c65d3bb9d8805f66249d0bb6ba380b8aa2508fd53a073c1bc3dece82e00fe472`. Independent validation recomputes all 18 raw digests, the summary CSV digest, canonical aggregate, exact pair invariants, and headline percentages from committed bytes.

Verification passed CPU CTest 14/14 and pytest 272/47, liburing/direct CTest 15/15 and pytest 278/41, ASan/UBSan liburing CTest 15/15 plus five artifact-backed persistent tests, and CUDA CTest 23/23 with pytest 311/8. Compute Sanitizer reported `ERROR SUMMARY: 0 errors` for `aurora-persistent + expert-major + cuda-custom`. CPU cursor memory safety is established by ASan/UBSan; the CUDA check covers the target path rather than claiming GPU instrumentation of CPU state code.

Public branch and pull-request correctness runs `31340338639` and `31340340063` passed. PR #23 was rebase-merged at public integration head `30bbf7a8`, and post-merge `main` correctness run `31340476396` passed. The GitHub Actions run emitted a Node.js 20 deprecation warning for `actions/checkout@v4` and `actions/setup-python@v5`; it did not alter the successful benchmark or correctness evidence and remains a CI maintenance item.

## B-0019 — Milestone 18 exact transient CUDA AURORA drafting

- Date: 2026-08-10.
- Commit: provider contract `89ca6c6`; runtime ownership `785f73e`; separated telemetry `df72718`; measurement evidence `7257280`.
- Hardware: AMD Ryzen 7 9800X3D and NVIDIA GeForce RTX 5080 16 GB under WSL2 Ubuntu 24.04.4, CUDA 13.3 native `sm_120`.
- Model/checkpoint: runner-generated temporary synthetic natural Top-16 K3X artifact, SHA-256 `6604d1ec65f8056f6d4f04d09fa357a442c7c2f7a46faf56899caf31671d2ca7`.
- Mode: CPU natural Top-16 target; persistent fixed draft Top-4; disabled L1; blocking `pread + buffered`; 4 prompt tokens; 6 generated tokens; 3 warmups and 20 measured samples. CUDA draft identity is FP32, reused, transient, grouped, `ffn-block`, synchronous, fusion `none`, zero resident/pinned capacity.
- Cases: natural greedy plus matched CPU/CUDA fixed block-2 and adaptive pairs for token-major and CPU expert-major target verification.

| Case | Decode tok/s | Prefill tok/s | TTFT ms | Peak RSS | Acceptance | Target / draft Reader bytes | Draft H2D bytes | Draft kernel ms | Draft peak VRAM | Pair decode delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| natural greedy | 1168.1207 | 926.6565 | 82.6596 | 236,892,160 B | n/a | 1,294,992 / 0 | 0 | 0 | 0 | n/a |
| CPU fixed-2 token | 692.1847 | 910.3109 | 93.8958 | 237,637,632 B | 1.0000 | 1,294,992 / 785,808 | 0 | 0 | 0 | reference |
| CUDA fixed-2 token | 24.4660 | 911.3571 | 340.4560 | 509,886,464 B | 1.0000 | 1,294,992 / 785,808 | 5,843,840 | 37.4711 | 44,448 B | -96.465% |
| CPU adaptive token | 671.2625 | 910.9660 | 94.8104 | 237,625,344 B | 0.5000 | 1,294,992 / 805,392 | 0 | 0 | 0 | reference |
| CUDA adaptive token | 21.4149 | 912.3053 | 339.0534 | 509,935,616 B | 0.5000 | 1,294,992 / 805,392 | 6,428,224 | 54.0608 | 44,448 B | -96.810% |
| CPU fixed-2 expert | 724.4920 | 922.9024 | 95.2674 | 237,772,800 B | 1.0000 | 1,102,416 / 785,808 | 0 | 0 | 0 | reference |
| CUDA fixed-2 expert | 21.7318 | 907.7394 | 340.8668 | 510,214,144 B | 1.0000 | 1,102,416 / 785,808 | 5,843,840 | 54.5497 | 44,448 B | -97.000% |
| CPU adaptive expert | 562.5927 | 924.9426 | 94.5504 | 237,916,160 B | 0.5000 | 1,197,072 / 805,392 | 0 | 0 | 0 | reference |
| CUDA adaptive expert | 21.2731 | 903.5146 | 338.5157 | 510,361,600 B | 0.5000 | 1,197,072 / 805,392 | 6,428,224 | 52.9858 | 44,448 B | -96.219% |

Every matched CPU/CUDA pair has identical proposed, accepted, and committed draft-token counts; strict target tokens, final KDA/MLA state, committed routes, and acceptance are exact. The target stays on CPU, and its kernel, H2D, and peak-VRAM counters remain zero. CUDA draft rows record 13 allocations, 410 or 451 synchronizations, and predominantly weight H2D: 5,756,160 of 5,843,840 bytes for fixed rows and 6,331,776 of 6,428,224 bytes for adaptive rows. Cache hits/misses/bypasses are zero because this experiment deliberately excludes residency.

The result rejects transient synchronous CUDA drafting as a default. It isolates repeated weight H2D and fine-grained synchronous GPU work as the next measured AURORA bottleneck; it does not reject bounded resident drafting, persistent larger kernels, or later quality-measured reduced precision. GPU utilization, memory bandwidth, physical PCIe GB/token, physical NVMe GB/token, coding quality, and full-model throughput remain unmeasured.

Raw JSON/CSV and independently cross-checked summaries are under `results/b0019-cuda-aurora-draft-wsl/`. Runner SHA-256 is `fb7bded3cb3edd5b2f626801ec38edd246ba0c19a990e6301955e32d0642d52f`; canonical aggregate-record SHA-256 is `ce1a599eb04077f3b0c1b8350254b126a58f4dc311421bfc38fc8f7a78478c59`; summary JSON/CSV SHA-256 is `3750254294385cecf503f2efcd69f8d23953a7e982006fe969c4c9ac9ee2913f` / `1b6234889c8997486e5b268d277af3ca892b2b6c152489b4f85cea81717edd1f`. Independent validation recomputes all 18 raw digests, the summary CSV digest, canonical aggregate, exact pair invariants, and headline deltas from committed bytes.

Verification passes CPU CTest 14/14 and pytest 278/50, liburing/direct CTest 15/15 and pytest 284/44, ASan/UBSan liburing CTest 15/15, and CUDA CTest 23/23 with pytest 319/9. Compute Sanitizer reports `ERROR SUMMARY: 0 errors` for the exact CUDA draft plus CPU expert-major target path.

Public push and pull-request correctness runs `31343260116` and `31343261633` passed. PR #25 was rebase-merged at public integration head `7899a7ae`, and post-merge `main` correctness run `31343401178` passed.

## B-0020 — Milestone 19 bounded exact CUDA AURORA residency

- Date: 2026-08-10.
- Commit: provider contract `f3e3c6c`; CLI ownership `6fa0806`; telemetry `26686c6`; measurement runner `1bc2b1b`; evidence `f676957`.
- Hardware: AMD Ryzen 7 9800X3D and NVIDIA GeForce RTX 5080 16 GB under WSL2 Ubuntu 24.04.4, CUDA 13.3 native `sm_120`.
- Model/checkpoint: runner-generated temporary synthetic natural Top-16 K3X artifact, SHA-256 `47795886397106b3d1a029fefb86e58776be659cb0470ceb7c9998851aedcf26`.
- Mode: CPU natural Top-16 target; persistent fixed draft Top-4; disabled L1; blocking `pread + buffered`; 4 prompt tokens; 6 generated tokens; 3 warmups and 20 measured samples. CUDA draft identity is FP32, reused, grouped, `ffn-block`, synchronous, fusion `none`, zero pinned capacity, and either transient weights or exact resident weights with an 8,388,608-byte hard cap.
- Cases: natural greedy plus matched transient/resident fixed block-2 and adaptive pairs for token-major and CPU expert-major target verification.

| Case | Decode tok/s | Prefill tok/s | TTFT ms | Peak RSS | Acceptance | Target / draft Reader bytes | Draft weight H2D | Draft kernel ms | Draft peak VRAM | Pair decode delta |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| natural greedy | 1172.6545 | 930.9483 | 82.1574 | 236,945,408 B | n/a | 1,294,992 / 0 | 0 | 0 | 0 | n/a |
| transient fixed-2 token | 21.7783 | 903.6918 | 337.9615 | 509,939,712 B | 1.0000 | 1,294,992 / 785,808 | 5,756,160 B | 54.7252 | 44,448 B | reference |
| resident fixed-2 token | 25.1719 | 918.2254 | 340.1868 | 510,492,672 B | 1.0000 | 1,294,992 / 785,808 | 644,160 B | 45.6142 | 688,608 B | +15.582% |
| transient adaptive token | 21.5013 | 912.5053 | 337.8708 | 509,739,008 B | 0.5000 | 1,294,992 / 805,392 | 6,331,776 B | 52.7462 | 44,448 B | reference |
| resident adaptive token | 20.9506 | 913.7784 | 335.7986 | 510,492,672 B | 0.5000 | 1,294,992 / 805,392 | 647,424 B | 54.1255 | 691,872 B | -2.561% |
| transient fixed-2 expert | 21.5694 | 912.7434 | 339.4491 | 509,898,752 B | 1.0000 | 1,102,416 / 785,808 | 5,756,160 B | 55.2515 | 44,448 B | reference |
| resident fixed-2 expert | 26.4599 | 911.6050 | 339.6024 | 510,259,200 B | 1.0000 | 1,102,416 / 785,808 | 644,160 B | 44.3741 | 688,608 B | +22.673% |
| transient adaptive expert | 21.8668 | 908.1380 | 339.1982 | 510,115,840 B | 0.5000 | 1,197,072 / 805,392 | 6,331,776 B | 51.1403 | 44,448 B | reference |
| resident adaptive expert | 23.0844 | 909.6943 | 340.7877 | 510,324,736 B | 0.5000 | 1,197,072 / 805,392 | 647,424 B | 53.2099 | 691,872 B | +5.569% |

All matched pairs preserve proposed, accepted, and committed draft-token counts; strict target tokens `[56, 55, 18, 11, 11, 13]`; final KDA/MLA state; and committed routing. Target CUDA residency, H2D, kernel, and VRAM counters remain zero. Average target Top-K is 16 and the fixed draft executes Top-4. Expert-major fixed rows observe 122 unique expert payloads across two verification blocks, or 61 per block; adaptive rows observe 180 across three blocks, or 60 per block.

The resident fixed rows record 666 hits, 214 misses, 75.682% hit rate, 644,160 current/peak resident bytes, and zero bypasses. Adaptive rows record 748 hits, 220 misses, 77.273% hit rate, 647,424 current/peak resident bytes, and zero bypasses. Weight H2D falls by 88.809% fixed and 89.775% adaptive. Including activation traffic, total draft RAM-to-GPU traffic falls from 5,843,840 to 731,840 bytes per run for fixed rows and from 6,428,224 to 743,872 bytes per run for adaptive rows, or approximately 0.000974 to 0.000122 GB/generated-token and 0.001071 to 0.000124 GB/generated-token respectively.

Physical NVMe GB/token, GPU utilization, GPU memory bandwidth, and I/O stall time are not measured by this WSL2 synthetic benchmark. Reader bytes above are logical runtime bytes, peak RSS is process memory rather than total system-RAM residency, and the cache is a tiny no-eviction working set. Quality evidence is exact synthetic token/state/route parity only; coding/agentic quality and full-model behavior remain unmeasured. The mixed paired decode results reject promotion to a default and identify 410–451 synchronous waits plus fine-grained launches as the next isolated CUDA draft bottleneck.

Raw JSON/CSV and independently cross-checked summaries are under `results/b0020-cuda-aurora-residency-wsl/`. Runner SHA-256 is `9fd847ff95c0f3b9c3bb3bc90ff568381b3a3d540f80eebcf433551465d79daa`; canonical aggregate-record SHA-256 is `4bb84fe49cbbc735bc9ef8668ab4d2944fef3d4e3a0f1048a7973410b211df87`; summary JSON/CSV SHA-256 is `32d9795ab3da3107c8f4fe5573be439130795d91b6860bdda91cc7d84635a192` / `059ede44149da8490f8342061b4dc10e623abe08f420266f84d7ca73963e3a62`. Independent validation recomputes all 18 raw digests, the summary CSV digest, canonical aggregate, exact pair invariants, capacity and bypass gates, pair decode deltas, H2D reductions, and hit rates from committed bytes.

Fresh verification passes CPU CTest 14/14 with pytest 284 passed/53 skipped, liburing/direct CTest 15/15 with pytest 290 passed/47 skipped, ASan/UBSan liburing CTest 15/15, and CUDA CTest 23/23 with pytest 328 passed/9 skipped. Compute Sanitizer reports `ERROR SUMMARY: 0 errors` for both the 8 MiB full-fit path and the one-byte exact-bypass path; the latter records 880 misses, 880 bypasses, and zero resident bytes while preserving target tokens.

Public push and pull-request correctness runs `31346575341` and `31346587586` passed. PR #27 was rebase-merged at public integration head `c88456c0`, and post-merge `main` correctness run `31346725071` passed.

### Post-B-0020 Nsight launch diagnostic

- Date: 2026-08-10.
- Commit: public documentation head `01eac162`; runtime implementation is the unchanged Milestone 19 integration.
- Hardware: AMD Ryzen 7 9800X3D and NVIDIA GeForce RTX 5080 under WSL2 Ubuntu, CUDA 13.3 native `sm_120`, Nsight Systems 2026.1.3.
- Model/checkpoint: B-0020 Top-16 executable synthetic artifact.
- Mode: one instrumented fixed block-2 token-major run, CPU target, persistent CUDA Top-4 draft, exact 8 MiB resident weights, four prompt tokens and six generated tokens. No warmup or throughput sample is claimed because instrumentation changes timing.
- Observed CUDA API counts: 1,040 kernel launches, 1,346 `cudaMemcpyAsync`, 410 `cudaStreamSynchronize`, 1,288 event creates, and 389 allocation/free pairs.
- Observed GPU kernel instances: 520 cuBLAS GEMV, 360 native MXFP4 matvec, and 160 SiTU kernels; aggregate GPU kernel duration was approximately 1.13 ms.
- Instrumented host API totals: approximately 35.03 ms in `cudaLaunchKernel`, 71.55 ms in `cudaMemcpyAsync`, and 0.94 ms in `cudaStreamSynchronize`. These are diagnostic attribution values, not decode/prefill TPS or uninstrumented latency.
- Interpretation: after exact weight residency, fine-grained launches and activation copies are a stronger next boundary than stream-wait duration alone. No full-model, native-Linux, utilization, bandwidth, physical PCIe, physical NVMe, or quality claim is made.

## B-0021 — Milestone 20 resident CUDA expert grid

- Date: 2026-08-10.
- Commit: public evidence head `8e85ff3`; implementation and runner lineage ends at `5ed8e74` before the evidence commit.
- Hardware: AMD Ryzen 7 9800X3D and NVIDIA GeForce RTX 5080 16,303 MiB under WSL2 Ubuntu 24.04.4, CUDA 13.3 native `sm_120`.
- Model/checkpoint: runner-generated synthetic natural Top-16 K3X artifact, SHA-256 `7e12595e5e400b4c26946c75927b37f39ed3a0bcb8f90ca72b1e8f7c6cb95cad`.
- Mode: CPU natural Top-16 target; persistent exact resident CUDA Top-4 draft; grouped versus resident-grid draft batching; fixed block-2 and adaptive policies; token-major and CPU expert-major target verification; 4 prompt tokens; 6 generated tokens; 3 warmups and 20 measured samples.
- Quality scope: exact synthetic target token, final KDA/MLA state, committed route, proposal, acceptance, Reader-byte, and resident-weight-H2D parity. Coding/agentic quality and full-model quality are unmeasured.

| Pair | Grouped decode tok/s | Grid decode tok/s | Paired delta | MoE launches grouped → grid | Grid calls | Grid fallbacks | Total draft H2D grouped → grid |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed-2 token-major | 27.5269 | 30.4982 | +10.794% | 480 → 120 | 30 | 0 | 731,840 → 752,960 B |
| adaptive token-major | 23.1484 | 28.7238 | +24.086% | 528 → 132 | 33 | 0 | 743,872 → 767,104 B |
| fixed-2 expert-major | 22.5699 | 31.1475 | +38.005% | 480 → 120 | 30 | 0 | 731,840 → 752,960 B |
| adaptive expert-major | 23.0008 | 28.0281 | +21.857% | 528 → 132 | 33 | 0 | 743,872 → 767,104 B |

Natural greedy measures 1171.6814 decode tok/s, 1072.0700 prefill tok/s, 81.0359 ms TTFT, and 237,002,752-byte peak RSS on the CPU tiny graph. It is an environment anchor, not a paired CUDA-draft baseline. Grid draft peak VRAM is 671,744 bytes fixed and 675,008 bytes adaptive; grouped values are 688,608 and 691,872 bytes. Fixed acceptance is 1.0 and adaptive acceptance is 0.5. Average target Top-K is 16 and draft Top-K is 4.

Each grid call uses four MoE kernel launches. Descriptor H2D is 5,760 bytes across 30 fixed calls and 6,336 bytes across 33 adaptive calls. Exact resident weight H2D remains 644,160 bytes fixed and 647,424 bytes adaptive in both grouped and grid rows. The launch reduction is therefore not a weight-traffic reduction; activation and descriptor bytes increase total H2D slightly. Physical NVMe GB/token, GPU utilization, GPU memory bandwidth, PCIe counters, and I/O stall time are not measured. Reader bytes are logical runtime reads and WSL2 is not native-Linux performance authority.

Raw JSON/CSV and summaries are under `results/b0021-cuda-aurora-grid-wsl/`. Runner SHA-256 is `0497a53a6ba6045d911dbb685e7155ee698a7e83946059e7b611202918bd4aa8`; canonical aggregate SHA-256 is `a628064544cdae0d06af7177539bc253f264946840f59651508121146af2edda`; summary JSON/CSV SHA-256 is `8586f6a1939dfe209813c504727c0952149730a757eb5600b05fb6a02021877f` / `b87b26c1403a2f3d30fa46b5550837f1f72db18a1d162710a907419da8d64401`.

Fresh verification passes CPU CTest 14/14 with pytest 290 passed/55 skipped, liburing/direct CTest 15/15 with pytest 296 passed/49 skipped, ASan/UBSan liburing CTest 15/15, and CUDA CTest 24/24 with pytest 336 passed/9 skipped. Compute Sanitizer reports `ERROR SUMMARY: 0 errors` for the direct expert-grid test and 4x4 benchmark.

Public push and pull-request correctness runs `31351465644` and `31351486146` passed. PR #29 was rebase-merged at public integration head `90b20c87`, and post-merge `main` correctness run `31351649761` passed.

## B-0022 — Milestone 21 resident CUDA MoE layer

- Date: 2026-08-10.
- Commit: runner and schema head `6921ee1`; committed evidence head `78a7022`.
- Hardware: AMD Ryzen 7 9800X3D and NVIDIA GeForce RTX 5080 16,303 MiB under WSL2 Ubuntu 24.04.4, CUDA 13.3 native `sm_120`.
- Model/checkpoint: runner-generated synthetic natural Top-16 K3X artifact, SHA-256 `af52a83307f0c0ee9caf8d2e5662de45c3757cd7a294a2465aaa1146854f15b4`.
- Mode: CPU natural Top-16 target; persistent exact resident CUDA Top-4 draft; split resident-grid versus complete resident MoE-layer boundary; fixed block-2 and adaptive policies; token-major and CPU expert-major target verification; 4 prompt tokens; 6 generated tokens; 3 warmups and 20 measured samples.
- Quality scope: exact synthetic target token, final KDA/MLA state, committed route, proposal, acceptance, Reader-byte, and selected-expert parity. Coding/agentic quality and full-model quality are unmeasured.

| Pair | Grid decode tok/s | Layer decode tok/s | Paired delta | Sync grid → layer | Activation H2D grid → layer | D2H grid → layer | Total H2D grid → layer |
|---|---:|---:|---:|---:|---:|---:|---:|
| fixed-2 token-major | 27.3311 | 28.8668 | +5.619% | 470 → 380 | 108,800 → 93,920 B | 102,880 → 76,000 B | 752,960 → 738,464 B |
| adaptive token-major | 28.2119 | 27.4351 | -2.753% | 517 → 418 | 119,680 → 103,312 B | 113,168 → 83,600 B | 767,104 → 751,120 B |
| fixed-2 expert-major | 28.1488 | 27.8065 | -1.216% | 470 → 380 | 108,800 → 93,920 B | 102,880 → 76,000 B | 752,960 → 738,464 B |
| adaptive expert-major | 26.9613 | 28.0216 | +3.933% | 517 → 418 | 119,680 → 103,312 B | 113,168 → 83,600 B | 767,104 → 751,120 B |

Every fixed layer row executes 30 complete layer calls and every adaptive layer row executes 33. The measured synchronization deltas are exactly three per successful call: 90 and 99. Each call records thirteen layer operations, for 390 and 429 layer launches, with zero layer or grid fallback. Contribution H2D is 480 bytes fixed and 528 bytes adaptive.

The layer path admits one routed RMSNorm vector absent from the split CPU norm path. Weight H2D and resident-weight occupancy both rise by exactly 384 bytes in every pair, from 644,160 to 644,544 bytes fixed and 647,424 to 647,808 bytes adaptive. Activation savings are 14,880 or 16,368 bytes, so total H2D still falls by 14,496 or 15,984 bytes. D2H falls by 26,880 or 29,568 bytes. This validates the physical accounting correction in D-047 rather than hiding the new norm upload.

Natural greedy measures 1172.6645 decode tok/s, 854.6167 prefill tok/s, 84.7403 ms TTFT, and 236,945,408-byte peak RSS on the CPU tiny graph. Layer draft peak VRAM is 675,344 bytes fixed and 678,608 bytes adaptive. Fixed acceptance is 1.0 and adaptive acceptance is 0.5. These are process/runtime counters under WSL2, not native-Linux full-model or physical PCIe/NVMe measurements. GPU utilization, GPU memory bandwidth, physical NVMe GB/token, and coding quality remain unmeasured.

Raw JSON/CSV and summaries are under `results/b0022-cuda-aurora-moe-layer-wsl/`. Runner SHA-256 is `745fde3f062bcc886997ec1811e636dd1e5a19d3642c04869971855d441bce16`; canonical aggregate SHA-256 is `404064335dcf2adf6c580b7f99812627cb357693aa6d9bcd414f4d51b2b19a9b`; summary JSON/CSV SHA-256 is `46db38afb586a0a3807f98a69ee387bec377ba2d8b0fd9036af18b6ee5dbf8df` / `5852eeb4ee0c792524f31df92566bc8052212877a1be92f5367872a34b7e5e4d`.

The exact layer boundary remains opt-in. Two paired decode rows improve and two regress slightly, so the traffic and synchronization result does not justify a default change. The next execution-boundary decision must use representative dimensions and native Linux; CUDA Graph caching is not selected from this synthetic result alone.

Fresh verification passes CPU CTest 14/14 with pytest 295 passed/56 skipped, liburing/direct CTest 15/15 with pytest 301 passed/50 skipped, ASan/UBSan liburing CTest 15/15, and CUDA CTest 26/26 with pytest 341 passed/10 skipped. Compute Sanitizer reports `ERROR SUMMARY: 0 errors` for both the low-level MoE-layer operations and the complete resident layer. Focused B-0022 evidence plus CLI ownership verification passes 103/33.

Public branch and pull-request correctness runs `31355460022` and `31355471896` passed, as did CodeQL run `31355471922`. PR #31 was rebase-merged at public integration head `97eb3e4e`, and post-merge `main` correctness run `31355678835` passed.

## B-0023 — Milestone 22 released-dimension resident MoE-layer boundary

- Date: 2026-08-10.
- Commit: initial runner head `2ca2d66`; oracle-lifetime correction and committed evidence head `d0035ea`.
- Hardware: AMD Ryzen 7 9800X3D and NVIDIA GeForce RTX 5080 16,303 MiB under WSL2 Ubuntu 24.04.4, driver 591.86, CUDA 13.3.1 native `sm_120`.
- Model/checkpoint: one non-executable released-dimension storage expert reused under unique logical IDs, K3X SHA-256 `e087ff78284e99760a7d113cf744562878537a6379e7a63be95585eec8b9f1be`; deterministic FP32 released-size dense/vector fixture.
- Mode: `cuda-custom + fp32 + reused + resident + resident-grid + synchronous + fusion-none`, split `ffn-block` versus complete `moe-layer`, 1 GiB hard capacity, 3 warmups and 20 measured iterations.
- Context length: not applicable; direct layer-boundary invocation with `routing_semantics=false`.
- Decode tok/s, prefill tok/s, TTFT: not measured and not emitted.
- System RAM, physical NVMe GB/token, GPU utilization, GPU memory bandwidth, and cache hit rate: not measured.
- Average Top-K, speculative acceptance, unique experts per verification block, and cold rescue count: not applicable to this direct boundary.
- Quality: separate split CUDA oracle; maximum absolute error 0 in all rows; no full-model or coding-quality claim.
- Enabled optimizations: exact FP32 dense residency, native MXFP4 expert residency, resident expert grid, allocation reuse, synchronous transfer, and complete resident layer only in the layer rows.

| Experts | Split median | Layer median | Layer delta | Sync split → layer | Activation H2D split → layer | D2H split → layer | Resident bytes split → layer | Peak VRAM split → layer |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1,227,823 ns | 20,487,750 ns | +1568.624% | 80 → 20 | 1,720,320 → 574,480 B | 1,720,320 → 573,440 B | 487,309,312 → 487,323,648 B | 575,555,632 → 575,555,632 B |
| 4 | 2,370,565 ns | 20,953,689 ns | +783.911% | 80 → 20 | 1,720,320 → 577,600 B | 2,580,480 → 573,440 B | 539,951,104 → 539,965,440 B | 628,336,832 → 628,336,832 B |
| 16 | 5,681,151 ns | 24,422,315 ns | +329.883% | 80 → 20 | 1,720,320 → 590,080 B | 6,021,120 → 573,440 B | 750,518,272 → 750,532,608 B | 839,518,976 → 839,518,976 B |

Every selected row records zero warm weight H2D, zero weight-cache bypass, zero resident-grid fallback, and zero resident-layer fallback. Split rows execute 20 grid calls, 80 grid launches, four synchronizations per iteration, and no complete-layer calls. Layer rows execute 20 grid calls, 80 grid launches, 20 complete-layer calls, 260 layer launches, and one synchronization per iteration. The layer-minus-split cold-weight and resident-weight deltas are both exactly 14,336 bytes at every expert count.

The complete layer reduces measured activation H2D by 1,145,840/1,142,720/1,130,240 bytes and D2H by 1,146,880/2,007,040/5,447,680 bytes. Median latency nevertheless rises sharply. Aggregate kernel time across 20 iterations is 15,121,920→22,970,976 ns at one expert, 24,506,720→27,692,480 ns at four, and 58,395,968→61,886,528 ns at sixteen. The split oracle is released before the selected backend is constructed; `peak_vram_bytes` is therefore the maximum of two sequential phases, not an overlapping sum. The larger wall/kernel gap is consistent with, but does not independently prove, the per-call full dense-weight finiteness scan identified in source review.

Raw JSON and summaries are under `results/b0023-cuda-released-moe-layer-wsl/`. Runner SHA-256 is `3c2695fc31adc01040a992098180a83cb58947d85858412eab62282b66ec6baf`; canonical aggregate SHA-256 is `88c51b6a58340a4325b2b09faa0fb63d1bc5f4439542261383f6070dbe526ade`; summary JSON/CSV SHA-256 is `d67fe356735ddc38e238a9e86e7f46ec3729ef24bc27d1f286aacaaabf0af954` / `4a95494381c87862aa6933811248f1fd2ff35a28d88e576917da57e50e87d621`. Committed-evidence tests recompute all six raw digests, the aggregate, the summary CSV digest, every pair gate, and every reported percentage, including oracle peak-VRAM coverage.

Fresh pre-review verification passes CPU CTest 14/14 with pytest 305 passed/67 skipped, liburing/direct CTest 15/15 with pytest 307 passed/65 skipped, ASan/UBSan liburing CTest 15/15, and CUDA CTest 26/26 with pytest 362 passed/10 skipped. After the oracle-lifetime correction, CUDA CTest passes 26/26, focused live/evidence pytest passes 22/22, and the released one-expert complete-layer Compute Sanitizer run reports `ERROR SUMMARY: 0 errors`.

Public branch and pull-request correctness runs `31358991710` and `31359003481` passed, as did CodeQL run `31359003436`. PR #36 was rebase-merged at public integration head `e4820a18`, followed by successful post-merge `main` correctness run `31359158926` and CodeQL run `31359158878`.

The result rejects a default change and does not select CUDA Graphs. The next benchmark must preserve immutable-tensor validation while removing its repeated hot-path scan, then distinguish host validation, launch, synchronization, and kernel time before broadening the execution boundary.

## B-0024 — Milestone 23 admission validation attribution

- Date: 2026-08-10.
- Commit: implementation lineage `3560e9a` through profiler-off physical telemetry correction `7931d66`; committed evidence head `105a860`.
- Hardware: AMD Ryzen 7 9800X3D and NVIDIA GeForce RTX 5080 16,303 MiB under WSL2 Ubuntu 24.04.4, CUDA 13.3 native `sm_120`.
- Model/checkpoint: the existing non-executable released expert fixture, SHA-256 `e087ff78284e99760a7d113cf744562878537a6379e7a63be95585eec8b9f1be`, plus deterministic released-size FP32 dense/vector tensors.
- Mode: split `ffn-block + per-call`, complete `moe-layer + per-call`, and complete `moe-layer + admission`; profiler independently off/on; 1/4/16 repeated-view experts; 1 GiB hard capacity; 3 warmups and 20 measured iterations.
- Context length, decode tok/s, prefill tok/s, TTFT, average Top-K, speculative acceptance, and cold rescue: not applicable and not emitted because this directly invokes one released-dimension layer with `routing_semantics=false`.
- System RAM, physical NVMe GB/token, physical H2D/PCIe counters, GPU utilization, and GPU memory bandwidth: not measured.
- Quality: maximum absolute error 0 against the separately scoped split CUDA oracle in all 18 rows. Coding/agentic and full-model quality are unmeasured.

| Experts | Per-call median, profiler off | Admission median, profiler off | Paired change | Per-call warm validation bytes | Admission warm scans / hits |
|---:|---:|---:|---:|---:|---:|
| 1 | 19,570,019 ns | 1,246,879 ns | -93.629% | 9,395,527,680 B | 0 / 120 |
| 4 | 20,728,924 ns | 1,939,696 ns | -90.643% | 9,395,527,680 B | 0 / 120 |
| 16 | 24,518,749 ns | 5,220,560 ns | -78.708% | 9,395,527,680 B | 0 / 120 |

Each admission layer row records six cold scans and 469,776,384 cold validation bytes, followed by zero measured warm scan bytes. Every per-call layer row records 120 scans. All rows preserve zero warm weight H2D, zero cache bypass, zero grid/layer fallback, exact 80 versus 20 synchronization counts, exact launch counts, and profiler on/off parity for numerical and non-profiler physical counters. `kernel_nanoseconds` is JSON `null` when profiler collection is off.

Raw JSON and summaries are under `results/b0024-cuda-admission-validation-wsl/`. Runner SHA-256 is `952fd739d7654b8a4685e62c045d5727955b792244519dd09667f1e7acff441b`; canonical aggregate SHA-256 is `0747d22f6e409c81ab788cb936b65c9a75a13ed2f37255d6f468c7899f3026d1`; summary JSON/CSV SHA-256 is `4c4af570602d3322120ac445ad881c00f96ac0f63f3a39dc45a8032620cc8c82` / `a49614469b71f01b9d86ff93bf996937cb0d9e93ed272044a136491d6575b68f`. Committed-evidence verification recomputes all 18 raw hashes, exact case order, aggregate and summary CSV hashes, validation formulas, nullable-kernel contract, LF-only CSV, and reported admission percentage deltas.

Fresh verification passes CPU CTest 14/14 with pytest 311 passed/68 skipped, liburing/direct CTest 15/15 with pytest 313 passed/66 skipped, ASan/UBSan CTest 15/15, and CUDA CTest 26/26 with pytest 369 passed/10 skipped. Compute Sanitizer reports `ERROR SUMMARY: 0 errors` for both `test_cuda_moe_layer` and a released one-expert admission benchmark invocation.

Public push and pull-request correctness runs `31363433423` and `31363437230` passed, as did pull-request CodeQL run `31363437226`. PR #38 was rebase-merged at public integration head `e24cac2`, followed by successful post-merge `main` correctness run `31363673811` and CodeQL run `31363673857`.

The measurement establishes repeated host validation as the dominant B-0023 wall term and accepts admission validation as an exact opt-in path. It does not promote the general default because unchanged-pointer in-place mutation is outside the admission contract, and it does not select CUDA Graphs without ordered routed-set reuse and bounded graph-cache evidence.

## B-0025 — Milestone 24 bounded CUDA Graph cache attribution

- Date: 2026-08-10.
- Commit: implementation and canonical runner `77b461a`; committed evidence and verifier `708e555`.
- Hardware: AMD Ryzen 7 9800X3D and NVIDIA GeForce RTX 5080 16,303 MiB under WSL2 Ubuntu 24.04.4, CUDA 13.3 native `sm_120`.
- Model/checkpoint: the existing non-executable released expert fixture, SHA-256 `e087ff78284e99760a7d113cf744562878537a6379e7a63be95585eec8b9f1be`, reused as four resident logical views with five deterministic ordered permutations; released FP32 dense/vector tensors.
- Mode: exact `cuda-custom + fp32 + reused + resident + resident-grid + moe-layer + synchronous + fusion-none + admission`; direct `disabled`, whole-executable `update-1`, and ordered-set `cache-1|2|4`; 3 warmups and 20 measured iterations.
- Context length, decode tok/s, prefill tok/s, TTFT, average Top-K, speculative acceptance, and cold rescue: not applicable and not emitted because this directly invokes one released-dimension layer with `routing_semantics=false`.
- System RAM, physical NVMe GB/token, physical PCIe counters, GPU utilization, GPU memory bandwidth, and coding/agentic quality: not measured.
- Quality: maximum absolute error 0 against the separately scoped split CUDA oracle in all 15 rows.

| Trace | Mode | Median | Delta vs trace direct | Hits | Misses | Evictions | Update successes |
|---|---|---:|---:|---:|---:|---:|---:|
| stable-1 | disabled | 1,997,778 ns | 0% | 0 | 0 | 0 | 0 |
| stable-1 | update-1 | 1,967,566 ns | -1.512% | 0 | 0 | 0 | 20 |
| stable-1 | cache-1 | 1,964,179 ns | -1.682% | 20 | 0 | 0 | 0 |
| stable-1 | cache-2 | 2,010,994 ns | +0.662% | 20 | 0 | 0 | 0 |
| stable-1 | cache-4 | 2,035,124 ns | +1.869% | 20 | 0 | 0 | 0 |
| alternating-2 | disabled | 2,067,525 ns | 0% | 0 | 0 | 0 | 0 |
| alternating-2 | update-1 | 1,976,416 ns | -4.407% | 0 | 0 | 0 | 20 |
| alternating-2 | cache-1 | 2,159,855 ns | +4.466% | 0 | 20 | 20 | 0 |
| alternating-2 | cache-2 | 2,034,800 ns | -1.583% | 20 | 0 | 0 | 0 |
| alternating-2 | cache-4 | 2,005,247 ns | -3.012% | 20 | 0 | 0 | 0 |
| rotating-5 | disabled | 1,960,733 ns | 0% | 0 | 0 | 0 | 0 |
| rotating-5 | update-1 | 2,093,936 ns | +6.794% | 0 | 0 | 0 | 20 |
| rotating-5 | cache-1 | 2,099,426 ns | +7.074% | 0 | 20 | 20 | 0 |
| rotating-5 | cache-2 | 2,080,154 ns | +6.091% | 0 | 20 | 20 | 0 |
| rotating-5 | cache-4 | 2,187,663 ns | +11.574% | 0 | 20 | 20 | 0 |

Every row records zero measured warm weight H2D, zero cache bypass, zero grid/layer fallback, 577,600 activation-H2D bytes, 573,440 D2H bytes, 20 synchronizations, 20 complete-layer calls, 80 expert-grid launches, and 260 logical MoE-layer kernel launches. Ordered permutation changes graph identity without changing resident tensor IDs, so cache churn is not confounded by weight admission. Update succeeds on all 20 measured calls after the pre-measurement executable is established.

Raw JSON/CSV and summaries are under `results/b0025-cuda-graph-cache-wsl/`. Runner SHA-256 is `60ee26df29b05a9f6638323477abd7c277f22802fe67866bdb4b647ec7f85c21`; canonical aggregate SHA-256 is `f65ab85eb8b750a69330f10146f50f1644b5205e60c6752abcaf0d1deffa3bd8`; summary JSON/CSV SHA-256 is `da6a0e8cd336c131231d42597971e0c12a803dbad5ca05ca7f826310af6cad99` / `a53482f9386eaeb292c263f921c165fd8fb931aa156f4ebd2ecee2f74672d2ad`. The independent verifier recomputes every raw JSON/CSV digest, payload, graph formula, aggregate, summary CSV digest, case order, and artifact/runner digest. Staged Git blobs are separately checked for LF-only digest parity.

Fresh verification passes CPU CTest 15/15 with pytest 332 passed/70 skipped, liburing/direct CTest 16/16 with pytest 334 passed/68 skipped, ASan/UBSan CTest 16/16, and CUDA CTest 27/27 with pytest 386 passed/16 skipped. Compute Sanitizer reports `ERROR SUMMARY: 0 errors` for both a stable cache hit and an alternating capacity-one miss/eviction.

PR #40 was rebase-merged at public integration head `13a403f`. Branch and pull-request correctness runs `31371133295`/`31371136825` passed, as did pull-request CodeQL `31371136804`. Post-merge `main` correctness `31371387067` and CodeQL `31371387081` also passed.

The result retains direct execution as the default. Stable and alternating deltas are small and mixed, while rotating churn is consistently slower. Real K3 routed-set reuse, native-Linux end-to-end token timing, dynamic residency interaction, utilization, physical traffic, and quality remain unmeasured.

## B-0026 — Milestone 25 converter integrity audit

- Date: 2026-08-10.
- Commit: public canonical runner and implementation `35e4419`; committed evidence `cce5223`.
- Hardware: AMD Ryzen 7 9800X3D host under WSL2 Ubuntu 24.04.4. The RTX 5080 was present but unused. The audit used temporary filesystem storage and did not measure the P44 Pro as a physical device.
- Model/checkpoint: deterministic synthetic K3-compatible source checkpoint and K3X v1 output; no official Kimi K3 weight.
- Mode: `k3x-converter-integrity-audit-v1`, 257-byte source chunks, interruption after two committed extents, and an optional 8,192-byte orphan suffix.
- Context length, decode tok/s, prefill tok/s, TTFT, VRAM, H2D, cache hit rate, average Top-K, speculative acceptance, cold rescue, GPU utilization, GPU bandwidth, physical NVMe GB/token, and quality: not applicable or not measured.
- System RAM: peak RSS not measured. Output bytes are file size, not resident-memory use.

| Scenario | Wall time | Maximum source read | Output | Reused extents | Committed prefix | Orphan suffix | Reader valid |
|---|---:|---:|---:|---:|---:|---:|---:|
| fresh | 804,991,621 ns | 257 B | 1,421,568 B | 0 | 0 B | 0 B | yes |
| resume-clean | 800,116,522 ns | 257 B | 1,421,568 B | 2 | 20,736 B | 0 B | yes |
| resume-orphan | 887,550,657 ns | 257 B | 1,421,568 B | 2 | 20,736 B | 8,192 B | yes |

Artifact SHA-256 values for fresh, clean-resume, and orphan-resume are `7abe2955c1433b1b6087308b37eefaf6edbdf9a70a6b04c6fdf071e8bd209998`, `4597ed5559cf8a4c3f9da9e875dbf00d903413b1ef553a4006c76f1c80839e77`, and `22b0d40dc643bd48ca81e8e00da2a3d3d7fef2b32a130e26a0bc6c94f31d8ae2`. Root digests are `ba8fdade8492068af56a4ac51b6e7e4cb4e45e13b107881cec782b1e37db112d`, `c4707c4644eb66f1b4c0bc22a3fce22bd6fb29f00104ca1b641fde9ac06c1751`, and `71122ea28bdd8abfbf27deb2b589a85e72c0a95a7abe5931da02ecfc8433c751`.

Runner SHA-256 is `d292991ada21dd305078d2fe90116450d7a5184606962891bc1c383c47487ca0`; canonical aggregate SHA-256 is `4181e012dc0ccc1570f5ca18336ee3037327b32da63b8128bcc5423c0191100c`; summary JSON/CSV SHA-256 is `f78de6ef9bb3b47d1cb3d56af1969d1d3d465025a21ee0b25f2d97df27e38116` / `c98f12b39fdc7f76bd4cd824cb5fc9da9b44208dd27a954c958f6e3bf3b6ea6d`. The verifier checks schema, exact scenario order, aggregate and file digests, JSON/CSV parity, LF-only evidence, bounded reads, Reader validity, resume reuse, and absence of token/GPU/quality fields.

B-0026 is a correctness and recovery audit, not a performance forecast. It permits bounded official-source discovery to begin; it does not establish publisher authenticity, full-checkpoint conversion, cloud execution, token throughput, or a production memory ceiling.

## B-0027 — Milestone 26 official bounded expert conversion

- Date: 2026-08-10.
- Commit: discovery/transport implementation through `5b893a0`; live evidence recorded immediately afterward on the same code; final path and verifier review fixes through `fb7fb49`.
- Hardware: AMD Ryzen 7 9800X3D host under WSL2 Ubuntu 24.04.4. RTX 5080 was present but the B-0027 conversion did not execute GPU kernels.
- Model/checkpoint: official public `moonshotai/Kimi-K3` snapshot at `9f62e4e9fffbd0a83ddd60e1c209d828994b3569`; layer 1, expert 0 only.
- Mode: strict dry-run followed by one `transport-pinned-range` materialization. No complete shard, full checkpoint, or paid cloud resource.
- Context length, decode tok/s, prefill tok/s, TTFT, VRAM, H2D, cache hit rate, average Top-K, speculative acceptance, quality, GPU utilization, GPU bandwidth, and physical NVMe GB/token: not applicable or not measured.
- System RAM: peak RSS not measured. Response and artifact byte counts are not resident-memory measurements.

| Metric | Measured value |
|---|---:|
| Repository files / bytes | 118 / 1,560,998,984,390 |
| Index bytes / tensors / shards | 59,764,096 / 497,220 / 96 |
| Declared tensor bytes | 1,560,860,324,864 |
| HTTP requests | 11 |
| Metadata bytes | 59,799,719 |
| Safetensors length + header bytes | 818,704 |
| Tensor payload bytes | 17,547,264 |
| Maximum response bytes | 59,764,096 |
| Wall time | 14.972839499 s |
| Reader valid / optional feature | yes / `OPTIONAL_STORAGE_FIXTURE` |

The payload SHA-256 is `1d925fa7bd91331511783b7423204d20b6337cd672b403fd017b7b42f421c36c`; content-addressed microshard SHA-256 is `ed3f07d595f37d90b1688de21ba0cdc012ee92c67dd92c460c0c73b2ef374a34`; K3X root SHA-256 is `d585d283325e13e1316a0194c2d6274dd89ef75a28b96b02f02733290b7658be`. The six committed tensor digests are retained in the summary JSON.

Summary JSON/CSV SHA-256 is `57ebd9d85ed3ae55a4e2ab01f023bc451faf02cd7b6e69f478d11e3ea73e982a` / `7c6238b466aca5c4eb52e83c1ba17139af15f4634074ee23a970f1ade992bdd6`. The strict verifier checks repository/revision/snapshot/config/index/shard/expert identities, JSON/CSV parity, canonical record and CSV digests, exact deterministic range/traffic values, six tensor/artifact hashes, non-executable optional identity, provenance level, and absence of token/GPU/NVMe/quality claims. Nine consistently rehashed identity mutations are rejected.

Fresh final-review verification passes CPU CTest 15/15 with pytest 462 passed/70 skipped, liburing/direct CTest 16/16 with pytest 464 passed/68 skipped, ASan/UBSan CTest 16/16, and CUDA CTest 27/27 with pytest 516 passed/16 skipped. The unchanged released MoE-layer Compute Sanitizer path reports `ERROR SUMMARY: 0 errors`. The actual real-weight K3X exits 4 with `NON_EXECUTABLE_ARTIFACT` before graph execution.

The next bottleneck is not source compatibility or bounded conversion. It is the unimplemented dependency-closed real CUDA layer invocation. B-0027 makes no TPS or full-model performance claim.

PR #44 was rebase-merged at public implementation head `5b6345db`. Both branch and pull-request correctness passed, all pull-request CodeQL checks passed, and post-merge `main` correctness `31386873905` and CodeQL `31386873928` succeeded. Node.js 20 and CodeQL Action v3 deprecation annotations did not change the successful conclusions and remain separate workflow maintenance.

## B-0028 — Milestone 27 official expert CUDA execution

- Date: 2026-08-11.
- Commit: official identity `2e2acb9`, dedicated CUDA harness and runtime D2H correction `60f19e7`, strict runner/verifier `c39aac2`, committed evidence/documentation `e78fda0`; publicly integrated through PR #46 at `ec08b827`.
- Hardware: AMD Ryzen 7 9800X3D and NVIDIA GeForce RTX 5080 16,303 MiB under WSL2 Ubuntu 24.04.4, driver 591.86, CUDA 13.3.1 native `sm_120`.
- Model/checkpoint: official public `moonshotai/Kimi-K3` snapshot `9f62e4e9fffbd0a83ddd60e1c209d828994b3569`; layer 1, expert 0 only; ignored K3X artifact SHA-256 `e08293cd854ed11913bd8f1bc3a51d1eb577202fd5fd9b5b7e3c96ef1bccecc7` and root `d585d283325e13e1316a0194c2d6274dd89ef75a28b96b02f02733290b7658be`.
- Mode: exact native MXFP4 `cuda-custom + reused + ffn-block + synchronous + fusion-none`; transient versus exact 17,547,264-byte resident capacity; one cold call, three warmups, and twenty measured calls.
- Context length, decode tok/s, prefill tok/s, and TTFT: not applicable and not emitted because this invokes one expert FFN with `token_semantics=false` and `routing_semantics=false`.
- System RAM, physical NVMe GB/token, RAM-to-GPU GB/token, GPU utilization, GPU memory bandwidth, I/O stall time, and coding/agentic quality: not measured. H2D/D2H below are backend byte counters per twenty direct invocations, not per-token or physical-bus counters.
- Average Top-K, speculative acceptance, and cold rescue: not applicable. The invocation contains exactly one fixed expert and no router or speculative block.
- Cache hit rate: transient has no resident lookup; resident records 60 hits, zero misses, and zero bypasses during the measured interval after cold admission, or 100% of its measured tensor lookups.
- Quality: all 3,584 outputs are finite and both modes have `3.0267983675e-9` maximum absolute error against the independent portable CPU backend. Full-model and coding quality are unmeasured.

| Mode | Cold latency | Warm p05 | Warm median | Warm p95 | Kernel total | Cold weight H2D | Measured weight H2D | Activation H2D | D2H | Resident bytes | Peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Transient | 7,122,628 ns | 1,865,439 ns | 2,508,377 ns | 2,649,090 ns | 6,309,888 ns | 17,547,264 B | 350,945,280 B | 286,720 B | 286,720 B | 0 B | 5,914,624 B |
| Resident | 7,748,006 ns | 319,489 ns | 331,868 ns | 366,599 ns | 2,692,992 ns | 17,547,264 B | 0 B | 286,720 B | 286,720 B | 17,547,264 B | 23,461,888 B |

The bounded resident median is 86.77% lower than transient, or 7.56 times as fast at this one-expert boundary, while the single cold resident call is 8.78% slower. The measured result establishes exact reuse benefit only. It does not measure a natural Top-16 routed set, a shared expert, a complete MoE layer, tokens, native Linux, physical PCIe/NVMe traffic, utilization, bandwidth, or quality.

Raw JSON and summary artifacts are under `results/b0028-official-expert-cuda-wsl/`. Raw transient/resident SHA-256 values are `3b39610b5f5b6f4cfd5ec1da1bc3588e00c0af62f58438c81a7f9b3357093518` / `79c935869226108431f391bb61402e10b61a616493720bac75fc545512cc30bf`. Runner SHA-256 is `48f0f295ab7299af07f261522ffd2999814bd5967e12bfcc3e7b0b3d21b201fa`; canonical aggregate SHA-256 is `eb4580b74481855d04fdf9d3f7ed5921ea25b0e5b56408d561cd645a3ea99172`; summary JSON/CSV SHA-256 is `9c7aec65fe3f662c8a8e7ea08084e8d69901cff2707b32e10958b62439e69919` / `d339a8774283e49608393172ffd551d46692a076e00cb4d63e1e2a347ae42a91`.

Strict verification rehashes the ignored artifact and runner, checks fixed B-0027 provenance, raw payload/digests, exact case order, canonical aggregate, LF-only CSV parity, mode-specific traffic formulas, finite parity, and forbidden metric absence. Staged Git blob SHA-256 values match all four working files. A separate read-only cross-check recomputed raw-summary equality, `20 × 17,547,264` transient H2D, resident hit/residency values, activation/D2H formulas, aggregate, CSV order, and forbidden fields.

Fresh verification passes CPU CTest 16/16 with pytest 473 passed/76 skipped, liburing/direct CTest 17/17 with pytest 475 passed/74 skipped, ASan/UBSan CTest 17/17, and CUDA CTest 28/28 with pytest 531 passed/18 skipped. The actual-artifact focused suite passes 6/6, the committed B-0028 evidence suite passes 11/11, and the resident official-expert Compute Sanitizer run reports `ERROR SUMMARY: 0 errors`.

Public branch correctness `31455570571`, pull-request correctness `31455597581`, and pull-request CodeQL `31455597565` passed. PR #46 was rebase-merged at public integration head `ec08b827`; post-merge `main` correctness `31455776634` and CodeQL `31455776673` also passed.

The measured next bottleneck is no longer official single-expert compatibility. M28 must close a real MoE FFN sublayer with the official router, all 896 scores, natural Top-16 selection, exact selected routed experts, real shared expert, mixing, and residual parity before any cache-pressure, full-layer, or throughput conclusion.

## Pending benchmark gates

- Native Linux repetition of B-0002; WSL2 is the development path, not final performance authority.
- Native-Linux repetition of B-0004/B-0005/B-0006 and a larger KDA/MLA or decoder subgraph boundary.
- Native-Linux repetition of B-0008 with disclosed warm/cold preparation before selecting an L2 default.
- Native-Linux repetition of B-0009 with representative multi-expert pressure and controlled warm/cold preparation before selecting any deadline policy.
- Native-Linux repetition of B-0010 with a representative routing trace, full-size experts, and controlled warm/cold preparation before selecting any cache policy.
- Native-Linux repetition of B-0011 with repository-duration sessions and controlled helpful, stale, and adversarial priors before selecting any profile policy.
- Native-Linux repetition of B-0016 with physical NVMe accounting, GPU utilization, memory bandwidth, multi-expert/full-layer groups, and representative acceptance distributions before any speculative default claim.
- Representative native-Linux persistent AURORA measurement with physical I/O, realistic acceptance, coding quality, and resident-expert pressure before any self-speculative default claim.
- Persistent multi-token/multi-expert CUDA execution after B-0020 removes most repeated weight H2D; keep dynamic eviction/prediction and reduced precision as separate policy and quality axes.
- Native-Linux and representative-dimension repetition of B-0022 before selecting a MoE-layer boundary or CUDA Graph strategy as a default.
- Real K3 ordered routed-set reuse, native-Linux end-to-end graph timing, and dynamic residency interaction before reconsidering CUDA Graphs or a larger device-resident token boundary; B-0025 keeps direct execution as the default.
- Real router-selected Top-16 plus shared-expert FFN sublayer execution before claiming cache pressure, locality, or complete-layer behavior; B-0028 proves only one fixed expert.
- Complete-object or authenticated-chunk source verification before production conversion claims; B-0027/B-0028 retain explicit `transport-pinned-range` provenance.
- Native-Linux repetition of B-0028 before using its one-expert residency latency ratio in any runtime policy.
- L2 runtime physical NVMe, utilization, memory-bandwidth, and storage I/O-stall counters.

## Milestone 28 Task 2 verification — bounded official MoE materializer

- Date: 2026-08-11.
- Commit: `0b0c944`.
- Hardware/model: controlled local fake-range fixtures under WSL2; no official tensor payload and no GPU execution.
- Mode: two-phase always-active route derivation followed by exact selected-union planning, content-addressed resume/reuse, physical source assembly, and explicit CLI orchestration.
- Result: 27 focused materializer/CLI tests passed. The combined official-source, transport, safetensors, converter-resume, and source-manifest regression matrix passed 149 tests in 13.24 seconds. Python compile validation and `git diff --check` also passed.
- Traffic contract: every response is capped at 8 MiB; fresh, resumed, and reused content objects report actual response bytes independently from logical source-object bytes. No live response-byte total is recorded because no official M28 payload was requested.
- Decode tok/s, prefill tok/s, TTFT, VRAM, system RAM, physical NVMe GB/token, H2D GB/token, cache hit rate, average Top-K, speculative acceptance, quality, GPU utilization, and GPU bandwidth: not measured and not applicable to this verification-only unit.
- Interpretation: this entry records a correctness/recovery gate, not B-0029 and not a performance benchmark. The next measurable boundary is the portable official BF16/MXFP4 oracle followed by the native CUDA implementation over the bounded ignored fixture.

## Milestone 28 Task 3 verification — portable official MoE oracle

- Date: 2026-08-11.
- Commit: `8a13cf5`.
- Hardware/model: tiny dimension-driven literal BF16/MXFP4 fixture on the AMD Ryzen 7 9800X3D host under WSL2; no official tensor payload and no CUDA execution.
- Mode: pure Attention Residual, post-RMSNorm, all-score sigmoid routing, correction-only Top-K selection, exact MXFP4 experts, FP32 weighted accumulation, routed normalization/up-projection, shared SiTU-GLU, combination, and prefix addition.
- Correctness result: every named C++ intermediate boundary matches independently computed PyTorch values at `1e-6` absolute tolerance. Scalar BF16 decoding covers positive/signed zero, a finite normal, infinity, and NaN. Malformed dimensions, route cardinality, duplicate IDs, missing experts, non-finite contributions, and non-unit contribution mass fail closed.
- Verification result: CPU CTest 17/17 passed. The complete `tests/python/test_cpp_parity.py` run passed 113 tests with 32 capability skips in 21.75 seconds. `git diff --check` passed.
- Decode tok/s, prefill tok/s, TTFT, VRAM, system RAM, physical NVMe GB/token, H2D GB/token, cache hit rate, average Top-K, speculative acceptance, quality, GPU utilization, and GPU bandwidth: not measured and not applicable to this tiny correctness oracle.
- Interpretation: this is the independent CPU authority for Task 4 CUDA parity. It is not B-0029 and cannot support a full-size latency, traffic, cache, token, or quality claim.

## Milestone 28 Task 4 verification — native BF16 CUDA official MoE boundary

- Date: 2026-08-11.
- Commit: `bb634e1`.
- Hardware/model: NVIDIA GeForce RTX 5080 under WSL2 with a tiny dimension-driven literal BF16/MXFP4 fixture; no official M28 tensor payload.
- Mode: dedicated opt-in transient and bounded exact resident CUDA boundary over routed down, two selected expert FFNs, ordered weighted mix, routed norm/up, shared SiTU-GLU, combination, and prefix addition.
- Correctness result: both modes match the Task 3 portable oracle within `2e-2`, retain exact selected-expert order, leave caller buffers unchanged, and reject tensor-ID aliasing, duplicate expert IDs, contribution-count mismatch, and insufficient resident capacity.
- Traffic result: each successful call performs one final output-vector D2H. A second resident call adds zero weight H2D, increases resident hits, and retains nonzero resident bytes. Raw BF16 and native MXFP4 bytes are admitted without a host FP32 weight copy.
- Verification result: CPU CTest 17/17 and CUDA CTest 30/30 passed. `/usr/local/cuda-13.3/bin/compute-sanitizer --tool memcheck --error-exitcode=99 ./build-cuda/test_cuda_official_moe` reported `ERROR SUMMARY: 0 errors`. `git diff --check` passed before commit.
- Decode tok/s, prefill tok/s, TTFT, latency distribution, VRAM peak, system RAM, physical NVMe GB/token, physical H2D GB/token, average Top-K, speculative acceptance, quality, GPU utilization, and GPU bandwidth: not measured and not applicable to this tiny correctness gate.
- Interpretation: this verifies the CUDA contract needed by the pinned harness. It is not B-0029 and does not establish official full-size performance or a runtime default.

## Milestone 28 Task 5 verification — pinned official MoE harness

- Date: 2026-08-11.
- Commits: `a109409`, `bdab0da`.
- Hardware/model: WSL2 synthetic CLI/identity fixtures plus the tiny CUDA boundary; no official M28 multi-expert payload.
- Mode: strict manifest parsing and fixed identity, checksum Reader/root binding, released tensor metadata/order checks, route/contribution recomputation, portable oracle preparation, and A/B/alternating transient/resident execution schema.
- Verification result: `tests/python/test_cuda_official_moe.py` passes 18 tests with 3 ignored-real-fixture skips. CUDA CTest passes 30/30. Focused Compute Sanitizer reports `ERROR SUMMARY: 0 errors`. The harness source also compiles with `-Wall -Wextra -Wpedantic -Werror`.
- Decode tok/s, prefill tok/s, TTFT, official latency, VRAM, system RAM, physical NVMe GB/token, physical H2D GB/token, cache hit rate, quality, GPU utilization, and GPU bandwidth: not measured.
- Interpretation: only the harness and fail-closed gates are verified. The three skipped smoke cases must run on the bounded ignored fixture before B-0029 or any official execution claim.

## Milestone 28 Task 6 verification — B-0029 evidence tooling

- Date: 2026-08-11.
- Commit: `ba3a0d2`.
- Hardware/model: controlled fake runner records plus the synthetic harness suite; no official M28 multi-expert payload.
- Mode: fixed A transient, A resident, and alternating resident orchestration with strict raw/summary/CSV schema, traffic, parity, digest, and case-order verification.
- Verification result: combined runner/harness pytest passes 28 tests with 3 ignored-real-fixture skips. Python compile validation and `git diff --check` pass.
- At this tooling-only commit, all performance and quality fields were not measured and no B-0029 rows existed yet.
- Interpretation: this historical entry records evidence-pipeline correctness before the formal run documented below.

## B-0029 — Official layer-1 MoE FFN sublayer

- Date: 2026-08-11.
- Evidence commit: `bf147fa`; final verification fix: `bdfc0b6`.
- Hardware: AMD Ryzen 7 9800X3D, NVIDIA GeForce RTX 5080 16,303 MiB, driver 591.86, CUDA 13.3.1, WSL2 Ubuntu 24.04.4 on Windows 11.
- Model/checkpoint: bounded non-executable layer-1 artifact from public `moonshotai/Kimi-K3` commit `9f62e4e9fffbd0a83ddd60e1c209d828994b3569`; two deterministic natural Top-16 routes with a 32-expert union; K3X root `1287d84bbfa02e849ab786808107fbfbfe14459477bf79e3048b2ebb6bdff288`.
- Mode: byte-native official BF16 trunk/shared tensors, native MXFP4 experts, FP32 accumulation, exact natural routing, 3 warmups, 20 measured iterations, one subprocess per row.

| Case | Median | p05 | p95 | Kernel total | Weight H2D | Activation H2D | D2H | Resident weights | Hits | Peak VRAM | Max error |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| A transient | 97,095,781 ns/call | 89,552,049 ns | 104,365,582 ns | 147,674,816 ns | 12,955,299,840 B | 1,163,520 B | 573,440 B | 0 B | 0 | 648,845,120 B | 0 |
| A resident | 10,153,939 ns/call | 9,856,963 ns | 10,613,355 ns | 147,954,240 ns | 0 B | 1,163,520 B | 573,440 B | 647,764,992 B | 1,080 | 648,845,120 B | 0 |
| Alternating resident | 20,201,466 ns/A+B sequence | 19,769,028 ns | 20,888,597 ns | 294,440,320 ns | 0 B | 2,327,040 B | 1,146,880 B | 928,521,216 B | 2,160 | 929,601,344 B | 0.00048828125 |

- A transient weight split: 7,340,175,360 BF16 bytes and 5,615,124,480 MXFP4 bytes over twenty calls. Its transient device allocation count is 2,040 after the corrected 102-per-call accounting.
- A resident median is 89.54% lower than A transient in this single formal run. This is a measured FFN sublayer comparison, not a projected token speedup.
- Formal-run audit: the first attempt stopped before writing output when contribution validation incorrectly required exact float equality. The second attempt also stopped before output when the transient allocation formula omitted per-call temporaries. Commits `7bfd152` and `1afda75` corrected those fail-closed defects. The third run produced the sole published matrix; it was not rerun to select timing.
- Evidence SHA-256: artifact `96b2919cce9a0c8bc835cb6707753a550dc3728528eda80cbc2d57c52d85c4d5`; manifest `7116a03b79fb14d25c8b7d71de0bb4333aa869ff7d4b1a8a5eb2ec01e119ee27`; runner `59b90d6b2da4498f2b8cb0c5057462e802729b0149fe957efd1f3c0711c86f9e`; aggregate `2a1a758493791e5a417fda694dc0ee2a3e9adb2d92f71c39e7589fdc2683be39`; summary JSON file `0518bbda69d7f3b0040446c8ba8e7d1847bd8a0a14c782352e2a8f8adf202cfb`; summary CSV `b251aea5cccbe8cba2417e4cc3a97f9127cdd52fc1a07904d32d170fc7f64a95`.
- Raw SHA-256: A transient `c0d197bc366772e06f792dc8829002246c82e5590d8bd955d187392a70ac6994`; A resident `1c349fdef629cf286734a3cf6e3ebdb665e914859313733f78e7b17a7591d588`; alternating resident `e6de5bfa0b6f2a7567c8736b5e8fa5c1287dee191073bfc7f99c7f6a685bb059`.
- Decode tok/s: not measured.
- Prefill tok/s: not measured.
- TTFT: not measured.
- System RAM: not measured by the benchmark schema.
- NVMe GB/token and physical RAM-to-GPU GB/token: not measured. H2D fields above are logical CUDA-copy counters.
- Expert-cache hit rate: not emitted as a percentage; exact hit counts are shown above and misses/bypasses are zero in measured resident iterations.
- Average Top-K: fixed natural Top-16, not adaptive.
- Speculative acceptance and unique experts per verification block: not applicable; speculation is disabled.
- Quality benchmark, GPU utilization, and GPU memory bandwidth: not measured.
- Enabled optimizations: exact CUDA residency only in resident rows. No proxy, pruning, adaptive Top-K, speculative verification, CUDA Graph default, or lossy quantization is enabled.
- Interpretation: B-0029 proves one dependency-complete official MoE FFN sublayer with exact routing, real shared-expert tensors, two changing routes, bounded exact residency, and independent parity. It does not include KDA, MLA, attention, a complete transformer layer, tokens, coding quality, physical storage traffic, or native-Linux performance authority.

## Milestone 28 final verification matrix

- Date: 2026-08-11.
- CPU: CTest 17/17; Python 507 passed, 97 skipped.
- liburing/direct capability build: CTest 18/18; Python 509 passed, 95 skipped.
- ASan/UBSan: CTest 18/18.
- CUDA with actual bounded artifact: CTest 30/30; Python 592 passed, 12 skipped.
- Actual alternating resident Compute Sanitizer: `ERROR SUMMARY: 0 errors`; maximum absolute error `0.00048828125`; warm weight H2D zero.
- A full-matrix precursor exposed two incomplete BF16 source-integrity fields in a Python test helper. Commit `bdfc0b6` added the source/tensor digests and canonical tensor order; focused CPU and liburing BF16 tests then passed 2/2 before this fresh matrix.
- Public integration: branch correctness `31465297042`, pull-request correctness `31465320780`, and pull-request CodeQL `31465320778` passed. PR #48 rebase-merged at `eb2c20860ee9c7c612b9b74984170bd8b4443ba1`; post-merge `main` correctness `31465590414` and CodeQL `31465590416` passed.

## Milestone 29 design metadata gate — not a benchmark

- Date: 2026-08-11.
- Evidence commit: `5f04768`.
- Hardware: metadata-only HTTPS inspection from the current development host; no GPU execution.
- Model/checkpoint: `moonshotai/Kimi-K3` revision `9f62e4e9fffbd0a83ddd60e1c209d828994b3569`.
- Observation: pinned `modeling_kimi_linear.py` is 51,506 bytes and recomputes to Git blob `b8c41e8bfce768d74d8da3a37e693f5ee43876a0`; 17 layer-1 KDA/self-Attention-Residual header entries total 887,843,840 payload bytes in `model-00002-of-000096.safetensors`.
- Network scope: repository API, 59,764,096-byte index metadata, 818,696-byte safetensors header, 51,506-byte pinned source, and small config metadata only. No tensor payload, complete shard, checkpoint, or paid resource was used.
- Decode tok/s, prefill tok/s, TTFT, VRAM, system RAM, NVMe GB/token, H2D GB/token, cache hit rate, Top-K average, speculative acceptance, quality, layer latency, utilization, and bandwidth: not measured.
- Interpretation: this entry establishes the bounded M29 planning envelope and the `A_log[128]` checkpoint contract. It is not B-0030 and supports no performance conclusion.

## Milestone 29 Task 1 verification — official KDA metadata planner

- Date: 2026-08-11.
- Model/checkpoint: pinned metadata for `moonshotai/Kimi-K3` revision `9f62e4e9fffbd0a83ddd60e1c209d828994b3569`.
- Verification: official source/MoE/layer/CLI pytest passes 64 tests; Python compile validation and `git diff --check` pass.
- Live metadata-only result: layer 1, shard `model-00002-of-000096.safetensors`, 17 KDA tensors, 887,843,840 KDA bytes, 1,267,744,256 base bytes, 1,829,256,704 maximum two-token bytes, and source blob `b8c41e8bfce768d74d8da3a37e693f5ee43876a0`.
- Negative coverage: `A_log[96]`, malformed layer-list schemas, config drift, tensor metadata drift, source blob drift, and shard/range drift fail closed before payload use.
- All performance, traffic, memory, state-parity, route, quality, and token fields: not measured.
- Interpretation: this is a planning correctness gate only. It is not official tensor materialization, KDA execution, complete-layer parity, or B-0030.

## Milestone 29 Task 2 verification — independent scalar KDA oracle

- Date: 2026-08-11.
- Hardware/model: CPU PyTorch tiny literal fixture with `hidden=4`, `heads=2`, `head_dim=2`, and convolution width 3; no official tensor payload.
- Mode: BF16 projection and convolution boundaries, F32 channel-wise decay, scalar-per-head beta, FP32 recurrence, V-first state publication, and full two-token versus incremental A-then-B execution.
- Verification: focused official/synthetic KDA and model pytest passes 17 tests in 2.10 seconds. The combined official source/MoE/layer/CLI plus KDA/model regression passes 81 tests in 4.46 seconds; Python compile validation and `git diff --check` pass.
- Parity: BF16 outputs and convolution histories match exactly; final FP32 recurrence matches within absolute and relative tolerance `1e-6`. A separate nonzero-state calculation independently reconstructs the paper recurrence and verifies unchanged input state.
- Negative coverage: weight dtype/shape, non-finite values, convolution-history width, empty sequence, and recurrent-state layout fail closed.
- Decode tok/s, prefill tok/s, TTFT, VRAM, system RAM, NVMe GB/token, H2D GB/token, cache hit rate, average Top-K, speculative acceptance, quality, official layer latency, utilization, and bandwidth: not measured.
- Interpretation: this is a tiny correctness gate. It is not official payload execution, complete-layer parity, CUDA evidence, B-0030, or a performance benchmark.

## Milestone 29 Task 3 verification — bounded complete-layer manufacturing

- Date: 2026-08-11.
- Model/checkpoint: pinned metadata for `moonshotai/Kimi-K3` revision `9f62e4e9fffbd0a83ddd60e1c209d828994b3569`; no M29 tensor payload.
- Mode: content-addressed KDA/MoE trunk objects, full/incremental KDA route-state publication before selected experts, exact execution-order source assembly, existing non-executable optional-feature guard, and `kda-layer` CLI dry-run/materialization contracts.
- Verification: layer, MoE, CLI, official source, converter resume, and source-manifest integrity pytest passes 123 tests in 10.72 seconds. Python compile validation and `git diff --check` pass.
- Live metadata-only result: 12.867-second wall time, revision `9f62e4e9fffbd0a83ddd60e1c209d828994b3569`, source blob `b8c41e8bfce768d74d8da3a37e693f5ee43876a0`, 17 KDA tensors, 887,843,840 KDA bytes, 1,267,744,256 base bytes, 1,829,256,704 maximum two-token bytes, and zero tensor-payload bytes.
- Traffic scope: nine metadata/header HTTP requests; 59,799,738 metadata bytes; 818,704 header bytes; maximum response 59,764,096 bytes from the index. The 8 MiB cap applies to future tensor-range responses, not the separately verified index metadata response.
- Negative/ordering coverage: source blob and complete-plan drift fail before payload; all 28 pre-route objects precede route derivation; the route-state manifest exists before selected expert planning; only the selected union is requested; source order begins with all 17 KDA tensors and ends with the existing M28 shared expert order.
- Decode tok/s, prefill tok/s, TTFT, VRAM, system RAM, NVMe GB/token, H2D GB/token, cache hit rate, average Top-K, speculative acceptance, quality, official route union, layer latency, utilization, and bandwidth: not measured.
- Interpretation: this verifies manufacturing code and a live zero-payload planning boundary. It is not official materialization, complete-layer execution, CUDA evidence, B-0030, or a performance result.

## Milestone 29 Task 4 verification — portable C++ KDA oracle

- Date: 2026-08-11.
- Hardware/model: CPU tiny literal `hidden=4`, `heads=2`, `head_dim=2`, convolution width 3 fixture; no official tensor payload.
- Mode: native BF16 word views, F32 short-convolution weights, channel-wise decay, scalar-per-head beta, FP32 recurrence with V-first owned state, and full two-token versus incremental A-then-B execution.
- Verification: full CPU build succeeds; CPU CTest passes 18/18; focused official expert/MoE/KDA CTest passes 3/3; Python C++ parity passes 114 tests with 32 capability skips in 21.83 seconds; source and test compile with `-Wall -Wextra -Wpedantic -Werror`; `git diff --check` passes.
- Parity: every projected, convolved, normalized Q/K/V, decay, beta, recurrent, gated, output, convolution-state, and recurrent-state field matches the independent PyTorch oracle within `1e-6`; BF16 outputs and convolution-state words are exact.
- Negative coverage: malformed A-log length, non-finite F32 weight, recurrent-state length drift, checked dimension products, native BF16 finiteness, and derived non-finite values fail before result publication. Input state remains unchanged.
- Decode tok/s, prefill tok/s, TTFT, VRAM, system RAM, NVMe GB/token, H2D GB/token, cache hit rate, average Top-K, speculative acceptance, quality, official layer latency, utilization, and bandwidth: not measured.
- Interpretation: this is a tiny portable correctness oracle. It is not Reader integration, official payload execution, complete-layer parity, CUDA evidence, B-0030, or a performance benchmark.

## Milestone 29 Task 5A verification — portable complete-layer composition

- Date: 2026-08-11.
- Hardware/model: CPU tiny literal `hidden=4`, `heads=2`, `head_dim=2`, convolution width 3, three router experts, and two materialized native-MXFP4 experts; no official tensor payload.
- Mode: explicit self Attention Residual, input RMSNorm, portable KDA, BF16 prefix accumulation, MLP Attention Residual, post RMSNorm, natural Top-2 routing, portable MoE, and final prefix accumulation. The reduced Top-K belongs only to the tiny fixture.
- Verification: full CPU build succeeds; CPU CTest passes 19/19; focused official expert/MoE/KDA/layer CTest passes 4/4; Python C++ parity passes 115 tests with 32 capability skips in 21.90 seconds; source and test compile with `-Wall -Wextra -Wpedantic -Werror`; `git diff --check` passes.
- Parity: full two-token and incremental A-then-B execution match at both Attention Residual outputs, both normalized inputs, all KDA boundaries and final V-first state, natural routes `[0,1]` then `[1,0]`, route contributions, every MoE intermediate, and final outputs. An independent PyTorch graph matches all exposed values within `1e-6`; BF16 boundaries are exact.
- Decode tok/s, prefill tok/s, TTFT, VRAM, system RAM, NVMe GB/token, H2D GB/token, cache hit rate, average official Top-K, speculative acceptance, quality, official layer latency, utilization, and bandwidth: not measured.
- Interpretation: this is a tiny pure-composition correctness gate. It does not read a K3X artifact, validate a pinned manifest, construct a CUDA backend, execute official payload bytes, or constitute B-0030.

## Milestone 29 Task 5B verification — pinned fail-before-backend preflight

- Date: 2026-08-11.
- Hardware/model: CPU and RTX 5080 build-capability validation with synthetic manifests/artifacts plus a live metadata/header-only official range cross-check; no official tensor payload.
- Mode: strict JSON and fixed manifest identity, explicit V-first state chain, fixed input hashes, natural route-union ordering, exact header-derived source ranges, K3X checksum/root/source-fingerprint verification, exact tensor order/dtype/shape, per-data/per-auxiliary SHA-256 verification, and reconstructed microshard SHA-256 before backend construction.
- Verification: focused preflight/Reader/converter pytest passes 36 tests with 4 capability skips in 7.52 seconds; the wider official layer/MoE/discovery/source/converter-resume/source-integrity regression passes 123 tests in 11.13 seconds; full CPU build and CTest pass 19/19; full CUDA build and CTest pass 32/32; strict source/test warning compilation and `git diff --check` pass.
- Live metadata-only result: the pinned header reproduces all 17 KDA and 11 MoE trunk source ranges used by the preflight constants. Network scope is repository/index/config/header metadata only; no tensor range was requested.
- Fail-closed boundary: duplicate keys, non-finite JSON numbers, fixed identity drift, state-chain drift, duplicate or reordered route unions, object-name/range drift, generic artifact identity, root/source-fingerprint drift, tensor metadata drift, and tensor payload digest drift are rejected before any backend factory can be called. A fully valid preflight currently terminates with explicit `BACKEND_UNAVAILABLE` because official CPU/CUDA execution is not implemented yet.
- Decode tok/s, prefill tok/s, TTFT, VRAM, system RAM, NVMe GB/token, H2D GB/token, cache hit rate, average Top-K, speculative acceptance, quality, layer latency, utilization, and bandwidth: not measured.
- Interpretation: this is a correctness/provenance gate, not official payload execution, CUDA parity, B-0030, or a performance benchmark.

## Milestone 29 Task 5C verification — bounded official-weight portable oracle

- Date: 2026-08-11.
- Hardware/model: Ryzen 7 9800X3D host under WSL2, bounded official Kimi K3 layer-1 artifact with 17 KDA tensors, 11 MoE trunk tensors, and the exact two-route 32-expert MXFP4 union. No complete shard or checkpoint was downloaded.
- Mode: checksum Reader, exact tensor/source/root binding, source-byte PyTorch oracle sidecar, portable C++ full and incremental complete-layer execution, natural Top-16 routing, and intentional fail-before-CUDA boundary.
- Materialization identity: source-object bytes 1,829,256,704; K3X bytes 1,829,310,720; microshard SHA-256 `bd5146441a7db4cfea9b965070c027fb022c4d491b40993f73514c73043d637d`; K3X root `49e70d84f358c81616314597ccd72a432c39f59473a01bc56efdfafdf87be22a`; converter source fingerprint `5d9a1add1f5ab5e867c81e1d9418681cedbcf8a4cf1758ac0658f256a2ce5fe0`; oracle SHA-256 `199b80b401cfd1bd5aba88204a766541ee0653bd78789516edd036d05e7da4af` over 6,541,344 bytes.
- Reuse observation: the verified regeneration reused 60 range objects and downloaded zero tensor-payload bytes. Its 155.389146494-second wall time includes metadata/header transport, object rehash, source assembly, and conversion; it is not model performance.
- Correctness: full and incremental portable paths agree; both official natural routes contain the exact expected 16 IDs. Source-byte versus portable maximum absolute differences are `1.52588e-05` output, `0.0078125` Q/K convolution state, `0.00390625` V convolution state, `4.39133e-05` recurrent state, and `1.06171e-06` contribution. Corresponding maximum relative differences are `42.6`, `0.293532`, `0.101695`, `0.0526316`, `131.126`, and `1.7391e-05`; the large output/recurrent ratios occur at near-zero references and are not acceptance gates. The valid artifact reaches the intended `BACKEND_UNAVAILABLE` only after these gates pass.
- Regression verification: CPU CTest 19/19, CUDA CTest 32/32, and focused Python 67 passed with 4 skipped. The skip count is build-capability related. No native official-layer CUDA test or sanitizer result exists yet.
- Decode tok/s, prefill tok/s, TTFT, VRAM, system RAM, NVMe GB/token, H2D GB/token, cache hit rate, average Top-K, speculative acceptance, complete-layer CUDA latency, utilization, bandwidth, and quality: not measured.
- Interpretation: this is official bounded-payload CPU correctness and provenance evidence. It is not B-0030, native CUDA execution, token throughput, physical traffic, or a quality result.

## Milestone 29 Task 6 verification — native CUDA complete-layer capability

- Date: 2026-08-11.
- Hardware/model: Ryzen 7 9800X3D and RTX 5080 under WSL2; bounded 1,829,310,720-byte official Kimi K3 layer-1 K3X artifact with the exact disjoint 32-expert union. No complete shard/checkpoint or paid cloud resource was used.
- Mode: host self/MLP Attention Residual, RMS normalization, and exact all-896 natural routing; native `sm_120` BF16/F32 KDA projections, short convolution, channel decay, V-first recurrence, output projection, and exact native-MXFP4 MoE FFN. The recorded official smoke is `ab-incremental`, exact resident, zero warmups, one iteration.
- Correctness: transient/resident tiny CUDA paths match the portable complete layer; full two-token and incremental A-to-B outputs and final state agree. The official resident smoke preserves both exact Top-16 routes and reports complete-layer maximum absolute output error `0.00048828125`.
- Verification: CPU CTest 19/19; CUDA CTest 34/34; focused harness/Reader/C++ parity Python 175 passed and 8 capability skips; `test_cuda_official_kda` and `test_cuda_official_layer` Compute Sanitizer each report `ERROR SUMMARY: 0 errors`.
- Cold capability counters: 381,907,507 ns wall time; 32,897,536 ns aggregate profiled device time; 32 KDA kernel launches over two KDA calls; 1,816,322,048 weight-H2D/resident bytes; 13,198,976 activation-H2D bytes; 13,139,968 total D2H bytes; 13,025,280 KDA-state bytes in each direction; 57,344 KDA-output D2H bytes; 1,824,612,416 tracked peak device bytes.
- Decode tok/s, prefill tok/s, TTFT, VRAM from an independent process observer, system RAM, physical NVMe GB/token, physical H2D GB/token, cache hit rate, average full-model Top-K, speculative acceptance, utilization, memory bandwidth, coding quality, and end-to-end model quality: not measured.
- Interpretation: this is one cold capability/correctness smoke with zero warmups, not B-0030. It proves the dependency-closed official layer fits and executes on RTX 5080 but supports no warm performance, token-rate, physical-traffic, quality, native-Linux, or default-policy conclusion.

## Milestone 29 Task 7 verification — B-0030 evidence tooling

- Date: 2026-08-11.
- Mode: fixed A transient, A-to-B incremental resident, and A+B full resident order; strict 3 warmups/20 samples for official verification; fsynced atomic partial-directory publication; canonical raw JSON, LF-only CSV, summary, aggregate, and artifact/manifest/runner digests. The closed schema includes BF16/F32/MXFP4 weight traffic, process peak RSS, and Reader logical/storage counters.
- Verification: focused runner tests pass 8/8; Python compile validation passes; focused official MoE/KDA/layer CUDA CTest passes 3/3. Actual one-sample capability probes confirm all closed traffic formulas and exact full/incremental BF16 output plus V-first final-state digest parity.
- Negative coverage: forbidden token/TTFT/quality/physical-traffic metrics, schema drift, route/contribution drift, traffic drift, numerical divergence, non-finite output, resident warm weight H2D, raw/CSV/aggregate mutation, row-order drift, and full/incremental output/state divergence fail before final publication.
- All performance fields: not recorded as formal evidence. No B-0030 output directory exists at this implementation checkpoint.
- Interpretation: this verifies the non-ranking evidence pipeline only. It is not B-0030 and supports no performance or default-policy conclusion.

## B-0030 — bounded official KDA complete-layer CUDA

- Date: 2026-08-11.
- Evidence commit: `bbdccb9` from runner implementation `8ace63d`.
- Hardware: AMD Ryzen 7 9800X3D, NVIDIA GeForce RTX 5080 16 GB, driver 591.86, CUDA 13.3.1, WSL2 Ubuntu 24.04.4 on Windows 11.
- Model/checkpoint: bounded official layer-1 fixture from `moonshotai/Kimi-K3` revision `9f62e4e9fffbd0a83ddd60e1c209d828994b3569`; 1,829,256,704 source-object bytes and 1,829,310,720 K3X bytes; no complete shard/checkpoint.
- Mode: fixed A transient, A-to-B incremental resident, and A+B full resident order; three warmups and twenty measured calls or sequences; exact natural Top-16 routes with a disjoint 32-expert union.
- Context length: two fixed layer-boundary positions. This benchmark has no token semantics.

| Row | Median | p05 | p95 | Kernel total / per sequence | Orchestration total / per sequence | Warm weight H2D | Peak VRAM | Peak RSS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A transient | 262,801,334 ns | 240,377,634 ns | 276,623,171 ns | 315,483,680 / 15,774,184 ns | 4,864,858,937 / 243,242,946.85 ns | 30,711,316,480 B | 896,091,200 B | 2,198,085,632 B |
| AB incremental resident | 168,577,563 ns | 164,557,269 ns | 172,811,430 ns | 634,547,296 / 31,727,364.8 ns | 2,741,768,725 / 137,088,436.25 ns | 0 B | 1,824,612,416 B | 2,204,311,552 B |
| AB full resident | 114,804,882 ns | 111,352,514 ns | 117,765,462 ns | 631,906,211 / 31,595,310.55 ns | 1,662,409,908 / 83,120,495.4 ns | 0 B | 1,825,310,016 B | 2,198,372,352 B |

- Resident bytes: 1,816,322,048 for both AB rows; cache hits are 2,720 incremental and 2,440 full with zero misses or bypasses.
- KDA traffic: incremental records 40 calls, 640 launches, and 260,505,600 state bytes in each direction across twenty sequences. Full records 20 calls, 480 launches, and 130,252,800 state bytes in each direction.
- Activation H2D / total D2H: 263,979,520 / 262,799,360 bytes incremental; 133,726,720 / 132,546,560 bytes full.
- Reader scope: each harness process records 440 calls and 3,658,513,408 requested, completed, storage-submitted, and storage-completed bytes. These are logical Reader counters, not physical NVMe traffic.
- Correctness: all rows are finite with maximum absolute error `0.00048828125`. Full and incremental output SHA-256 are both `3bc173301781ec02502c29a1d8ac2951139ba51cfef593f858bbac65cd748617`; final V-first state SHA-256 are both `5f0ce4680ca343648838ef274cc3f8526c5174eba9922b44b8f37715c2901073`.
- Derived comparison: full resident wall median is 31.897887% below incremental resident, while per-sequence kernel time is 0.416216% lower. This supports a host/orchestration/validation attribution experiment, not a causal claim about one subcomponent.
- Evidence SHA-256: artifact `9f0c29fcb18b8cdab5aeeec67d8e5e0113b8dffb7352a2dcdac1ae41ae5198c6`; manifest `cf0dd554d5dfc7db640cb3313f7527e6c354a6fd74f9011cd747348b247168d4`; runner `253af0dfa411b771913997f9685c3bb4c5d5877ae68d7fe263eaff6e67f2b1b9`; aggregate `86f0007af7da007d6646dec6fa8fba4008c1bf7bedff53971d5d31926c9f6452`; summary CSV `1e5af9bb7d5b9abb16f62962bbce3584b62014873b12ce7642868e919770a635`.
- Raw JSON SHA-256: A transient `2bea4a645ed3c91c7dfe0449b5367fd79a3faee4095d5dce3a89e359a127e4cc`; incremental resident `aee70f024bc6c0f8326b5cb9d46464ec506698e8498888a0be99b8bf54bf21f8`; full resident `31ac8b44329f929bab9f0f83c20d03efee13b1e31222afb3a0246b05c284b632`.
- Decode tok/s, prefill tok/s, TTFT, physical NVMe GB/token, physical H2D GB/token, GPU utilization, GPU memory bandwidth, coding quality, speculative acceptance, and adaptive Top-K: not measured.
- Enabled optimizations: exact resident BF16/F32/MXFP4 weights in resident rows only. No proxy, pruning, adaptive Top-K, speculation, CUDA Graph default, or lossy quantization.

## Milestone 29 final local verification matrix

- Date: 2026-08-11.
- CPU: CTest 19/19; Python 541 passed, 119 skipped.
- liburing/direct capability: CTest 20/20; Python 543 passed, 117 skipped with `K3X_TEST_IO_URING=1`.
- ASan/UBSan: CTest 20/20.
- CUDA with the actual bounded artifact: CTest 34/34; Python 639 passed, 21 skipped.
- Committed B-0030 verifier: 9/9 passed; strict artifact/manifest/runner/raw/CSV/aggregate rehash passed.
- Actual AB incremental resident Compute Sanitizer: `ERROR SUMMARY: 0 errors`; launch attach used `--launch-timeout 0` because the checksum/preflight oracle exceeds the tool's 10-second default attach timeout. The valid run reports zero warm weight H2D and maximum absolute error `0.00048828125`.
- Production guard: `k3x_run` exits 4 with `NON_EXECUTABLE_ARTIFACT` on the bounded layer fixture.
- Public integration: branch correctness `31487723904`, pull-request correctness `31487748354`, and pull-request CodeQL `31487748339` passed. PR #50 rebase-merged at `2a4bfaf40284204ab314938f8112b280915f77df`; post-merge `main` correctness `31488078940` and CodeQL `31488078974` passed.

## B-0031 — official KDA immutable admission validation

- Date: 2026-08-11.
- Evidence commit: `fb33d84`; implementation commits `192b4da`, `b2a27ed`, `c39b3bb`, and direct-run fix `038b427`.
- Hardware: AMD Ryzen 7 9800X3D, NVIDIA GeForce RTX 5080 16 GB, driver 591.86, CUDA 13.3.1, WSL2 Ubuntu 24.04.4 on Windows 11.
- Model/checkpoint: the unchanged bounded official layer-1 fixture from `moonshotai/Kimi-K3` revision `9f62e4e9fffbd0a83ddd60e1c209d828994b3569`; 1,829,256,704 source-object bytes and 1,829,310,720 K3X bytes; no complete shard/checkpoint.
- Mode: exact resident A-to-B incremental and A+B full execution, each crossed with `per-call` and `admission`; three warmups and twenty measured two-position sequences; natural Top-16 routes and disjoint 32-expert union unchanged.
- Context length: two fixed layer-boundary positions. This benchmark has no token semantics.

| Row | Median | p05 | p95 | Kernel total / sequence | Orchestration total / sequence | Validation total / sequence | Scan or hit count |
|---|---:|---:|---:|---:|---:|---:|---:|
| Incremental per-call | 175,667,985 ns | 170,075,440 ns | 181,255,901 ns | 680,383,008 / 34,019,150.4 ns | 2,835,219,248 / 141,760,962.4 ns | 2,077,482,534 / 103,874,126.7 ns | 560 scans |
| Incremental admission | 70,584,413 ns | 68,989,767 ns | 72,530,729 ns | 677,780,608 / 33,889,030.4 ns | 733,963,129 / 36,698,156.45 ns | 0 / 0 ns | 560 hits |
| Full per-call | 121,067,320 ns | 116,080,515 ns | 137,283,784 ns | 681,833,504 / 34,091,675.2 ns | 1,808,664,255 / 90,433,212.75 ns | 1,114,634,419 / 55,731,720.95 ns | 280 scans |
| Full admission | 67,236,923 ns | 66,278,446 ns | 68,450,482 ns | 679,179,681 / 33,958,984.05 ns | 667,805,396 / 33,390,269.8 ns | 0 / 0 ns | 280 hits |

- Validation bytes: incremental per-call scans 35,512,033,280 bytes; full per-call scans 17,756,016,640 bytes. Admission cold execution scans 887,800,832 bytes once, then the measured interval scans zero bytes.
- Derived paired changes: admission lowers incremental wall median 59.819421% and full wall median 44.463194%. Paired aggregate kernel totals change -0.382490% and -0.389219%. The admission incremental/full median gap is 3,347,490 ns.
- Correctness and traffic: every row has output SHA-256 `3bc173301781ec02502c29a1d8ac2951139ba51cfef593f858bbac65cd748617`, final V-first state SHA-256 `5f0ce4680ca343648838ef274cc3f8526c5174eba9922b44b8f37715c2901073`, maximum absolute error `0.00048828125`, 1,816,322,048 resident weight bytes, zero measured weight H2D, zero cache misses/bypasses, and finite output.
- Incremental rows record 40 KDA calls, 640 launches, 260,505,600 state bytes in each direction, 263,979,520 activation-H2D bytes, and 262,799,360 total D2H bytes. Full rows record 20 calls, 480 launches, 130,252,800 state bytes in each direction, 133,726,720 activation-H2D bytes, and 132,546,560 total D2H bytes.
- Peak tracked VRAM / process RSS: incremental 1,824,612,416 / 2,204,409,856 or 2,204,418,048 bytes; full 1,825,310,016 / 2,198,142,976 or 2,197,835,776 bytes.
- Reader scope: each process records 440 calls and 3,658,513,408 requested/completed/storage-submitted/storage-completed bytes. These are logical Reader counters, not physical NVMe traffic.
- Evidence SHA-256: artifact `9f0c29fcb18b8cdab5aeeec67d8e5e0113b8dffb7352a2dcdac1ae41ae5198c6`; manifest `cf0dd554d5dfc7db640cb3313f7527e6c354a6fd74f9011cd747348b247168d4`; runner binary `a710b7189220256189fa682e9be371e412ce4b27bedf63ffa5e2dac81c864685`; aggregate `5d6ba38a0d959902c5ab8e7f7bce4f13254f018644430a18490864c173b30a1f`; summary CSV `ea4bcd6b6f613d4a07bdb4e8aa14cb989f2ef11a3fcdfd473112a846e0883501`; summary JSON `9ddad853b3a99ea84efd189f0294814d736d7418c5264bbcf06865e6f94d4fc4`.
- Raw JSON SHA-256: incremental per-call `4c3a850f812d3451a05485fe7e728a370d3db3b8ed24515d4ad9ae5f7df588e8`; incremental admission `761123186491dbef38a942189d784d38c4c178944d167c347832efb6b970a6a4`; full per-call `faa2fdfb7ec8a096ff1d694373d68fa1aa01214c0047f03b01964641095bb9d9`; full admission `02581a42977049cf42479b853ba022cb63e0839f1e2e1d1aac2fafcf5ebb1ee3`.
- Decode tok/s, prefill tok/s, TTFT, physical NVMe GB/token, physical H2D GB/token, GPU utilization, GPU memory bandwidth, coding quality, speculative acceptance, adaptive Top-K, and quality benchmarks: not measured.
- Enabled optimization: exact resident immutable-weight admission only in admission rows. No proxy, pruning, adaptive Top-K, speculation, graph default, reduced precision, or routing change.

## Milestone 30 final local verification matrix

- Date: 2026-08-11.
- CPU: CTest 19/19; Python 558 passed, 122 skipped.
- liburing/direct capability: CTest 20/20; Python 560 passed, 120 skipped with `K3X_TEST_IO_URING=1`.
- ASan/UBSan: CTest 20/20.
- CUDA with the actual bounded artifact: CTest 34/34; Python 659 passed, 21 skipped.
- B-0031 evidence-tool plus B-0030 regression tests: 26/26 passed; strict B-0031 artifact/manifest/runner/raw/CSV/aggregate rehash passed.
- Actual admission-mode AB incremental resident Compute Sanitizer: `ERROR SUMMARY: 0 errors` with `--launch-timeout 0`; measured interval records 28 hits, zero scans/bytes/time, zero warm weight H2D, and maximum error `0.00048828125`.
- Production guard: `k3x_run` exits 4 with `NON_EXECUTABLE_ARTIFACT` on the unchanged bounded layer fixture.
- Public integration: branch correctness `31493248372`, pull-request correctness `31493267425`, and pull-request CodeQL `31493267404` passed. PR #52 rebase-merged at `51182575b32b49afa4b1fb2586f31df058a74155`; post-merge `main` correctness `31493550970` and CodeQL `31493549669` passed.

## B-0032 — official KDA device-state handoff

- Date: 2026-08-11.
- Evidence commit: `992e0de`; implementation commits `fa79e86`, `a830fa9`, `84a5d0d`, and undefined-mode guard `9999918`.
- Hardware: AMD Ryzen 7 9800X3D, NVIDIA GeForce RTX 5080 16 GB, driver 591.86, CUDA 13.3.1, WSL2 Ubuntu 24.04.4 on Windows 11.
- Model/checkpoint: the unchanged bounded official layer-1 fixture from `moonshotai/Kimi-K3` revision `9f62e4e9fffbd0a83ddd60e1c209d828994b3569`; 1,829,256,704 source-object bytes and 1,829,310,720 K3X bytes; no complete shard/checkpoint.
- Mode: exact resident admission with incremental host round trip, incremental device handoff, and full host round trip; three warmups and twenty measured two-position sequences; natural Top-16 routes and disjoint 32-expert union unchanged.
- Context length: two fixed layer-boundary positions. This benchmark has no token semantics.

| Row | Median | p05 | p95 | Kernel / sequence | Orchestration / sequence | State H2D / D2H |
|---|---:|---:|---:|---:|---:|---:|
| Incremental host | 73,192,169 ns | 70,663,570 ns | 74,723,294 ns | 33,772,262.4 ns | 39,023,029.3 ns | 260,505,600 / 260,505,600 B |
| Incremental device | 69,835,612 ns | 67,384,546 ns | 71,854,965 ns | 33,887,020.8 ns | 35,815,361.7 ns | 130,252,800 / 130,252,800 B |
| Full host | 68,224,527 ns | 66,958,871 ns | 72,100,858 ns | 33,734,271.8 ns | 35,167,668.75 ns | 130,252,800 / 130,252,800 B |

- Derived paired changes: device handoff lowers the incremental wall median by 4.585951%, or 3,356,557 ns. Aggregate kernel time changes +0.339801%, orchestration falls 3,207,667.6 ns per sequence, and the device-incremental/full-host median gap is 1,611,085 ns.
- Correctness: every row has output SHA-256 `3bc173301781ec02502c29a1d8ac2951139ba51cfef593f858bbac65cd748617`, final V-first state SHA-256 `5f0ce4680ca343648838ef274cc3f8526c5174eba9922b44b8f37715c2901073`, maximum absolute error `0.00048828125`, finite output, and exact route/contribution parity.
- Traffic: incremental host/device activation H2D is 263,979,520/133,726,720 bytes and total D2H is 262,799,360/132,546,560 bytes. Device handoff removes exactly 130,252,800 bytes in each direction over twenty sequences. Every row has zero measured weight H2D, 1,816,322,048 resident weight bytes, zero cache misses/bypasses, and 440 logical Reader calls for 3,658,513,408 requested/completed bytes.
- State operations: the device row records 20 seeds, 20 continuations, 20 publications, and zero invalidations. Host rows record no device-state operation. Incremental rows execute 40 KDA calls and 640 launches; full executes 20 calls and 480 launches.
- Peak tracked VRAM / process RSS: host incremental 1,824,612,416 / 2,200,453,120 bytes; device incremental 1,824,612,416 / 2,193,940,480 bytes; full host 1,825,310,016 / 2,193,661,952 bytes.
- Evidence SHA-256: artifact `9f0c29fcb18b8cdab5aeeec67d8e5e0113b8dffb7352a2dcdac1ae41ae5198c6`; manifest `cf0dd554d5dfc7db640cb3313f7527e6c354a6fd74f9011cd747348b247168d4`; runner `1422a1776dab47b8f26673876ec64c2d3991e7922af5a16c4751828c5df56225`; aggregate `88db7c3e8210035204a3e6679c482782f8223de1b49edd39f5d407ad3edab339`; summary JSON `42cb8809b6e7b0b0a23f152f8377cdaffa1d5a6d0efd31d7182322d146963d5f`; summary CSV `9abfb06e0bc936211e258b39f2aa2cc0bf88e1c3ec553dd68e9427291fc79c11`.
- Raw JSON SHA-256: host incremental `06dce7ab0cf4523ed2e42c1c0a2a053a1e20753e835b865841a558c60b03b9df`; device incremental `5b0a7e9b0d1dab70226fdff7aa5d5e0efe835fbdfc65dfc8060a8d5f24e052f0`; full host `e47de6eb60682e7b03dff7d8292fa30be896356ec818689d9edac0f41475ce67`.
- Decode tok/s, prefill tok/s, TTFT, physical NVMe GB/token, physical H2D GB/token, GPU utilization, GPU memory bandwidth, coding quality, speculative acceptance, adaptive Top-K, and quality benchmarks: not measured.
- Enabled optimization: exact device-state handoff only in the device row. Admission and resident weights are common controls. No proxy, pruning, adaptive Top-K, speculation, reduced precision, routing change, or production default changed.

## Milestone 31 final local verification matrix

- Date: 2026-08-11.
- CPU: CTest 19/19; Python 578 passed, 125 skipped.
- liburing/direct capability: CTest 20/20; Python 584 passed, 119 skipped with `K3X_TEST_IO_URING=1 K3X_TEST_DIRECT=1`.
- ASan/UBSan: CTest 20/20.
- CUDA with the actual bounded artifact: CTest 34/34; Python 688 passed, 15 skipped with `K3X_TEST_CUDA=1`.
- B-0032/B-0031/B-0030 evidence-tool tests: 46/46 passed; strict B-0032 artifact/manifest/runner/raw/CSV/aggregate rehash passed.
- Actual device-state AB incremental resident-admission Compute Sanitizer: `ERROR SUMMARY: 0 errors` with `--launch-timeout 0`; measured state H2D/D2H is 6,512,640/6,512,640 bytes, seeds/continuations/publications are 1/1/1, invalidations are zero, warm weight H2D is zero, and maximum error is `0.00048828125`.
- Production guard: `k3x_run` exits 4 with `NON_EXECUTABLE_ARTIFACT` and creates no output on the unchanged bounded layer fixture.
- Public integration: PR #54 rebase-merged at `e1233891537f14785373f47e9f736fed43598c46`. Branch correctness `31501537039`, pull-request correctness `31501569778`, pull-request CodeQL `31501569789`, post-merge `main` correctness `31501949124`, and post-merge CodeQL `31501949081` all succeeded. The only workflow annotations were the Node 20 and CodeQL Action v3 deprecation notices plus the existing C++ overlay-base fallback warning; every job conclusion was `success`.

## Milestone 32 Task 3 correctness verification

- Date: 2026-08-12.
- Hardware: AMD Ryzen 7 9800X3D, NVIDIA GeForce RTX 5080 16 GB, CUDA 13.3.1, WSL2 Ubuntu 24.04.4.
- Model/checkpoint: tiny synthetic official-layer fixture plus the unchanged bounded official layer-1 artifact; no complete shard/checkpoint.
- Mode: exact incremental device KDA state, resident admission, explicit device route preparation, natural Top-16, exact MXFP4 FFN.
- Correctness: tiny CUDA wrapper and cleanup test passed; route, missing-expert, and FFN failures each discarded the prepared token and invalidated the KDA token. Compute Sanitizer reported `ERROR SUMMARY: 0 errors`.
- Actual-artifact smoke: explicit host and device route paths each passed one bounded two-position execution. The device path preserved exact route IDs, contributions, output, and final state, returned 7,168 logical router-logit D2H bytes, consumed two prepared tokens, recorded zero discards/invalidations on success, and transferred zero warm weight bytes.
- Performance: not measured. The smoke used zero warmups and one correctness iteration, so no latency, TPS, TTFT, quality, utilization, bandwidth, or physical traffic claim is recorded.

## B-0033 — official MoE device route preparation

- Date: 2026-08-12.
- Evidence commit: `3d5d96c`; implementation commits `34f71cb`, `5b1cf18`, `b9094fa`, and strict evidence commit `53e88a9`.
- Hardware: AMD Ryzen 7 9800X3D, NVIDIA GeForce RTX 5080 16 GB, driver 591.86, CUDA 13.3.1, WSL2 Ubuntu 24.04.4 on Windows 11.
- Model/checkpoint: unchanged bounded official layer-1 fixture from `moonshotai/Kimi-K3` revision `9f62e4e9fffbd0a83ddd60e1c209d828994b3569`; no complete shard/checkpoint.
- Mode: exact AB incremental device KDA state, resident admission, host versus device route preparation, natural Top-16, three warmups, and twenty measured two-position sequences.
- Context length: two fixed layer-boundary positions. This benchmark has no token semantics.

| Row | Median | p05 | p95 | Kernel / sequence | Orchestration / sequence | Resident weights |
|---|---:|---:|---:|---:|---:|---:|
| Host route | 64,210,407 ns | 63,435,171 ns | 65,424,822 ns | 31,551,662.4 ns | 32,877,229.05 ns | 1,816,322,048 B |
| Device route | 63,767,134 ns | 62,450,616 ns | 64,669,895 ns | 40,163,417.6 ns | 23,597,271.55 ns | 1,829,210,112 B |

- Derived changes: device route median is 443,273 ns or 0.690344% lower. Aggregate kernel time is 27.294141% higher, while orchestration is 9,279,957.5 ns lower per sequence.
- Correctness: both rows have output SHA-256 `3bc173301781ec02502c29a1d8ac2951139ba51cfef593f858bbac65cd748617`, state SHA-256 `5f0ce4680ca343648838ef274cc3f8526c5174eba9922b44b8f37715c2901073`, exact route/contribution parity, finite output, and maximum error `0.00048828125`.
- Traffic: both rows transfer zero measured weight H2D and 133,726,720 activation H2D bytes over twenty sequences. Device route adds 143,360 logical D2H bytes total, exactly 7,168 per sequence, and 12,888,064 resident weight bytes. Tracked peak VRAM rises from 1,824,612,416 to 1,837,618,752 bytes.
- Prepared operations: device route records 40 prepare calls, 80 kernels, 143,360 logit bytes, 40 seeds, 40 consumes, zero discards, and zero invalidations. Host route records zero for every prepared operation.
- Evidence SHA-256: artifact `9f0c29fcb18b8cdab5aeeec67d8e5e0113b8dffb7352a2dcdac1ae41ae5198c6`; manifest `cf0dd554d5dfc7db640cb3313f7527e6c354a6fd74f9011cd747348b247168d4`; runner `20025307cedb0856f82847e6f57b380d833f3d26a6288160c383d1fac1c256d8`; aggregate `709fe13d67d144d025f32e17d6cafdfeef3d6e52e901d4234b7e54d7c9342d61`; summary JSON `86cb016d1764747233536614137566cfa8098cbfcfb10bdf5d5dc596d37c9ace`; summary CSV `d83d7cd34bc6601e39af3b55635d1ffb00f4eac79c742152ee669dd29915ee1d`.
- Raw JSON SHA-256: host `82ddb5c5e4b185b3f79f80af366c78a2bf32bb5e3063056509c68085270d0095`; device `67b9d9ee76e9e882377be7ea3d8e0a0014a66e42b75f852781f6cb1a297c2724`.
- Decode tok/s, prefill tok/s, TTFT, physical NVMe/H2D GB/token, GPU utilization, GPU bandwidth, quality, speculative acceptance, adaptive Top-K, and coding quality: not measured.
- Decision: retain host route preparation as default. The bounded device path is exact but its 0.690344% wall change is mixed with materially higher kernel time and is not sufficient for a default or full-model claim.

## Milestone 32 final local verification matrix

- Date: 2026-08-12.
- CPU: CTest 19/19; Python 597 passed, 128 skipped.
- liburing/direct capability: CTest 20/20; Python 603 passed, 122 skipped with `K3X_TEST_IO_URING=1 K3X_TEST_DIRECT=1`.
- ASan/UBSan: CTest 20/20.
- CUDA with actual bounded artifacts: CTest 34/34; Python 710 passed, 15 skipped with `K3X_TEST_CUDA=1`.
- B-0033/B-0032/B-0031/B-0030 evidence tests: 65/65 passed; strict B-0033 rehash and independent aggregate/formula/LF checks passed.
- Actual device-route Compute Sanitizer: `ERROR SUMMARY: 0 errors`; exact routes/output/state, 7,168 logit D2H bytes, two seeds/consumes, zero discard/invalidation, and zero warm weight H2D.
- Production guard: actual bounded artifact returns `NON_EXECUTABLE_ARTIFACT` and creates no output.
- Public integration: PR #56 rebase-merged at `ab0ecb19ade01e7989bcba9f6dbcd1c853c43432`. Branch correctness `31510344481`, pull-request correctness `31510368444`, pull-request CodeQL `31510368390`, post-merge `main` correctness `31510749958`, and post-merge CodeQL `31510749973` all succeeded. Workflow annotations are limited to the known Node 20 and CodeQL Action v3 deprecation notices.

## B-0034 — official two-layer device closure

- Date: 2026-08-13.
- Public evidence commit: `ead4371`; implementation/correctness head before evidence: `c86225b`.
- Hardware: AMD Ryzen 7 9800X3D, NVIDIA GeForce RTX 5080 16 GB, driver 591.86, CUDA 13.3.1, WSL2 Ubuntu 24.04.4 on Windows 11.
- Model/checkpoint: 3,641,057,536-byte bounded official layers 1 and 2 fixture from `moonshotai/Kimi-K3` revision `9f62e4e9fffbd0a83ddd60e1c209d828994b3569`; 119 verified range objects from two pinned shards; no complete shard/checkpoint.
- Mode: exact A1→A2→B1→B2 KDA plus natural Top-16 Stable LatentMoE, host round trip versus experimental device closure, 4 GiB resident admission, three warmups, and twenty measured two-position sequences.
- Context length: two fixed layer-boundary positions. This benchmark has no token semantics.

| Row | Median | p05 | p95 | Maximum error | Resident weights | Peak device bytes |
|---|---:|---:|---:|---:|---:|---:|
| Host round trip | 96,102,951 ns | 92,875,813 ns | 98,275,077 ns | 0.000976562 | 3,640,872,960 B | 3,655,794,240 B |
| Device closure | 109,388,034 ns | 104,939,491 ns | 112,149,396 ns | 0.00195312 | 3,640,958,976 B | 3,656,052,288 B |

- Derived changes: device closure is 13,285,083 ns or 13.823803% slower at the median. It adds 86,016 resident bytes and 258,048 tracked peak-device bytes.
- Correctness: both rows preserve identical natural Top-16 expert IDs for all four steps. Each row independently passes expert-ID-keyed contribution tolerance `2e-5`, final BF16 output tolerance `2e-3`, BF16 convolution-state tolerance `8e-3`, and recurrent-state tolerance `5e-4`. Observed contribution, output, and state digests are retained and are not falsely normalized to oracle identities.
- Traffic: both rows transfer zero measured warm weight H2D. Host round trip records 57,344 logical inter-layer H2D and 57,344 logical inter-layer D2H bytes per sequence; device closure records zero for both. These are logical counters, not physical PCIe measurements.
- Evidence SHA-256: artifact `ebc33ef266f43e3acf46b20bd79f966595ae8ef5b1b017652d6cabd896034dd5`; K3X root `2a0139cdfaddb57c95d4d50462ed79ff56b248e2946f49c773ddbb1e3c31913b`; manifest `8f4f65efa2ef7c2e9b178dee1796cf3cab221bd75fa9004f775457afd44af6b4`; oracle `f6e85aa6f5e0612d5b305e9f7b01199bdbe9c65703e3c02e8af6ea194bb4f824`; runner `96a3ec613a3561a77cc517bb895eda8e2f33529282e69bd67a3d3b00440beaa1`; aggregate `016df479907678a76cac22d4f178ab271c5ce5e074639971d22a2b8fa03cc258`; summary JSON `37e02f52a56995a820517f4419565613f649003816da11175b049fa8afce522b`; summary CSV `b151d5065af784b2805eedcea53010802c4bf34944f02e8921f24ab2301bc7d7`.
- Raw JSON SHA-256: host `162967f6901031457ee39f3d313c62eaf2bd321289d04e03032c48ca537fdc42`; device `cd8dfb2326b04c46b9770baa813f112f46a5c966329335cbc59760c103a10730`.
- Decode tok/s, prefill tok/s, TTFT, VRAM process-wide allocation, system RAM, physical NVMe GB/token, physical H2D GB/token, GPU utilization, GPU bandwidth, cache hit rate, average Top-K, speculative acceptance, unique experts per verification block, quality, and coding quality: not measured.
- Decision: retain host round trip as the default. Attribute or fuse device front/tail kernel and synchronization overhead before attempting a wider closure.

## Milestone 33 final local verification matrix

- Date: 2026-08-13.
- Portable CPU: CTest 20/20; Python 652 passed, 134 skipped.
- liburing/direct capability: CTest 17/17.
- ASan/UBSan with liburing: CTest 17/17.
- CUDA: CTest 36/36; actual official two-layer harness 6/6.
- Evidence regressions: B-0030 through B-0034 focused suite 93/93; strict committed B-0034 rehash passed.
- Actual device-closure Compute Sanitizer: `ERROR SUMMARY: 0 errors`; maximum error `0.00195312`, zero inter-layer hidden H2D/D2H, closed state/prepared lifetimes, and exact B-0034 route/output/state identities.
- Production guard: the actual bounded two-layer artifact emits `NON_EXECUTABLE_ARTIFACT` and creates no output. The direct shell wrapper does not preserve the child exit code, while the existing production regression asserts child exit 4.
- The complete CUDA-enabled Python matrix exceeded the 10-minute command limit and is not claimed as passed. Its affected actual path is covered independently by CUDA CTest 36/36, harness 6/6, evidence 93/93, and Compute Sanitizer zero errors.
- Public integration: PR #58 rebase-merged at `9ce513f9d79b6cee88b7bb2de176e6fbbb79f43b`. Branch correctness `31673610347`, pull-request correctness `31673636564`, pull-request CodeQL `31673636680`, post-merge `main` correctness `31673888294`, and post-merge CodeQL `31673888289` all succeeded. Remaining annotations are the known Node 20 and CodeQL Action v3 deprecation notices plus the existing C++ overlay-base fallback warning.

## B-0035 — official two-layer closure attribution

- Date: 2026-08-13.
- Local evidence commit: `129012c`; implementation/correctness head before evidence: `a9efb73`.
- Hardware: AMD Ryzen 7 9800X3D, NVIDIA GeForce RTX 5080 16 GB, driver 591.86, CUDA 13.3.1, WSL2 Ubuntu 24.04.4 on Windows 11.
- Model/checkpoint: the same 3,641,057,536-byte bounded official layers 1 and 2 fixture used by B-0034, from `moonshotai/Kimi-K3` revision `9f62e4e9fffbd0a83ddd60e1c209d828994b3569`; 119 verified range objects from two pinned shards; no complete shard/checkpoint.
- Mode: exact A1→A2→B1→B2 KDA plus natural Top-16 Stable LatentMoE, host round trip versus experimental device closure with opt-in profiler-snapshot attribution, 4 GiB resident admission, three warmups, and twenty measured two-position sequences.
- Context length: two fixed layer-boundary positions. This benchmark has no token semantics.

| Row | Median | p05 | p95 | Front wall mean | Route wall mean | Tail wall mean | Remainder mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| Host round trip | 99,316,205 ns | 95,310,862 ns | 103,227,516 ns | 0 ns | 0 ns | 0 ns | 99,341,079 ns |
| Device closure | 110,701,472 ns | 106,969,304 ns | 114,098,378 ns | 69,822,990 ns | 39,036 ns | 41,060,877 ns | 23,582 ns |

- Device breakdown: front, route, tail, and remainder are 62.934%, 0.035%, 37.010%, and 0.021% of device-closure attributed wall time. Existing-event CUDA time averages 52,571,374 ns for front and 30,057,734 ns for tail; the corresponding wall-minus-device gaps are 17,251,616 and 11,003,143 ns.
- Correctness: both rows preserve the exact B-0034 natural Top-16 expert IDs and measured output/state/contribution identities. Maximum errors are `0.000976562` for host round trip and `0.00195312` for device closure.
- Traffic and memory: both rows transfer zero warm weight H2D. Host round trip records 57,344 logical inter-layer bytes in each direction; device closure records zero. Host/device resident weights are 3,640,872,960/3,640,958,976 B and tracked peak device bytes are 3,655,794,240/3,656,052,288 B. These are logical/runtime counters, not process-wide VRAM or physical PCIe measurements.
- Evidence SHA-256: artifact `ebc33ef266f43e3acf46b20bd79f966595ae8ef5b1b017652d6cabd896034dd5`; K3X root `2a0139cdfaddb57c95d4d50462ed79ff56b248e2946f49c773ddbb1e3c31913b`; manifest `8f4f65efa2ef7c2e9b178dee1796cf3cab221bd75fa9004f775457afd44af6b4`; oracle `f6e85aa6f5e0612d5b305e9f7b01199bdbe9c65703e3c02e8af6ea194bb4f824`; runner `35ffd0d40c3ce07a66a14de924f84a318626ad6b9fad4009d5c077dadbd70d93`; aggregate `db3ef6e688fbe5d064cce555ba9fe1ceae0c9f8f939da44245402dd62a58c32a`; summary JSON `a0f4cb9da16692794850dd9fd53deeffa471eda54b825f4418fd59ce4d1828c5`; summary CSV `da17f480255221d961ee4c016c30a24bb38f59afd851fec55bda5b4df904d2a0`.
- Raw JSON SHA-256: host `b347813181d23c09f6e1b79334d34a89737b10f8e5e334d0c97754208e40ec84`; device `bb7083470ffb086922f67afd4ca832c64a3dc61714bf36b33e6048bea3b72602`.
- Decode tok/s, prefill tok/s, TTFT, process-wide VRAM, system RAM, physical NVMe GB/token, physical H2D GB/token, GPU utilization, GPU bandwidth, cache hit rate, average Top-K, speculative acceptance, unique experts per verification block, quality, and coding quality: not measured.
- Decision: retain host round trip as the default. Attribute existing front/tail profiler events by operation before selecting any fusion target. B-0035 and B-0034 were separate runs with different runner hashes, so their medians are not treated as a paired overhead measurement.

## Milestone 34 final local verification matrix

- Date: 2026-08-13.
- Portable CPU: CTest 20/20; Python 658 passed, 136 skipped.
- liburing/direct capability: CTest 21/21.
- ASan/UBSan with liburing: CTest 21/21.
- CUDA: CTest 36/36; focused attribution plus actual official two-layer harness 14/14.
- Evidence: B-0030 through B-0035 regressions 99/99 before the formal run; strict committed B-0035 artifact/manifest/oracle/runner/raw/CSV/aggregate rehash passed after the formal run.
- Actual attribution Compute Sanitizer: `ERROR SUMMARY: 0 errors`; zero warm weight H2D, exact B-0034 route/output/state identities, and a closed attribution formula.
- Production guard: the actual bounded two-layer artifact emits `NON_EXECUTABLE_ARTIFACT` and creates no output. The direct shell wrapper reports a generic failure status, while the portable Python regression asserts child exit 4.
- Public integration: PR #60 rebase-merged at `a7ba52044e4d0ccbb23371cac54b5db87d4002f1`. Push correctness `31677396649`, pull-request correctness `31677408262`, pull-request CodeQL `31677408278`, post-merge correctness `31677651704`, and post-merge CodeQL `31677651706` all succeeded. Remaining annotations are the known Node 20 and CodeQL Action v3 deprecation notices.

## B-0036 — official two-layer operation attribution

- Date: 2026-08-13.
- Local evidence commit: `4a41223`; implementation/correctness head before evidence: `af758f8`.
- Hardware: AMD Ryzen 7 9800X3D, NVIDIA GeForce RTX 5080 16 GB, driver 591.86, CUDA 13.3.1, WSL2 Ubuntu 24.04.4 on Windows 11.
- Model/checkpoint: the same 3,641,057,536-byte bounded official layers 1 and 2 fixture used by B-0034/B-0035, from `moonshotai/Kimi-K3` revision `9f62e4e9fffbd0a83ddd60e1c209d828994b3569`; 119 verified range objects from two pinned shards; no complete shard/checkpoint.
- Mode: exact A1→A2→B1→B2 KDA plus natural Top-16 Stable LatentMoE, host round trip versus experimental device closure with opt-in existing-event operation attribution, 4 GiB resident admission, three warmups, and twenty measured two-position sequences.
- Context length: two fixed layer-boundary positions. This benchmark has no token semantics.

| Row | Median | p05 | p95 | KDA device mean | Route-prepare device mean | MoE FFN device mean | Unclassified |
|---|---:|---:|---:|---:|---:|---:|---:|
| Host round trip | 102,157,295 ns | 100,049,678 ns | 104,824,512 ns | 0 ns | 0 ns | 0 ns | 0 ns |
| Device closure | 116,049,550 ns | 114,109,526 ns | 116,827,943 ns | 36,345,792 ns | 19,075,499 ns | 31,698,525 ns | 0 ns |

- Device breakdown: KDA, device route preparation, and MoE FFN account for 41.719%, 21.896%, and 36.385% of the 87,119,816 ns classified existing-event CUDA time per sequence. Within front device time, KDA/route preparation are 65.581%/34.419%. Front and tail regional device means are 55,421,291/31,698,525 ns and both close with zero unclassified time.
- Wall breakdown: device closure averages 73,578,176 ns front wall, 41,220 ns canonical host route wall, 42,052,397 ns tail wall, and 27,660 ns remainder. These are 63.594%, 0.036%, 36.346%, and 0.024% of attributed wall time.
- Correctness: both rows preserve the exact B-0034/B-0035 natural Top-16 expert IDs and measured output/state/contribution identities. Maximum errors are `0.000976562` for host round trip and `0.00195312` for device closure.
- Traffic and memory: both rows transfer zero warm weight H2D. Host round trip records 57,344 logical inter-layer bytes in each direction; device closure records zero. Host/device resident weights are 3,640,872,960/3,640,958,976 B and tracked peak device bytes are 3,655,794,240/3,656,052,288 B. These are logical/runtime counters, not process-wide VRAM or physical PCIe measurements.
- Evidence SHA-256: artifact `ebc33ef266f43e3acf46b20bd79f966595ae8ef5b1b017652d6cabd896034dd5`; K3X root `2a0139cdfaddb57c95d4d50462ed79ff56b248e2946f49c773ddbb1e3c31913b`; manifest `8f4f65efa2ef7c2e9b178dee1796cf3cab221bd75fa9004f775457afd44af6b4`; oracle `f6e85aa6f5e0612d5b305e9f7b01199bdbe9c65703e3c02e8af6ea194bb4f824`; runner `0f458b77ddd93958265f2473eb4e25d14deda6ccc9076227db831278724a29b3`; aggregate `373e8cd33a56dd9b35100b0a679c4e0594e3802ded3f4022c6ff673f5b4dd9af`; summary JSON `aa51f7def3cca5cc6223d6d76adb36a76a1ea0a93a85c92547182559362a3899`; summary CSV `83148fa62b35552608e46350b2a72788689aebab72b380d6dc568bdc7f6d176f`.
- Raw JSON SHA-256: host `795039a4a1d7cba54dd70061bc00e18c455955a79fbae4895ea50d41dbef758f`; device `7063c03c700e3d9a0a6bc0516a92a1cdf894c56dd9ea3fa20ba317301da14315`.
- Decode tok/s, prefill tok/s, TTFT, process-wide VRAM, system RAM, physical NVMe GB/token, physical H2D GB/token, GPU utilization, GPU bandwidth, cache hit rate, average Top-K, speculative acceptance, unique experts per verification block, quality, and coding quality: not measured.
- Decision: KDA is the largest single operation but not a majority. Add KDA-internal attribution before selecting a fusion. Retain host round trip as the default. B-0035 and B-0036 use different runner hashes and are not treated as a paired overhead measurement.

## Milestone 35 final local verification matrix

- Date: 2026-08-13.
- Portable CPU: CTest 20/20; Python 664 passed, 139 skipped.
- liburing/direct capability: CTest 21/21.
- ASan/UBSan with liburing: CTest 21/21.
- CUDA: CTest 36/36; actual default/M34/M35 schema compatibility 3/3 plus fast parser/contract regressions 10/10.
- Evidence: B-0030 through B-0036 focused suite 105/105; strict committed B-0036 artifact/manifest/oracle/runner/raw/CSV/aggregate rehash passed.
- Actual operation-attribution Compute Sanitizer: `ERROR SUMMARY: 0 errors`; exact routes/output/state, regional device closure, and zero unclassified device time.
- Production guard: the actual bounded two-layer artifact emits `NON_EXECUTABLE_ARTIFACT` and creates no output. The portable Python regression asserts child exit 4.

## Milestone 37 portable 3-bit correctness gate

- Date: 2026-08-13.
- Commit: `0b8a1e5`.
- Hardware: local 9800X3D development host; portable CPU execution under WSL2. The CUDA-enabled binary was also built, but the test selected the CPU backend.
- Model: deterministic synthetic K3-compatible checkpoint.
- Mode: group-32 signed 3-bit routed experts, incremental greedy generation, exact FP32 trunk.
- Correctness: prefill layer outputs and logits match the independently constructed quantized Python model at `1e-6` absolute and relative tolerance; six greedy tokens match exactly.
- Verification: C++ `ops` 1/1; focused Python/converter/reader/runtime regression 28/28; CUDA-enabled `k3x_run` build plus the same 3-bit CPU execution gate 1/1.
- Performance and traffic fields: not measured. No decode tok/s, prefill tok/s, TTFT, VRAM, RAM, NVMe, H2D, cache, Top-K, speculation, or quality result is claimed.

## Milestone 37 direct-packed CUDA correctness gate

- Date: 2026-08-13.
- Commit: `5f61c59`.
- Hardware: NVIDIA RTX 5080 under WSL2, CUDA 13.3, native `sm_120` build.
- Model: one literal 2-by-32 group-wise 3-bit matrix and the deterministic synthetic K3-compatible checkpoint.
- Mode: scalar transient direct-packed CUDA matvec with FP32 activation and output.
- Correctness: literal output matches the portable decoder/matvec at `1e-5`; complete synthetic prefill layers and logits match the quantized Python model at `1e-4`; six incremental greedy tokens match exactly.
- Transfer contract: the literal test requires weight H2D to equal the 24 packed bytes plus four BF16 scale bytes, activation H2D to equal 128 bytes, and D2H to equal eight bytes. No host FP32 weight extent is transferred.
- Verification: CPU `ops|backend` 2/2; focused CPU Python 27/27; CUDA `cuda_mxfp4|cuda_quant3` 2/2; synthetic CUDA parity 1/1; Compute Sanitizer `ERROR SUMMARY: 0 errors`.
- Performance fields: not measured. Test wall time is not reported as an inference benchmark.

## Milestone 37 bounded released-expert 3-bit quality gate

- Date: 2026-08-13.
- Commit: `eb02a4b`.
- Hardware: local 9800X3D development host under WSL2; this quality proxy ran on CPU.
- Model/checkpoint: released Kimi K3 layer 1, expert 0 from revision `9f62e4e9fffbd0a83ddd60e1c209d828994b3569`; source SHA-256 `ed3f07d595f37d90b1688de21ba0cdc012ee92c67dd92c460c0c73b2ef374a34`.
- Mode: native MXFP4 reference versus group-32 signed 3-bit with three least-squares group-scale fitting iterations; four deterministic random-normal input samples, seed `20260813`.
- Storage: 17,547,264 native tensor bytes become 14,450,688 3-bit tensor bytes and a 14,471,424-byte Reader-valid K3X artifact. The K3X file UUID is intentionally regenerated by conversion, so the stable evidence identity is the pinned source SHA-256 plus canonical named 3-bit payload SHA-256 `cade06a1c12ef136edee8ec1fe12b6bf488f918d057c06719304e48eacbd2594`. The pinned byte-recipe record SHA-256 is `6ca55a34579f89bb9fb4fcc56af1b5d5b3dde4e04a8c49fdcce91af8237cb2ae`.

| Boundary | Cosine | Relative L2 | RMSE | Maximum absolute error |
|---|---:|---:|---:|---:|
| Gate projection | 0.978184 | 0.207768 | 0.307734 | 1.281939 |
| Up projection | 0.978590 | 0.205833 | 0.304966 | 1.247936 |
| Down projection, same activation | 0.979802 | 0.200007 | 0.403610 | 1.733368 |
| Complete expert | 0.945909 | 0.325174 | 0.656195 | 2.751330 |

- Matrix relative L2 is 0.207333/0.207401/0.208238 for gate/up/down. Scale fitting improves complete-expert relative L2 from the initial max-scale codec's 0.369519 to 0.325174, but does not establish acceptable model quality.
- Quality scope: deterministic random-normal single-expert proxy only. Token agreement, perplexity, coding quality, routing changes, decode tok/s, TTFT, VRAM, RAM, physical NVMe/H2D traffic, and GPU utilization were not measured.
- Decision: fail the loss-minimization launch gate and keep the full 96-shard manufacture disabled pending a less lossy recipe or end-to-end evidence.

## Milestone 38 local quality manufacture start

- Date: 2026-08-13.
- Implementation commits: `abf52c6`, `35ec442`, and `84bdc9b`.
- Hardware: AMD Ryzen 7 9800X3D host, C-drive P44 Pro destination, D-drive two-slot staging, WSL2 converter. RTX 5080 was not used by these first two fragment conversions.
- Source: pinned `moonshotai/Kimi-K3` revision `9f62e4e9fffbd0a83ddd60e1c209d828994b3569` through authenticated HF Xet.
- Mode: native MXFP4 routed experts, group-128 signed 8-bit selected trunk matrices, sensitive BF16/F32 passthrough, 8 MiB bounded chunks, per-extent fsync/CRC, final root SHA-256, strict Reader reopen, and deletion only after ledger publication.

| Shard | Source bytes | K3X bytes | Quant8 matrices | Native expert tensors | Conversion seconds | Output SHA-256 |
|---|---:|---:|---:|---:|---:|---|
| 1 | 2,341,216,112 | 1,189,290,240 | 13 | 0 | 75.091 | `48ba2a106782c539e5b23d6412bbc797a5e779129aa4c4edd3cabf0923ec35aa` |
| 2 | 16,990,911,504 | 16,373,248,256 | 15 | 5,376 | 978.222 | `70ca52402d1583e93ebaeba25c00ac63c9ade6477e29a2700aa9947eb88d2fea` |
| 3 | 16,990,911,504 | 16,373,248,256 | 15 | 5,376 | not recorded | `40004c7b90c1f0606c1d63edb3dfc70541cbdf46d206e03c83c4dd1815277557` |
| 4 | 16,567,501,776 | 16,157,917,440 | 13 | 5,376 | not recorded | `9acfbd97ccb746d694951e3c878eae1239fd7e8dda842329fac9cd9dadcade91` |
| 5 | 16,990,911,504 | 16,373,248,256 | 15 | 5,376 | 735.660 | `e7eeece7da23f79d2271aa96d98c3c20698fb778552bf24c016f76ddfc294ed9` |
| 6 | 16,990,911,504 | 16,373,248,256 | not recorded | not recorded | not recorded | `6415fe154a539ecc96a5e722066564861d08884c706e2a1c4d7f453b2b8d7e43` |
| 7 | 16,990,911,504 | 16,373,248,256 | 15 | 5,376 | 598.707 | `ac077f3804a920ce24a333d538be2e4b3b02f1ae93de71bb27637e6365fd4b31` |
| 8 | 16,567,501,776 | 16,157,917,440 | 13 | 5,376 | 398.151 | `2a9f7c0d701e37c8c8f61b63a87f323a1f25547b796412a3762fecfd43e87e05` |
| 37 | 16,990,911,504 | 16,373,248,256 | 15 | 5,376 | 471.855 | `f4bae47cc6d2b0036939126e79e29ecdc1d418368d6aeed8b2eb184bd4ed5100` |
| 38 | 16,990,911,504 | 16,373,248,256 | 15 | 5,376 | 697.560 | `f816129454d08041042dc729740b4360f2bb0493af5b3aa2361d0b99be4c3383` |
| 39 | 16,990,911,504 | 16,373,248,256 | 15 | 5,376 | 452.074 | `00759cdb5a974d16d9526fd954db11ee8037f7cefcbdca6ddbd1b4118bde283e` |
| 40 | 16,567,507,176 | 16,157,917,440 | 13 | 5,376 | 401.198 | `f5c6953670fd96a25e16b7c9f77894aa9508d2c51aefb1088e825c41f6329894` |
| 41 | 16,990,916,912 | 16,373,248,256 | 15 | 5,376 | 719.230 | `474151aff5df919137c24537b3d2204f1ec301320ecfb2382bbf9cfb6c6017a9` |
| 67 | 16,990,911,504 | 16,373,248,256 | 15 | 5,376 | 408.273 | `e554928b86c77946125850597aa8578aa74756b7203e2f31edbee22f1dea53c5` |
| 68 | 16,567,501,776 | 16,157,917,440 | 13 | 5,376 | 694.020 | `aa57235520b41b75ce4c2aa6d24ce3376ca8411f5cf165058005fc09651d5e67` |
| 69 | 16,990,911,504 | 16,373,248,256 | 15 | 5,376 | 449.727 | `2b76f65baf9b53f4e97587a3703a97671ab238e78d107ee52c07bee96f2a5847` |
| 70 | 16,990,911,504 | 16,373,248,256 | 15 | 5,376 | 601.916 | `b603c8cc0459cf6f27c2c68382535a4c843b35b5eb9404995dcbb6e96e60e3ff` |
| 71 | 16,990,916,912 | 16,373,248,256 | 15 | 5,376 | 401.689 | `9664b1c70ffdae03c201b6c15fface6ae1fc3be2457e6db9c519a2e2f4cf5a31` |
| 72 | 16,567,507,176 | 16,157,917,440 | 13 | 5,376 | 397.259 | `3577f94e87be31af66f06225f8ef9b3f32e7091a4853aa34b3e3fea9b7576129` |

- Both source objects matched their official SHA-256. Both source files were deleted only after the quality IMMORTAL ledger recorded their output identity; they remain recoverable by checksum-bound redownload.
- Shards 1 through 7, 37 through 39, and 67 through 70 are complete and source-cleaned, for 14 durable ledger units. Shards 37 and 67 resumed existing partial artifacts, so their lower elapsed values are not comparable to full fresh conversion. Fresh shards 39 and 69 completed in 452.074/449.727 seconds after the combined audit, immutable-source deletion, and RAM-staging changes, 35.2%/35.2% below fresh shards 38/68. This combined-path observation does not isolate any one change and remains manufacturing elapsed, not inference throughput. Shard 70 overlapped a competing cross-worker copy before D-090 and took 601.916 seconds. Conversion timing was not emitted for shards 3, 4, or 6, so no elapsed value is reconstructed from timestamps.
- Shards 2 and 5 have identical 16,990,911,504-byte sources, 16,373,248,256-byte outputs, 15 Q8 matrices, 5,376 native expert tensors, and 2,716 K3X records. Shard 5 with D-077 batching plus D-079 source-alias streaming completed 242.562 seconds or 24.796% sooner, a 1.3297x elapsed-time ratio. The result does not isolate either optimization and is manufacturing throughput, not inference throughput.
- The strict Python Reader reopened official shard 1 in 11.401 seconds and then loaded finite `[7168]` BF16 input norm and `[12288,7168]` group-128 Q8 q-projection tensors by canonical name. The artifact contains 23 records. This one-run WSL2 `/mnt/c` diagnostic is not startup authority, decode throughput, or a token benchmark.
- The explicit sealed-set directory-open policy opened the same official shard-1 artifact and exposed the same 23 records in 0.004774 seconds without rescanning payload CRC/root bytes. Strict/default and sealed-open focused regressions pass 4/4. The roughly 2,388x open-elapsed ratio compares integrity policies and is neither token throughput nor evidence that periodic strict audits can be removed.
- Decode tok/s, prefill tok/s, TTFT, VRAM, GPU utilization, coding quality, token agreement, and physical inference traffic are not measured.
- Three-conductor download scheduling initially shared one Xet cache and serialized active Python children. After B/C restarted with per-conductor caches, a 20-second Windows Ethernet counter interval increased from 466,554,359,934 to 468,982,205,548 received bytes, or 121.39 MB/s and 971.14 Mb/s. This is host link traffic, includes protocol overhead, and is not effective checkpoint payload throughput.
- Three simultaneous independent-cache downloads later created three concurrent HDD assembly tails. D-093 bounded full download transactions to two; the first live 55-second two-slot interval increased Ethernet receive from 622,116,018,109 to 628,742,771,063 bytes, or 120.49 MB/s and 963.89 Mb/s. Only shard 10 and 43 `hf.exe` processes existed while shard 74 waited, confirming the capacity boundary.
- After replacing leaked semaphore capacity with D-094 process-owned mutex slots, conductors restarted at shards 11, 44, and 75 with the ledger unchanged at 25/96. Exactly two `hf.exe` downloads ran while the third conductor waited. Two host receive samples measured 2,256,574,867 bytes over 20 seconds, 107.60 MiB/s, and 108.49 MiB/s over 15 seconds. These are host-link observations with protocol overhead, not effective checkpoint payload throughput, conversion throughput, or inference throughput.
- Shard 44 then completed under the full D-094 and 128 MiB audit-read path in 398.970 seconds. Its 16,157,917,440-byte K3X fragment has SHA-256 `eaa3da6fbf5cb771c63f034d192ce9cf4a3eb858d971a2427e97591d07e80034`; the source was deleted after durable ledger publication, increasing completion to 26/96. This matches the prior clean 397–406 second manufacturing band but does not isolate either scheduling change and is not inference throughput.
- D-095 followed two live HF children that retained zero CPU, network, and D-drive I/O for roughly 10 and 15 minutes. After bounded restart and the PowerShell empty-exit-code correction, shard 45 completed in 407.314 seconds, published a 16,373,248,256-byte fragment with SHA-256 `dd96349e42b413c7d81a3c39bb9780e1095e7e4876fec9f0d3e449076533472f`, deleted its authenticated source, and raised the ledger to 29/96. No isolated timeout speedup is claimed.
- `Start-Job` prefetch remained at zero CPU, network, and write bytes even with explicit per-conductor Xet environment. D-096 replaced the job runspace with an owned child process. A live 25-second interval during shard 47/79 download and shard 14/78 conversion measured 116.65 MiB/s host receive. Shard 78 then completed in 589.095 seconds, published 16,373,248,256 bytes with SHA-256 `8fac53b47f135c758f7151bb8159ae3b8617a1f64c1a9437e388988f9ccd2749`, deleted its source, and raised completion to 35/96. Neither measurement is inference throughput or an isolated prefetch speedup.
- The same three-conductor path subsequently completed shards 14–17, 47–50, and 79–82. At the 2026-08-14 checkpoint the durable ledger contains 47/96 Reader-valid fragments totaling 751,990,071,040 bytes; every corresponding source shard was deleted only after official source SHA-256, finalized K3X root/file SHA-256, and ledger publication passed. Shards 50 and 82 completed in 1,019.505 and 1,017.078 seconds with output SHA-256 `1a506c7eac15ef789f4e4371b3f4072b2613ccc6da264c5eecac42b88660ead6` and `13d20f31c9db27f6035df5e6117a8c066cd605b78e82fe3cb9a748a3fff37752`. These elapsed values include serialized RAM staging and finalized-output audit waits; they are manufacturing wall times, not inference throughput or isolated optimization results.
- The durable ledger later reached 60/96 Reader-valid fragments totaling 964,196,305,920 bytes, covering shards 1–22, 37–54, and 67–86. Shards 22, 54, and 86 completed in 989.893, 1,005.372, and 975.209 seconds with output SHA-256 `f15cbb60e53acaeb548198975a560284d18340e2f55dc4009a59daa7ae0defc3`, `f2664455caa76be6f0639522eda95f039b6f36542b1b3501e54ed8544de51eb1`, and `1f9963653e45c712c5750ea4bed0f031567307b23f99cec9e6c19929d20c841d`. All corresponding source files were removed only after the unchanged correctness gates. These are manufacturing wall times with shared waits, not inference measurements.

## B-0046 complete official K3X first token

- Date: 2026-08-14.
- Commit: execution used `bd3bb5f`; result and documentation are committed in the following evidence commit.
- Hardware: AMD Ryzen 7 9800X3D, NVIDIA RTX 5080 16 GB, 96 GB host RAM, WSL2, K3X fragments on the local C-drive P44 Pro volume.
- Model/checkpoint: `moonshotai/Kimi-K3` revision `9f62e4e9fffbd0a83ddd60e1c209d828994b3569`; 96-fragment `K3XSET1`, 1,507,512,467,456 payload bytes, manifest record SHA-256 `f5c7443fd9ea9b4a2f0c95010f148182eefaedec4f29c094ca24e6bc4e61cefe`.
- Mode: natural Top-16, native MXFP4 routed experts, selected group-128 Q8 trunk matrices restored to their logical BF16 graph dtype, exact cold expert selection, no proxy, no pruning, Python compatibility execution.
- Context: one input token, ID 1. All layers 0–92 and the complete LM head executed.
- Correctness: K3X greedy token 9689 matches the independent original-precision token 9689. Selected logits are 8.290502548217773 and 8.307021141052246, for 0.016518592834472656 absolute and 0.1988509786% relative difference. All 93 layer records, the set identity, 449 final-state tensor digests, and the head record were verified.
- TTFT: 1,891 seconds from observed process start to result publication, with one-second timestamp resolution. This includes set finalization, per-layer Python processes, repeated metadata retrieval, state publication, storage, transfer, compute, and head scan.
- Layer intervals: 563.791264 seconds summed pre-expert load/attention/routing and 161.543706 seconds summed expert/dense compute. The remaining wall time is uninstrumented orchestration and repeated metadata/process overhead.
- CUDA memory: peak tracked allocation 3,602,630,144 bytes; peak tracked reservation 4,982,833,152 bytes. An eight-second diagnostic observed 4–7% SM utilization but is not an average utilization measurement.
- Traffic: downloaded model payload 0 bytes. The 136,990,317,568-byte sum is source-equivalent logical request accounting, not physical K3X/NVMe traffic. Physical NVMe GB/token, H2D GB/token, host RAM peak, cache hit rate, speculation, and memory bandwidth were not measured.
- Throughput and quality: decode tok/s and prefill tok/s were not measured; one TTFT token must not be reported as steady throughput. Token agreement is not a coding-quality or perplexity result.
- Evidence: head file SHA-256 `a9b3b32ffb65f4c4c341fd18defc9c613d18e3fa746bfe58a9258ab53439fb27`; comparison record SHA-256 `2ff90f75a63dfa467b67b60e3c0fefeac648aba9bdaf047820743545600c6dee`; layer-file chain SHA-256 `7fd29743f541b7c8c9f9817193df4f5cabbd77c5a04cd8f3eebd6f7c008869ab`.

## B-0047 official metadata cache boundary

- Date: 2026-08-14.
- Hardware: same local host under WSL2; cache stored below the D-drive first-token object directory.
- Boundary: official snapshot, 59.8 MB index, config, and the 5,404-tensor `model-00002-of-000096` header.
- Cold: 10.191660 seconds, nine HTTP requests, 60,618,467 response bytes.
- Warm: 1.751957 seconds, zero HTTP requests, zero response bytes.
- Change: 82.809896% lower elapsed at this isolated metadata boundary. Index/config digests and header tensor count are identical.
- Correctness: cached bodies are SHA-256 validated; the focused regression proves reuse across transport instances and corruption-triggered refetch. Seven focused transport/local-shard tests pass.
- Scope: component timing only. No token, tok/s, physical storage traffic, or full-run speedup is claimed.
- Evidence: `results/b0047-official-metadata-cache/summary.json`, record SHA-256 `a35a12b65b9057885cf2a045a5c6fa8237a471ff82f9e44db972bf98782f43c5`.

## B-0048 persistent official K3X first-token runtime

- Date: 2026-08-14.
- Commit: execution used d367487; implementation was committed through that head before the uninterrupted run.
- Hardware: AMD Ryzen 7 9800X3D, NVIDIA RTX 5080 16 GB, 96 GB host RAM, WSL2; sealed K3X fragments on the local C-drive P44 Pro volume and prefix-state files on D.
- Model/checkpoint: the same 96-fragment official K3X set and pinned Kimi K3 revision as B-0046.
- Mode: one Python process, natural Top-16, native MXFP4 experts, group-128 Q8 trunk restored to logical graph dtype, exact rescue, no proxy, no pruning, no speculation.
- Context: one input token, ID 1; layers 0 through 92 and the complete LM head.
- Correctness: all 93 layer output and state-manifest digests match B-0046. Greedy token 9689, selected logit 8.290502548217773, final normalized-hidden digest, and final state-manifest digest are identical.
- Timing: 1,156.152598 seconds full wall versus the 1,891-second compatibility baseline, a 734.847402-second or 38.860254% reduction and 1.635597x speedup. Stage wall was 21.898687 seconds for layer 0, 1,125.904976 seconds for layers 1 through 92, and 8.339971 seconds for the head.
- Layer attribution: recorded load/decode intervals sum to 529.554997 seconds and expert/dense compute intervals to 159.084610 seconds.
- Memory and traffic: maximum tracked CUDA allocation/reservation was 3,602,630,144/4,982,833,152 bytes. Downloaded payload was zero. A live process-I/O sample after layer 82 showed 79,236,866,016 physical read bytes; it is only a lower bound, not complete NVMe GB/token. Source-equivalent requested bytes were 134,641,464,320 and are not physical traffic. H2D bytes, bandwidth, average utilization, and complete host RAM peak were not measured.
- Throughput and quality: decode tok/s and prefill tok/s remain unmeasured because this is still one first token. No inverse-TTFT TPS or broad coding-quality claim is made.
- Evidence: results/b0048-persistent-official-runtime/summary.json SHA-256 a951b00ecc530359369f6181631f81c99950cc4d7f389ca2ea25ac84e01d60d1; full timing SHA-256 0776d86c1af9d1166b613109040fab0ef38a389e0a78e3c0b9010725732f29ea; full token SHA-256 fa108dfe21aecf25eed6acd0057f5526e21f527cd3bca67f8b76f6293892f8ec; 93-file chain SHA-256 75cc7591b02baeaf326a45a095376ee0ad8fa58eb070bfb5ee75c91d2dd06a53.

## B-0049 shared official runtime context

- Date: 2026-08-14.
- Commit: execution used 6a569c1.
- Hardware/model/mode: same host, sealed official K3X set, input token, natural Top-16, Q8 trunk, native MXFP4 experts, exact rescue, and one-process Python graph as B-0048. The only intended change is shared immutable metadata/header/store ownership.
- Correctness: zero mismatch across all 93 layer output/state records. Token 9689, logit 8.290502548217773, final normalized hidden, and final state digest match B-0048.
- Timing: 913.336487 seconds full wall, down 242.816110 seconds or 21.002081% from B-0048, for 1.265856x speedup. Relative to B-0046 the measured reduction is 51.700873% and speedup is 2.070431x.
- Attribution: layer load/decode intervals total 430.579377 seconds and compute intervals total 159.572791 seconds. Stage wall is 14.853027 seconds for layer 0, 883.793521 seconds for layers 1 through 92, and 6.248858 seconds for the head.
- Memory/traffic: maximum tracked CUDA allocation/reservation remains 3,602,630,144/4,982,833,152 bytes. Downloaded payload is zero. Requested source-equivalent bytes are 134,641,464,320 and are not physical traffic. Complete physical NVMe/H2D traffic and average utilization were not measured.
- Throughput/quality: this remains one-token TTFT. Decode tok/s, prefill tok/s, and coding quality are not measured, and no inverse-TTFT TPS is claimed.
- Evidence: summary SHA-256 a30f81225d8b699dfe11c529d832874984733fe7d8e762a0ac235a757284e6fb; full timing SHA-256 61fdc4e76b5184f7a8267215c98c6950c93fa9332171a17c9655ce1b3bcbb633; full token SHA-256 8fab7bb4565d144eac54e4e0f47a2468fd26b72206aca0f42a6e4c93ab4d46e7; 93-file chain SHA-256 ff23ff5eabd394997bf44e2a97dcb85b2401344c937981645902459c35ace90d.

## B-0044 official original-precision first token

- Date: 2026-08-14.
- Commit: execution spanned the pinned M36/M37 official graph on branch `codex/official-end-to-end-token`; compact result file SHA-256 is `e4b57d6dde9f59e205c2e6b40be8908b45d255ef9d29e8b4465fcb634c13beb9`.
- Hardware: AMD Ryzen 7 9800X3D host, NVIDIA RTX 5080 under WSL2, D-drive content-addressed range cache, C-drive durable prefix state.
- Model: `moonshotai/Kimi-K3` revision `9f62e4e9fffbd0a83ddd60e1c209d828994b3569`, exact 93-layer topology record `6ca55a34579f89bb9fb4fcc56af1b5d5b3dde4e04a8c49fdcce91af8237cb2ae`.
- Mode: original released precision, natural Top-16 routing, input token ID 1, exact KDA/MLA recurrent state, chunked LM-head greedy selection.
- Result: all layers 0 through 92 completed; generated token ID `9689` with FP32 logit `8.307021141052246`. Final normalized hidden SHA-256 is `84f7422be9d329bfdb0a3971f066de7c67ac137befd2e4e083afd051e1652b6a`.
- Head boundary: 2,348,853,248 requested/downloaded payload bytes, 283 range requests, 91.725 seconds wall, and 209,960,960 peak CUDA allocated bytes. This is a one-off chunked-head correctness execution with cold payload materialization, not TTFT or steady-state throughput.
- Decode tok/s, prefill tok/s, TTFT, end-to-end full-token wall, process-wide VRAM, RAM, physical NVMe GB/token, physical H2D GB/token, utilization, bandwidth, cache hit rate, and coding quality: not measured.

## B-0050 device-side Q8 decode

- Date: 2026-08-14.
- Commit: execution used `7cb9498`.
- Hardware/model/mode: same 9800X3D, RTX 5080 16 GB, 96 GB RAM, WSL2, sealed official K3X set, input token 1, natural Top-16, exact native MXFP4 experts, no proxy/pruning/speculation, and one-process graph as B-0049. The intended change is compressed Q8 transfer and CUDA BF16 reconstruction.
- Released-tensor boundary: layer-0 q-projection shape 12,288 by 7,168, 88,080,384 code bytes and 1,376,256 scale bytes. CPU decode-plus-copy median was 0.520519 seconds; CUDA decode median was 0.441014 seconds, a 1.180276x speedup. BF16 output is bit-exact and CUDA peak allocation was 177,537,024 bytes. Physical H2D bytes were not measured.
- Correctness: all 93 layer output/state records match B-0049. Token 9689, logit 8.290502548217773, final normalized hidden, and final state-manifest digest are identical. All 96 raw record digests validate.
- Timing: 583.658078 seconds full wall versus 913.336487 seconds for B-0049, a 329.678409-second or 36.096052% reduction and 1.564849x speedup. Relative to B-0046, the reduction is 69.134951% and speedup is 3.239911x. Stage wall is 7.679511 seconds for layer 0, 567.702803 seconds for layers 1 through 92, and 6.030117 seconds for the head.
- Attribution: recorded load/decode intervals total 368.422976 seconds, 14.435527% below B-0049. Compute intervals total 157.256886 seconds, 1.451316% below B-0049.
- Memory/traffic: maximum tracked CUDA allocation/reservation remains 3,602,630,144/4,982,833,152 bytes. Downloaded payload is zero. Layer source-equivalent requests total 134,641,464,320 bytes and the head adds 2,348,853,248 bytes; neither is physical traffic. Complete physical NVMe/H2D traffic, host RAM peak, and average utilization were not measured.
- Throughput/quality: this is still one-token TTFT. Decode tok/s, prefill tok/s, and coding quality remain unmeasured; no inverse-TTFT TPS is claimed.
- Evidence: `results/b0050-device-q8-decode/summary.json` SHA-256 `7685846e780f594791bfb6e86354f5b65f91efc00d1e8cdec4d7345ff7b17c3d`; full timing SHA-256 `a8777e874e07599811623aae1f9f1281b3190e03bb054cc3f9366b85dca8443c`; full token SHA-256 `f3f84f487df462e72882c459d79688865e9de2a83d033fe3afd51e09290c1d63`; 93-file chain SHA-256 `f51ca63ea36d356b91cb46c7e7ad181320710824b2ba46dc225f365cabaa2b86`; microbenchmark SHA-256 `d3a08597ff1a021c1b8e84dbbbf3fe38327696c4bd9ab6915cb3bf21ed5645c8`.

## B-0051 direct-packed Q8 matvec

- Date: 2026-08-14.
- Commit: execution used `f06d078`.
- Hardware/model: same 9800X3D, RTX 5080 16 GB, 96 GB RAM, WSL2, sealed official K3X set, and input token 1 as B-0050.
- Mode: experimental `--direct-q8`; large dense-MLP and Stable LatentMoE group-128 Q8 matrices execute from packed codes/scales. KDA/MLA attention tensors, natural Top-16 routing, native MXFP4 experts, and the LM head retain their B-0050 paths. No proxy, pruning, or speculation.
- Released q-projection: cold materialized/direct medians are 0.463762/0.397872 seconds, a 1.165607x speedup. Resident materialized/direct medians are 0.0116363/0.000288751 seconds, a 40.298662x speedup. Direct peak allocation is 123,103,232 bytes versus 650,131,456 bytes materialized. BF16 exact ratio is 57.4788%, maximum absolute error 0.03125, and mean absolute error 0.00147309.
- Full timing: 566.002323 seconds versus B-0050's 583.658078 seconds, a 17.655755-second or 3.025017% reduction and 1.031194x speedup. Stage wall is 7.503913 seconds layer 0, 550.634495 seconds layers 1 through 92, and 5.546185 seconds head.
- Attribution: recorded load/decode falls from 368.422976 to 260.212333 seconds, while compute rises from 157.256886 to 256.880227 seconds because cold packed reads/H2D now occur inside lazy matvec calls. This boundary movement is why only full wall is decisive.
- Routing/quality: 83/92 layers retain the same Top-16 set; 36/92 retain the same ordered tuple. Mean/minimum expert overlap is 15.9022/15 of 16. Final-hidden cosine is 0.9999314, exact-element ratio 17.7316%, maximum error 0.046875, and mean error 0.00671871. Greedy token remains 9689; logit changes from 8.290502548217773 to 8.28560733795166, absolute delta 0.00489521. Coding quality is not measured, so this mode is non-default.
- Memory/traffic: layer peak CUDA allocation/reservation falls from 3,602,630,144/4,982,833,152 to 1,381,369,344/1,589,641,216 bytes. Downloaded model payload is zero. Complete physical NVMe/H2D traffic and host RAM peak are not measured.
- Throughput: this is one-token TTFT, not steady decode. Decode/prefill tok/s remain unmeasured.
- Evidence: `results/b0051-direct-q8-matvec/summary.json` SHA-256 `8af87c939adde4abd312dfc4a79dc3b3b20c734869fa30c43b21939ea6b3c0cd`; full timing SHA-256 `eb9aca8d0b1593473e0fe6fc1f5fd9ef767d817ea800a95a1bf93719ad84d2f1`; full token SHA-256 `588a12bc8ca8a449a6c27644181b0ab96953fc474f936831ac09b6b412366b57`; 93-file chain SHA-256 `95b5a88107ced328ed319d842d717fe76ac6b6969b1b305c609e2847378ebc39`; q-projection SHA-256 `8c5ea2eaa270577f2fd3a360899763869b583e5ee1157b6420b6cddddc5d95cb`.

## B-0052 packed Q8 residency

- Date: 2026-08-14.
- Hardware/model: AMD Ryzen 7 9800X3D, RTX 5080 16 GB, 96 GB RAM, WSL2, sealed official K3X set, input token 1, official layer 0.
- Mode: experimental direct-Q8 dense MLP, 1 GiB device budget, zero host budget, stable first admission, one shared runtime context, no eviction.
- Result: cold/warm wall is 11.149552/2.638550 seconds, a 4.2256x ratio. Three packed matrices occupy 738,017,280 bytes; the cold pass records three misses and three admissions, and the warm pass records three device hits with no new miss. Output and KDA-state digests match across the two runs.
- Scope: layer-0 boundary only. Physical NVMe/H2D bytes, token throughput, prefill, TTFT, and coding quality were not measured.
- Evidence: `results/b0052-packed-q8-residency/summary.json` SHA-256 `3442625c2b13a43bb39f59e214b05c001de757a744d29ee77b070e72f93637e5`.

## B-0055 layer-0 direct KDA and packed Q8 residency

- Date: 2026-08-14.
- Hardware/model: same local hardware, sealed checkpoint, input token, and layer boundary as B-0052.
- Mode: experimental direct-packed group-128 Q8 for the eight KDA and three dense-MLP projection matrices, 4 GiB device budget, zero host budget, stable first admission, one shared runtime context.
- Result: cold wall is 7.345563 seconds. Five warm walls are 0.213557, 0.209281, 0.210865, 0.217803, and 0.215707 seconds; the median is 0.213557 seconds and the cold/median ratio is 34.3963x. Eleven matrices occupy 1,188,528,640 bytes; every warm pass records 11/11 L0 hits. All six output/KDA-state digests match. Peak tracked CUDA allocation/reservation is 1,259,244,032/1,272,971,264 bytes on warm passes.
- Quality versus B-0052: final-hidden cosine 0.9999964, exact-element ratio 54.4224%, maximum absolute error 0.000244141, and mean absolute error 0.0000201179. Recurrent-state maximum/mean absolute error is 0.00170143/0.000000437602. Coding quality and full-token routing are unmeasured, so this remains experimental.
- Scope: layer-0 boundary only; no decode/prefill tok/s, TTFT, physical NVMe/H2D traffic, utilization, or bandwidth claim.
- Evidence: implementation commit `7f8fc05`, median-harness commit `300fb75`, summary SHA-256 `8de699de6aca2c0ff21f9dc93e50fcfa44112b1dd3a97a691f8fcb7736260b54`, and quality record canonical SHA-256 `f097abfcc6db8cec2a97a144bfbb7ec7b36b2f62cbb5884c0ecec8fa4ccf42cb`.

## B-0056 native MXFP4 expert residency

- Date: 2026-08-14.
- Hardware/model: AMD Ryzen 7 9800X3D, RTX 5080 16 GB, 96 GB RAM, WSL2, sealed official K3X set, layer 1 expert 0.
- Mode: experimental native packed MXFP4 CUDA matvec, three matrix-granular admissions, explicit device cache, one CPU oracle, one cold call, and five warm calls.
- Result: CPU oracle wall is 0.229109 seconds and cold CUDA wall is 0.305275 seconds. Warm walls are 0.013616, 0.013732, 0.014179, 0.013833, and 0.013668 seconds; median is 0.013732 seconds and cold/median ratio is 22.2308x. Three matrices occupy 17,547,264 bytes and every warm call records three device hits.
- Quality: cosine is 0.9999999999998764, maximum absolute error is 0.0000014007, and BF16 exact-element ratio is 99.9721% against the CPU oracle.
- Scope: one expert FFN only. No token throughput, TTFT, physical traffic, utilization, or coding quality was measured.
- Evidence: implementation commit `7f72416`, harness commit `42b0e17`, summary SHA-256 `06d45f8ea8bc2e6ac34b25e08880a876637343b49caed68eb26a896f7a331797`.

## B-0057 through B-0059 native MXFP4 scalar experiments

- Date: 2026-08-14.
- Hardware/model: same official expert boundary and local hardware as B-0056.
- B-0057: warp reduction plus bitwise E8M0 records a 0.013952-second warm median, 1.60% slower than B-0056. It was rejected and reverted. Evidence SHA-256 is `326d4662e9bb3a5facabb6e2c9fd2e176100ea6ca7d253ef17bf072e70d722e1`.
- B-0058: stage attribution reports roughly 4–5 ms each for gate, up, and down projection events and below 0.1 ms for SiTU on warm calls. Its cold wall is invalid for comparison because source reversion triggered lazy extension rebuild. Evidence SHA-256 is `5d47f614f0b11ea419186b3a17a06f573e564f5ebfeed2343d029ec4ef4192c5`.
- B-0059: isolated bitwise E8M0 records a 0.013976-second warm median, also slower than B-0056. It was rejected and reverted. Evidence SHA-256 is `61f16a8c391ab283df04782b05d99f062d39f62826d8224cdb1135069d103710`.
- Scope: scalar kernel diagnostics only. None is a tok/s result.

## B-0060 official layer-1 expert-major MXFP4 batch

- Date: 2026-08-14.
- Hardware/model: same local hardware and sealed official K3X set, input token 1, official KDA-MoE layer 1, natural Top-16 route.
- Mode: experimental direct Q8 where previously supported plus one native MXFP4 expert-major batch; 48 expert matrices resident in 280,756,224 bytes.
- Result: scalar direct reference wall is 6.639758 seconds. Five-run batch warm median is 2.845790 seconds and measured batch compute median is 0.239126 seconds, yielding a 2.333x scalar/warm ratio. Every warm pass records 48 MXFP4 cache hits.
- Quality: the batched output is BF16 bit-exact to the scalar direct reference for route `[498, 730, 748, 15, 14, 66, 873, 104, 394, 303, 236, 635, 162, 212, 814, 5]`.
- Scope: one layer only. Decode/prefill tok/s, TTFT, physical NVMe/H2D traffic, and coding quality were not measured.
- Evidence: implementation commit `fa92e40`, harness commit `5bf0313`, summary SHA-256 `d7f7c8404d5532a7d783d15bc765741d85ceedad88664afa5c18c9452f30ea2d`.

## B-0061 official layer-1 full packed residency

- Date: 2026-08-14.
- Hardware/model: same local hardware, sealed official K3X set, input token 1, official layer 1, natural Top-16 routing.
- Mode: experimental direct-packed Q8 for every self-attention projection plus native expert-major MXFP4. The shared context also executes layer 0 for recurrent-state setup.
- Result: exact-Q8 layer-1 reference wall is 5.449819 seconds and scalar-direct wall is 5.114390 seconds. Batch cold wall is 5.290414 seconds. Five warm walls are 0.458119, 0.453030, 0.448773, 0.425937, and 0.427000 seconds; median is 0.448773 seconds. Warm compute median is 0.240653 seconds. The B-0060-to-B-0061 warm-wall ratio is 6.341x and scalar-direct/warm ratio is 11.396x.
- Residency: 24 Q8 matrices across setup and layer execution occupy 1,825,408,000 bytes; layer 1 records 13 Q8 hits per warm pass. Forty-eight MXFP4 matrices occupy 280,756,224 bytes and record 48 hits per warm pass.
- Quality: route overlap is 16/16. The batch output is bit-exact to scalar direct. Against exact Q8, cosine is 0.9999961257, maximum absolute error is 0.0009765625, mean absolute error is 0.0000491009, and BF16 exact-element ratio is 56.3058%.
- Scope: one layer boundary, not a complete token. Decode/prefill tok/s, TTFT, full-model cache behavior, physical traffic, and coding quality remain unmeasured.
- Evidence: implementation commit `ad48c9a`, harness commit `a82bd05`, summary SHA-256 `7e58a75115ba4c737386eccb815a7e6f1ec7fb5e54c679086a1363e561d688ab`.

## B-0062 natural Top-16 in-memory decode

- Date: 2026-08-14.
- Hardware/model: AMD Ryzen 7 9800X3D, RTX 5080 16 GB, 96 GB RAM, WSL2, complete sealed official K3X set, input token 1, two generated tokens.
- Mode: experimental direct Q8, native expert-major MXFP4, natural Top-16, persistent in-memory KDA/MLA/hidden/block state, 20 GiB Q8 host plus 3 GiB device cache, and 20 GiB MXFP4 host plus 6 GiB device cache.
- Correctness: generated tokens are `[9689, 10]`; the first token matches B-0050. All 93 attention states persist and maximum MLA length is two. The direct-Q8 path remains experimental and coding quality is unmeasured.
- Result: first-token wall is 824.624471 seconds. The second token is 380.562451 seconds, yielding measured decode throughput of 0.00262769 tok/s.
- Cache result: final Q8 counters are 65 device hits, 456 host hits, 1,797 misses, and 59,868,323,840 rejected bytes. Final MXFP4 counters are 126 device hits, 498 host hits, 8,208 misses, and 20,097,466,368 rejected bytes.
- Resource result: peak CUDA allocation/reservation is 10,472,534,016/11,765,022,720 bytes. Physical NVMe/H2D traffic, utilization integration, prefill, and coding quality were not measured.
- Evidence: implementation commit `57bc29f`, summary SHA-256 `78b30848356e39cc609c801ac28b739a9908c1b2e43e526ab6ca0e96a02e1949`.

## B-0063 fixed Top-4 with RAM-resident Q8 trunk

- Date: 2026-08-14.
- Hardware/model: same local hardware and complete sealed official K3X set as B-0062, input token 1, two generated tokens.
- Mode: explicitly lossy fixed Top-4, experimental direct Q8, native expert-major MXFP4, persistent in-memory state, 60 GiB Q8 host plus 2 GiB device cache, zero MXFP4 host plus 7 GiB device cache.
- Quality: generated tokens are `[21339, 13500]`. The first token differs from B-0050's 9689, so this mode fails exact greedy-token parity. Coding/agentic quality remains unmeasured.
- Result: first-token wall is 445.455813 seconds. The second token is 71.624887 seconds, yielding 0.0139616 tok/s. This is 5.313x B-0062 throughput and remains 358x below the 5 tok/s target.
- Cache result: all 1,159 Q8 matrices fit across RAM and VRAM; the second token adds zero Q8 misses and records 41 device plus 1,118 host hits. MXFP4 records 153 second-token device hits and 951 new misses, including 4,503,797,760 rejected bytes after the device cache fills.
- Resource result: peak CUDA allocation/reservation is 10,365,210,112/10,812,915,712 bytes. Physical NVMe/H2D traffic, utilization integration, prefill, and coding quality were not measured.
- Evidence: implementation commit `35c2eb2`, summary SHA-256 `c51840db3f8ae3f170b1639463e52137ae657cf277919926152c61b702f46c34`.

## B-0064 fixed Top-4 while populating the ext4 extent hot bank

- Date: 2026-08-14.
- Hardware/model: AMD Ryzen 7 9800X3D, RTX 5080 16 GB, 96 GB RAM, WSL2, complete sealed official K3X set on DrvFS, 64 GiB persistent ext4 cache, input token 1, three generated tokens.
- Mode: explicitly lossy fixed Top-4, experimental direct Q8, native expert-major MXFP4, persistent in-memory state, 60 GiB Q8 host plus 2 GiB device cache, zero MXFP4 host plus 7 GiB device cache.
- Quality: generated tokens are `[21339, 13500, 17830]`; the first token differs from exact 9689. Coding/agentic quality remains unmeasured.
- Result: token walls are 459.182465, 78.840365, and 65.266149 seconds. The two decode tokens total 144.106514 seconds, or 0.0138786 tok/s; the last token alone is 0.0153219 tok/s.
- Cache result: 2,782 extents occupy 18,369,727,840 bytes, with 365 hits, 2,782 misses, and zero rejected bytes at completion. All Q8 matrices remain resident across RAM/VRAM after first use.
- Resource result: physical NVMe/H2D traffic, utilization integration, prefill, and coding quality were not measured.
- Evidence: implementation commit `6d649d5`, canonical record SHA-256 `564713f213b437e07fd65a73bd03952eec7a081b24b8e44339bd0a0d52fcf402`, summary-file SHA-256 `f7c58fdd285536c68dd611153d8b9823b1d8a0cc21aa7ab530cdf5d5e90d51a5`.

## B-0065 fixed Top-4 full ext4 hot-bank replay

- Date: 2026-08-14.
- Hardware/model: same local hardware, sealed set, cache budgets, and three-token input boundary as B-0064; the 18,369,727,840-byte ext4 extent bank is already populated.
- Mode: same explicitly lossy fixed Top-4 path as B-0064.
- Quality: generated tokens are again `[21339, 13500, 17830]`, proving deterministic replay but not exact or coding-quality parity.
- Result: token walls are 416.128496, 47.048368, and 41.308851 seconds. The two decode tokens total 88.357219 seconds, yielding 0.0226354 tok/s. The last token alone is 0.0242079 tok/s. Average throughput is 1.621x B-0063 and remains about 221x below 5 tok/s.
- Cache result: 3,147 persistent extent hits, zero misses, zero admissions, and zero rejected bytes. Q8 records 82 device and 2,236 host hits across the two decode tokens; MXFP4 records 288 device hits but still cycles nonresident experts through CUDA.
- Resource result: peak CUDA allocation/reservation is 10,365,603,328/10,812,915,712 bytes. Physical NVMe/H2D traffic, utilization integration, prefill, and coding quality were not measured.
- Bottleneck: the fully hot last-token wall is consistent with roughly 92 times the B-0061 resident layer boundary. Storage is no longer the leading explanation; Q8 trunk H2D, 1,159 matvec calls, expert execution, and Python orchestration are next.
- Evidence: implementation commit `6d649d5`, canonical record SHA-256 `3bde0691f9c70f7734f566fa101349eaa32acc123df274c29dfbf08e94c8e445`, summary-file SHA-256 `d15ca5866c7ec89610d5c0ca8125fbd9c4b91d88a3bdf2e9720502c330ea0b4a`.
