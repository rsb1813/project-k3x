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

## Pending benchmark gates

- Native Linux repetition of B-0002; WSL2 is the development path, not final performance authority.
- Native-Linux repetition of B-0004/B-0005/B-0006 and a larger KDA/MLA or decoder subgraph boundary.
- Native-Linux repetition of B-0008 with disclosed warm/cold preparation before selecting an L2 default.
- Native-Linux repetition of B-0009 with representative multi-expert pressure and controlled warm/cold preparation before selecting any deadline policy.
- Native-Linux repetition of B-0010 with a representative routing trace, full-size experts, and controlled warm/cold preparation before selecting any cache policy.
- Native-Linux repetition of B-0011 with repository-duration sessions and controlled helpful, stale, and adversarial priors before selecting any profile policy.
- Native-Linux repetition of B-0016 with physical NVMe accounting, GPU utilization, memory bandwidth, multi-expert/full-layer groups, and representative acceptance distributions before any speculative default claim.
- Persistent-state AURORA parity and representative native-Linux measurement with physical I/O, realistic acceptance, coding quality, and resident-expert pressure before any self-speculative default claim.
- Multi-expert or full-layer bounded slices before claiming cache-pressure or locality behavior.
- L2 runtime physical NVMe, utilization, memory-bandwidth, and storage I/O-stall counters.
