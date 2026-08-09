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

## Derived bottleneck model — not a benchmark

The released dimensions imply 17,547,264 bytes per native MXFP4 routed expert. With no cache reuse, natural Top-16 across 92 MoE layers implies 25,829,572,608 expert bytes/token. Applying the P44 Pro published 7.0 GB/s sequential figure gives a derived expert-only ceiling of about 0.271 tok/s and implies roughly 94.6% expert NVMe-byte avoidance for a 5 tok/s target.

These values are capacity and traffic estimates. They are not inserted into B-0001's NVMe field and must be replaced by Linux block-I/O measurements when the tiered runtime exists.

## Pending benchmark gates

- Native Linux repetition of B-0002; WSL2 is the development path, not final performance authority.
- Native-Linux repetition of B-0004/B-0005/B-0006 and a larger KDA/MLA or decoder subgraph boundary.
- Full-dimension bounded-slice runtime before any full-model throughput claim.
- L2 runtime physical NVMe, utilization, memory-bandwidth, and storage I/O-stall counters.
