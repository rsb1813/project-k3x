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

## Derived bottleneck model — not a benchmark

The released dimensions imply 17,547,264 bytes per native MXFP4 routed expert. With no cache reuse, natural Top-16 across 92 MoE layers implies 25,829,572,608 expert bytes/token. Applying the P44 Pro published 7.0 GB/s sequential figure gives a derived expert-only ceiling of about 0.271 tok/s and implies roughly 94.6% expert NVMe-byte avoidance for a 5 tok/s target.

These values are capacity and traffic estimates. They are not inserted into B-0001's NVMe field and must be replaced by Linux block-I/O measurements when the tiered runtime exists.

## Pending benchmark gates

- Native Linux repetition of B-0002; WSL2 is the development path, not final performance authority.
- Native-Linux repetition of B-0004 and a larger KDA/MLA or decoder subgraph boundary.
- Full-dimension bounded-slice runtime before any full-model throughput claim.
- Tiered-runtime NVMe, pinned H2D, cache-hit, utilization, memory-bandwidth, and I/O-stall counters.
