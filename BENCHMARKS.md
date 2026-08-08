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

## Derived bottleneck model — not a benchmark

The released dimensions imply 17,547,264 bytes per native MXFP4 routed expert. With no cache reuse, natural Top-16 across 92 MoE layers implies 25,829,572,608 expert bytes/token. Applying the P44 Pro published 7.0 GB/s sequential figure gives a derived expert-only ceiling of about 0.271 tok/s and implies roughly 94.6% expert NVMe-byte avoidance for a 5 tok/s target.

These values are capacity and traffic estimates. They are not inserted into B-0001's NVMe field and must be replaced by Linux block-I/O measurements when the tiered runtime exists.

## Pending benchmark gates

- Native Linux repetition of B-0002; WSL2 is the development path, not final performance authority.
- Persistent-buffer and layer/block-batched CUDA ablation against B-0002.
- Full-dimension bounded-slice runtime before any full-model throughput claim.
- Tiered-runtime NVMe, pinned H2D, cache-hit, utilization, memory-bandwidth, and I/O-stall counters.
