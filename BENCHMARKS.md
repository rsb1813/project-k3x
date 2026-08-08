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

## Derived bottleneck model — not a benchmark

The released dimensions imply 17,547,264 bytes per native MXFP4 routed expert. With no cache reuse, natural Top-16 across 92 MoE layers implies 25,829,572,608 expert bytes/token. Applying the P44 Pro published 7.0 GB/s sequential figure gives a derived expert-only ceiling of about 0.271 tok/s and implies roughly 94.6% expert NVMe-byte avoidance for a 5 tok/s target.

These values are capacity and traffic estimates. They are not inserted into B-0001's NVMe field and must be replaced by Linux block-I/O measurements when the tiered runtime exists.

## Pending benchmark gates

- Milestone 1 CPU profiler baseline.
- Milestone 1 RTX 5080 cuBLASLt dense throughput baseline. FP32/BF16-rounded literal correctness, exact transfer-byte accounting, and CUDA-event instrumentation are implemented, but no throughput result is recorded yet.
- Milestone 1 RTX 5080 custom MXFP4 throughput baseline. Native-byte literal correctness, exact transfer-byte accounting, stride coverage, and CUDA-event instrumentation are implemented, but no throughput result is recorded yet.
- Milestone 1 end-to-end synthetic CPU versus CUDA comparison.
