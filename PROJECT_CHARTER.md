# Project K3X: TITAN Charter

## Constitutional role

This document preserves the stable goals and constraints of Project K3X: TITAN. Implementation details belong in `ARCHITECTURE.md`; current progress belongs in `PROJECT_STATE.md`; accepted and rejected choices belong in `DECISIONS.md`; measurements belong in `BENCHMARKS.md`.

Changes to this charter require an explicit user instruction and a corresponding decision record. Ordinary implementation progress must not rewrite it.

## Mission

Design and implement, from first principles, a Kimi K3-specific out-of-core inference engine that runs as fast as practical on one consumer PC while preserving measurable correctness and agentic/coding quality.

K3X is not a llama.cpp or vLLM fork with a few added options. Existing projects and papers may supply ideas and kernels, but K3X owns a K3-specific runtime, storage format, scheduling model, and evidence trail.

## Priority order

1. Correctness.
2. Measured end-to-end tokens per second.
3. Minimum NVMe, system-RAM, and PCIe traffic.
4. Preservation of coding and agentic quality.
5. Checkpoint size.
6. General benchmark performance.

No tokens-per-second result may be promised before measurement. If hardware makes a target impossible, the project reports the measured bottleneck and an explicitly labeled theoretical ceiling.

## Target hardware and environments

- CPU: AMD Ryzen 7 9800X3D.
- GPU: NVIDIA RTX 5080 with 16 GB VRAM.
- System RAM: 96 GB DDR5-4200.
- Storage: Solidigm P44 Pro 2 TB NVMe.
- Primary final runtime: Linux native.
- Conversion and calibration: bounded shard work on Google Cloud Run Jobs or low-cost GPU VMs.
- The full source checkpoint is never assumed to fit in RAM or VRAM.
- Paid cloud resources are never provisioned automatically.

## Correctness constitution

- A small deterministic synthetic K3-compatible model must exercise KDA, MLA, Attention Residual, Stable LatentMoE, routing, native MXFP4 expert decode, incremental decoding, persistent KV/KDA state, and greedy generation.
- PyTorch is the numerical reference for the synthetic graph.
- Layer outputs, logits, persistent state, and final tokens must be compared with explicit tolerances.
- Every optimization must have a switchable reference mode that disables it.
- Prefetch and residency may change latency but must not change natural routing in exact modes.
- Native MXFP4 expert payloads must remain available without forced dequantization and requantization.
- Permanent pruning and cold-expert proxies are optional lossy experiments, never correctness defaults.
- Quality changes must be measured alongside speed changes.

## K3X checkpoint constitution

The K3X format sacrifices generic interchangeability for K3 execution and streaming. It must support fixed superblocks, versioning, tensor and per-layer directories, aligned extents, checksums, crash-safe and resumable conversion, hot/cold expert separation, task-profile metadata, quantization metadata, per-expert random access, and large sequential reads.

Weights may be packed by execution and prefetch order rather than tensor identifier. An optional profile-guided repacker may colocate experts that are observed together. Conversion must read, transform, write, verify, and release bounded shard or layer units so a worker with roughly 32 GB RAM can make progress without whole-model residency.

## Precision constitution

- Trunk and non-expert weights may use sensitivity-aware 2-, 3-, 4-, 6-, or 8-bit formats or BF16 passthrough.
- Calibration allocates bits by tensor, layer, or channel sensitivity and must preserve higher precision for routers, norms, and sensitive outliers when justified.
- The tooling must generate measured quality-versus-bytes Pareto results.
- Native expert MXFP4 remains an exact supported path.

## Runtime memory constitution

The intended hierarchy is L0 RTX 5080 VRAM, L1 system RAM, and L2 NVMe. Cache decisions must combine observed and predicted usefulness rather than defaulting blindly to LRU or LFU. Candidate signals include global, task, and session frequency; expert transitions; recency; predicted next use; transfer latency; size; current residency; and speculative-block usefulness.

Least-Stale expert caching must be reproduced from the original SpecMD source and benchmarked against LRU and LFU. Policies remain runtime-switchable.

Runtime-only task/session metadata may initialize cache priors without being inserted into K3 prompt tokens. Observed routing gradually outweighs the prior. Persistent profiles may retain expert frequency, transitions, hot banks, prefix and KDA metadata, and repository-specific statistics.

Full 896-way router information should remain available in fast modes. A high-scoring nonresident expert triggers exact MXFP4 rescue through NVMe, RAM, and GPU. Repeated use may promote it. Dynamic residency is distinct from permanent pruning.

## Adaptive quality constitution

Natural Top-16 is the reference. Fast modes may support Top-4, 6, 8, 12, or 16 and token/layer-adaptive selection using routing entropy, cumulative mass, confidence, speculative acceptance, logit entropy, and observed agent failures.

Repeated compiler, test, tool, or agent failure may raise quality from fast execution toward K8, K12, or exact K16. This escalation must be observable and reversible.

The stable quality modes are as follows.

| Mode | Constitutional behavior |
|---|---|
| `QUALITY` | High-precision trunk, natural Top-16, exact expert routing, strict speculative verification, no proxy |
| `BALANCED` | Mixed quantization, adaptive K, exact cold rescue, full routing, cost-aware prefetch |
| `HYPERTURBO` | Aggressive mixed quantization, adaptive K4/6/8, task hot bank, experimental verifier budgeting, expert-aware speculation, high-score exact rescue |
| `EXTREME` | Explicitly lossy proxy or pruning experiments and tighter expert budgets |

Any later `AUTO` mode must select among these behaviors transparently and retain escalation to `QUALITY`.

## I/O and scheduling constitution

- GPU compute must not synchronously wait for storage by design.
- The minimum intended pipeline overlaps layer `N` compute, RAM-to-VRAM transfer for `N+1`, and NVMe-to-RAM transfer for `N+2`.
- Linux `io_uring`, `O_DIRECT`, pinned memory, asynchronous CUDA copies, CUDA Graphs, and persistent kernels are experiments whose defaults are chosen by measured end-to-end results.
- Prefetch uses estimated use time, fetch latency, and current residency to schedule by deadline rather than FIFO.
- Expert prediction optimizes recall and must have an exact-prefetch mode where a miss changes performance but not output.
- Transition tables, task-conditioned transitions, and a tiny learned predictor must be compared before selection.

## CUDA constitution

- Kernels target K3 dimensions and the actual RTX 5080 architecture.
- Build and runtime detect supported GPU architecture.
- Native cubins are generated for the target architecture.
- Candidate fusion includes MXFP4 load/unpack, projection, activation, expert scaling, accumulation, and residual work.
- KDA, MLA, and Attention Residual implementations must be compared with inspected original optimized implementations before adoption.
- Generic library, custom kernel, CUDA Graph, and persistent-kernel choices become defaults only through measured evidence.

## Speculative decoding constitution

K3X must expose a DSpark-compatible draft/target interface and support MoE expert-major block verification. For a candidate block it computes all routing decisions, forms layer-level unique expert unions, fetches an expert once, and batches every candidate token that needs it.

Cost-aware speculation evaluates acceptance probability, marginal new-expert cost, and residency. EcoSpec, MoE-Spec, and AcceptMoE concepts are separate experimental modes reproduced from their original sources. Natural-routing strict verification always remains available, and no quality-changing verifier optimization becomes default without quality benchmarks.

## Prefix and persistent state constitution

The recurrent-attention structure must support reusable prefix, KDA, MLA, system prompt, tool definition, repository instruction, and stable conversation state. Long-context compression is an independent feature flag.

## SKYFORGE constitution

SKYFORGE is the cloud-side K3X manufacturing system.

- The Conductor discovers source shards, creates manifests, allocates bounded idempotent units, records progress, retries failures, and merges artifacts.
- Foundry Workers process one bounded shard or layer range, stream source tensors from object storage, transform or calibrate them, upload verified outputs, and release temporary memory.
- The IMMORTAL Ledger persists unit state, checksums, converter version, and configuration so termination never invalidates completed work.
- Cloud Run local disks, whole-model RAM, whole-model VRAM, and one long-lived process are never assumed.
- Source, intermediate, and final artifacts use an explicitly configured persistent object store.
- The initial cloud target may use up to three independent RTX PRO 6000 Blackwell workers only after the user explicitly provisions billable resources.
- Dockerfile, reproducible build, entrypoint, manifest, allocator, retries, resumable transfers, progress, checksum verification, dry run, and cost accounting are required.
- The first SKYFORGE milestone operates only on the synthetic model.

## Benchmark constitution

Meaningful runs record decode and prefill throughput, TTFT, GPU utilization, GPU memory bandwidth, VRAM, host RAM, NVMe GB/token, RAM-to-GPU GB/token, expert-cache hit rate, speculative acceptance, unique experts per verification block, average adaptive Top-K, cold rescues, kernel time, I/O stall time, quality results, and enabled optimizations.

Optimizations must be independently switchable. An automated ablation runner stores JSON and CSV and produces comparison reports. Missing measurements are labeled not measured; derived estimates are never inserted into measured fields.

## Engineering targets

- First meaningful target: at least 5 warm coding decode tok/s if measurements show it is achievable.
- Strong result: 5–8 tok/s with quality measured simultaneously.
- Stretch result: 8–10 tok/s with quality measured simultaneously.

These are engineering targets, not forecasts.

## Development order

1. Reference graph and tests.
2. K3X streaming format and compiler.
3. Exact CPU/GPU runtime.
4. Profiler.
5. Basic GPU backend.
6. Tiered asynchronous storage.
7. Expert cache policies.
8. Task/session profiles.
9. Adaptive Top-K and rescue.
10. Fused CUDA kernels.
11. Speculative block verification.
12. Expert-major scheduling.
13. EcoSpec, MoE-Spec, and AcceptMoE experiments.
14. Quantization calibration.
15. Proxy and pruning experiments.
16. Cloud Run conversion pipeline.
17. Full ablation and quality suite.

After every stage, applicable tests and benchmarks run, results are documented, and work does not advance solely on theoretical speedups.

## Source discipline

Official Kimi K3 implementation and report, vLLM Kimi K3 support, `kimi-k3-in-c`, `kimi-k3-mlx`, DSpark, EcoSpec, AcceptMoE, MoE-Spec, SpecMD, and Least-Stale caching must be inspected at their original paper or implementation before their techniques are claimed or adopted. Expected numbers in planning text are hypotheses until reproduced.
