# K3X Official Expert CUDA Execution Design

## Status

Accepted for Milestone 27 under the standing authorization to continue non-billable pre-Cloud-Run work. This milestone proves one exact official expert FFN, not a complete MoE layer or token-generation graph.

## Goal

Execute the pinned official Kimi K3 layer-1 expert-0 native-MXFP4 weights on the RTX 5080, compare every output element with the independent portable CPU backend, and measure cold and warm CUDA traffic and latency without weakening the storage-fixture generation guard.

## Evidence already observed

A read-only exploratory run used the existing released-dimension batch executable against the ignored B-0027 K3X artifact. One scalar expert FFN completed with 7,212,040 ns wall time, 1,962,624 ns accumulated CUDA kernel time, 17,547,264 weight-H2D bytes, and maximum absolute CPU-oracle error `3.0267983675e-9`. These values establish feasibility only. They are not B-0028 and must not be copied into measured benchmark evidence.

## Alternatives

### Reuse the released-dimension executable unchanged

This is the smallest implementation change, but its schema identifies every input as `released_dimension_single_expert` and does not bind the official K3X root or ordered expert payload. It cannot distinguish the synthetic released fixture from the official B-0027 artifact and is rejected as formal evidence.

### Add official options to the released-dimension executable

This shares more code, but it mixes two evidence contracts and lets caller-provided labels imply official provenance. It is rejected because a generic switch should not turn a synthetic benchmark into official-checkpoint evidence.

### Add a dedicated pinned official-expert executable

This is accepted. A CUDA-only `k3x_cuda_official_expert_bench` loads the existing storage fixture through the strict Reader, verifies fixed B-0027 identities before backend creation, computes an independent CPU oracle, and measures one exact CUDA expert FFN. The executable remains outside `k3x_run`, so generation from `OPTIONAL_STORAGE_FIXTURE` continues to fail with `NON_EXECUTABLE_ARTIFACT`.

## Components

- `official_expert.hpp` and `official_expert.cpp` own the pinned identity constants and a pure validation function over root digest, ordered digest, feature bits, layer, expert, payload bytes, and the three matrix shapes. They perform no I/O and are covered by CPU tests.
- `cuda_official_expert_bench.cpp` owns CLI parsing, strict Reader/load sequencing, CPU oracle execution, CUDA measurement, correctness gates, and canonical JSON output.
- `ablate_official_expert_cuda.py` owns the two-case B-0028 run, raw/summary publication, digest checks, and strict verification. It never copies or commits the real K3X input.

## Fixed identity

The executable accepts `--model`, `--weight-mode transient|resident`, `--warmup`, and `--iterations`. Official identity is not caller-configurable.

- Repository: `moonshotai/Kimi-K3`.
- Resolved revision: `9f62e4e9fffbd0a83ddd60e1c209d828994b3569`.
- Layer and expert: layer 1, expert 0.
- K3X root SHA-256: `d585d283325e13e1316a0194c2d6274dd89ef75a28b96b02f02733290b7658be`.
- Gate/up/down ordered payload SHA-256: `4e23bd960dfb5e8b10def10e12a94bac1119500f72918698986bd332d56d33ff`.
- Payload bytes: 17,547,264.
- Shapes: gate `[3072,3584]`, up `[3072,3584]`, down `[3584,3072]`, group size 32.
- Activation: SiTU beta 4.0 and linear beta 25.0.

`Reader::open` uses full checksum verification. The superblock root, optional storage identity, tensor records, loaded byte count, and ordered payload digest must all match before CPU or CUDA backend construction. A mismatch exits nonzero and launches no CUDA work.

## Execution flow

The input is a deterministic 3,584-element FP32 vector where element `i` is `((i % 17) - 8) * 0.01`. The executable builds one `Mxfp4MlpView` directly over the six loaded extents.

The portable CPU backend executes `mxfp4_situ_mlp_group` once outside CUDA timing and produces the 3,584-element oracle. CPU oracle wall time is recorded separately.

The CUDA backend uses `cuda-custom`, reused allocations, the FFN-block boundary, synchronous transfer, no routed accumulation, and one of two explicit weight modes.

- `transient` uses zero resident capacity and transfers the exact expert weights on every execution.
- `resident` uses a hard capacity of exactly 17,547,264 bytes. The untimed cold call admits all three matrices; measured calls must report zero weight-H2D bytes and exact residency without bypass.

One cold call is always recorded separately. Requested warmups then run without entering the sample set. Each measured call produces one output vector. Any non-finite output, incorrect output length, or maximum absolute CPU-oracle error above `1e-6` is fatal.

## Telemetry contract

One canonical JSON record is written to stdout. It includes fixed source identity, weight mode, input/output sizes, payload and root digests, CPU oracle time, cold wall time and traffic, measured median/p05/p95 wall time, accumulated kernel time, weight and activation H2D bytes, D2H bytes, allocation and synchronization deltas, cache hit/miss/bypass deltas, current/peak resident bytes, peak VRAM, maximum absolute error, all-finite status, warmup count, and iteration count.

The schema explicitly sets `token_semantics=false`, `routing_semantics=false`, and `full_moe_layer=false`. It contains no decode tok/s, prefill tok/s, TTFT, quality, physical NVMe, or full-checkpoint fields.

## B-0028 runner and verifier

`tools/ablate_official_expert_cuda.py` runs exactly two rows, `transient` and `resident`, with three warmups and 20 measured calls. It writes LF-only raw JSON/CSV plus summary JSON/CSV. The real K3X artifact remains under ignored `artifacts/`; only bounded result records are committed.

The strict verifier independently binds the executable digest, B-0027 summary digest, fixed official identities, case order, iteration counts, parity threshold, resident/transient traffic invariants, raw file digests, summary parity, and absence of forbidden token/quality claims. It recomputes aggregates from raw rows rather than trusting the summary.

## Testing

- Pure C++ identity tests accept the exact root/digest/shape contract and reject one-bit root, ordered-digest, payload-size, layer, expert, optional-feature, and shape mutations.
- CLI tests reject missing model, invalid mode, zero iterations, and a synthetic storage fixture before CUDA backend creation.
- Local RTX 5080 tests execute both official modes and compare them with the CPU oracle.
- Full CPU, liburing, ASan/UBSan, CUDA, and Compute Sanitizer gates run before publication.

## Non-goals

This milestone does not execute real routed-down/up, shared-expert, router, KDA, MLA, attention, residual, logits, or token generation. It does not download another range, complete shard, or full checkpoint. It does not alter K3X v1, cache policy, routing, production defaults, or `NON_EXECUTABLE_ARTIFACT`. The smallest real dependency-closed multi-expert or complete MoE-layer slice is a separate Milestone 28 decision based on B-0028.
