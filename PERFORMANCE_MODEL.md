# K3X Performance Model

## What this model is

This is a byte-traffic model for the released Kimi K3 text decoder on the target PC. It is not a throughput benchmark and it does not promise a token rate. Weight totals are derived from released dimensions rather than scanned from the full checkpoint, which Milestone 0 intentionally does not download.

Decimal GB is used for bandwidth equations; GiB is shown where capacity planning benefits from it. Hardware maxima are vendor specifications or interface ceilings, not sustained K3X measurements.

## Target hardware envelope

| Resource | Published or configured ceiling | Qualification |
|---|---:|---|
| RTX 5080 VRAM | 16 GB GDDR7 | Capacity, before runtime allocations |
| RTX 5080 memory bandwidth | 960 GB/s | NVIDIA architecture specification |
| DDR5-4200 dual-channel | 67.2 GB/s | Theoretical transfer rate, not measured STREAM bandwidth |
| PCIe 5.0 x16 | about 64 GB/s each direction | Raw encoding-level ceiling; payload and topology reduce it |
| Solidigm P44 Pro sequential read | up to 7.0 GB/s | Vendor maximum; random reads and contention are lower |

The Ryzen platform shares finite I/O and memory-controller resources between the GPU, NVMe device, and CPU. The production profiler must measure concurrent transfer rather than adding headline bandwidths.

## Native routed-expert bytes

One routed expert has three matrices.

```text
gate: 3,072 × 3,584
up:   3,072 × 3,584
down: 3,584 × 3,072
total values = 33,030,144
```

MXFP4 stores two E2M1 values per byte and one E8M0 scale byte per group of 32 values.

```text
packed values = 33,030,144 / 2 = 16,515,072 bytes
scales        = 33,030,144 / 32 = 1,032,192 bytes
one expert    = 17,547,264 bytes = 16.734375 MiB
```

With natural Top-16 and no reuse, one MoE layer requests 280,756,224 bytes. Across 92 MoE layers, routed experts alone request 25,829,572,608 bytes/token, or 25.83 GB/token. All 896 experts for one layer occupy about 14.64 GiB; all routed expert instances across 92 layers occupy about 1.446 TB decimal. These values exclude alignment and directory overhead.

Milestone 7 physically materialized one expert and confirmed the 17,547,264-byte native payload exactly: 16,515,072 packed bytes plus 1,032,192 scale bytes. Its six released-dimension extents are each divisible by the WSL2 ext4 direct-I/O offset alignment of 512 bytes, so B-0008 measured zero direct byte amplification for this one-expert slice. This confirms serialization size and alignment for the bounded artifact only; the 25.83 GB/token figure remains a derived uncached full-model total.

## Trunk estimate

The following BF16 estimate includes embeddings/LM head, attention, norms, residual projections, latent projections, shared experts, and the dense first layer. It excludes routed experts and MoonViT-V2. Exact tensor tying and implementation details can move the total, so the converter must replace this estimate with a checkpoint-derived manifest before full-scale planning.

| Group | Derived BF16 values or bytes |
|---|---:|
| 69 KDA layer trunks | 17.907 billion values |
| 24 MLA layer trunks | 5.573 billion values |
| 92 MoE non-routed trunks | 17.474 billion values |
| Dense first-layer MLP | 0.727 billion values |
| Embedding/final/LM group | 1.174 billion values |
| **Estimated trunk total** | **42.858 billion values = 85.72 GB = 79.83 GiB** |

The estimate demonstrates the capacity problem even before exact conversion. A 96 GB host cannot safely dedicate 85.72 GB to BF16 trunk weights and still retain the OS, runtime state, page cache, conversion buffers, and a useful expert bank. Ideal 4-bit trunk storage would reduce the value payload to about 21.43 GB before scales, outliers, and alignment, which is why sensitivity-aware mixed quantization is a prerequisite for a useful L1 expert cache.

## Persistent state

KDA state is constant with context length. With FP32 state storage, the released dimensions imply approximately 434,110,464 bytes for the recurrent matrices plus 30,523,392 bytes for three convolution histories, or 464,633,856 bytes total.

MLA state grows with context. Storing main keys and values for 96 heads plus the shared extra key gives the following across 24 MLA layers.

```text
FP32 MLA growth = 2,365,440 bytes per context token
BF16 MLA growth = 1,182,720 bytes per context token
```

These figures omit allocator padding and future cache compression. K3X will record the actual allocated bytes and state precision in benchmarks.

## Uncached bandwidth ceilings

The worst-case expert-only ceilings divide headline bandwidth by 25.83 GB/token.

| Path | Arithmetic ceiling | Why it is not a prediction |
|---|---:|---|
| P44 Pro to RAM | 0.27 tok/s | Uses 7.0 GB/s sequential maximum and assumes perfectly sequential expert layout |
| DDR5-4200 RAM reads | 2.60 tok/s | Uses theoretical 67.2 GB/s and ignores CPU/trunk/state traffic |
| PCIe 5.0 x16 RAM to GPU | 2.48 tok/s | Uses raw 64 GB/s and ignores protocol, copies, topology, and contention |

If the estimated 85.72 GB BF16 trunk and 25.83 GB routed experts were both read from HBM for every token, the 960 GB/s HBM ceiling would be about 8.61 tok/s before compute, state, activation, and kernel overhead. The trunk cannot all reside in 16 GB VRAM, so this merely bounds an impossible idealized stream; it is not an achievable end-to-end rate.

## Traffic budget for the engineering target

At the P44 Pro's 7.0 GB/s published maximum, a 5 tok/s target permits at most 1.40 GB of NVMe traffic per token if storage does nothing else. Compared with the uncached 25.83 GB expert demand, at least 94.6% of routed-expert bytes must avoid NVMe. At 8 tok/s the budget falls to 0.875 GB/token, requiring 96.6% avoidance.

This is the first material bottleneck. Prefetch alone cannot overcome insufficient bandwidth; the system needs high L1/L0 hit rates, physical expert locality, speculative-block reuse, and likely a quantized trunk that leaves RAM for experts. Exact cold rescue remains necessary for correctness when the router selects a nonresident expert.

## Capacity intuition

- The full 16 GB VRAM capacity could hold at most roughly 900 routed-expert instances if it held nothing else. Real capacity is lower because trunk tiles, activations, state, CUDA workspaces, and buffers also reside there.
- A hypothetical 60 GB host expert bank would hold roughly 3,400 layer-specific expert instances. This is only an illustration; actual available RAM must be measured after the mixed-precision trunk and runtime allocations exist.
- Expert identity is layer-specific. Caching expert 42 in layer 10 does not satisfy expert 42 in layer 11.

## Milestone 0 measurements

The following values were measured on 2026-08-08 using the deterministic tiny synthetic model, a Windows 11 AMD64 host, the independent C++20 CPU runtime, an MSVC Debug build, three warmups, and 20 measured process runs.

| Metric | Measured result |
|---|---:|
| Prefill throughput | 405.11 tok/s |
| Incremental decode throughput | 558.89 tok/s |
| Process-level TTFT median | 86.20 ms |
| Observed peak child RSS | 6,270,976 bytes |
| Artifact bytes read / generated token | 110,936 bytes |
| Synthetic KDA state | 19,200 bytes |
| Synthetic MLA KV state at measured context | 2,592 bytes |
| Median layer times | 2.191, 5.495, 5.511, 5.038 ms |

`TTFT` includes process startup, strict CRC32C/directory SHA-256/root SHA-256 verification of the complete artifact, lazy tensor loads, prompt prefill, and JSON output. Decode throughput times exactly the five incremental forwards after the first generated token; that first token comes from the prompt prefill logits. File bytes/token counts the runtime's logical tensor reads after open-time integrity verification and is not an operating-system disk-counter measurement. The benchmark does not use the target RTX 5080 and says nothing about full Kimi K3 token throughput.

Reproduce the record with the commands in [`README.md`](README.md). JSON and CSV are generated under an ignored `build-results/` directory so host-specific results are not mistaken for portable project data.

## Milestone 13–15 speculative verification accounting

The accepted token-major verifier is a correctness reference, not a traffic optimization. For a proposal with `p` candidate tokens and `a` accepted candidates, it performs `a + 1` target forwards and commits `a + 1` tokens, including the target bonus token. Rejected suffix candidates are never executed. Its target work and committed KDA/MLA state therefore match ordinary greedy decoding; proposal and lifecycle overhead can only make this reference equal or slower.

B-0014 confirms this accounting on the synthetic CPU fixture. Greedy, perfect block-2, and mixed block-2 each perform five target decode forwards and read 665,616 logical bytes. Perfect acceptance is 1.0 and mixed acceptance is 0.25, but neither changes target work or traffic. Their measured +1.55% and +1.05% decode deltas are therefore treated as fixture variation rather than an amortization result.

Milestone 14 now implements the first exact CPU expert-major reference. For each MoE layer, its relevant traffic variable is the exact unique expert union across candidate-token routing decisions, not `block_tokens × Top-K` by assumption. The runtime loads each union payload once and reports unique loads, assignments, evaluated positions, and discarded positions separately.

B-0015 demonstrates why acceptance belongs in the traffic model. Perfect block-2 execution loads 24 unique payloads for 30 assignments, reuses six assignments, and reduces logical Reader bytes from 665,616 to 655,824 relative to token-major. The mixed trace loads 39 payloads for 48 assignments but evaluates eight positions to commit five, raising logical Reader bytes to 680,304. On this fixture the block traffic benefit is therefore the union reuse saved on committed work minus payloads spent on rejected suffix positions. These are logical synthetic Reader bytes, not physical P44 Pro or H2D measurements.

Milestone 15 adds physical CUDA H2D evidence for one expert reused across multiple tokens. For a native expert payload of `E = 17,547,264` bytes and a group of `b` tokens, repeated transient scalar execution transfers `bE` weight bytes per iteration, while the batch primitive transfers `E`. B-0016 observes exactly this identity for `b=2` and `b=4`: 20 iterations use 701,890,560 versus 350,945,280 bytes and 1,403,781,120 versus 350,945,280 bytes. Activation H2D and result D2H stay proportional to token count and are identical within each scalar/batch pair.

The released-dimension batch reduces median boundary latency by 49.55% for two tokens and 60.75% for four tokens, while aggregate kernel time falls by 33.58% and 42.48%. This isolates expert-weight reuse but does not include routing, multi-expert union formation, KDA/MLA, rejection cost, physical NVMe, or full-layer concurrency. The synthetic CUDA graph confirms exact integration: perfect block-2 slightly reduces weight H2D from 4,981,824 to 4,972,032 bytes, while the mixed row increases it to 6,627,744 bytes because three rejected positions are evaluated.

B-0017 adds a real reduced-K replay distribution without changing the target. On the Top-16 fixture, fixed block-2 accepts all three proposed tokens, while adaptive token/expert rows accept two of four. The best replay row, fixed block-2 expert-major, reduces target logical Reader bytes from 1,294,992 to 1,102,416 through union reuse but adds 1,454,112 draft bytes, so combined logical traffic rises to 2,556,528 bytes. Complete-prefix replay also repeats 13 context-token positions and measures 46.35% lower decode than natural. Adaptive rows replay 20 context-token positions, read 2,181,168 draft bytes, and regress by 60.77% to 62.52%.

B-0018 implements that persistent boundary and compares matched replay/persistent pairs. Fixed block-2 persistent state reduces draft Reader bytes from 1,454,112 to 785,808 (-45.96%) for both target verification modes. Adaptive persistent state reduces them from 2,181,168 to 805,392 (-63.08%), replaces 20 repeated prefix positions with one five-token context prefill, and records one exact rollback plus one cropped MLA position. The KDA checkpoint-copy counters are 57,600 bytes for fixed and 76,800 bytes for adaptive runs; these are in-memory state-copy accounting, not storage traffic.

Persistent decode improves over its replay pair by 14.97%/14.55% for token/expert fixed block-2 and 41.75%/27.08% for adaptive token/expert. It still measures 38.08% to 52.26% below the tiny natural greedy baseline because reduced-K drafting itself executes additional model work. The next boundary is therefore reducing or overlapping per-token draft execution through measured residency, GPU execution, or precision changes while preserving target quality. Expected bytes per committed token must still combine acceptance, per-layer unique-expert union, expert group-size distribution, and current residency. Natural non-speculative execution remains the default until representative drafting, native-Linux physical I/O, coding quality, and full-layer CUDA evidence exist.

B-0019 isolates exact GPU placement without residency or precision changes. Fixed CPU/CUDA draft pairs transfer 5,843,840 CUDA-draft bytes per run, of which 5,756,160 are weights; adaptive pairs transfer 6,428,224 bytes, of which 6,331,776 are weights. Peak draft VRAM is only 44,448 bytes because allocation reuse is enabled, but transient weights are retransferred at every operation. The paired decode regressions of 96.22% to 97.00% show that low peak allocation is not the relevant bound here: repeated H2D plus 410 to 451 synchronous stream waits and small-kernel launch overhead dominate this tiny graph. These are logical CUDA transfer counters under WSL2, not full-model PCIe GB/token or physical NVMe measurements.

B-0020 executes that bounded exact residency experiment without changing FP32 arithmetic, reduced Top-4 proposals, routing, target verification, scheduler, or kernels. The 8 MiB cap holds the complete observed working set at 644,160 bytes for fixed rows and 647,424 bytes for adaptive rows. Cache hit rate is 75.68% or 77.27%, misses are the first exact admissions, and bypasses are zero. Weight H2D falls from 5,756,160 to 644,160 bytes for fixed rows (-88.81%) and from 6,331,776 to 647,424 bytes for adaptive rows (-89.78%). Total draft H2D still includes 87,680 or 96,448 activation bytes.

Removing repeated weight transfer does not remove fine-grained execution. The fixed resident rows still record 410 synchronizations and the adaptive rows 451. Their aggregate draft kernel time ranges from 44.37 to 54.13 ms on the tiny graph. Paired decode changes are +15.58% fixed token, -2.56% adaptive token, +22.67% fixed expert-major, and +5.57% adaptive expert-major, so residency alone is not a universal throughput win. The next isolated model boundary is persistent multi-token/multi-expert CUDA work that amortizes launches and synchronization while preserving the exact resident table and target contract.

A production performance model still needs eviction pressure under a realistic expert working set, expert union size across speculative blocks, target acceptance, native-Linux PCIe throughput, physical NVMe traffic, GPU utilization/memory bandwidth, and full-layer occupancy. Reduced precision remains deferred so traffic reduction is not conflated with quality divergence.

## Milestone 20 resident-grid launch and traffic model

The exact resident grid changes execution granularity, not weight precision or routed-expert demand. For `E` selected experts, each call uploads three 16-byte matrix descriptors per expert, or `48E` bytes. The synthetic Top-4 AURORA path therefore adds 192 descriptor bytes per grid call: 5,760 bytes across 30 fixed-policy calls and 6,336 bytes across 33 adaptive-policy calls.

The grouped reference launches gate, up, SiTU, and down once per expert. The resident grid launches each operation once per complete expert set. B-0021 reduces measured MoE launches from 480 to 120 for fixed rows and from 528 to 132 for adaptive rows, exactly 75% in every matched pair. This is launch amortization, not a reduction in exact native-MXFP4 weight bytes: paired weight H2D remains 644,160 bytes fixed and 647,424 bytes adaptive.

Milestone 21's layer boundary adds one real cold L0 weight absent from the split path: the routed RMSNorm vector, or `latent_width * sizeof(float)` bytes in FP32 per distinct layer admitted. Hidden inputs, expert descriptors, and router contributions remain activation traffic. B-0022 measures exactly 384 additional weight-H2D and resident bytes in every matched pair, confirming that the norm is accounted as a physical cold L0 admission.

Activation and descriptor traffic make total draft H2D rise from 731,840 to 752,960 bytes in fixed rows and from 743,872 to 767,104 bytes in adaptive rows. Despite that increase, paired synthetic decode improves by 10.79% to 38.00%. The result isolates launch granularity on the tiny WSL2 graph; it does not establish that multi-token batching caused the gain because current AURORA calls the grid with one token, and it does not project full-model throughput.

The next traffic boundary is keeping intermediate activations and non-FFN draft state on device. KDA, MLA, routing, residual/state work, descriptor copies, activation transfers, and synchronization remain outside the exact resident-weight saving.

## Milestone 21 resident MoE-layer traffic model

For `C` successful MoE-layer calls, the split path performs four synchronizations per call that the complete layer can collapse to one. B-0022 observes the exact identity `sync_split - sync_layer = 3C`: 90 fewer waits for 30 fixed calls and 99 fewer for 33 adaptive calls.

The complete layer reduces activation H2D by 496 bytes per call on this fixture and D2H by 896 bytes per call. After the one-time 384-byte norm admission, total H2D falls by 14,496 bytes fixed and 15,984 bytes adaptive. This is a measured synthetic traffic identity, not a full-dimension extrapolation. The full model still needs representative hidden/latent widths, resident pressure, PCIe counters, and native-Linux timing before the same ratios can be used for capacity planning.

Paired decode changes are mixed from -2.75% to +5.62%. Therefore reduced traffic and synchronization are established, while a throughput default is not. The next model must distinguish host orchestration/launch overhead from transfer savings at representative dimensions rather than projecting a TPS gain from bytes alone.

## Milestone 22 released-dimension MoE-layer model

The bounded released-size fixture fixes hidden width 7,168, routed latent width 3,584, and expert intermediate width 3,072. Its FP32 dense/vector layer weights occupy 469,776,384 bytes. Adding repeated-view native MXFP4 experts gives 487,323,648 resident bytes at one expert, 539,965,440 at four, and 750,532,608 at sixteen. The split path is exactly 14,336 bytes smaller because its routed RMSNorm remains on the CPU.

| Experts | Split median | Layer median | Layer delta | Kernel total split → layer | Activation H2D reduction | D2H reduction |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1,227,823 ns | 20,487,750 ns | +1568.624% | 15,121,920 → 22,970,976 ns | 1,145,840 B | 1,146,880 B |
| 4 | 2,370,565 ns | 20,953,689 ns | +783.911% | 24,506,720 → 27,692,480 ns | 1,142,720 B | 2,007,040 B |
| 16 | 5,681,151 ns | 24,422,315 ns | +329.883% | 58,395,968 → 61,886,528 ns | 1,130,240 B | 5,447,680 B |

Every row uses 20 measured iterations after three warmups. Split synchronization is 80 and layer synchronization is 20; measured warm weight H2D is zero; maximum error, capacity bypass, and fallback are zero. The oracle backend is released before selected-backend construction, and reported peak VRAM is the larger of the sequential oracle and selected phases: 575,555,632, 628,336,832, and 839,518,976 bytes at 1, 4, and 16 experts. The table is a layer-boundary measurement and has no token, prefill, TTFT, physical PCIe, or NVMe interpretation.

The B-0023 layer wall regression is much larger than its aggregate kernel-time increase. Source inspection identified one unamortized term: each complete-layer call performed finiteness scans across all 469,776,384 immutable dense/vector bytes before resident lookup and launch. Milestone 23 below measures this attribution while retaining the malformed/non-finite rejection contract.

## Milestone 23 validation traffic attribution

For `C` complete released-size layer calls, per-call validation reads `C × 469,776,384` host bytes. At `C = 20`, B-0024 measures the exact logical scan volume of 9,395,527,680 bytes per row. Admission mode reads 469,776,384 bytes once during cold setup and then performs six constant-time identity lookups per warm call, so measured warm validation bytes are zero.

| Experts | Per-call median | Admission median | Paired change | Per-call validation host time over 20 calls |
|---:|---:|---:|---:|---:|
| 1 | 19,570,019 ns | 1,246,879 ns | -93.629% | 368,716,067 ns |
| 4 | 20,728,924 ns | 1,939,696 ns | -90.643% | 375,352,908 ns |
| 16 | 24,518,749 ns | 5,220,560 ns | -78.708% | 382,753,782 ns |

These profiler-off medians establish host validation as the dominant B-0023 wall term at this boundary. They do not model token throughput, system-RAM bandwidth under a full checkpoint, PCIe contention, NVMe traffic, or native-Linux scheduling. After admission, expert-count-dependent CUDA work and host orchestration become the next visible terms.

## Milestone 27 official expert residency model

B-0028 replaces the released repeated-view fixture with one pinned official layer-1 expert-0 payload. The payload remains exactly `E = 17,547,264` bytes. Both modes perform one cold admission of `E`; over `C = 20` measured calls, transient execution transfers `CE = 350,945,280` weight bytes while exact residency transfers zero measured weight bytes and reports `3C = 60` tensor hits.

| Mode | Cold latency | Warm median | Kernel total, 20 calls | Weight H2D, 20 calls | Activation H2D | D2H | Resident / peak VRAM |
|---|---:|---:|---:|---:|---:|---:|---:|
| Transient | 7,122,628 ns | 2,508,377 ns | 6,309,888 ns | 350,945,280 B | 286,720 B | 286,720 B | 0 / 5,914,624 B |
| Resident | 7,748,006 ns | 331,868 ns | 2,692,992 ns | 0 B | 286,720 B | 286,720 B | 17,547,264 / 23,461,888 B |

The resident warm median is 86.77% lower in this bounded run, or 7.56 times the transient median. The single cold resident call is 8.78% slower than the single cold transient call, so residency is valuable only when reuse occurs. These wall and kernel observations come from one deterministic WSL2 run and must not be projected to a full layer or tok/s.

The arithmetic natural Top-16 routed payload remains `16E = 280,756,224` bytes per MoE layer before shared-expert, router, descriptor, activation, and workspace bytes. B-0028 proves that one real expert can occupy exact L0 residency and avoid repeated H2D; it does not prove that a changing natural Top-16 set fits a useful L0 policy or avoids L1/L2 traffic. M28 must therefore measure a real router-selected Top-16 set plus the real shared-expert path as one dependency-closed FFN sublayer before extrapolating cache hit rate or layer latency.

## Required production measurements

Before selecting a default storage or kernel path, the Linux target must record decode and prefill rates, TTFT, GPU utilization and memory bandwidth, VRAM and host RAM, NVMe and RAM-to-GPU GB/token, expert-cache hit rate, speculative acceptance, unique experts per block, adaptive Top-K, cold rescues, per-kernel time, and I/O stall time. Every result must carry an ablation configuration and quality mode.

## Primary hardware sources

- [NVIDIA GeForce RTX 5080 specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5080/).
- [NVIDIA Blackwell GPU architecture whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf).
- [Solidigm P44 Pro product brief](https://www.solidigm.com/content/dam/solidigm/en/site/products/client/d6/product-brief/p44-pro-product-brief/documents/p44-pro-product-brief.pdf).
- [AMD Ryzen 7 9800X3D product page](https://www.amd.com/en/products/processors/desktops/ryzen/9000-series/amd-ryzen-7-9800x3d.html).
