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

## Required production measurements

Before selecting a default storage or kernel path, the Linux target must record decode and prefill rates, TTFT, GPU utilization and memory bandwidth, VRAM and host RAM, NVMe and RAM-to-GPU GB/token, expert-cache hit rate, speculative acceptance, unique experts per block, adaptive Top-K, cold rescues, per-kernel time, and I/O stall time. Every result must carry an ablation configuration and quality mode.

## Primary hardware sources

- [NVIDIA GeForce RTX 5080 specifications](https://www.nvidia.com/en-us/geforce/graphics-cards/50-series/rtx-5080/).
- [NVIDIA Blackwell GPU architecture whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf).
- [Solidigm P44 Pro product brief](https://www.solidigm.com/content/dam/solidigm/en/site/products/client/d6/product-brief/p44-pro-product-brief/documents/p44-pro-product-brief.pdf).
- [AMD Ryzen 7 9800X3D product page](https://www.amd.com/en/products/processors/desktops/ryzen/9000-series/amd-ryzen-7-9800x3d.html).
