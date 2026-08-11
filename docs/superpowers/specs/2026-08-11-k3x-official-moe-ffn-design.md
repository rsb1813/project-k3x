# K3X Milestone 28 Official MoE FFN Design

## Status

Accepted on 2026-08-11 under the standing authorization to continue all non-billable work before Cloud Run provisioning. This milestone closes one official Kimi K3 layer-1 feed-forward boundary over two deterministic inputs. It does not claim authentic token activations, attention execution, a complete transformer layer, token throughput, or full-checkpoint execution.

## Goal

Build one content-addressed K3X fixture containing exactly the official layer-1 tensors required to execute and verify:

1. Attention Residual preparation for the MLP input.
2. Post-attention RMS normalization.
3. All 896 router scores and natural Top-16 selection.
4. The exact routed experts selected by two deterministic inputs.
5. Stable LatentMoE down projection, routed mixing, routed RMS normalization, and up projection.
6. The two-expert shared SiTU-GLU MLP.
7. Routed-plus-shared addition and the layer prefix residual.

The portable source-byte PyTorch oracle, portable C++ CPU implementation, and native `sm_120` CUDA implementation must agree within explicit routing and numerical gates.

## Official evidence

The pinned official implementation is `moonshotai/Kimi-K3` revision `9f62e4e9fffbd0a83ddd60e1c209d828994b3569`.

- [`modeling_kimi_linear.py`](https://huggingface.co/moonshotai/Kimi-K3/blob/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/modeling_kimi_linear.py) defines sigmoid routing, correction-bias selection, natural Top-16 renormalization, Stable LatentMoE, shared experts, Attention Residual preparation, and prefix accumulation.
- [`config.json`](https://huggingface.co/moonshotai/Kimi-K3/blob/9f62e4e9fffbd0a83ddd60e1c209d828994b3569/config.json) fixes hidden width 7,168, routed latent width 3,584, 896 experts, Top-16, two shared experts, 3,072 intermediate values per expert, SiTU beta 4, linear beta 25, routed normalization, and RMS epsilon `1e-5`.
- The pinned index SHA-256 remains `a1c5210650ce71d2d3ae9ec5a101ac4afd3cf4b10091be589853437eb967febd`.

Read-only header inspection confirms that all layer-1 tensors in this milestone belong to `model-00002-of-000096.safetensors`. The complete shard is not downloaded or treated as verified.

## Approaches considered

### Accepted: one dependency-closed K3X MoE slice

One K3X artifact owns the always-active tensors and the union of experts naturally selected by the two fixed inputs. The artifact has one root identity, preserves per-expert extents, supports exact Reader validation, and can feed the existing resident CUDA layer boundary.

### Rejected for M28: one trunk artifact plus many expert artifacts

Splitting always-active and expert objects would improve later reuse, but this milestone would also need a multi-Reader atomic identity, multi-file resume transaction, and runtime lifetime contract. Those concerns belong to the production out-of-core model, not this first complete official sublayer proof.

### Deferred: authentic layer-1 activation generation

Producing a real post-attention activation requires closing layer 0, layer-1 KDA, recurrent state, and Attention Residual history. That is a complete transformer dependency expansion and would prevent independent isolation of the MoE boundary.

## Deterministic boundary inputs

The fixture owns two cases named `A` and `B`. Each case contains a 7,168-value `prefix_sum` and one 7,168-value Attention Residual bank row. For zero-based index `i`, the exact formulas are:

- `A.prefix_sum[i] = (((17*i + 3) % 257) - 128) / 1024`;
- `A.block_residual[i] = (((29*i + 11) % 251) - 125) / 1024`;
- `B.prefix_sum[i] = (((31*i + 7) % 263) - 131) / 1024`;
- `B.block_residual[i] = (((43*i + 19) % 269) - 134) / 1024`.

The power-of-two denominator makes every generated value exactly representable in FP32. The canonical byte string is little-endian IEEE-754 FP32 in index order. No language RNG or platform-specific distribution is permitted.

The input contract is versioned and hashed. The materializer records:

- generator name and version;
- case order `A`, then `B`;
- SHA-256 of every FP32 input byte string;
- natural Top-16 expert IDs and normalized contribution weights;
- the ordered union of expert IDs by first use across `A`, then `B`.

The two route sets must differ. If they do not, materialization fails before expert payload download. The implementation must not search seeds for a preferred expert or cache result.

## Exact graph

For each case, use the following official sequence.

1. Let `prefix_sum` be the residual output entering the layer-1 MLP half and let `block_residual` contain one prior block row.
2. Concatenate `block_residual` and `prefix_sum`, normalize each row in FP32 with `mlp_res_norm`, score with `mlp_res_proj`, softmax the two scores, and form the weighted Attention Residual input.
3. Apply `post_attention_layernorm` with epsilon `1e-5`.
4. Compute all 896 router logits in FP32 from the BF16 router weight and apply sigmoid.
5. Add `e_score_correction_bias` only for expert selection. Select natural Top-16. Gather the unadjusted sigmoid scores, renormalize them to sum to one, and multiply by routed scale 1.
6. Project the normalized hidden vector from 7,168 to 3,584 with the BF16 routed-down matrix.
7. Execute each selected native-MXFP4 expert over the 3,584-value latent input with gate/up 3,072, strict SiTU-GLU, and down 3,584.
8. Multiply each expert result by its normalized router contribution and accumulate in one canonical route order.
9. Apply the 3,584-value routed RMSNorm and the BF16 3,584-to-7,168 routed-up projection.
10. In parallel semantics, apply the BF16 shared gate/up projections from 7,168 to 6,144, strict SiTU-GLU, and the shared down projection to 7,168.
11. Add routed and shared results, then add that MoE result to the original `prefix_sum`.

The official PyTorch call uses `torch.topk(..., sorted=False)`. M28 requires exact selected-set equality and records a canonical descending adjusted-score order with expert ID as the final tie-break. The source oracle and both K3X backends use that canonical order for deterministic accumulation. This preserves natural selection while removing an unspecified ordering dependency.

## Required official tensors

The always-active source payload is exactly 379,900,416 bytes.

| Tensor suffix below `language_model.model.layers.1` | Source dtype | Shape | Bytes |
|---|---|---:|---:|
| `block_sparse_moe.gate.e_score_correction_bias` | F32 | 896 | 3,584 |
| `block_sparse_moe.gate.weight` | BF16 | 896 × 7,168 | 12,845,056 |
| `block_sparse_moe.routed_expert_down_proj.weight` | BF16 | 3,584 × 7,168 | 51,380,224 |
| `block_sparse_moe.routed_expert_norm.weight` | BF16 | 3,584 | 7,168 |
| `block_sparse_moe.routed_expert_up_proj.weight` | BF16 | 7,168 × 3,584 | 51,380,224 |
| `block_sparse_moe.shared_experts.down_proj.weight` | BF16 | 7,168 × 6,144 | 88,080,384 |
| `block_sparse_moe.shared_experts.gate_proj.weight` | BF16 | 6,144 × 7,168 | 88,080,384 |
| `block_sparse_moe.shared_experts.up_proj.weight` | BF16 | 6,144 × 7,168 | 88,080,384 |
| `mlp_res_norm.weight` | BF16 | 7,168 | 14,336 |
| `mlp_res_proj.weight` | BF16 | 1 × 7,168 | 14,336 |
| `post_attention_layernorm.weight` | BF16 | 7,168 | 14,336 |

Each selected routed expert contributes the existing six native-MXFP4 extents and exactly 17,547,264 bytes. If the two cases select `U` unique experts, total unaligned tensor payload is `379,900,416 + 17,547,264 × U`, where `16 <= U <= 32`. A single route requires 660,656,640 bytes; the actual two-route total is recorded only after natural routing is computed.

## K3X format evolution

M28 extends K3X v1 without changing record sizes or the major/minor version.

- Add tensor dtype `BF16 = 3` with quantization `NONE` and exact two-byte logical storage.
- Add required feature `BF16_TENSORS = 1 << 0`. Writers set it whenever a BF16 tensor exists. Readers that do not support it fail at the superblock before interpreting directories.
- Add optional feature `OFFICIAL_MOE_FIXTURE = 1 << 1`.
- The M28 artifact sets both the existing `STORAGE_FIXTURE` bit and `OFFICIAL_MOE_FIXTURE`. The existing production `k3x_run` guard therefore remains fail-closed with `NON_EXECUTABLE_ARTIFACT`.
- Add source manifest format `k3-official-moe-slice-v1` and artifact kind `official_moe_fixture`.

BF16 bytes pass through conversion unchanged. Expanding BF16 source weights to FP32 in K3X is rejected because it doubles L2/L1 bytes and forces a later BF16 reconstruction before CUDA execution.

## Physical layout

The writer packs extents by first-use order rather than source offset or tensor ID.

1. Attention Residual norm and projection.
2. Post-attention norm.
3. Router weight and correction bias.
4. Routed-down projection.
5. Selected experts in first-use union order, each as gate packed/scale, up packed/scale, and down packed/scale.
6. Routed norm and routed-up projection.
7. Shared gate, up, and down projections.

Per-expert directory records point directly to every selected expert. Experts not selected by either case are absent and cannot be substituted.

## Bounded materialization

Materialization is a two-phase, restartable local manufacturing job.

### Phase 1: always-active tensors and route derivation

1. Rediscover and revalidate the fixed snapshot, index, config, shard identity, and required tensor metadata.
2. Download each required tensor in ordered chunks no larger than 8 MiB.
3. Write each tensor into a content-addressed partial object, fsync it, verify SHA-256 and exact length, then atomically rename it.
4. Build the two source-byte PyTorch routes from the verified always-active objects.
5. Require two different natural Top-16 sets and publish a route manifest atomically.

### Phase 2: selected expert union and K3X conversion

1. Plan the exact six extents for every union expert from the pinned index and shard header.
2. Download only those extents in chunks no larger than 8 MiB.
3. Verify per-tensor and per-expert ordered SHA-256 identities and atomically publish content-addressed source objects.
4. Generate the complete source manifest only after every required object is present and verified.
5. Run the existing crash-safe K3X converter with BF16 support and resumable extent commits.
6. Reopen the final artifact with both Python and C++ Readers and verify root, directories, required/optional features, tensors, routes, and source fingerprints.

A completed content-addressed object is reused only after rehashing. A missing or corrupt object is fetched again. Partial objects, manifests, and K3X outputs are never accepted as completed work.

Provenance remains `transport-pinned-ranges`. The complete 16.99 GB shard LFS SHA-256 is metadata only and is not claimed as locally recomputed.

## Runtime boundary

Add a dedicated `k3x_cuda_official_moe_bench` executable. It is not a new `k3x_run` mode.

The executable performs these steps before constructing a CUDA backend.

1. Verify the fixed official snapshot/config/index/shard identities recorded in the artifact.
2. Verify the K3X root, BF16 required feature, both fixture optional bits, deterministic input digests, tensor set, shapes, dtypes, expert union, and route manifest.
3. Recompute both natural routes from K3X tensors and require exact selected-set and canonical-order equality.

The CPU path decodes BF16 source words explicitly and applies official output-dtype boundaries. Attention Residual and RMSNorm compute their normalization and scoring in FP32 and round their BF16 outputs. BF16 Linear operations use BF16 inputs and weights with FP32 accumulation and round the operation output to BF16. Native-MXFP4 experts receive BF16 latent input, round expert outputs to BF16, mix contributions in FP32, and round the mixed latent output to BF16 before routed normalization. Routed/shared addition and final prefix addition round to BF16. The CUDA path consumes BF16 K3X bytes directly and implements the same boundaries with the existing native-MXFP4 kernels plus native BF16 views; it must not rebuild a BF16 host vector on every call.

M28 keeps Attention Residual preparation, post normalization, and routing on the CPU. Their wall time and bytes are included in the sublayer record. Routed projections, selected experts, routed normalization/up, shared MLP, routed/shared addition, final prefix residual addition, and one final D2H execute through the dependency-closed CUDA MoE boundary.

## Correctness gates

The synthetic graph and all previous artifacts remain unchanged. New gates require:

- every unsupported required feature fails before directory use;
- BF16 source bytes round-trip exactly through Python and C++ Readers;
- wrong BF16 length, quantization, feature bit, CRC, or logical length fails closed;
- fixture conversion resumes without recomputing verified extents and rejects source drift;
- both official routes contain exactly 16 unique valid IDs and differ from each other;
- all 896 router scores are finite;
- selected expert sets and canonical order match the source-byte PyTorch oracle exactly;
- normalized router contributions match within `1e-6` absolute and relative tolerance;
- CPU and CUDA final 7,168-value outputs are finite and match the source-byte oracle with `atol=rtol=2e-2`;
- every intermediate boundary records maximum absolute and relative error;
- transient and resident modes produce the same routes and outputs within the same tolerance;
- the resident path reports zero measured warm weight H2D after exact cold admission;
- production `k3x_run` still returns `NON_EXECUTABLE_ARTIFACT`;
- no source or K3X real-weight artifact is tracked by Git.

## B-0029 measurement

B-0029 runs three warmups and twenty measured iterations for these fixed cases.

1. Case A transient.
2. Case A exact resident.
3. Alternating A/B exact resident with capacity for the complete fixture.

The runner fixes order and does not rerun to select preferred timing. It records cold and warm latency, p05/median/p95, kernel time, router time, Attention Residual time, CPU orchestration time, source/K3X bytes, BF16 and MXFP4 weight H2D separately, activation H2D, D2H, resident and peak VRAM, hit/miss/bypass counters, selected IDs, union size, route overlap, and numerical errors.

Decode tok/s, prefill tok/s, TTFT, token quality, coding quality, physical NVMe traffic, physical PCIe traffic, GPU utilization, and GPU memory bandwidth remain explicitly not measured. B-0029 is a complete official MoE FFN sublayer benchmark over synthetic boundary states, not a token benchmark.

## Failure behavior

All identity, shape, route, feature, and source checks happen before CUDA allocation. Network, checksum, conversion, route, backend, allocation, and numerical failures are fatal and preserve prior completed content-addressed objects. No failure falls back to a different expert, reduced K, synthetic tensor, dequantized replacement artifact, or CPU result presented as CUDA.

## Scope boundary after M28

Successful M28 evidence proves one official layer-1 Stable LatentMoE FFN boundary with two naturally changing routes. It does not prove attention/KDA/MLA integration or tokens. The next architecture decision must use B-0029 to choose between:

- closing one complete official transformer layer with authentic recurrent/attention state; or
- first tracing multiple real MoE layers to measure expert overlap and residency pressure.

No complete shard, full checkpoint, paid VM, Cloud Run Job, default runtime change, adaptive Top-K, proxy, pruning, or throughput projection is authorized by this design.
