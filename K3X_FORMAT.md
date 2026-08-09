# K3X Checkpoint Format Version 1.0

K3X is a little-endian, execution-ordered checkpoint format for the Kimi K3 text decoder. Version 1 deliberately omits generic tensor names. Every tensor is addressed by the FNV-1a 64-bit hash of its canonical source name, with collisions rejected during conversion.

## File order

1. One 4 KiB superblock.
2. 4 KiB-aligned tensor data and optional quantization auxiliary extents in execution order.
3. A 4 KiB-aligned tensor directory.
4. A 4 KiB-aligned layer directory.
5. A 4 KiB-aligned expert directory.
6. One 4 KiB-aligned fixed model-config record.

All integers and floats use little-endian encoding. Reserved bytes must be zero. Readers reject checked-addition overflow, ranges outside the declared file length, overlapping extents, unaligned extents, unsupported required feature bits, invalid enum values, and checksum failures.

## Superblock

The superblock is exactly 4096 bytes. Bytes 232 through 4091 are reserved.

| Offset | Type | Field |
|---:|---|---|
| 0 | `char[8]` | `K3XCHKPT` |
| 8 | `u16` | major version, 1 |
| 10 | `u16` | minor version, 0 |
| 12 | `u32` | superblock bytes, 4096 |
| 16 | `u32` | extent alignment, 4096 |
| 20 | `u32` | state, 0 partial or 1 finalized |
| 24 | `u64` | required feature bits |
| 32 | `u64` | optional feature bits |
| 40 | `u8[16]` | file UUID |
| 56 | `u8[32]` | source fingerprint |
| 88 | `u64` | tensor directory offset |
| 96 | `u64` | tensor directory length |
| 104 | `u64` | layer directory offset |
| 112 | `u64` | layer directory length |
| 120 | `u64` | expert directory offset |
| 128 | `u64` | expert directory length |
| 136 | `u64` | model config offset |
| 144 | `u64` | model config length, 256 |
| 152 | `u64` | first payload offset |
| 160 | `u64` | finalized file length |
| 168 | `u8[32]` | directory SHA-256 |
| 200 | `u8[32]` | root file SHA-256 |
| 4092 | `u32` | superblock CRC32C over bytes 0 through 4091 |

The directory digest hashes the exact tensor-directory bytes, layer-directory bytes, expert-directory bytes, and model-config bytes in that order, excluding alignment padding. The root digest hashes the declared finalized file length while treating bytes 200 through 231 and 4092 through 4095 as zero. The CRC is written last.

Optional feature bit 0 identifies a non-executable `STORAGE_FIXTURE`. Readers may inspect its tensor and expert extents, but model runtimes must reject graph execution before reading model tensors. Unknown optional bits remain ignorable. Required feature handling is unchanged.

## Directory headers

Every directory begins with a 16-byte header containing a four-byte tag, `u32 record_size`, and `u64 record_count`. Tags are `TENS`, `LAYR`, and `EXPT`.

## Tensor record

Tensor records are 128 bytes.

| Offset | Type | Field |
|---:|---|---|
| 0 | `u64` | tensor ID |
| 8 | `u32` | role, version 1 uses 0 for hash-addressed generic tensors |
| 12 | `u16` | dtype, 1 FP32 or 2 UINT8 |
| 14 | `u16` | quantization, 0 none or 1 native MXFP4 |
| 16 | `u8` | rank, 0 through 4 |
| 17 | `u8` | flags |
| 18 | `u16` | reserved |
| 20 | `i32` | layer ID or -1 |
| 24 | `i32` | expert ID or -1 |
| 28 | `u32` | reserved |
| 32 | `u64[4]` | dimensions, unused entries zero |
| 64 | `u64` | data offset |
| 72 | `u64` | data length |
| 80 | `u64` | logical byte length after decode |
| 88 | `u64` | auxiliary offset, zero when absent |
| 96 | `u64` | auxiliary length, zero when absent |
| 104 | `u32` | data CRC32C |
| 108 | `u32` | auxiliary CRC32C, zero when absent |
| 112 | `u8[16]` | reserved |

For native MXFP4, data contains low-nibble-first E2M1 codes and the auxiliary extent contains E8M0 group scales. Group size is stored in model config.

## Layer record

Layer records are 64 bytes and contain `layer_index`, `attention_kind`, `ffn_kind`, the first tensor index and count, the first expert index and count, an Attention Residual write index or -1, flags, and 32 reserved bytes.

Attention kinds are 1 KDA and 2 MLA. FFN kinds are 1 dense and 2 Stable LatentMoE.

## Expert record

Expert records are 64 bytes and contain `layer_index`, `expert_id`, `physical_order`, flags, tensor IDs for gate/up/down matrices, a Q32 profile-frequency prior, and 16 reserved bytes.

## Model config record

The model config is a fixed 256-byte record containing the synthetic or real graph dimensions in the following order.

`vocabulary_size`, `hidden_size`, `layer_count`, `kda_head_count`, `kda_head_dimension`, `short_conv_kernel_size`, `mla_head_count`, `q_lora_rank`, `kv_lora_rank`, `qk_nope_head_dimension`, `qk_rope_head_dimension`, `value_head_dimension`, `expert_count`, `top_k`, `shared_expert_count`, `routed_latent_size`, `expert_intermediate_size`, `dense_intermediate_size`, `attention_residual_block_size`, `mxfp4_group_size`, `rms_norm_epsilon`, `kda_gate_lower_bound`, `routed_scaling_factor`, `activation_situ_beta`, `activation_situ_linear_beta`, `absolute_tolerance`, `relative_tolerance`, and `mla_flags`.

`mla_flags` bit 0 means NoPE and bit 1 means output gating. Remaining bytes are zero.

## Conversion transaction

Conversion writes only `<output>.partial` plus an atomic JSON resume ledger until every extent has been flushed, read back, and CRC-verified. A work unit is identified by its canonical tensor hash and `data` or `auxiliary` suffix. Resume requires exact source, converter-version, and configuration fingerprints. Completed entries must form a canonical prefix of the conversion plan and match the expected ID, aligned offset, source length, current-source CRC32C, and partial-file CRC32C before reuse. Finalization writes directories and the superblock, closes the file, then atomically renames the partial artifact to the requested path.

The bounded `STORAGE_FIXTURE` source contract additionally requires a content-addressed shard, manifest-last publication, one declared shard SHA-256, and a SHA-256 for each of its six source tensors. Conversion verifies all declared digests with bounded reads before creating or resuming output. Unreferenced files do not participate in source identity.

If termination occurs after the final rename but before ledger deletion, the next invocation verifies the finalized artifact's complete integrity, source fingerprint, and model configuration before deleting the stale ledger. A missing partial file is otherwise an error and never causes an unverified final artifact to be accepted.
