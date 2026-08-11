# K3X Milestone 29 Official KDA Transformer-Layer Design

## Status

Accepted on 2026-08-11 under the standing authorization to continue non-billable work before Cloud Run provisioning. This milestone closes one official Kimi K3 layer-1 transformer boundary over a deterministic two-token state sequence. It does not include layer 0, embeddings, logits, authentic prompt tokens, a complete shard, the full checkpoint, token throughput, or a production-runtime default.

## Goal

Build one bounded, content-addressed, non-executable K3X fixture that executes the official layer-1 sequence from the self-attention Attention Residual input through the final MoE prefix accumulation.

The milestone must prove all of the following.

1. Exact official tensor planning and bounded range materialization for layer-1 KDA and both Attention Residual halves.
2. A source-byte PyTorch oracle for a two-token KDA sequence with explicit convolution and recurrent state.
3. Full two-token execution and token-by-token incremental execution with the same final output, KDA state, natural Top-16 routes, and selected expert contributions.
4. Portable C++ parity for every dependency-complete boundary.
5. Native `sm_120` CUDA execution with exact native-MXFP4 experts and byte-native BF16/F32 trunk tensors.
6. Measured complete-layer latency, logical weight/state traffic, residency, and numerical error without presenting the result as tokens per second.

## Official authority and discovered inconsistency

The model authority is `moonshotai/Kimi-K3` revision `9f62e4e9fffbd0a83ddd60e1c209d828994b3569`.

- The pinned `modeling_kimi_linear.py` is 51,506 bytes and its computed Git blob ID is `b8c41e8bfce768d74d8da3a37e693f5ee43876a0`, exactly matching repository metadata.
- The pinned configuration fixes 93 layers, hidden width 7,168, KDA layer 1, 96 heads, head dimension 128, short-convolution width 4, full-rank output gating, gate lower bound `-5.0`, Attention Residual block size 12, and RMS epsilon `1e-5`.
- The Kimi Linear report defines KDA decay as channel-wise and the recurrent update as `(I - beta k k^T) Diag(alpha) S + beta k v^T`.
- The pinned checkpoint header stores `self_attn.A_log` as F32 shape `[128]`, which agrees with channel-wise decay.
- The pinned Python constructor creates `A_log` with shape `[num_heads]`, or `[96]`. K3X must not silently reproduce this source/checkpoint mismatch. The checkpoint header and KDA paper are authoritative for M29, and every planner/oracle/backend must require `[128]`.

The implementation fails closed if the pinned revision, source blob, configuration, shard header, tensor dtype, tensor shape, or source range differs from this contract.

## Approaches considered

### Accepted: layer-1 state boundary with external causal inputs

Provide the layer-1 hidden vector, its layer-0 Attention Residual source-bank row, and initial KDA state as explicit boundary inputs. Execute layer 1 completely. This is the smallest causal closure that exposes recurrent attention, both Attention Residual operations, natural routing, MoE, final output, and state traffic without importing layer-0 weights.

### Rejected for M29: include embeddings and layer 0

This would create more authentic activations, but it would also introduce a separate MLA/dense-layer closure, substantially more payload, KV state, and additional correctness questions. It would make the first complete-layer bottleneck hard to attribute.

### Staging gate only: official KDA without the FFN

A KDA-only oracle is required to isolate recurrence and precision defects before composing the layer, but it does not satisfy D-060 and cannot be reported as M29 completion.

## Boundary inputs and state

The fixture owns two deterministic tokens named `A` and `B`. Each token has one 7,168-value `hidden_input` and one 7,168-value `block_source`. Values are generated in FP32 by fixed integer formulas with power-of-two denominators, serialized as little-endian IEEE-754 FP32, and hashed. No language RNG, seed search, or route search is permitted.

For zero-based index `i`, the inputs are fixed as follows.

- `A.hidden_input[i] = (((17*i + 3) % 257) - 128) / 1024`.
- `A.block_source[i] = (((29*i + 11) % 251) - 125) / 1024`.
- `B.hidden_input[i] = (((31*i + 7) % 263) - 131) / 1024`.
- `B.block_source[i] = (((43*i + 19) % 269) - 134) / 1024`.

These are the existing M28 formulas under layer-input and source-bank names. Reusing them preserves a known deterministic input identity; it does not require or assert preservation of M28's post-attention routes.

The initial state is exactly zero.

- Three BF16 convolution histories each have shape `[1, 3, 12288]` and together occupy 221,184 bytes.
- The mathematical recurrent state is key-by-value `[K,V]`, while the pinned call requests transposed V-first storage. K3X therefore serializes FP32 `[1,96,128V,128K]`, records the layout identity explicitly, and occupies 6,291,456 bytes. Equal K/V dimensions must not be used to ignore the layout flag.
- The source-bank input for each token has shape `[1, 1, 7168]`.

The oracle executes both tokens in a single two-token call and as two incremental calls. Token B must consume token A's returned convolution and recurrent state. The two paths must match at every layer output, final state, route set, and contribution vector within operation-specific tolerances.

## Exact layer graph

For each token, execute the pinned layer-1 graph in this order.

1. Set `prefix_sum = hidden_input`.
2. Apply self-attention Attention Residual over `[block_source, prefix_sum]` using `self_attention_res_norm` and `self_attention_res_proj`, with variance, scoring, softmax, and weighted accumulation in FP32 followed by the official output-dtype boundary.
3. Apply `input_layernorm` with epsilon `1e-5`.
4. Compute BF16 Q, K, and V projections from width 7,168 to 12,288.
5. Apply independent width-4 depthwise short convolutions with F32 weights and SiLU, consuming and updating the three convolution histories.
6. Reshape Q, K, and V to `[96, 128]`; L2-normalize Q and K with the official KDA scale.
7. Compute the rank-128 forget projection, combine it with F32 `dt_bias`, and apply channel-wise decay from F32 `A_log[128]` with the configured safe lower bound.
8. Compute scalar-per-head `beta` from `b_proj` and sigmoid.
9. Update FP32 recurrent state using the KDA recurrence and compute the recurrent output.
10. Compute the full-rank BF16 output gate, apply sigmoid-gated head-wise RMSNorm using F32 `o_norm`, then project 12,288 to 7,168 with BF16 `o_proj`.
11. Add KDA output to `prefix_sum` using the official output-dtype boundary.
12. Apply MLP Attention Residual over `[block_source, prefix_sum]` using the existing official M28 tensors.
13. Apply `post_attention_layernorm`, all-896 sigmoid routing with correction-bias selection, natural Top-16 contribution normalization, routed-down, exact native-MXFP4 experts, routed norm/up, shared expert, routed/shared addition, and final prefix accumulation exactly as M28.

The portable scalar recurrence is the semantic authority. An upstream FLA recurrent/chunk implementation may be used as an additional comparison oracle, but it cannot replace the independent recurrence or weaken checkpoint-shape validation.

## Required new official tensors

Header-only inspection confirms all 17 new tensors are in `model-00002-of-000096.safetensors`. Their exact unaligned payload is 887,843,840 bytes.

| Tensor suffix below `language_model.model.layers.1` | Dtype | Shape | Bytes |
|---|---|---:|---:|
| `self_attn.A_log` | F32 | 128 | 512 |
| `self_attn.dt_bias` | F32 | 12,288 | 49,152 |
| `self_attn.k_conv1d.weight` | F32 | 12,288 x 1 x 4 | 196,608 |
| `self_attn.o_norm.weight` | F32 | 128 | 512 |
| `self_attn.q_conv1d.weight` | F32 | 12,288 x 1 x 4 | 196,608 |
| `self_attn.v_conv1d.weight` | F32 | 12,288 x 1 x 4 | 196,608 |
| `input_layernorm.weight` | BF16 | 7,168 | 14,336 |
| `self_attention_res_norm.weight` | BF16 | 7,168 | 14,336 |
| `self_attention_res_proj.weight` | BF16 | 1 x 7,168 | 14,336 |
| `self_attn.b_proj.weight` | BF16 | 96 x 7,168 | 1,376,256 |
| `self_attn.f_a_proj.weight` | BF16 | 128 x 7,168 | 1,835,008 |
| `self_attn.f_b_proj.weight` | BF16 | 12,288 x 128 | 3,145,728 |
| `self_attn.g_proj.weight` | BF16 | 12,288 x 7,168 | 176,160,768 |
| `self_attn.k_proj.weight` | BF16 | 12,288 x 7,168 | 176,160,768 |
| `self_attn.o_proj.weight` | BF16 | 7,168 x 12,288 | 176,160,768 |
| `self_attn.q_proj.weight` | BF16 | 12,288 x 7,168 | 176,160,768 |
| `self_attn.v_proj.weight` | BF16 | 12,288 x 7,168 | 176,160,768 |

The final unaligned tensor payload is `1,267,744,256 + 17,547,264 * U` bytes, where `U` is the natural two-token expert union and `16 <= U <= 32`. The maximum is 1,829,256,704 bytes. The actual union and aligned artifact length are recorded only after route derivation; no route is chosen to minimize bytes.

## Manufacturing and physical layout

Manufacturing extends the M28 two-phase protocol rather than creating a second trust model.

1. Revalidate snapshot, config, index, pinned source blob, shard identity, and all 28 always-active layer tensor metadata entries before payload access.
2. Materialize the 17 new tensors as content-addressed bounded range objects with an 8 MiB maximum request, exact length, per-object SHA-256, fsync, and atomic rename.
3. Build the independent KDA oracle, derive both incremental MoE routes, and publish the route/state manifest before expert fetching.
4. Reuse every matching M28 content object only after rehashing; fetch only missing natural-union experts.
5. Assemble one final K3X artifact in execution order: self Attention Residual, input norm, Q/K/V and short conv, forget/beta/output gates, KDA output norm/projection, MLP Attention Residual, router, routed-down, selected experts, routed norm/up, and shared expert.
6. Bind snapshot, config, source blob, header, input, initial/final state, route, tensor, source-manifest, and final K3X root identities.

The complete 16.99 GB shard is never downloaded. Provenance remains pinned transport ranges rather than a locally recomputed complete-shard LFS digest.

## Runtime boundary

Add a dedicated complete-layer benchmark executable. Do not make the fixture executable through `k3x_run`.

Implementation proceeds through three independently testable layers.

1. Portable KDA scalar oracle over tiny and official dimensions.
2. Portable complete-layer composition using the existing official MoE oracle.
3. CUDA complete-layer execution, initially allowing host-orchestrated Attention Residual/routing while all large projections, recurrence, and MoE kernels execute on the GPU.

All identities and tensor metadata must be verified before backend allocation. No failure may silently replace official tensors, change K, omit experts, reset state, substitute CPU output for CUDA, or claim a partial boundary as complete-layer execution.

## Correctness gates

- Pinned source Git blob, config, index, shard, header, range, dtype, shape, and length validation all fail closed.
- `A_log` must be exactly F32 `[128]`; `[96]` is a negative regression case.
- Tiny independent PyTorch and portable C++ KDA recurrence agree for zero and nonzero initial states.
- Full two-token and incremental A-then-B paths agree on both outputs, all convolution histories, recurrent state, selected expert IDs, and contributions.
- Every one of 896 router scores is finite and both routes contain 16 unique expert IDs.
- Source-byte PyTorch, portable C++, and CUDA complete-layer outputs and final states agree within separately recorded maximum absolute and relative tolerances.
- Transient and exact-resident modes preserve outputs, state, routes, and contributions.
- Resident warm calls report zero logical weight H2D for the complete resident set.
- Production `k3x_run` continues to return `NON_EXECUTABLE_ARTIFACT`.
- No real tensor object or K3X artifact is tracked by Git.

## Measurement contract

The formal milestone benchmark is B-0030 and is run once after all correctness and sanitizer gates pass. Its fixed rows compare one-token cold/transient, two-token incremental resident, and two-token full-sequence resident execution. It records layer latency, KDA/MoE/kernel/orchestration time, weight and state H2D/D2H, resident/peak VRAM, system RAM, Reader traffic, final-state bytes, route union, numerical error, and operation counters.

Decode tok/s, prefill tok/s, TTFT, coding quality, physical NVMe/PCIe traffic, GPU utilization, and bandwidth remain unmeasured unless a later executable token loop supplies those observations. B-0030 must not derive or imply a token-rate figure from isolated layer latency.

## Completion boundary

M29 is complete only after the ignored bounded fixture, independent full/incremental parity, CPU/CUDA complete-layer execution, sanitizers, fixed B-0030 evidence, synchronized README and TITAN Ledger, public integration, and post-merge CI all pass.

After M29, measured evidence chooses between local whole-layer kernel fusion and the first bounded multi-layer routing/residency trace. Neither choice is made in advance.
