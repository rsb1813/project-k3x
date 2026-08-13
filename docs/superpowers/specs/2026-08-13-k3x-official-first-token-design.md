# K3X Official First-Token Design

## Goal

Execute one text-only token through the released 93-layer Kimi K3 checkpoint on the target PC without downloading or loading the complete checkpoint.

## Boundary

- Pin the released Hugging Face revision and derive an exact topology manifest from `config.json`, `model.safetensors.index.json`, and bounded safetensors headers.
- Fetch only one embedding row, each layer's always-active tensors, the natural Top-16 routed experts selected at that layer, and chunked LM-head rows.
- Keep KDA recurrent state, MLA KV state, and Attention Residual block sources for the single-token execution.
- Reuse the existing exact KDA, Stable LatentMoE, native MXFP4, and CUDA primitives. Add the missing released-dimension dense layer, Gated MLA, embedding, output Attention Residual, final RMSNorm, and LM-head scan.
- Store every downloaded range as a verified, resumable content-addressed object. Never require a complete source shard, whole checkpoint, whole-model RAM, or whole-model VRAM.
- Treat the first run as a correctness and feasibility result. Record no TPS until repeated decode measurement exists.

## Execution order

1. Resolve one token embedding row.
2. Execute layers 0 through 92 in released order. Layer 0 uses the dense SiTU MLP; layers 1 through 92 use Stable LatentMoE. Layers `3, 7, ..., 91, 92` use Gated MLA and all other layers use KDA.
3. For every MoE layer, load its router and latent/shared trunk first, compute the canonical 896-way router, then fetch exactly the natural Top-16 expert triplets.
4. Apply the output Attention Residual and final RMSNorm.
5. Stream LM-head row chunks, retain the exact FP32 maximum, and emit the greedy token ID.

## Failure contract

Missing tensors, source revision drift, header drift, range digest mismatch, unsupported dtype/shape, non-finite output, or incomplete persistent state fails closed. A partial run retains only verified range objects and an atomic progress ledger; it never publishes a token.

## Validation

Development uses one focused topology/runtime test after implementation and one end-to-end integration gate before publication. Existing broad synthetic and historical evidence suites are not rerun unless the focused gate exposes a shared-path regression.
