# K3X Official First-Token Implementation Plan

> **For agentic workers:** Execute inline in this session. Subagents are explicitly disabled by user instruction.

**Goal:** Run one exact text-only greedy token through the released 93-layer Kimi K3 checkpoint with bounded range streaming.

**Architecture:** A pinned topology manifest drives a resumable range cache and the smallest complete text decoder. Existing K3X exact primitives remain the numerical base; missing graph nodes are added without unrelated optimization.

**Tech Stack:** Python 3, C++20, CUDA 13.3, safetensors range transport, K3X v1.

**Spec:** `docs/superpowers/specs/2026-08-13-k3x-official-first-token-design.md`

## Global Constraints

- No complete checkpoint, shard, RAM, or VRAM residency requirement.
- No paid cloud resource provisioning.
- No token-throughput claim from a one-token feasibility run.
- Focused verification once, then one integration gate.

### Task 1: Exact topology manifest

- [ ] Add a metadata-only planner for global tensors, layer 0, KDA layers, MLA layers, MoE trunks, and expert families.
- [ ] Emit canonical JSON with source identities, tensor dtype/shape/shard, total byte classes, and missing/extra contract failures.
- [ ] Verify one live pinned metadata run without downloading tensor payloads.

### Task 2: Resumable selected-range cache

- [ ] Reuse the fixed-authority HTTPS transport and content-addressed range objects.
- [ ] Bind each object to revision, shard LFS hash, tensor identity, offset, length, and SHA-256.
- [ ] Make interrupted retrieval restart from verified objects only.

### Task 3: Missing exact graph nodes

- [ ] Add released-dimension embedding-row, dense SiTU MLP, Gated MLA, final Attention Residual/RMSNorm, and chunked LM-head reference execution.
- [ ] Connect 93-layer dispatch to the existing exact KDA and MoE paths.
- [ ] Preserve natural Top-16 and native MXFP4 expert bytes.

### Task 4: First-token run

- [ ] Run the focused topology/runtime test once.
- [ ] Execute a fixed text token through the pinned released revision with restartable progress.
- [ ] Record token ID, layer completion, downloaded bytes, wall time, peak RAM/VRAM, and source/artifact digests.
- [ ] Run one integration gate, update the ledger documents, commit, push, and publish through GitHub.
