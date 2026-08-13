# K3X Local Foundry Design

## Goal

Manufacture a 1.28 TB local-only K3X checkpoint on the target PC without ever storing the complete source checkpoint, while preserving a measured path back to native MXFP4 for sensitive tensors.

## Accepted boundary

- Download complete official shards with authenticated `hf_xet` and `HF_XET_HIGH_PERFORMANCE=1`.
- Keep at most two source shards on the 1 TB HDD: one converting and one downloading.
- Use bounded RAM buffers and the RTX 5080 for statistics and quantization.
- Write finalized extents directly into a `.partial` K3X artifact on the P44 Pro. Final publication is a same-volume rename, not a second 1.28 TB copy.
- Reserve 200 GiB on the destination volume and 100 GiB on the staging volume. Refuse to start or continue when either reserve would be violated.
- Preserve router, norms, KDA, MLA, Attention Residual, embeddings, LM head, shared experts, and sensitivity-selected routed experts at their accepted precision. Only accepted low-sensitivity routed experts may use the new 3-bit representation.
- Do not begin the full checkpoint download until the synthetic 3-bit codec round trip, metadata contract, and reference runtime decode pass.

## Data flow

1. The Conductor pins the official revision and creates 96 idempotent shard units.
2. The Xet stage downloads the next shard to `D:` and verifies its official LFS SHA-256.
3. The Foundry reads bounded tensor ranges, collects statistics, applies the frozen precision recipe, and appends aligned K3X extents to `C:`.
4. The IMMORTAL ledger records source identity, output extents, hashes, converter version, and resource accounting before a source shard is deleted.
5. The next unit reuses the released HDD space. A restart resumes only from verified ledger entries.
6. Final directories and the superblock are written only after every unit and the global byte budget close.

## Failure and quality contract

- Authentication, revision drift, source hash mismatch, insufficient disk reserve, non-finite statistics, codec error, output checksum failure, or budget overflow fails closed.
- Source shards are deleted only after their output extents and ledger update are durable.
- The 1.28 TB target is a hard upper bound, not a reason to silently demote sensitive tensors. If calibration cannot meet both quality and size, the job stops and reports the minimum safe size.
- No quality loss, token rate, or completion time is claimed before measured evaluation.

## Verification

- One focused synthetic test covers disk preflight, deterministic 3-bit packing/decode, ledger resume, and source deletion eligibility.
- One final integration gate converts the synthetic K3-compatible checkpoint through the exact local data flow and compares layer outputs and greedy tokens with its reference mode.
- The full official download begins only after both gates pass.
