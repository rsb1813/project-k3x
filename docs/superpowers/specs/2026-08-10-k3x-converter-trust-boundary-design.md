# K3X Converter Trust-Boundary Design

## Status and scope

Milestone 25 hardens the existing streaming converter before any externally supplied Kimi K3 shard is trusted. D-028 already verifies the bounded storage fixture's shard/tensor hashes and canonical resume extent prefix, so this milestone does not reimplement that closed work. It extends the same fail-closed discipline to the generic source index, safetensors metadata, resume-ledger schema, and crash residue around the committed extent prefix.

This milestone does not download Kimi K3 weights, define signed supply-chain provenance, change K3X v1 on-disk records, provision cloud resources, or claim power-loss durability. The next milestone owns bounded real-checkpoint discovery and its source-manifest production contract.

## Accepted approach

Use a narrow trust-boundary audit rather than either a no-op repetition of D-028 or a premature signed-manifest v2.

1. Parse the source manifest as untrusted structured data and reject malformed or duplicate JSON fields with stable `K3XError` codes.
2. Resolve every referenced shard through one containment helper. Absolute paths, parent traversal, non-file targets, and symlink escapes outside the declared source directory are rejected before shard inspection.
3. Inspect each referenced safetensors shard independently and prove exact ownership: every manifest tensor appears in exactly its declared shard, no referenced shard supplies an undeclared tensor, and duplicate tensor names across shards are rejected rather than silently overwritten.
4. Validate safetensors header structure, tensor names, dtype, shape, offsets, byte length, and non-overlap without reading tensor payloads into memory. Duplicate JSON keys and malformed metadata fail with a stable source-header error.
5. Parse the resume ledger through a strict schema boundary. Required top-level and extent keys, lowercase fixed-width hex strings, non-boolean integers, canonical extent order, and valid UUID length are verified before reuse.
6. Treat the ledger as the commit record. After every committed extent and partial-file CRC is verified, truncate and fsync any bytes after the aligned end of the committed prefix. This recovers a crash after payload fsync but before ledger publication without preserving orphan bytes or allowing unbounded suffix growth.

## Source manifest contract

The accepted source formats remain `synthetic-k3-source-v1` and `k3-storage-slice-v1`. Both require exactly one `weight_map` dictionary whose keys and values are non-empty strings. Shard values must be normalized single relative paths beneath the source directory. Multiple tensors may reference one shard, but each tensor is owned by the exact shard named in `weight_map`.

The generic synthetic format continues to derive `source_sha256` from the canonical manifest bytes plus all referenced shard bytes because it is a deterministic test format. The bounded storage fixture additionally retains D-028's declared full-shard and per-tensor SHA-256 verification. M25 does not mistake a locally computed digest for publisher authenticity; it proves conversion identity and corruption detection only.

## Safetensors inspection contract

`inspect_shard()` remains metadata-only and bounded by the header length. It rejects the following before returning any `SourceTensor`.

- duplicate JSON keys, non-object root/header entries, empty or reserved tensor names;
- missing or extra tensor metadata fields outside `dtype`, `shape`, and `data_offsets`;
- unsupported metadata types, booleans where integers are required, negative dimensions, invalid offset pairs, and overlapping extents;
- dtype/shape/extent byte-count disagreement for the currently supported source dtypes `F32` and `U8`;
- any declared data range outside the shard or any unreferenced bytes inside a tensor range.

The header length is rejected when it exceeds either the file payload boundary or the official safetensors 100 MB default header limit. The limit is a denial-of-service boundary rather than a measured K3 layout assumption and is checked before allocating or reading the declared header. This follows the upstream safetensors format and implementation boundary rather than inventing a smaller project-specific cap.

## Resume ledger and recovery contract

`read_resume_manifest()` returns only a fully validated `ResumeManifest` or raises `K3XError("INVALID_RESUME_MANIFEST")`. It never leaks `KeyError`, `TypeError`, `ValueError`, JSON decoding errors, or duplicate-key ambiguity across the converter boundary.

The existing writer still proves that `completed` is a canonical prefix of expected execution-order extents and that source CRC32C equals the ledger value. It then proves the committed bytes in the partial file have the same CRC32C. Only after both proofs succeed may the writer truncate the partial file to `align_up(last.offset + last.length)`, or to `SUPERBLOCK_BYTES` for an empty ledger, and fsync it. Missing committed bytes remain an error; only an uncommitted suffix is discarded.

## Error behavior

New stable errors are limited to the trust boundary.

- `INVALID_SOURCE_MANIFEST` for malformed JSON/schema/duplicate fields.
- `SOURCE_SHARD_PATH_ESCAPE` for absolute, traversal, or resolved path escape.
- `SOURCE_TENSOR_SHARD_MISMATCH` for wrong-shard ownership, duplicates, or undeclared tensors.
- Existing `INVALID_SOURCE_HEADER`, `INVALID_SOURCE_EXTENT`, `OVERLAPPING_SOURCE_EXTENT`, and dtype errors remain where their meanings already fit.
- `INVALID_RESUME_MANIFEST` for malformed ledger syntax or schema.

No malformed input may create or modify the output, partial artifact, or resume ledger. Resume suffix truncation is permitted only after the source identity, ledger schema, canonical extent prefix, source CRCs, and partial CRCs have all passed.

## Testing and evidence

Tests use real temporary manifests, safetensors shards, ledgers, and partial files rather than mocks.

- Source tests witness RED for `../` and absolute shard paths, a symlink escape where supported, tensor declared in shard A but physically found in shard B, and the same tensor duplicated across referenced shards.
- Safetensors tests witness RED for duplicate JSON keys, malformed metadata types, negative dimensions, and dtype/shape/extent byte-count mismatch.
- Ledger tests witness RED for malformed JSON, missing/extra keys, booleans as integers, invalid/uppercase/wrong-length hex, duplicate JSON keys, and non-list `completed`.
- Crash recovery writes a valid unledgered suffix after the committed extent, resumes conversion, and proves byte-identical output and removal of the suffix. A corrupt committed byte must still fail without truncating or rewriting the partial.
- Focused tests run after each RED/GREEN cycle. Final gates run CPU, liburing/direct, ASan/UBSan, CUDA, committed-evidence verifiers, and applicable sanitizer coverage. Because this milestone changes converter Python only, Compute Sanitizer is a regression gate for the unchanged CUDA runtime rather than direct coverage of the new parser.

## Documentation and next boundary

`ARCHITECTURE.md` will describe the implemented trust boundary, `DECISIONS.md` will supersede the stale M25 gap wording without weakening D-028, and `BENCHMARKS.md` will record correctness/resource measurements without inventing tok/s. `PROJECT_STATE.md` is updated last.

After public integration, Milestone 26 may discover the official Kimi K3 checkpoint index and build a bounded, content-addressed conversion manifest without downloading the full checkpoint.
