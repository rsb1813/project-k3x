# K3X Official Range Discovery Design

## Status and scope

Milestone 26 is the first bounded interaction with the official Kimi K3 checkpoint. It resolves and records the official Hugging Face snapshot, verifies the model index and released text configuration, inspects one safetensors header through HTTP ranges, and optionally materializes exactly one real native-MXFP4 routed expert as a non-executable K3X storage fixture.

The user's standing approval covers non-billable work before Cloud Run. This milestone therefore may fetch bounded public metadata and one explicitly capped expert range, but it must not download a complete 17 GB shard, the 1.56 TB checkpoint, or any paid resource. Real tensor bytes and derived artifacts remain local and untracked. Only code, tests, canonical metadata, hashes, and measurements may enter Git.

## Verified official snapshot

The accepted authority is the public, ungated Hugging Face repository `moonshotai/Kimi-K3`, which the official MoonshotAI repository links as the released weights. Discovery resolved `main` on 2026-08-10 to commit `9f62e4e9fffbd0a83ddd60e1c209d828994b3569`.

The Hugging Face model API reported 118 files and 1,560,998,984,390 repository bytes. `model.safetensors.index.json` is 59,764,096 bytes with LFS SHA-256 `a1c5210650ce71d2d3ae9ec5a101ac4afd3cf4b10091be589853437eb967febd`. The index reports 497,220 tensors, 96 weight shards, and `metadata.total_size = 1,560,860,324,864` tensor bytes.

Layer 1 expert 0 resides wholly in `model-00002-of-000096.safetensors`, whose API-declared size is 16,990,911,504 bytes and LFS SHA-256 is `26a3284e1d2cb567934ebef002e6a1813551d646739e8bcb1e9e3fe7f878e0f5`. An eight-byte range request returned HTTP 206, exact total size, and safetensors header length 818,696. The header range is therefore file bytes `[8, 818704)`.

The six selected tensors are contiguous in the shard data section.

| Official tensor suffix | K3X role | Shape | Data offsets | Bytes |
|---|---|---:|---:|---:|
| `w1.weight_packed` | gate packed | 3,072 × 1,792 | 1,267,744,256–1,273,249,280 | 5,505,024 |
| `w1.weight_scale` | gate scale | 3,072 × 112 | 1,273,249,280–1,273,593,344 | 344,064 |
| `w2.weight_packed` | down packed | 3,584 × 1,536 | 1,273,593,344–1,279,098,368 | 5,505,024 |
| `w2.weight_scale` | down scale | 3,584 × 96 | 1,279,098,368–1,279,442,432 | 344,064 |
| `w3.weight_packed` | up packed | 3,072 × 1,792 | 1,279,442,432–1,284,947,456 | 5,505,024 |
| `w3.weight_scale` | up scale | 3,072 × 112 | 1,284,947,456–1,285,291,520 | 344,064 |

The physical payload range adds the 818,704-byte data-section base, giving `[1,268,562,960, 1,286,110,224)` and exactly 17,547,264 bytes. The official `modeling_kimi_linear.py` defines w1 as gate, w2 as down, and w3 as up; this mapping is not inferred from conventional names.

## Alternatives considered

### API tree only

This downloads almost no checkpoint metadata and is suitable for a repository inventory, but it cannot prove tensor-to-shard placement or identify byte ranges. It cannot produce a conversion smoke and is rejected as the complete milestone.

### Full-shard download first

Downloading shard 2 would permit complete LFS SHA-256 verification before tensor extraction. It costs 16.99 GB and delays the first real-byte evidence. It remains the required provenance level for later production conversion but is deferred from the first bounded smoke.

### Pinned index plus exact HTTP ranges

This approach downloads and verifies the 59.76 MB index, the 0.82 MB shard header, and one 17.55 MB contiguous expert payload. It reaches real native-MXFP4 bytes quickly while retaining strict byte ceilings and exact source identities. It cannot recompute the full shard's LFS SHA-256, so its provenance is explicitly `transport-pinned-range`, never `full-shard-verified`. This is the accepted Milestone 26 design.

## Component boundaries

### Official snapshot discovery

`converter/k3x_converter/official_source.py` owns repository metadata, revision resolution, canonical JSON parsing, allowed-host checks, and bounded HTTP requests. The production authority is fixed to `moonshotai/Kimi-K3`; tests inject a transport rather than arbitrary production URLs.

The discovery record uses schema `k3x-official-discovery-v1` and contains the repository, requested revision, resolved 40-character lowercase commit, API observation fields, index identity, all 96 shard path/size/LFS identities, canonical snapshot digest, and provenance level. Observation time may appear in an audit envelope but is excluded from the canonical identity digest.

### Index and configuration validation

The pinned index is fetched atomically with a 64 MiB ceiling, hashed while streaming, and accepted only when size and SHA-256 equal the API record. JSON duplicate keys, non-standard constants, wrong top-level keys, non-string tensor/shard names, unsupported shard paths, inconsistent shard counts, missing declared shards, and non-integer `metadata.total_size` fail closed.

The pinned `config.json` is capped at 1 MiB, hashed locally, and bound to the API-declared Git blob identity by recomputing `SHA-1("blob " + decimal_length + NUL + payload)`. It is then checked for the released text dimensions needed by the slice: 93 layers, one dense layer, 896 experts, natural Top-16, hidden width 7,168, routed latent width 3,584, expert intermediate width 3,072, MXFP4 group size 32, SiTU constants 4.0 and 25.0, and routed scaling 1.0. Blob-identity or configuration mismatch stops before any tensor range.

### Expert range planning

`plan_official_expert(index, layer_id=1, expert_id=0)` requires exactly six official U8 tensors in one declared shard. It validates the official names, w1/w2/w3 role mapping, shapes, lengths, contiguity, and 17,547,264-byte union. It returns immutable source ranges and canonical K3X names under `model.layers.1.feed_forward.experts.0.{gate,down,up}`.

M26 intentionally fixes the canonical live selection to layer 1 expert 0. General layer/expert selection is accepted only in unit tests until another live identity is measured. This prevents a floating CLI argument from silently broadening network cost.

### Bounded range transport

Every network operation has a 120-second timeout, at most five redirects, and an allowed final host of `huggingface.co` or a hostname ending in `.hf.co`. Metadata API bodies are capped at 4 MiB, the index at 64 MiB, config at 1 MiB, a safetensors header at 100,000,000 bytes, and selected tensor payload at 32 MiB.

Range requests require HTTP 206 and an exact `Content-Range` start, end, and full shard size. HTTP 200, missing or mismatched range metadata, short/long bodies, revision drift, LFS identity drift, untrusted redirect hosts, and limit overruns fail before publication. Downloaded range SHA-256 values are recorded, but they do not replace the unverified full-shard LFS digest.

### Real storage-slice materialization

`materialize_official_expert_slice()` writes a temporary local safetensors microshard containing the exact six official byte sequences under canonical K3X tensor names. It preserves the official U8 shapes, computes per-tensor and local microshard SHA-256, fsyncs, and atomically renames the content-addressed shard.

The accompanying existing `k3-storage-slice-v1` source manifest retains `artifact_kind = storage_fixture`, released config, packed shapes, exact local hashes, and a new ignored-by-v1-reader `source_provenance` object containing official repository/revision/index/shard/range identities and `verification = transport-pinned-range`. The existing converter then verifies all local bytes and emits a K3X artifact with `OPTIONAL_STORAGE_FIXTURE`; graph execution must continue to reject it.

Failure removes temporary index, header, microshard, manifest, and K3X partials created by that invocation. An already published content-addressed object is immutable and may be reused only after its local digest is reverified.

## CLI and evidence

`python -m tools.discover_official_kimi_k3` defaults to `--dry-run`. Dry-run resolves the official snapshot, verifies config and index, plans the fixed expert, and writes only canonical discovery JSON. It must report zero tensor-payload bytes.

`--materialize-expert` performs the exact header and 17,547,264-byte payload ranges, creates the local source fixture, converts it to K3X, validates it with `K3XReader`, and records B-0027 JSON/CSV summary evidence. Real source and K3X artifacts live below a user-supplied untracked output directory and are never copied into `results/` or Git.

B-0027 records resolved revision, all expected upstream sizes and digests, HTTP request count, metadata bytes, header bytes, tensor-payload bytes, maximum response bytes, local tensor/microshard/K3X hashes, wall time, Reader validity, and optional-feature identity. It explicitly omits token throughput, GPU metrics, physical NVMe attribution, quality, and full-shard verification.

## Test strategy

Unit tests use an injected deterministic transport and witness RED/GREEN for revision drift, malformed API records, duplicate/non-standard JSON, index hash or size mismatch, wrong tensor ownership, wrong w1/w2/w3 shape or role, noncontiguous ranges, oversized bodies, HTTP 200 on a range, wrong `Content-Range`, short reads, redirect-host escape, config mismatch, atomic cleanup, and accidental payload access during dry-run.

Integration tests construct a tiny local HTTP fixture with redirects and Range support, materialize a six-tensor microshard, convert it through the unchanged writer, verify tensor payload digests and `OPTIONAL_STORAGE_FIXTURE`, and prove runtime rejection before graph execution.

The live official test is opt-in through `K3X_TEST_OFFICIAL_DISCOVERY=1`. Canonical B-0027 runs once after all mock and local integration tests pass. No live payload request occurs during ordinary CI.

## Completion boundary

Milestone 26 completes only when dry-run and exact-range modes have passing tests, one digest-backed live B-0027 record exists without committed real bytes, the K3X output is byte-verified and non-executable, all applicable local/CI gates pass, and the TITAN Ledger distinguishes transport-pinned range evidence from full-shard verification.

Milestone 27 may then use the bounded real expert artifact for the first local real out-of-core CUDA layer execution. Production conversion still requires complete source-object verification or an equivalently strong authenticated chunk scheme.
