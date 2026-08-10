# K3X Official Range Discovery Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Discover the pinned official Kimi K3 checkpoint, materialize exactly one bounded native-MXFP4 expert range, and round-trip it through K3X without downloading a complete shard or committing real weights.

**Architecture:** A standard-library HTTPS boundary resolves a fixed public Hugging Face repository and enforces byte, redirect, host, revision, and exact-range limits. Pure parsers bind API metadata, index, config, and safetensors header identities before a materializer atomically publishes one content-addressed six-tensor microshard and invokes the unchanged K3X converter. Ordinary tests inject transport responses; only one opt-in live test may fetch the real 17,547,264-byte payload.

**Tech Stack:** Python 3.12, standard-library `urllib.request`, `hashlib`, strict JSON decoding, pytest 9.1, existing K3X converter/Reader, C++20 runtime rejection gate.

## Global Constraints

- Production repository authority is exactly `moonshotai/Kimi-K3` at resolved commit `9f62e4e9fffbd0a83ddd60e1c209d828994b3569` for B-0027.
- Do not download a complete 16,990,911,504-byte shard, the 1.56 TB checkpoint, or provision any paid resource.
- Cap API bodies at 4 MiB, index at 64 MiB, config at 1 MiB, safetensors header at 100,000,000 bytes, and tensor payload at 32 MiB.
- Each request uses a 120-second timeout, at most five redirects, and only `huggingface.co` or a hostname ending in `.hf.co`.
- Exact ranges require HTTP 206, exact `Content-Range`, exact body length, and the API-declared full object size.
- Dry-run must perform zero tensor-payload requests and report zero tensor-payload bytes.
- The live selection is fixed to layer 1, expert 0; generic layer/expert behavior is unit-test-only.
- Real bytes, source fixtures, and K3X files stay under a user-supplied ignored directory and never enter Git or `results/`.
- Range evidence is `transport-pinned-range`, never `full-shard-verified`.
- Every new Python source file starts with a one-line Korean role comment.
- Every production behavior follows a witnessed RED/GREEN cycle and each logical task ends in a semantic commit.

---

## File structure

- `converter/k3x_converter/official_transport.py` owns bounded HTTPS, redirect-host validation, exact range responses, and byte counters.
- `converter/k3x_converter/official_source.py` owns the fixed repository schema, strict discovery/index/config/header parsing, expert planning, atomic materialization, and canonical records.
- `converter/k3x_converter/safetensors_reader.py` exposes one metadata parser shared by local shard inspection and remote-header planning.
- `tools/discover_official_kimi_k3.py` owns dry-run/materialize CLI orchestration and canonical audit output.
- `tools/verify_official_discovery.py` verifies committed B-0027 metadata without needing untracked weights.
- `tests/python/test_official_transport.py` tests the network trust boundary with deterministic responses and a local HTTP server.
- `tests/python/test_official_source.py` tests pure discovery, parsing, planning, and materialization.
- `tests/python/test_official_discovery_cli.py` tests CLI/evidence behavior and the opt-in live boundary.

### Task 1: Bounded HTTPS transport and snapshot identity

**Files:**
- Create: `converter/k3x_converter/official_transport.py`
- Create: `tests/python/test_official_transport.py`
- Create: `converter/k3x_converter/official_source.py`
- Create: `tests/python/test_official_source.py`

**Interfaces:**
- Produces: `HttpResponse(status: int, final_url: str, headers: Mapping[str, str], body: bytes)`.
- Produces: `Transport.get(url: str, *, headers: Mapping[str, str], max_bytes: int, timeout_seconds: float) -> HttpResponse`.
- Produces: `UrllibTransport(max_redirects: int = 5)` and `TransportStats(requests, response_bytes, maximum_response_bytes)`.
- Produces: `discover_official_snapshot(transport: Transport) -> OfficialSnapshot` with fixed repository, resolved revision, file identities, and canonical digest.

- [ ] **Step 1: Write failing transport tests.**

  Add tests which prove a bounded body is accepted, `max_bytes + 1` is rejected with `OFFICIAL_BODY_LIMIT`, an untrusted redirect target is rejected with `UNTRUSTED_OFFICIAL_HOST`, more than five redirects is rejected, HTTP status drift is rejected, and counters include only returned response bytes. The local server must return deterministic byte bodies and must never contact the public internet.

- [ ] **Step 2: Run the transport RED gate.**

  Run `python -m pytest tests/python/test_official_transport.py -q` and record failure because `k3x_converter.official_transport` does not exist.

- [ ] **Step 3: Implement the minimum transport.**

  Implement a validating `HTTPRedirectHandler`, exact host predicate, capped streaming read of `max_bytes + 1`, lower-cased response headers, and immutable stats. Do not add retries, authentication, arbitrary production repositories, or a third-party HTTP dependency.

- [ ] **Step 4: Run the transport GREEN and regression gates.**

  Run `python -m pytest tests/python/test_official_transport.py -q` and `python -m pytest tests/python/test_source_manifest_integrity.py tests/python/test_safetensors_integrity.py -q`.

- [ ] **Step 5: Write failing snapshot tests.**

  Use a small injected `FakeTransport` to provide API JSON. Assert exact repository/revision, lowercase 40-hex resolved commit, 118 files, index size/LFS digest, shard 2 size/LFS digest, canonical digest stability when observation time changes, and rejection of duplicate keys, booleans-as-sizes, missing LFS identity, path traversal, duplicate paths, revision drift, and oversized API data.

- [ ] **Step 6: Run the snapshot RED gate.**

  Run `python -m pytest tests/python/test_official_source.py -k snapshot -q` and confirm failure because `discover_official_snapshot` is absent.

- [ ] **Step 7: Implement strict snapshot discovery.**

  Add immutable `OfficialFile` and `OfficialSnapshot` records, duplicate-key/non-standard-constant JSON rejection, fixed API endpoint construction, 40-hex commit validation, LFS SHA-256 parsing, normal Git blob ID retention, normalized relative POSIX paths, and canonical sorted JSON SHA-256 excluding observation time.

- [ ] **Step 8: Run snapshot GREEN and commit Task 1.**

  Run `python -m pytest tests/python/test_official_transport.py tests/python/test_official_source.py -k 'transport or snapshot' -q`, then commit with `feat: add bounded official snapshot discovery`.

### Task 2: Strict index, config, header, and expert plan

**Files:**
- Modify: `converter/k3x_converter/safetensors_reader.py`
- Modify: `converter/k3x_converter/official_source.py`
- Modify: `tests/python/test_safetensors_integrity.py`
- Modify: `tests/python/test_official_source.py`

**Interfaces:**
- Produces: `TensorMetadata(name: str, dtype: str, shape: tuple[int, ...], offset: int, length: int)`.
- Produces: `parse_safetensors_header(header: bytes, *, data_start: int, file_size: int) -> dict[str, TensorMetadata]`.
- Produces: `load_official_index(snapshot, transport) -> OfficialIndex` and `load_official_config(snapshot, transport) -> OfficialConfig`.
- Produces: `inspect_official_shard_header(snapshot, shard, transport) -> Mapping[str, TensorMetadata]`.
- Produces: `plan_official_expert(index, metadata, *, layer_id: int, expert_id: int) -> ExpertPlan`.

- [ ] **Step 1: Write failing metadata-parser tests.**

  Build in-memory headers with U8 tensors and assert absolute offsets from `data_start`, exact shape-derived lengths, gap/overlap/trailing rejection, duplicate/non-standard JSON rejection, bounded extents, and identical metadata to `inspect_shard()` for an existing local fixture.

- [ ] **Step 2: Run the parser RED gate.**

  Run `python -m pytest tests/python/test_safetensors_integrity.py -k parse_safetensors_header -q` and confirm the import fails for the missing helper.

- [ ] **Step 3: Extract the shared parser minimally.**

  Move only existing header validation into `parse_safetensors_header`; keep the current error codes. Make `inspect_shard()` read the length/header, call the helper, and map metadata to `SourceTensor(path, ...)`. Do not weaken whole-file gap/trailing validation.

- [ ] **Step 4: Run parser GREEN and all source-reader regressions.**

  Run `python -m pytest tests/python/test_safetensors_integrity.py tests/python/test_source_manifest_integrity.py tests/python/test_converter_resume.py -q`.

- [ ] **Step 5: Write failing index/config tests.**

  Assert index size and SHA-256 match the API LFS record, exactly 96 declared shard names cover every `weight_map` value, `metadata.total_size` is a non-boolean integer, unsafe paths and malformed keys fail closed, and `config.json` recomputes the API Git blob SHA-1 before validating all released dimensions and SiTU/scaling constants. Assert config failure happens before any range request.

- [ ] **Step 6: Run index/config RED.**

  Run `python -m pytest tests/python/test_official_source.py -k 'index or config' -q` and confirm the new functions are missing.

- [ ] **Step 7: Implement index/config validation.**

  Stream through the injected transport ceilings, use the same strict JSON decoder, recompute index SHA-256 and `sha1(b"blob " + str(len(body)).encode("ascii") + b"\0" + body)` for config, and return immutable canonical records. Reject unknown top-level index keys rather than silently ignoring them.

- [ ] **Step 8: Write failing exact-range and expert-plan tests.**

  Assert the first 8-byte request and header request each require exact 206/`Content-Range`; then assert the six w1/w2/w3 U8 tensors map to gate/down/up, have exact official shapes and lengths, belong to one shard, form `[1_268_562_960, 1_286_110_224)`, and total 17,547,264 bytes. Cover wrong role/shape, missing tensor, mixed shard, noncontiguity, wrong full size, HTTP 200, and short body.

- [ ] **Step 9: Run expert-plan RED.**

  Run `python -m pytest tests/python/test_official_source.py -k 'range or expert_plan' -q` and confirm the planning path is absent.

- [ ] **Step 10: Implement range inspection and planning.**

  Parse `Content-Range` strictly, fetch eight bytes before the header, enforce the 100,000,000-byte header limit and declared full shard size, call the shared parser, and produce six immutable `PlannedTensor` records in official physical order with canonical K3X names. Production CLI callers must reject any layer/expert other than 1/0.

- [ ] **Step 11: Run Task 2 GREEN and commit.**

  Run `python -m pytest tests/python/test_safetensors_integrity.py tests/python/test_source_manifest_integrity.py tests/python/test_converter_resume.py tests/python/test_official_source.py -q`, then commit with `feat: plan exact official expert ranges`.

### Task 3: Atomic real-byte materialization and K3X round trip

**Files:**
- Modify: `converter/k3x_converter/official_source.py`
- Modify: `tests/python/test_official_source.py`
- Modify: `tests/python/test_storage_fixture.py`

**Interfaces:**
- Produces: `materialize_official_expert_slice(snapshot, config, plan, transport, output_dir: Path, *, chunk_bytes: int) -> MaterializationReport`.
- `MaterializationReport` exposes source directory, manifest path, content-addressed microshard path, K3X path, six tensor digests, payload range digest, microshard digest, K3X root digest, and maximum in-memory chunk.

- [ ] **Step 1: Write failing materialization tests.**

  Feed six deterministic tensor byte ranges through `FakeTransport`; assert exactly one 17,547,264-byte union payload request, bounded chunk writes, canonical gate/up/down local names, exact per-tensor hashes, `source_provenance.verification == "transport-pinned-range"`, content-addressed filename, and no `.partial` files after success.

- [ ] **Step 2: Run materialization RED.**

  Run `python -m pytest tests/python/test_official_source.py -k materialize -q` and confirm the function is missing.

- [ ] **Step 3: Implement atomic microshard publication.**

  Build the canonical safetensors header from the six planned tensors, stream the exact contiguous official payload into a unique sibling partial, split and hash each tensor while writing, `flush`/`fsync`, rename to `<sha256>.safetensors`, and atomically write `source-manifest.json`. Reverify an existing content-addressed object before reuse and remove invocation-owned partials on every failure.

- [ ] **Step 4: Run materialization GREEN.**

  Run `python -m pytest tests/python/test_official_source.py -k materialize -q`.

- [ ] **Step 5: Write failing round-trip and cleanup tests.**

  Invoke the existing `convert()` and assert `K3XReader.open()` succeeds, `OPTIONAL_STORAGE_FIXTURE` is present, three MXFP4 records have the exact released shapes, source bytes match planned digests, and the existing runtime guard rejects generation as `NON_EXECUTABLE_ARTIFACT`. Inject failures during payload read, microshard publish, manifest publish, and conversion; assert no newly created partial artifact survives and a prior valid content-addressed object remains readable.

- [ ] **Step 6: Run round-trip RED.**

  Run `python -m pytest tests/python/test_official_source.py -k 'round_trip or cleanup' -q` and confirm the orchestration is absent or incomplete.

- [ ] **Step 7: Implement conversion orchestration.**

  Call the unchanged converter only after local source hashes pass, verify the finalized K3X with `K3XReader`, record root/source identities, and preserve the existing non-executable optional-feature boundary. Do not add real-model graph execution to M26.

- [ ] **Step 8: Run Task 3 GREEN and commit.**

  Run `python -m pytest tests/python/test_official_source.py tests/python/test_storage_fixture.py tests/python/test_k3x_format.py tests/python/test_cpp_reader.py -q`, then commit with `feat: materialize bounded official expert slice`.

### Task 4: Dry-run CLI, live B-0027, and evidence verifier

**Files:**
- Create: `tools/discover_official_kimi_k3.py`
- Create: `tools/verify_official_discovery.py`
- Create: `tests/python/test_official_discovery_cli.py`
- Create: `results/b0027-official-range/summary.json`
- Create: `results/b0027-official-range/summary.csv`

**Interfaces:**
- CLI defaults to `--dry-run`; mutually exclusive `--materialize-expert` requires `--output-dir` outside tracked `results/`.
- Live network access requires `K3X_TEST_OFFICIAL_DISCOVERY=1` and the pinned production revision.
- Verifier accepts the summary JSON/CSV and rechecks schema, row parity, canonical digests, exact byte counts, zero forbidden metrics, and provenance labels.

- [ ] **Step 1: Write failing CLI tests.**

  Inject the fake transport into `main(argv, transport=...)`; assert default dry-run writes discovery JSON, performs zero payload requests, records `tensor_payload_bytes = 0`, rejects materialization without an output directory, rejects output below `results/`, and rejects floating or noncanonical layer/expert inputs.

- [ ] **Step 2: Run CLI RED.**

  Run `python -m pytest tests/python/test_official_discovery_cli.py -q` and confirm the tool module is missing.

- [ ] **Step 3: Implement CLI and canonical evidence writer.**

  Keep human progress on stderr and machine JSON on stdout/file. Write LF-only JSON/CSV through sibling partials and atomic rename. Fields include revision, upstream sizes/digests, request counts, metadata/header/payload bytes, maximum response bytes, local hashes, wall time, Reader validity, optional feature, and provenance. Do not emit token, GPU, NVMe, quality, or full-shard-verified fields.

- [ ] **Step 4: Run CLI GREEN.**

  Run `python -m pytest tests/python/test_official_discovery_cli.py -q`.

- [ ] **Step 5: Write and witness verifier RED/GREEN.**

  First add tests that mutate one JSON count, one CSV value, one digest, one provenance label, and insert a forbidden `decode_tok_s` field. Run them to fail for the absent verifier, implement the minimal verifier, then run `python -m pytest tests/python/test_official_discovery_cli.py -k verifier -q` to pass.

- [ ] **Step 6: Run dry-run against the official pinned snapshot.**

  Run with `PYTHONPATH=converter;reference` and an untracked output directory. Confirm resolved commit `9f62e4e9fffbd0a83ddd60e1c209d828994b3569`, index size 59,764,096, header length 818,696, planned payload 17,547,264, and zero downloaded tensor-payload bytes. Preserve only canonical metadata evidence.

- [ ] **Step 7: Run the sole live payload materialization.**

  Only after all prior tests pass, set `K3X_TEST_OFFICIAL_DISCOVERY=1` and run `--materialize-expert` once. Confirm the exact 17,547,264-byte range, six tensor digests, content-addressed local source, Reader-valid K3X, and `OPTIONAL_STORAGE_FIXTURE`. Do not retry automatically if any identity changes; investigate first.

- [ ] **Step 8: Verify B-0027 and commit.**

  Run `python -m tools.verify_official_discovery results/b0027-official-range/summary.json results/b0027-official-range/summary.csv`, prove `git status --short` contains no real byte artifact, then commit code/tests/canonical evidence with `bench: record bounded official expert conversion`.

### Task 5: Full verification, TITAN Ledger, review, and publication

**Files:**
- Modify: `README.md`
- Modify: `ARCHITECTURE.md`
- Modify: `DECISIONS.md`
- Modify: `BENCHMARKS.md`
- Modify: `PROJECT_STATE.md` last
- Modify: `checklist.md`
- Modify: `context-notes.md`

**Interfaces:**
- Produces a public M26 commit whose documentation distinguishes real bounded bytes from full-model execution and points M27 at the untracked artifact recipe rather than a machine-local path.

- [ ] **Step 1: Run focused Python verification.**

  Run `python -m pytest tests/python/test_official_transport.py tests/python/test_official_source.py tests/python/test_official_discovery_cli.py tests/python/test_safetensors_integrity.py tests/python/test_storage_fixture.py tests/python/test_k3x_format.py -q`.

- [ ] **Step 2: Run the full CPU matrix.**

  Run the repository's recorded CPU CTest and full pytest commands from `PROJECT_STATE.md`; record exact pass/skip counts from fresh output.

- [ ] **Step 3: Run Linux I/O, sanitizer, CUDA, and Compute Sanitizer gates.**

  Use the existing WSL2 ext4/liburing, ASan/UBSan, CUDA, and unchanged Compute Sanitizer commands. Do not claim native-Linux P44 Pro performance from WSL2.

- [ ] **Step 4: Synchronize documents from measured evidence.**

  Add the implemented M26 boundary to README/ARCHITECTURE, record the range-provenance decision in DECISIONS, record only B-0027 measured bytes/time/hash validity in BENCHMARKS, and update PROJECT_STATE last with the latest measured bottleneck and M27 next task. Keep all proposed TITAN components marked proposed.

- [ ] **Step 5: Self-review the complete diff.**

  Run `git diff --check`, search for secrets/signed URLs/local paths/real artifact extensions, verify no TODO/TBD placeholders, recompute every committed evidence digest, and inspect every changed line against the M26 design.

- [ ] **Step 6: Request one Critical/Important review and apply at most one fix batch.**

  Reviewer acceptance requires no trust-boundary bypass, no accidental payload in dry-run, no full-shard claim, no committed real bytes, exact raw/summary parity, and passing focused regressions.

- [ ] **Step 7: Reverify, commit ledger state, publish, and merge.**

  Re-run affected tests after fixes, commit documentation with `docs: complete bounded official discovery`, push the branch, open a public PR, wait for correctness and CodeQL, merge only when green, and verify post-merge `main`. Update `PROJECT_STATE.md` only with evidence actually observed.

## Self-review record

- Spec coverage: every design section maps to Tasks 1 through 5, including config Git blob binding, exact HTTP ranges, atomic cleanup, non-executable K3X, opt-in live access, and evidence exclusions.
- Placeholder scan: no implementation step delegates unspecified validation or uses TODO/TBD placeholders.
- Type consistency: transport, snapshot, metadata, plan, materialization, CLI, and verifier interfaces are introduced before their consumers and retain the same names throughout.
- Scope control: M26 stops at one bounded real expert storage artifact; real CUDA execution remains M27 and paid Cloud Run remains later explicit work.
