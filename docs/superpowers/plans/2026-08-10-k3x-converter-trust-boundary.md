# K3X Converter Trust-Boundary Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make every external source-manifest, referenced safetensors shard, resume ledger, and partial-file reuse decision fail closed before bounded real Kimi K3 shard work begins.

**Architecture:** Add one focused source-manifest boundary that owns strict JSON parsing, path containment, and exact tensor-to-shard ownership. Keep safetensors structural validation in `safetensors_reader.py`, keep ledger schema validation in `resume.py`, and let `writer.py` orchestrate only already-validated objects plus committed-prefix recovery.

**Tech Stack:** Python 3.12, pathlib, json, hashlib, google-crc32c, safetensors test fixtures, pytest 9.1.1, existing K3X v1 converter and reader.

## Global Constraints

- Do not download Kimi K3 weights or provision cloud resources.
- Do not change the K3X v1 superblock, directory, tensor, layer, or expert record layout.
- Preserve streaming reads and `maximum_source_read_bytes <= chunk_bytes`.
- Treat manifest, shard headers, ledgers, and partial files as untrusted data.
- No malformed input may create or modify output, partial, or resume files.
- Keep `synthetic-k3-source-v1` and `k3-storage-slice-v1` as the only accepted source formats.
- Preserve D-028 shard/tensor SHA-256 and canonical extent-prefix checks.
- New Python source files begin with a one-line Korean role comment.
- Each production behavior requires a witnessed failing test before implementation.
- Update `PROJECT_STATE.md` after every other ledger document.

---

### Task 1: Strict source manifest and shard ownership

**Files:**
- Create: `converter/k3x_converter/source_manifest.py`
- Modify: `converter/k3x_converter/writer.py:62-159`
- Create: `tests/python/test_source_manifest_integrity.py`

**Interfaces:**
- Produces: `load_source_manifest(source: Path) -> dict[str, object]`.
- Produces: `inspect_manifest_tensors(source: Path, manifest: dict[str, object]) -> dict[str, SourceTensor]`.
- Consumes: `inspect_shard(path: Path) -> dict[str, SourceTensor]`.
- Preserves: `_load_plans(source: Path) -> tuple[dict, list[_TensorPlan]]` for writer callers.

- [x] **Step 1: Write path and ownership RED tests**

Create the test file with the required Korean header and real synthetic shards. Mutate only the copied manifest or copied shard.

```python
# 외부 source manifest의 경로와 tensor 소유권을 검증합니다.
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from safetensors.torch import load_file, save_file

from k3x_converter.format import K3XError
from k3x_converter.writer import convert


def _manifest(source: Path) -> dict[str, object]:
    return json.loads((source / "source-manifest.json").read_text(encoding="utf-8"))


def _write_manifest(source: Path, value: dict[str, object]) -> None:
    (source / "source-manifest.json").write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )


def test_converter_rejects_parent_traversal_shard(
    synthetic_source: Path, tmp_path: Path
) -> None:
    manifest = _manifest(synthetic_source)
    name, original = next(iter(manifest["weight_map"].items()))
    shutil.copy2(synthetic_source / original, synthetic_source.parent / "outside.safetensors")
    manifest["weight_map"][name] = "../outside.safetensors"
    _write_manifest(synthetic_source, manifest)

    with pytest.raises(K3XError, match="SOURCE_SHARD_PATH_ESCAPE"):
        convert(synthetic_source, tmp_path / "escape.k3x", chunk_bytes=257)


def test_converter_rejects_tensor_mapped_to_wrong_referenced_shard(
    synthetic_source: Path, tmp_path: Path
) -> None:
    manifest = _manifest(synthetic_source)
    name, original = next(iter(manifest["weight_map"].items()))
    other = next(value for value in set(manifest["weight_map"].values()) if value != original)
    manifest["weight_map"][name] = other
    _write_manifest(synthetic_source, manifest)

    with pytest.raises(K3XError, match="SOURCE_TENSOR_SHARD_MISMATCH"):
        convert(synthetic_source, tmp_path / "wrong-owner.k3x", chunk_bytes=257)
```

Add separate tests for an absolute path, duplicate tensor physically present in two referenced shards, undeclared tensor, duplicate manifest JSON key, non-dictionary `weight_map`, empty tensor/shard names, and a symlink resolving outside the source directory where the platform permits symlink creation.

- [x] **Step 2: Run Task 1 RED**

Run:

```powershell
rtk .\.venv\Scripts\python.exe -m pytest tests/python/test_source_manifest_integrity.py -q
```

Expected: traversal and ownership cases fail because current `_load_plans()` joins raw paths and merges shard dictionaries with `dict.update()`; schema cases leak non-`K3XError` exceptions or pass.

- [x] **Step 3: Implement the minimal source boundary**

Create `source_manifest.py` with this public shape.

```python
# 외부 source manifest와 참조 shard의 신뢰 경계를 검증합니다.
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath, PureWindowsPath

from .format import K3XError
from .safetensors_reader import SourceTensor, inspect_shard


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise K3XError("INVALID_SOURCE_MANIFEST")
        result[key] = value
    return result


def load_source_manifest(source: Path) -> dict[str, object]:
    try:
        manifest = json.loads(
            (source / "source-manifest.json").read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
        )
    except K3XError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise K3XError("INVALID_SOURCE_MANIFEST") from error
    if not isinstance(manifest, dict):
        raise K3XError("INVALID_SOURCE_MANIFEST")
    return manifest


def inspect_manifest_tensors(
    source: Path, manifest: dict[str, object]
) -> dict[str, SourceTensor]:
    weight_map = manifest["weight_map"]
    tensors: dict[str, SourceTensor] = {}
    for shard_name in sorted(set(weight_map.values())):
        shard_path = _resolve_source_shard(source, shard_name)
        for name, tensor in inspect_shard(shard_path).items():
            if name in tensors or weight_map.get(name) != shard_name:
                raise K3XError("SOURCE_TENSOR_SHARD_MISMATCH", name)
            tensors[name] = tensor
    if set(tensors) != set(weight_map):
        raise K3XError("SOURCE_TENSOR_SHARD_MISMATCH")
    return tensors
```

Implement `_resolve_source_shard()` so POSIX and Windows absolute forms, `..`, non-canonical separators, non-files, and resolved paths outside `source.resolve()` raise `SOURCE_SHARD_PATH_ESCAPE`. Validate source format, `config`, `packed_shapes`, and non-empty string `weight_map` entries before resolving any path. Change `_load_plans()` to call these two functions and remove the raw `json.loads`, raw path join, and `dict.update()` path.

- [x] **Step 4: Run Task 1 GREEN and regression tests**

Run:

```powershell
rtk .\.venv\Scripts\python.exe -m pytest tests/python/test_source_manifest_integrity.py tests/python/test_k3x_format.py tests/python/test_storage_fixture.py tests/python/test_converter_resume.py -q
```

Expected: all pass; malformed cases leave output, `.partial`, and `.resume.json` absent.

- [x] **Step 5: Commit Task 1**

```powershell
rtk git add converter/k3x_converter/source_manifest.py converter/k3x_converter/writer.py tests/python/test_source_manifest_integrity.py
rtk git commit -m "fix: bind tensors to contained source shards"
```

---

### Task 2: Strict safetensors metadata inspection

**Files:**
- Modify: `converter/k3x_converter/safetensors_reader.py`
- Create: `tests/python/test_safetensors_integrity.py`

**Interfaces:**
- Preserves: `inspect_shard(path: Path) -> dict[str, SourceTensor]`.
- Preserves: `iter_tensor_chunks(tensor: SourceTensor, chunk_bytes: int) -> Iterator[bytes]`.
- Adds no payload residency and no new dependency.

- [ ] **Step 1: Write safetensors RED tests**

Use a small raw shard helper so malformed headers are not normalized by the safetensors library.

```python
# safetensors header와 extent 구조를 엄격하게 검증합니다.
import json
import struct
from pathlib import Path

import pytest

from k3x_converter.format import K3XError
from k3x_converter.safetensors_reader import inspect_shard


def _write_raw(path: Path, header: bytes, payload: bytes) -> None:
    path.write_bytes(struct.pack("<Q", len(header)) + header + payload)


def test_rejects_f32_shape_extent_length_mismatch(tmp_path: Path) -> None:
    shard = tmp_path / "bad.safetensors"
    header = json.dumps(
        {"x": {"dtype": "F32", "shape": [2], "data_offsets": [0, 4]}},
        separators=(",", ":"),
    ).encode()
    _write_raw(shard, header, bytes(4))
    with pytest.raises(K3XError, match="INVALID_SOURCE_EXTENT"):
        inspect_shard(shard)
```

Add one-behavior tests for duplicate JSON tensor keys, non-object root, non-object tensor metadata, missing/extra metadata keys, empty/reserved tensor name, boolean/negative shape dimension, invalid offset types, a gap before/between/after tensors, overlap, and `U8` byte-count mismatch.

- [ ] **Step 2: Run Task 2 RED**

Run:

```powershell
rtk .\.venv\Scripts\python.exe -m pytest tests/python/test_safetensors_integrity.py -q
```

Expected: length, duplicate-key, negative-shape, and gap cases pass incorrectly or leak generic exceptions.

- [ ] **Step 3: Implement structural validation**

In `safetensors_reader.py`, parse with a duplicate-key rejecting hook that translates all syntax/schema failures to `INVALID_SOURCE_HEADER`. Require exact tensor metadata keys, string dtype, list shape, two-item offset list, and non-boolean integers. For `F32` and `U8`, compute expected bytes with checked multiplication and require exact agreement with `end - start`. Sort ranges and require exact contiguous coverage from `data_start` through `size`; overlap or holes raise `INVALID_SOURCE_EXTENT` except overlap retains `OVERLAPPING_SOURCE_EXTENT`.

Do not read payload bytes and do not add an arbitrary header-size cap.

- [ ] **Step 4: Run Task 2 GREEN and source regressions**

```powershell
rtk .\.venv\Scripts\python.exe -m pytest tests/python/test_safetensors_integrity.py tests/python/test_source_manifest_integrity.py tests/python/test_k3x_format.py tests/python/test_storage_fixture.py -q
```

Expected: all pass and valid synthetic/bounded shards remain accepted.

- [ ] **Step 5: Commit Task 2**

```powershell
rtk git add converter/k3x_converter/safetensors_reader.py tests/python/test_safetensors_integrity.py
rtk git commit -m "fix: validate safetensors metadata structure"
```

---

### Task 3: Strict resume-ledger schema

**Files:**
- Modify: `converter/k3x_converter/resume.py`
- Modify: `tests/python/test_converter_resume.py`

**Interfaces:**
- Preserves: `CompletedExtent` and `ResumeManifest` dataclasses.
- Changes: `read_resume_manifest(path: Path) -> ResumeManifest` now raises only `K3XError("INVALID_RESUME_MANIFEST")` for syntax/schema failures.
- Preserves: `write_resume_manifest(path: Path, manifest: ResumeManifest) -> None` canonical LF JSON publication.

- [ ] **Step 1: Write ledger schema RED tests**

Append parameterized corruptions after producing a one-extent interrupted conversion.

```python
@pytest.mark.parametrize(
    "mutate",
    (
        lambda value: value.pop("file_uuid"),
        lambda value: value.__setitem__("unexpected", 1),
        lambda value: value.__setitem__("file_uuid", value["file_uuid"].upper()),
        lambda value: value["completed"][0].__setitem__("offset", True),
        lambda value: value.__setitem__("completed", {}),
    ),
)
def test_resume_rejects_noncanonical_manifest_schema(
    synthetic_source: Path, tmp_path: Path, mutate
) -> None:
    output = tmp_path / "synthetic.k3x"
    convert(synthetic_source, output, chunk_bytes=257, stop_after_extents=1)
    resume = output.with_suffix(".k3x.resume.json")
    value = json.loads(resume.read_text(encoding="utf-8"))
    mutate(value)
    resume.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(K3XError, match="INVALID_RESUME_MANIFEST"):
        convert(synthetic_source, output, chunk_bytes=257)
```

Add raw duplicate-key and malformed-JSON cases. Assert the partial and ledger bytes are unchanged after every rejection.

- [ ] **Step 2: Run Task 3 RED**

```powershell
rtk .\.venv\Scripts\python.exe -m pytest tests/python/test_converter_resume.py -q
```

Expected: cases leak `KeyError`, `TypeError`, `ValueError`, or are accepted.

- [ ] **Step 3: Implement strict ledger parsing**

Validate the exact five top-level keys and exact four extent keys. Require `source_fingerprint` and `configuration_fingerprint` to match `[0-9a-f]{64}`, `file_uuid` to match `[0-9a-f]{32}`, `converter_version` to be a non-empty string, `completed` to be a list, extent IDs to match `[0-9a-f]{16}:(data|auxiliary)`, and offset/length/crc32c to be non-boolean integers in unsigned ranges. Catch JSON, Unicode, OS, type, key, and value errors and raise `K3XError("INVALID_RESUME_MANIFEST")` from the original error.

- [ ] **Step 4: Run Task 3 GREEN**

```powershell
rtk .\.venv\Scripts\python.exe -m pytest tests/python/test_converter_resume.py tests/python/test_k3x_format.py -q
```

Expected: all pass and the canonical writer output round-trips byte-identically.

- [ ] **Step 5: Commit Task 3**

```powershell
rtk git add converter/k3x_converter/resume.py tests/python/test_converter_resume.py
rtk git commit -m "fix: validate resume ledger schema"
```

---

### Task 4: Recover only the committed partial prefix

**Files:**
- Modify: `converter/k3x_converter/writer.py:364-403`
- Modify: `tests/python/test_converter_resume.py`

**Interfaces:**
- Adds private `_committed_partial_length(completed: Sequence[CompletedExtent]) -> int`.
- Preserves conversion and report public signatures.
- Guarantees truncation only after source identity, ledger schema, source extent CRCs, and partial extent CRCs pass.

- [ ] **Step 1: Write orphan-suffix RED tests**

```python
def test_resume_discards_uncommitted_partial_suffix(
    synthetic_source: Path, tmp_path: Path
) -> None:
    output = tmp_path / "resumed.k3x"
    clean = tmp_path / "clean.k3x"
    convert(synthetic_source, output, chunk_bytes=257, stop_after_extents=1)
    partial = output.with_suffix(".k3x.partial")
    with partial.open("ab") as stream:
        stream.write(b"uncommitted" * 701)
    convert(synthetic_source, output, chunk_bytes=257)
    convert(synthetic_source, clean, chunk_bytes=257)
    assert output.stat().st_size == clean.stat().st_size
```

Add a second test that corrupts one committed byte, appends a suffix, captures the entire partial/ledger bytes, expects `RESUME_EXTENT_CRC_MISMATCH`, and proves neither file changed.

- [ ] **Step 2: Run Task 4 RED**

```powershell
rtk .\.venv\Scripts\python.exe -m pytest tests/python/test_converter_resume.py -q
```

Expected: the valid orphan-suffix output is larger than clean output; committed corruption already fails and remains unchanged.

- [ ] **Step 3: Implement post-validation truncation**

After the existing loop verifies every committed partial CRC, compute the aligned end of the last completed extent or `SUPERBLOCK_BYTES` for an empty prefix. Reject a partial shorter than that boundary. If it is longer, open it `r+b`, `truncate(boundary)`, `flush()`, and `os.fsync()`. Seek subsequent new writes from that boundary. Do not truncate before all validation loops finish.

- [ ] **Step 4: Run Task 4 GREEN and full converter tests**

```powershell
rtk .\.venv\Scripts\python.exe -m pytest tests/python/test_converter_resume.py tests/python/test_source_manifest_integrity.py tests/python/test_safetensors_integrity.py tests/python/test_k3x_format.py tests/python/test_storage_fixture.py -q
```

Expected: all pass; clean and recovered output lengths match; corruption is fail-atomic.

- [ ] **Step 5: Commit Task 4**

```powershell
rtk git add converter/k3x_converter/writer.py tests/python/test_converter_resume.py
rtk git commit -m "fix: truncate uncommitted resume suffix"
```

---

### Task 5: Audit evidence, full verification, and public integration

**Files:**
- Create: `tools/audit_converter_integrity.py`
- Create: `tests/python/test_converter_integrity_audit.py`
- Create: `results/b0026-converter-integrity-wsl/summary.json`
- Create: `results/b0026-converter-integrity-wsl/summary.csv`
- Modify: `README.md`
- Modify: `ARCHITECTURE.md`
- Modify: `DECISIONS.md`
- Modify: `BENCHMARKS.md`
- Modify: `checklist.md`
- Modify: `context-notes.md`
- Modify last: `PROJECT_STATE.md`

**Interfaces:**
- Produces: a non-token `k3x-converter-integrity-audit-v1` summary for `fresh`, `resume-clean`, and `resume-orphan` synthetic conversions.
- Records: wall nanoseconds, maximum source read bytes, output bytes, reused extent count, artifact validity/root digest, artifact/runner SHA-256, environment, and exact scenario identity.
- Explicitly omits: decode tok/s, prefill tok/s, TTFT, Top-K, speculative acceptance, GPU metrics, physical NVMe, and quality claims.

- [ ] **Step 1: Write audit-runner RED tests**

Test exact three-scenario order, bounded `maximum_source_read_bytes`, equal final output lengths, nonzero reuse for resume cases, K3X reader validation, JSON/CSV parity, LF-only CSV, artifact/runner digests, and absence of token fields. The initial import must fail before the tool exists.

- [ ] **Step 2: Implement and smoke the audit runner**

Use the existing deterministic synthetic source generator and converter. Do not add performance gates. Measure wall time with `time.perf_counter_ns()` and current/peak RSS with `psutil.Process().memory_info().rss` sampling only if the runner can do so without a background monitor; otherwise record RSS as not measured rather than inventing a peak.

Run one canonical WSL2 audit and commit raw summaries. Treat timings as converter audit measurements, not throughput forecasts.

- [ ] **Step 3: Run the full local verification matrix**

```powershell
rtk wsl -d Ubuntu-24.04 -- bash -lc 'cd /mnt/c/Users/jolib/Documents/project-k3x/.worktrees/milestone-twenty-four-cuda-graph-cache && cmake --build build-cpu -j && ctest --test-dir build-cpu --output-on-failure && K3X_BUILD_DIR=build-cpu /home/jolib/.venvs/k3x-m1/bin/python -m pytest -q'
rtk wsl -d Ubuntu-24.04 -- bash -lc 'cd /mnt/c/Users/jolib/Documents/project-k3x/.worktrees/milestone-twenty-four-cuda-graph-cache && cmake --build build-liburing -j && ctest --test-dir build-liburing --output-on-failure && K3X_BUILD_DIR=build-liburing /home/jolib/.venvs/k3x-m1/bin/python -m pytest -q'
rtk wsl -d Ubuntu-24.04 -- bash -lc 'cd /mnt/c/Users/jolib/Documents/project-k3x/.worktrees/milestone-twenty-four-cuda-graph-cache && cmake --build build-asan -j && ctest --test-dir build-asan --output-on-failure'
rtk wsl -d Ubuntu-24.04 -- bash -lc 'cd /mnt/c/Users/jolib/Documents/project-k3x/.worktrees/milestone-twenty-four-cuda-graph-cache && cmake --build build-cuda -j && ctest --test-dir build-cuda --output-on-failure && K3X_BUILD_DIR=build-cuda /home/jolib/.venvs/k3x-m1/bin/python -m pytest -q'
```

Expected: every suite passes. Re-run the unchanged stable CUDA Graph hit Compute Sanitizer as a cross-boundary regression gate and report it separately from Python parser coverage.

- [ ] **Step 4: Synchronize the TITAN Ledger in protocol order**

Update `ARCHITECTURE.md`, D-051 in `DECISIONS.md`, B-0026 in `BENCHMARKS.md`, `README.md`, checklist, and context notes. Update `PROJECT_STATE.md` last. State that D-028 was already implemented, distinguish local integrity from publisher authenticity, and retain no-full-weight/no-cloud/no-tok/s caveats.

- [ ] **Step 5: Review, publish, and merge**

Run `rtk git diff --check`, committed-evidence verification, and a focused Critical/Important read-only review. Fix verified findings once, rerun affected gates, commit semantic units, push `codex/milestone-twenty-five-converter-integrity`, open a ready public PR, wait for correctness and CodeQL, rebase merge, and verify post-merge `main` gates. Reconcile public documentation if the PR number or integration head is not yet recorded.

---

## Plan self-review result

- Every design requirement maps to Task 1, 2, 3, or 4; Task 5 supplies measured audit evidence and continuity publication.
- Public signatures remain stable except stricter documented failures for malformed external data.
- Source ownership, header structure, ledger schema, and suffix recovery have independent RED/GREEN commits.
- No step downloads weights, provisions cloud resources, changes K3X v1, or claims publisher authenticity.
- No placeholder, undefined production interface, or unresolved naming mismatch remains.
