# K3X 변환 중단 뒤 검증된 extent만 재사용하는지 검증합니다.
import hashlib
import json
import shutil
from pathlib import Path

import google_crc32c
import pytest

import k3x_converter.writer as writer
from k3x_converter.format import SUPERBLOCK_BYTES, K3XError
from k3x_converter.resume import read_resume_manifest
from k3x_converter.writer import convert
from k3x_ref.storage_fixture import write_bounded_expert_source


def test_conversion_batches_durable_resume_checkpoints(
    synthetic_source: Path, tmp_path: Path, monkeypatch
) -> None:
    checkpoints = []
    write_checkpoint = writer.write_resume_manifest

    def capture(path, manifest):
        checkpoints.append(len(manifest.completed))
        write_checkpoint(path, manifest)

    monkeypatch.setattr(writer, "write_resume_manifest", capture)
    writer.convert(synthetic_source, tmp_path / "batched.k3x", chunk_bytes=257)

    assert checkpoints[0] == 0
    assert checkpoints[-1] > 1
    assert len(checkpoints) == 3


def _interrupted_output(synthetic_source: Path, tmp_path: Path) -> Path:
    output = tmp_path / "interrupted.k3x"
    report = convert(synthetic_source, output, chunk_bytes=257, stop_after_extents=1)
    assert report.completed is False
    return output


def _assert_schema_rejected_without_mutation(source: Path, output: Path) -> None:
    partial = output.with_suffix(".k3x.partial")
    resume = output.with_suffix(".k3x.resume.json")
    partial_before = partial.read_bytes()
    resume_before = resume.read_bytes()

    with pytest.raises(K3XError, match="INVALID_RESUME_MANIFEST"):
        convert(source, output, chunk_bytes=257)

    assert partial.read_bytes() == partial_before
    assert resume.read_bytes() == resume_before


def _copy_interrupted_state(source: Path, target: Path) -> None:
    for suffix in (".partial", ".resume.json"):
        shutil.copy2(source.with_suffix(source.suffix + suffix), target.with_suffix(target.suffix + suffix))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize(
    "mutate",
    (
        lambda value: value.pop("file_uuid"),
        lambda value: value.__setitem__("unexpected", 1),
        lambda value: value.__setitem__("source_fingerprint", "0" * 63),
        lambda value: value.__setitem__("configuration_fingerprint", "G" * 64),
        lambda value: value.__setitem__("file_uuid", value["file_uuid"].upper()),
        lambda value: value.__setitem__("converter_version", ""),
        lambda value: value.__setitem__("completed", {}),
        lambda value: value.__setitem__("completed", [[]]),
        lambda value: value["completed"][0].pop("crc32c"),
        lambda value: value["completed"][0].__setitem__("unexpected", 1),
        lambda value: value["completed"][0].__setitem__("extent_id", "bad"),
        lambda value: value["completed"][0].__setitem__("offset", True),
        lambda value: value["completed"][0].__setitem__("length", -1),
        lambda value: value["completed"][0].__setitem__("length", 2**64),
        lambda value: value["completed"][0].__setitem__("crc32c", True),
        lambda value: value["completed"][0].__setitem__("crc32c", 2**32),
    ),
)
def test_resume_rejects_invalid_ledger_schema_without_mutating_files(
    synthetic_source: Path, tmp_path: Path, mutate
) -> None:
    output = _interrupted_output(synthetic_source, tmp_path)
    resume = output.with_suffix(".k3x.resume.json")
    ledger = json.loads(resume.read_text(encoding="utf-8"))
    mutate(ledger)
    resume.write_text(json.dumps(ledger), encoding="utf-8")

    _assert_schema_rejected_without_mutation(synthetic_source, output)


@pytest.mark.parametrize("constant", ("NaN", "Infinity", "-Infinity"))
def test_resume_rejects_non_standard_json_constant_without_mutating_files(
    synthetic_source: Path, tmp_path: Path, constant: str
) -> None:
    output = _interrupted_output(synthetic_source, tmp_path)
    resume = output.with_suffix(".k3x.resume.json")
    raw = resume.read_text(encoding="utf-8")
    resume.write_text(raw.replace("{", f'{{"unexpected":{constant},', 1), encoding="utf-8")

    _assert_schema_rejected_without_mutation(synthetic_source, output)


@pytest.mark.parametrize("raw", ("{", "[]"))
def test_resume_rejects_malformed_or_non_object_json_without_mutating_files(
    synthetic_source: Path, tmp_path: Path, raw: str
) -> None:
    output = _interrupted_output(synthetic_source, tmp_path)
    resume = output.with_suffix(".k3x.resume.json")
    resume.write_text(raw, encoding="utf-8")

    _assert_schema_rejected_without_mutation(synthetic_source, output)


def test_resume_rejects_duplicate_json_key_without_mutating_files(
    synthetic_source: Path, tmp_path: Path
) -> None:
    output = _interrupted_output(synthetic_source, tmp_path)
    resume = output.with_suffix(".k3x.resume.json")
    raw = resume.read_text(encoding="utf-8")
    duplicate = '"file_uuid":"' + "0" * 32 + '","file_uuid":'
    resume.write_text(raw.replace('"file_uuid":', duplicate, 1), encoding="utf-8")

    _assert_schema_rejected_without_mutation(synthetic_source, output)


def test_conversion_resumes_without_rewriting_completed_extents(
    synthetic_source: Path, tmp_path: Path
) -> None:
    output = tmp_path / "synthetic.k3x"
    first = convert(synthetic_source, output, chunk_bytes=257, stop_after_extents=3)
    assert first.completed is False
    before = read_resume_manifest(output.with_suffix(".k3x.resume.json"))

    second = convert(synthetic_source, output, chunk_bytes=257)
    assert second.completed is True
    assert second.reused_extent_ids == tuple(item.extent_id for item in before.completed)
    assert second.maximum_source_read_bytes <= 257
    assert output.exists()
    assert not output.with_suffix(".k3x.partial").exists()


def test_resume_truncates_orphan_suffix_to_exact_last_extent_end(
    synthetic_source: Path, tmp_path: Path
) -> None:
    clean = tmp_path / "clean.k3x"
    orphan = tmp_path / "orphan.k3x"
    convert(synthetic_source, clean, chunk_bytes=257, stop_after_extents=2)
    ledger = read_resume_manifest(clean.with_suffix(".k3x.resume.json"))
    boundary = ledger.completed[-1].offset + ledger.completed[-1].length
    assert boundary % 4096 != 0
    assert clean.with_suffix(".k3x.partial").stat().st_size == boundary
    _copy_interrupted_state(clean, orphan)
    with orphan.with_suffix(".k3x.partial").open("ab") as stream:
        stream.write(b"orphan-suffix" * 701)

    convert(synthetic_source, clean, chunk_bytes=257)
    convert(synthetic_source, orphan, chunk_bytes=257)

    assert clean.read_bytes() == orphan.read_bytes()
    assert _sha256(clean) == _sha256(orphan)


def test_resume_rejects_committed_corruption_with_suffix_without_mutation(
    synthetic_source: Path, tmp_path: Path
) -> None:
    output = tmp_path / "corrupt.k3x"
    convert(synthetic_source, output, chunk_bytes=257, stop_after_extents=2)
    partial = output.with_suffix(".k3x.partial")
    resume = output.with_suffix(".k3x.resume.json")
    item = read_resume_manifest(resume).completed[-1]
    with partial.open("r+b") as stream:
        stream.seek(item.offset)
        original = stream.read(1)
        stream.seek(item.offset)
        stream.write(bytes([original[0] ^ 1]))
        stream.seek(0, 2)
        stream.write(b"orphan-suffix" * 701)
    partial_before = partial.read_bytes()
    resume_before = resume.read_bytes()

    with pytest.raises(K3XError, match="RESUME_EXTENT_CRC_MISMATCH"):
        convert(synthetic_source, output, chunk_bytes=257)

    assert partial.read_bytes() == partial_before
    assert resume.read_bytes() == resume_before


def test_resume_truncates_empty_committed_prefix_before_replay(
    synthetic_source: Path, tmp_path: Path
) -> None:
    clean = tmp_path / "empty-clean.k3x"
    orphan = tmp_path / "empty-orphan.k3x"
    convert(synthetic_source, clean, chunk_bytes=257, stop_after_extents=2)
    resume = clean.with_suffix(".k3x.resume.json")
    ledger = json.loads(resume.read_text(encoding="utf-8"))
    ledger["completed"] = []
    resume.write_text(json.dumps(ledger), encoding="utf-8")
    assert clean.with_suffix(".k3x.partial").stat().st_size > SUPERBLOCK_BYTES
    _copy_interrupted_state(clean, orphan)
    with orphan.with_suffix(".k3x.partial").open("ab") as stream:
        stream.write(b"orphan-suffix" * 701)

    convert(synthetic_source, clean, chunk_bytes=257)
    convert(synthetic_source, orphan, chunk_bytes=257)

    assert clean.read_bytes() == orphan.read_bytes()
    assert _sha256(clean) == _sha256(orphan)


def test_resume_rejects_changed_source(
    synthetic_source: Path, tmp_path: Path
) -> None:
    output = tmp_path / "synthetic.k3x"
    convert(synthetic_source, output, chunk_bytes=257, stop_after_extents=1)
    shard = next(synthetic_source.glob("*.safetensors"))
    with shard.open("r+b") as stream:
        stream.seek(-1, 2)
        original = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([original[0] ^ 1]))
    with pytest.raises(K3XError, match="SOURCE_FINGERPRINT_MISMATCH"):
        convert(synthetic_source, output, chunk_bytes=257)


def test_resume_recovers_crash_after_final_rename(
    synthetic_source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    output = tmp_path / "synthetic.k3x"
    resume = output.with_suffix(".k3x.resume.json")
    original_unlink = Path.unlink

    def interrupt_ledger_cleanup(path: Path, *args, **kwargs) -> None:
        if path == resume:
            raise RuntimeError("simulated crash after final rename")
        original_unlink(path, *args, **kwargs)

    monkeypatch.setattr(Path, "unlink", interrupt_ledger_cleanup)
    with pytest.raises(RuntimeError, match="simulated crash"):
        convert(synthetic_source, output, chunk_bytes=257)
    assert output.exists()
    assert resume.exists()
    assert not output.with_suffix(".k3x.partial").exists()

    monkeypatch.setattr(Path, "unlink", original_unlink)
    recovered = convert(synthetic_source, output, chunk_bytes=257)
    assert recovered.completed is True
    assert not resume.exists()


def test_full_dimension_storage_fixture_resumes_exact_extents(
    tmp_path: Path,
) -> None:
    source = tmp_path / "bounded-source"
    write_bounded_expert_source(source, chunk_bytes=257 * 1024)
    output = tmp_path / "bounded.k3x"

    first = convert(
        source,
        output,
        chunk_bytes=193 * 1024,
        stop_after_extents=2,
    )
    before = read_resume_manifest(output.with_suffix(".k3x.resume.json"))
    second = convert(source, output, chunk_bytes=193 * 1024)

    assert first.completed is False
    assert len(before.completed) == 2
    assert second.completed is True
    assert second.reused_extent_ids == tuple(
        item.extent_id for item in before.completed
    )
    assert second.maximum_source_read_bytes <= 193 * 1024


def test_full_dimension_storage_fixture_resume_rejects_changed_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "bounded-source"
    report = write_bounded_expert_source(source, chunk_bytes=257 * 1024)
    output = tmp_path / "bounded.k3x"
    convert(source, output, chunk_bytes=193 * 1024, stop_after_extents=1)

    with report.shard_path.open("r+b") as stream:
        stream.seek(-1, 2)
        original = stream.read(1)
        stream.seek(-1, 2)
        stream.write(bytes([original[0] ^ 1]))

    with pytest.raises(K3XError, match="SOURCE_SHARD_SHA256_MISMATCH"):
        convert(source, output, chunk_bytes=193 * 1024)


@pytest.mark.parametrize("corruption", ("duplicate", "unknown", "zero", "offset"))
def test_storage_fixture_resume_rejects_noncanonical_extent_ledger(
    tmp_path: Path, corruption: str
) -> None:
    source = tmp_path / "bounded-source"
    write_bounded_expert_source(source, chunk_bytes=257 * 1024)
    output = tmp_path / "bounded.k3x"
    convert(source, output, chunk_bytes=193 * 1024, stop_after_extents=2)
    resume = output.with_suffix(".k3x.resume.json")
    ledger = json.loads(resume.read_text(encoding="utf-8"))

    if corruption == "duplicate":
        ledger["completed"][1]["extent_id"] = ledger["completed"][0]["extent_id"]
    elif corruption == "unknown":
        ledger["completed"][0]["extent_id"] = "0000000000000000:data"
    elif corruption == "zero":
        ledger["completed"][0]["length"] = 0
    else:
        ledger["completed"][0]["offset"] += 4096
    resume.write_text(json.dumps(ledger), encoding="utf-8")

    with pytest.raises(K3XError, match="INVALID_RESUME_EXTENT"):
        convert(source, output, chunk_bytes=193 * 1024)


def test_storage_fixture_resume_rejects_partial_bytes_not_matching_source(
    tmp_path: Path,
) -> None:
    source = tmp_path / "bounded-source"
    write_bounded_expert_source(source, chunk_bytes=257 * 1024)
    output = tmp_path / "bounded.k3x"
    convert(source, output, chunk_bytes=193 * 1024, stop_after_extents=1)
    resume = output.with_suffix(".k3x.resume.json")
    partial = output.with_suffix(".k3x.partial")
    ledger = json.loads(resume.read_text(encoding="utf-8"))
    item = ledger["completed"][0]

    with partial.open("r+b") as stream:
        stream.seek(item["offset"])
        original = stream.read(1)
        stream.seek(item["offset"])
        stream.write(bytes([original[0] ^ 1]))
        stream.seek(item["offset"])
        checksum = google_crc32c.Checksum()
        remaining = item["length"]
        while remaining:
            chunk = stream.read(min(193 * 1024, remaining))
            checksum.update(chunk)
            remaining -= len(chunk)
    item["crc32c"] = int.from_bytes(checksum.digest(), "big")
    resume.write_text(json.dumps(ledger), encoding="utf-8")

    with pytest.raises(K3XError, match="RESUME_SOURCE_EXTENT_MISMATCH"):
        convert(source, output, chunk_bytes=193 * 1024)
