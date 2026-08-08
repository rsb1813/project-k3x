# K3X 변환 중단 뒤 검증된 extent만 재사용하는지 검증합니다.
from pathlib import Path

import pytest

from k3x_converter.format import K3XError
from k3x_converter.resume import read_resume_manifest
from k3x_converter.writer import convert


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
