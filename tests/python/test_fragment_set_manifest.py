# Python 공식 실행 경로가 봉인된 K3X fragment set만 여는지 검증합니다.
from pathlib import Path

from k3x_converter.fragment_set import (
    read_fragment_set_manifest,
    write_fragment_set_manifest,
)
from k3x_converter.writer import convert


def test_fragment_set_manifest_round_trips(
    synthetic_source: Path, tmp_path: Path
) -> None:
    fragment = tmp_path / "model-00001-of-000096.k3x"
    convert(synthetic_source, fragment, chunk_bytes=257)
    manifest = tmp_path / "model.k3xset"
    digest = write_fragment_set_manifest(
        manifest, [fragment], plan_sha256="a" * 64
    )

    parsed = read_fragment_set_manifest(manifest)

    assert parsed.plan_sha256 == "a" * 64
    assert parsed.fragments == (fragment,)
    assert parsed.record_sha256 == digest
