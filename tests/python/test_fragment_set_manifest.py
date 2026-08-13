# Python 공식 실행 경로가 봉인된 K3X fragment set만 여는지 검증합니다.
from pathlib import Path

import pytest

import k3x_converter.fragment_tensor_store as fragment_tensor_store
from k3x_converter.format import K3XError
from k3x_converter.fragment_set import (
    read_fragment_set_manifest,
    write_fragment_set_manifest,
)
from tools.official_k3x_source import (
    k3x_set_identity,
    open_official_fragment,
    require_k3x_state_identity,
)
from k3x_converter.writer import convert


def test_fragment_set_manifest_round_trips(
    synthetic_source: Path, tmp_path: Path, monkeypatch
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
    open_kwargs = []
    reader_open = fragment_tensor_store.K3XReader.open

    def capture(path, **kwargs):
        open_kwargs.append(kwargs)
        return reader_open(path, **kwargs)

    monkeypatch.setattr(fragment_tensor_store.K3XReader, "open", capture)
    store = open_official_fragment(
        manifest, "model-00001-of-000096.safetensors"
    )
    assert store.tensors
    assert open_kwargs == [{"verify_payload": False, "verify_root": False}]
    assert k3x_set_identity(manifest) == digest
    require_k3x_state_identity({"k3x_set_manifest_sha256": digest}, digest)
    with pytest.raises(K3XError, match="K3X_STATE_SET_MISMATCH"):
        require_k3x_state_identity({}, digest)
