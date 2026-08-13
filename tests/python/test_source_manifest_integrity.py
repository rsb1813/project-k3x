# 외부 source manifest의 경로와 tensor 소유권을 검증합니다.
from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from k3x_converter.format import K3XError
from k3x_converter.source_manifest import inspect_manifest_tensors, load_source_manifest
from k3x_converter.writer import convert


def _manifest(source: Path) -> dict[str, object]:
    return json.loads((source / "source-manifest.json").read_text(encoding="utf-8"))


def _write_manifest(source: Path, value: dict[str, object]) -> None:
    (source / "source-manifest.json").write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")), encoding="utf-8"
    )


def _assert_rejected_without_output(source: Path, output: Path, code: str) -> None:
    with pytest.raises(K3XError, match=code):
        convert(source, output, chunk_bytes=257)
    assert not output.exists()
    assert not output.with_suffix(output.suffix + ".partial").exists()
    assert not output.with_suffix(output.suffix + ".resume.json").exists()


def _weight_map(manifest: dict[str, object]) -> dict[str, str]:
    return manifest["weight_map"]  # type: ignore[return-value]


def test_local_alias_manifest_selects_only_declared_physical_tensor(tmp_path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    shard = source / "official.safetensors"
    save_file(
        {
            "official.weight": torch.tensor([1.0], dtype=torch.float32),
            "unselected.weight": torch.tensor([2.0], dtype=torch.float32),
        },
        shard,
    )
    manifest = {
        "format": "k3-local-shard-v1",
        "config": {},
        "packed_shapes": {},
        "weight_map": {"model.weight": shard.name},
        "source_names": {"model.weight": "official.weight"},
    }
    _write_manifest(source, manifest)

    loaded = load_source_manifest(source)
    tensors = inspect_manifest_tensors(source, loaded)

    assert set(tensors) == {"model.weight"}
    assert tensors["model.weight"].name == "model.weight"
    assert tensors["model.weight"].length == 4


def test_converter_rejects_parent_traversal_shard(
    synthetic_source: Path, tmp_path: Path
) -> None:
    manifest = _manifest(synthetic_source)
    weight_map = _weight_map(manifest)
    name, original = next(iter(weight_map.items()))
    shutil.copy2(synthetic_source / original, synthetic_source.parent / "outside.safetensors")
    weight_map[name] = "../outside.safetensors"
    _write_manifest(synthetic_source, manifest)

    _assert_rejected_without_output(
        synthetic_source, tmp_path / "parent-traversal.k3x", "SOURCE_SHARD_PATH_ESCAPE"
    )


@pytest.mark.parametrize("shard_name", ("/tmp/outside.safetensors", r"C:\outside.safetensors"))
def test_converter_rejects_absolute_shard_path(
    synthetic_source: Path, tmp_path: Path, shard_name: str
) -> None:
    manifest = _manifest(synthetic_source)
    weight_map = _weight_map(manifest)
    name = next(iter(weight_map))
    weight_map[name] = shard_name
    _write_manifest(synthetic_source, manifest)

    _assert_rejected_without_output(
        synthetic_source, tmp_path / "absolute-path.k3x", "SOURCE_SHARD_PATH_ESCAPE"
    )


def test_converter_rejects_tensor_mapped_to_wrong_referenced_shard(
    synthetic_source: Path, tmp_path: Path
) -> None:
    manifest = _manifest(synthetic_source)
    weight_map = _weight_map(manifest)
    name, original = next(iter(weight_map.items()))
    other = next(value for value in set(weight_map.values()) if value != original)
    weight_map[name] = other
    _write_manifest(synthetic_source, manifest)

    _assert_rejected_without_output(
        synthetic_source, tmp_path / "wrong-owner.k3x", "SOURCE_TENSOR_SHARD_MISMATCH"
    )


def test_converter_rejects_duplicate_tensor_in_referenced_shards(
    synthetic_source: Path, tmp_path: Path
) -> None:
    manifest = _manifest(synthetic_source)
    weight_map = _weight_map(manifest)
    name, original = next(iter(weight_map.items()))
    duplicate = "duplicate.safetensors"
    shutil.copy2(synthetic_source / original, synthetic_source / duplicate)
    weight_map[name] = duplicate
    _write_manifest(synthetic_source, manifest)

    _assert_rejected_without_output(
        synthetic_source, tmp_path / "duplicate-physical.k3x", "SOURCE_TENSOR_SHARD_MISMATCH"
    )


def test_converter_rejects_undeclared_physical_tensor(
    synthetic_source: Path, tmp_path: Path
) -> None:
    manifest = _manifest(synthetic_source)
    weight_map = _weight_map(manifest)
    del weight_map[next(iter(weight_map))]
    _write_manifest(synthetic_source, manifest)

    _assert_rejected_without_output(
        synthetic_source, tmp_path / "undeclared-tensor.k3x", "SOURCE_TENSOR_SHARD_MISMATCH"
    )


def test_converter_rejects_duplicate_manifest_json_key(
    synthetic_source: Path, tmp_path: Path
) -> None:
    manifest_path = synthetic_source / "source-manifest.json"
    raw = manifest_path.read_text(encoding="utf-8")
    manifest_path.write_text(
        raw.replace(
            '"format":"synthetic-k3-source-v1"',
            '"format":"synthetic-k3-source-v1","format":"synthetic-k3-source-v1"',
            1,
        ),
        encoding="utf-8",
    )

    _assert_rejected_without_output(
        synthetic_source, tmp_path / "duplicate-key.k3x", "INVALID_SOURCE_MANIFEST"
    )


@pytest.mark.parametrize("constant", ("NaN", "Infinity", "-Infinity"))
def test_converter_rejects_non_standard_json_constant(
    synthetic_source: Path, tmp_path: Path, constant: str
) -> None:
    manifest_path = synthetic_source / "source-manifest.json"
    raw = manifest_path.read_text(encoding="utf-8")
    manifest_path.write_text(
        raw.replace('"config":{', f'"config":{{"unused":{constant},', 1),
        encoding="utf-8",
    )

    _assert_rejected_without_output(
        synthetic_source, tmp_path / f"{constant}.k3x", "INVALID_SOURCE_MANIFEST"
    )


def test_converter_rejects_non_dictionary_weight_map(
    synthetic_source: Path, tmp_path: Path
) -> None:
    manifest = _manifest(synthetic_source)
    manifest["weight_map"] = []
    _write_manifest(synthetic_source, manifest)

    _assert_rejected_without_output(
        synthetic_source, tmp_path / "invalid-weight-map.k3x", "INVALID_SOURCE_MANIFEST"
    )


@pytest.mark.parametrize("field", ("tensor", "shard"))
def test_converter_rejects_empty_tensor_or_shard_name(
    synthetic_source: Path, tmp_path: Path, field: str
) -> None:
    manifest = _manifest(synthetic_source)
    weight_map = _weight_map(manifest)
    name, shard = next(iter(weight_map.items()))
    if field == "tensor":
        del weight_map[name]
        weight_map[""] = shard
    else:
        weight_map[name] = ""
    _write_manifest(synthetic_source, manifest)

    _assert_rejected_without_output(
        synthetic_source, tmp_path / f"empty-{field}.k3x", "INVALID_SOURCE_MANIFEST"
    )


def test_converter_rejects_symlink_to_shard_outside_source(
    synthetic_source: Path, tmp_path: Path
) -> None:
    manifest = _manifest(synthetic_source)
    weight_map = _weight_map(manifest)
    name, original = next(iter(weight_map.items()))
    outside = synthetic_source.parent / "outside.safetensors"
    shutil.copy2(synthetic_source / original, outside)
    link = synthetic_source / "outside-link.safetensors"
    try:
        link.symlink_to(outside)
    except OSError as error:
        pytest.skip(f"symlink creation is unavailable: {error}")
    for mapped_name, shard in weight_map.items():
        if shard == original:
            weight_map[mapped_name] = link.name
    _write_manifest(synthetic_source, manifest)

    _assert_rejected_without_output(
        synthetic_source, tmp_path / "symlink-escape.k3x", "SOURCE_SHARD_PATH_ESCAPE"
    )


def test_converter_accepts_normalized_relative_shard_subdirectory(
    synthetic_source: Path, tmp_path: Path
) -> None:
    manifest = _manifest(synthetic_source)
    weight_map = _weight_map(manifest)
    original = next(iter(weight_map.values()))
    nested = synthetic_source / "nested" / "referenced.safetensors"
    nested.parent.mkdir()
    (synthetic_source / original).replace(nested)
    for name, shard in weight_map.items():
        if shard == original:
            weight_map[name] = "nested/referenced.safetensors"
    _write_manifest(synthetic_source, manifest)

    report = convert(synthetic_source, tmp_path / "nested.k3x", chunk_bytes=257)

    assert report.completed is True


def test_source_manifest_accepts_official_moe_fixture_format(
    synthetic_source: Path,
) -> None:
    manifest = _manifest(synthetic_source)
    manifest["format"] = "k3-official-moe-slice-v1"
    manifest["artifact_kind"] = "official_moe_fixture"
    _write_manifest(synthetic_source, manifest)

    loaded = load_source_manifest(synthetic_source)

    assert loaded["format"] == "k3-official-moe-slice-v1"
    assert loaded["artifact_kind"] == "official_moe_fixture"
