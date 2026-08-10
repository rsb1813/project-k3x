# safetensors header와 extent 구조를 엄격하게 검증합니다.
from __future__ import annotations

import json
import struct
from pathlib import Path

import pytest

from k3x_converter.format import K3XError
from k3x_converter.safetensors_reader import inspect_shard, parse_safetensors_header


def _write_raw(path: Path, header: bytes, payload: bytes = b"") -> None:
    path.write_bytes(struct.pack("<Q", len(header)) + header + payload)


def _header(value: object) -> bytes:
    return json.dumps(value, separators=(",", ":")).encode("utf-8")


def _tensor(
    dtype: object = "F32",
    shape: object = None,
    offsets: object = None,
) -> dict[str, object]:
    return {
        "dtype": dtype,
        "shape": [1] if shape is None else shape,
        "data_offsets": [0, 4] if offsets is None else offsets,
    }


@pytest.mark.parametrize(
    ("dtype", "shape", "offsets", "payload"),
    (("F32", [2], [0, 4], bytes(4)), ("U8", [2], [0, 1], bytes(1))),
)
def test_rejects_known_dtype_shape_extent_length_mismatch(
    tmp_path: Path, dtype: str, shape: list[int], offsets: list[int], payload: bytes
) -> None:
    shard = tmp_path / "bad-length.safetensors"
    _write_raw(shard, _header({"x": _tensor(dtype, shape, offsets)}), payload)

    with pytest.raises(K3XError, match="INVALID_SOURCE_EXTENT"):
        inspect_shard(shard)


def test_rejects_duplicate_tensor_json_key(tmp_path: Path) -> None:
    shard = tmp_path / "duplicate-key.safetensors"
    tensor = _header(_tensor()).decode("utf-8")
    _write_raw(shard, f'{{"x":{tensor},"x":{tensor}}}'.encode("utf-8"), bytes(4))

    with pytest.raises(K3XError, match="INVALID_SOURCE_HEADER"):
        inspect_shard(shard)


def test_rejects_non_object_header_root(tmp_path: Path) -> None:
    shard = tmp_path / "list-root.safetensors"
    _write_raw(shard, b"[]")

    with pytest.raises(K3XError, match="INVALID_SOURCE_HEADER"):
        inspect_shard(shard)


def test_rejects_non_object_tensor_metadata(tmp_path: Path) -> None:
    shard = tmp_path / "list-metadata.safetensors"
    _write_raw(shard, _header({"x": []}))

    with pytest.raises(K3XError, match="INVALID_SOURCE_HEADER"):
        inspect_shard(shard)


@pytest.mark.parametrize(
    "metadata",
    (
        {"dtype": "F32", "shape": [1]},
        {"dtype": "F32", "shape": [1], "data_offsets": [0, 4], "extra": 1},
    ),
)
def test_rejects_tensor_metadata_with_missing_or_extra_keys(
    tmp_path: Path, metadata: dict[str, object]
) -> None:
    shard = tmp_path / "metadata-keys.safetensors"
    _write_raw(shard, _header({"x": metadata}), bytes(4))

    with pytest.raises(K3XError, match="INVALID_SOURCE_HEADER"):
        inspect_shard(shard)


@pytest.mark.parametrize("name", ("", "__reserved"))
def test_rejects_empty_or_reserved_tensor_name(tmp_path: Path, name: str) -> None:
    shard = tmp_path / "reserved-name.safetensors"
    _write_raw(shard, _header({name: _tensor()}), bytes(4))

    with pytest.raises(K3XError, match="INVALID_SOURCE_HEADER"):
        inspect_shard(shard)


@pytest.mark.parametrize("metadata", ([], {"key": 1}))
def test_rejects_malformed_metadata_object(tmp_path: Path, metadata: object) -> None:
    shard = tmp_path / "metadata.safetensors"
    _write_raw(shard, _header({"__metadata__": metadata}))

    with pytest.raises(K3XError, match="INVALID_SOURCE_HEADER"):
        inspect_shard(shard)


@pytest.mark.parametrize("shape", ([True], [-1]))
def test_rejects_boolean_or_negative_shape_dimension(
    tmp_path: Path, shape: list[object]
) -> None:
    shard = tmp_path / "invalid-shape.safetensors"
    _write_raw(shard, _header({"x": _tensor(shape=shape, offsets=[0, 0])}))

    with pytest.raises(K3XError, match="INVALID_SOURCE_HEADER"):
        inspect_shard(shard)


@pytest.mark.parametrize("offsets", ([0], [0, True], ["0", 4]))
def test_rejects_malformed_offset_structure(
    tmp_path: Path, offsets: list[object]
) -> None:
    shard = tmp_path / "invalid-offsets.safetensors"
    _write_raw(shard, _header({"x": _tensor(offsets=offsets)}), bytes(4))

    with pytest.raises(K3XError, match="INVALID_SOURCE_HEADER"):
        inspect_shard(shard)


@pytest.mark.parametrize("offsets", ([-1, 0], [4, 0], [0, 8]))
def test_rejects_negative_reversed_or_out_of_file_offsets(
    tmp_path: Path, offsets: list[int]
) -> None:
    shard = tmp_path / "invalid-range.safetensors"
    _write_raw(shard, _header({"x": _tensor(offsets=offsets)}), bytes(4))

    with pytest.raises(K3XError, match="INVALID_SOURCE_EXTENT"):
        inspect_shard(shard)


@pytest.mark.parametrize(
    ("tensors", "payload"),
    (
        ({"x": _tensor(offsets=[4, 8])}, bytes(8)),
        (
            {"a": _tensor(offsets=[0, 4]), "b": _tensor(offsets=[8, 12])},
            bytes(12),
        ),
        ({"x": _tensor(offsets=[0, 4])}, bytes(8)),
    ),
)
def test_rejects_gap_before_between_or_after_tensor_ranges(
    tmp_path: Path, tensors: dict[str, object], payload: bytes
) -> None:
    shard = tmp_path / "gap.safetensors"
    _write_raw(shard, _header(tensors), payload)

    with pytest.raises(K3XError, match="INVALID_SOURCE_EXTENT"):
        inspect_shard(shard)


def test_rejects_real_tensor_range_overlap(tmp_path: Path) -> None:
    shard = tmp_path / "overlap.safetensors"
    tensors = {"a": _tensor(offsets=[0, 4]), "b": _tensor(offsets=[2, 6])}
    _write_raw(shard, _header(tensors), bytes(6))

    with pytest.raises(K3XError, match="OVERLAPPING_SOURCE_EXTENT"):
        inspect_shard(shard)


def test_rejects_oversized_declared_header_without_payload(tmp_path: Path) -> None:
    shard = tmp_path / "oversized-header.safetensors"
    shard.write_bytes(struct.pack("<Q", 100_000_001))

    with pytest.raises(K3XError, match="INVALID_SOURCE_HEADER"):
        inspect_shard(shard)


def test_accepts_leading_whitespace_and_trailing_header_padding(tmp_path: Path) -> None:
    shard = tmp_path / "padded-header.safetensors"
    _write_raw(shard, b" \n" + _header({"x": _tensor()}) + b"   ", bytes(4))

    assert inspect_shard(shard)["x"].length == 4


def test_accepts_scalar_tensor(tmp_path: Path) -> None:
    shard = tmp_path / "scalar.safetensors"
    _write_raw(shard, _header({"x": _tensor(shape=[], offsets=[0, 4])}), bytes(4))

    assert inspect_shard(shard)["x"].shape == ()


def test_accepts_empty_dimension_tensor(tmp_path: Path) -> None:
    shard = tmp_path / "empty-dimension.safetensors"
    _write_raw(shard, _header({"x": _tensor(shape=[0, 2], offsets=[0, 0])}))

    assert inspect_shard(shard)["x"].length == 0


def test_preserves_unsupported_string_dtype_for_writer_validation(tmp_path: Path) -> None:
    shard = tmp_path / "unsupported-dtype.safetensors"
    _write_raw(shard, _header({"x": _tensor("I16", [1], [0, 0])}))

    assert inspect_shard(shard)["x"].dtype == "I16"


def test_parse_safetensors_header_returns_absolute_metadata_offsets() -> None:
    header = _header(
        {
            "a": _tensor("U8", [2], [0, 2]),
            "b": _tensor("U8", [4], [2, 6]),
        }
    )

    tensors = parse_safetensors_header(header, data_start=100, file_size=106)

    assert tensors["a"].name == "a"
    assert tensors["a"].shape == (2,)
    assert tensors["a"].offset == 100
    assert tensors["a"].length == 2
    assert tensors["b"].offset == 102
    assert tensors["b"].length == 4


def test_parse_safetensors_header_matches_local_shard_inspection(tmp_path: Path) -> None:
    shard = tmp_path / "equivalent.safetensors"
    header = _header({"x": _tensor("U8", [4], [0, 4])})
    _write_raw(shard, header, b"data")
    data_start = 8 + len(header)

    metadata = parse_safetensors_header(
        header, data_start=data_start, file_size=shard.stat().st_size
    )["x"]
    local = inspect_shard(shard)["x"]

    assert (metadata.name, metadata.dtype, metadata.shape) == (
        local.name,
        local.dtype,
        local.shape,
    )
    assert (metadata.offset, metadata.length) == (local.offset, local.length)


@pytest.mark.parametrize(
    ("header", "file_size", "code"),
    [
        (_header({"x": _tensor("U8", [1], [1, 2])}), 102, "INVALID_SOURCE_EXTENT"),
        (_header({"x": _tensor("U8", [1], [0, 1])}), 102, "INVALID_SOURCE_EXTENT"),
        (
            b'{"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]},'
            b'"x":{"dtype":"U8","shape":[1],"data_offsets":[0,1]}}',
            101,
            "INVALID_SOURCE_HEADER",
        ),
    ],
)
def test_parse_safetensors_header_rejects_gap_trailing_or_duplicate(
    header: bytes, file_size: int, code: str
) -> None:
    with pytest.raises(K3XError, match=code):
        parse_safetensors_header(header, data_start=100, file_size=file_size)
