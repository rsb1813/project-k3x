# safetensors shard를 전체 적재 없이 제한된 chunk로 읽습니다.
from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from .format import K3XError

_MAX_HEADER_BYTES = 100_000_000
_METADATA_KEYS = {"dtype", "shape", "data_offsets"}
_KNOWN_DTYPE_BYTES = {"F32": 4, "U8": 1}


@dataclass(frozen=True)
class SourceTensor:
    path: Path
    name: str
    dtype: str
    shape: tuple[int, ...]
    offset: int
    length: int


@dataclass(frozen=True)
class TensorMetadata:
    name: str
    dtype: str
    shape: tuple[int, ...]
    offset: int
    length: int


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise K3XError("INVALID_SOURCE_HEADER")
        result[key] = value
    return result


def _reject_non_standard_constant(_: str) -> object:
    raise K3XError("INVALID_SOURCE_HEADER")


def _is_non_boolean_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _known_dtype_length(dtype: str, shape: list[int], data_bytes: int) -> int | None:
    item_bytes = _KNOWN_DTYPE_BYTES.get(dtype)
    if item_bytes is None:
        return None
    if 0 in shape:
        return 0
    length = item_bytes
    for dimension in shape:
        if length > data_bytes // dimension:
            raise K3XError("INVALID_SOURCE_EXTENT")
        length *= dimension
    return length


def parse_safetensors_header(
    header_bytes: bytes,
    *,
    data_start: int,
    file_size: int,
) -> dict[str, TensorMetadata]:
    if data_start < 0 or file_size < data_start:
        raise K3XError("INVALID_SOURCE_HEADER")
    try:
        header = json.loads(
            header_bytes,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_non_standard_constant,
        )
    except K3XError:
        raise
    except (json.JSONDecodeError, UnicodeDecodeError) as error:
        raise K3XError("INVALID_SOURCE_HEADER") from error
    if not isinstance(header, dict):
        raise K3XError("INVALID_SOURCE_HEADER")
    data_bytes = file_size - data_start
    result: dict[str, TensorMetadata] = {}
    ranges: list[tuple[int, int]] = []
    for name, metadata in header.items():
        if name == "__metadata__":
            if not isinstance(metadata, dict) or not all(
                isinstance(key, str) and isinstance(value, str)
                for key, value in metadata.items()
            ):
                raise K3XError("INVALID_SOURCE_HEADER")
            continue
        if not name or name.startswith("__"):
            raise K3XError("INVALID_SOURCE_HEADER")
        if not isinstance(metadata, dict) or set(metadata) != _METADATA_KEYS:
            raise K3XError("INVALID_SOURCE_HEADER")
        dtype = metadata["dtype"]
        shape = metadata["shape"]
        offsets = metadata["data_offsets"]
        if (
            not isinstance(dtype, str)
            or not isinstance(shape, list)
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or any(not _is_non_boolean_int(dimension) or dimension < 0 for dimension in shape)
            or any(not _is_non_boolean_int(offset) for offset in offsets)
        ):
            raise K3XError("INVALID_SOURCE_HEADER")
        start, end = offsets
        if start < 0 or end < start or end > data_bytes:
            raise K3XError("INVALID_SOURCE_EXTENT")
        known_length = _known_dtype_length(dtype, shape, data_bytes)
        if known_length is not None and end - start != known_length:
            raise K3XError("INVALID_SOURCE_EXTENT")
        if end > start:
            ranges.append((start, end))
        result[name] = TensorMetadata(
            name, dtype, tuple(shape), data_start + start, end - start
        )
    ranges.sort()
    cursor = 0
    for start, end in ranges:
        if start < cursor:
            raise K3XError("OVERLAPPING_SOURCE_EXTENT")
        if start > cursor:
            raise K3XError("INVALID_SOURCE_EXTENT")
        cursor = end
    if cursor != data_bytes:
        raise K3XError("INVALID_SOURCE_EXTENT")
    return result


def inspect_shard(path: Path) -> dict[str, SourceTensor]:
    size = path.stat().st_size
    with path.open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise K3XError("TRUNCATED_SOURCE_SHARD")
        header_length = struct.unpack("<Q", raw)[0]
        if header_length > _MAX_HEADER_BYTES:
            raise K3XError("INVALID_SOURCE_HEADER")
        if header_length > size - 8:
            raise K3XError("INVALID_SOURCE_HEADER")
        header_bytes = stream.read(header_length)
        if len(header_bytes) != header_length:
            raise K3XError("INVALID_SOURCE_HEADER")
    metadata = parse_safetensors_header(
        header_bytes, data_start=8 + header_length, file_size=size
    )
    return {
        name: SourceTensor(
            path, item.name, item.dtype, item.shape, item.offset, item.length
        )
        for name, item in metadata.items()
    }


def iter_tensor_chunks(tensor: SourceTensor, chunk_bytes: int) -> Iterator[bytes]:
    if chunk_bytes <= 0:
        raise K3XError("INVALID_CHUNK_SIZE")
    with tensor.path.open("rb") as stream:
        stream.seek(tensor.offset)
        remaining = tensor.length
        while remaining:
            chunk = stream.read(min(chunk_bytes, remaining))
            if not chunk:
                raise K3XError("TRUNCATED_SOURCE_SHARD")
            remaining -= len(chunk)
            yield chunk
