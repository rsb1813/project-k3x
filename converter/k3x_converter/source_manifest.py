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


def _reject_non_standard_constant(_: str) -> object:
    raise K3XError("INVALID_SOURCE_MANIFEST")


def load_source_manifest(source: Path) -> dict[str, object]:
    try:
        manifest = json.loads(
            (source / "source-manifest.json").read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_non_standard_constant,
        )
    except K3XError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise K3XError("INVALID_SOURCE_MANIFEST") from error
    if not isinstance(manifest, dict):
        raise K3XError("INVALID_SOURCE_MANIFEST")
    if manifest.get("format") not in {
        "synthetic-k3-source-v1",
        "k3-storage-slice-v1",
    }:
        raise K3XError("UNSUPPORTED_SOURCE_FORMAT")
    if not isinstance(manifest.get("config"), dict) or not isinstance(
        manifest.get("packed_shapes"), dict
    ):
        raise K3XError("INVALID_SOURCE_MANIFEST")
    weight_map = manifest.get("weight_map")
    if not isinstance(weight_map, dict) or any(
        not isinstance(name, str)
        or not name
        or not isinstance(shard_name, str)
        or not shard_name
        for name, shard_name in weight_map.items()
    ):
        raise K3XError("INVALID_SOURCE_MANIFEST")
    return manifest


def _resolve_source_shard(source: Path, shard_name: str) -> Path:
    posix = PurePosixPath(shard_name)
    windows = PureWindowsPath(shard_name)
    if (
        posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or windows.root
        or "\\" in shard_name
        or any(part in {"", ".", ".."} for part in shard_name.split("/"))
    ):
        raise K3XError("SOURCE_SHARD_PATH_ESCAPE")
    source_root = source.resolve()
    try:
        resolved = source_root.joinpath(*posix.parts).resolve(strict=True)
        resolved.relative_to(source_root)
    except (OSError, RuntimeError, ValueError) as error:
        raise K3XError("SOURCE_SHARD_PATH_ESCAPE") from error
    if not resolved.is_file():
        raise K3XError("SOURCE_SHARD_PATH_ESCAPE")
    return resolved


def inspect_manifest_tensors(
    source: Path, manifest: dict[str, object]
) -> dict[str, SourceTensor]:
    weight_map = manifest.get("weight_map")
    if not isinstance(weight_map, dict):
        raise K3XError("INVALID_SOURCE_MANIFEST")
    tensors: dict[str, SourceTensor] = {}
    for shard_name in sorted(set(weight_map.values())):
        if not isinstance(shard_name, str):
            raise K3XError("INVALID_SOURCE_MANIFEST")
        shard_path = _resolve_source_shard(source, shard_name)
        for name, tensor in inspect_shard(shard_path).items():
            if name in tensors or weight_map.get(name) != shard_name:
                raise K3XError("SOURCE_TENSOR_SHARD_MISMATCH", name)
            tensors[name] = tensor
    if set(tensors) != set(weight_map):
        raise K3XError("SOURCE_TENSOR_SHARD_MISMATCH")
    return tensors
