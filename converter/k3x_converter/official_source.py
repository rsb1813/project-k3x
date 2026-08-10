# 공식 Kimi K3 snapshot의 고정 파일 신원과 canonical digest를 생성합니다.
from __future__ import annotations

import hashlib
import json
import re
import struct
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Mapping, Protocol

from .format import K3XError
from .official_transport import HttpResponse
from .safetensors_reader import TensorMetadata, parse_safetensors_header


OFFICIAL_REPOSITORY = "moonshotai/Kimi-K3"
OFFICIAL_REQUESTED_REVISION = "main"
OFFICIAL_RESOLVED_REVISION = "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"
_API_LIMIT = 4 * 1024 * 1024
_INDEX_LIMIT = 64 * 1024 * 1024
_CONFIG_LIMIT = 1 * 1024 * 1024
_HEADER_LIMIT = 100_000_000
_PAYLOAD_LIMIT = 32 * 1024 * 1024
_TIMEOUT_SECONDS = 120.0
_HEX40 = re.compile(r"[0-9a-f]{40}")
_HEX64 = re.compile(r"[0-9a-f]{64}")


class Transport(Protocol):
    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        max_bytes: int,
        timeout_seconds: float,
        expected_status: int = 200,
    ) -> HttpResponse: ...


@dataclass(frozen=True)
class OfficialFile:
    path: str
    size: int
    blob_id: str
    lfs_sha256: str | None


@dataclass(frozen=True)
class OfficialSnapshot:
    repository: str
    requested_revision: str
    resolved_revision: str
    observed_at: str
    files: Mapping[str, OfficialFile]
    file_count: int
    repository_bytes: int
    canonical_sha256: str


@dataclass(frozen=True)
class OfficialIndex:
    total_size: int
    weight_map: Mapping[str, str]
    shard_paths: tuple[str, ...]
    tensor_count: int
    sha256: str


@dataclass(frozen=True)
class OfficialConfig:
    sha256: str
    git_blob_id: str
    hidden_size: int
    num_experts: int
    top_k: int
    routed_latent_size: int
    expert_intermediate_size: int


@dataclass(frozen=True)
class OfficialShardHeader:
    shard_path: str
    file_size: int
    header_length: int
    data_start: int
    tensors: Mapping[str, TensorMetadata]


@dataclass(frozen=True)
class PlannedTensor:
    official_name: str
    canonical_name: str
    role: str
    dtype: str
    shape: tuple[int, ...]
    offset: int
    length: int


@dataclass(frozen=True)
class ExpertPlan:
    layer_id: int
    expert_id: int
    shard_path: str
    payload_start: int
    payload_end: int
    payload_bytes: int
    tensors: tuple[PlannedTensor, ...]


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise K3XError("INVALID_OFFICIAL_API")
        result[key] = value
    return result


def _reject_non_standard_constant(_: str) -> object:
    raise K3XError("INVALID_OFFICIAL_API")


def _decode_api(body: bytes) -> dict[str, object]:
    try:
        value = json.loads(
            body,
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_non_standard_constant,
        )
    except K3XError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise K3XError("INVALID_OFFICIAL_API") from error
    if not isinstance(value, dict):
        raise K3XError("INVALID_OFFICIAL_API")
    return value


def _decode_document(body: bytes, code: str) -> dict[str, object]:
    def reject_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise K3XError(code)
            result[key] = value
        return result

    def reject_constant(_: str) -> object:
        raise K3XError(code)

    try:
        value = json.loads(
            body,
            object_pairs_hook=reject_pairs,
            parse_constant=reject_constant,
        )
    except K3XError:
        raise
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise K3XError(code) from error
    if not isinstance(value, dict):
        raise K3XError(code)
    return value


def _valid_path(value: object) -> bool:
    if not isinstance(value, str) or not value or "\\" in value:
        return False
    path = PurePosixPath(value)
    return not path.is_absolute() and all(part not in {"", ".", ".."} for part in path.parts)


def _is_int(value: object) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _parse_file(value: object) -> OfficialFile:
    if not isinstance(value, dict):
        raise K3XError("INVALID_OFFICIAL_FILE")
    path = value.get("rfilename")
    size = value.get("size")
    blob_id = value.get("blobId")
    if (
        not _valid_path(path)
        or not _is_int(size)
        or size < 0
        or not isinstance(blob_id, str)
        or _HEX40.fullmatch(blob_id) is None
    ):
        raise K3XError("INVALID_OFFICIAL_FILE")
    lfs_sha256: str | None = None
    lfs = value.get("lfs")
    if lfs is not None:
        if not isinstance(lfs, dict):
            raise K3XError("INVALID_OFFICIAL_FILE")
        lfs_size = lfs.get("size")
        lfs_digest = lfs.get("sha256")
        if (
            not _is_int(lfs_size)
            or lfs_size != size
            or not isinstance(lfs_digest, str)
            or _HEX64.fullmatch(lfs_digest) is None
        ):
            raise K3XError("INVALID_OFFICIAL_FILE")
        lfs_sha256 = lfs_digest
    if (
        path == "model.safetensors.index.json"
        or (isinstance(path, str) and path.endswith(".safetensors"))
    ) and lfs_sha256 is None:
        raise K3XError("INVALID_OFFICIAL_FILE")
    return OfficialFile(path, size, blob_id, lfs_sha256)


def _canonical_snapshot(
    resolved_revision: str, files: Mapping[str, OfficialFile]
) -> bytes:
    value = {
        "format": "k3x-official-discovery-v1",
        "repository": OFFICIAL_REPOSITORY,
        "requested_revision": OFFICIAL_REQUESTED_REVISION,
        "resolved_revision": resolved_revision,
        "files": [
            {
                "path": item.path,
                "size": item.size,
                "blob_id": item.blob_id,
                "lfs_sha256": item.lfs_sha256,
            }
            for item in (files[path] for path in sorted(files))
        ],
    }
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")


def discover_official_snapshot(
    transport: Transport,
    *,
    observed_at: str | None = None,
) -> OfficialSnapshot:
    url = (
        "https://huggingface.co/api/models/moonshotai/Kimi-K3/revision/"
        "main?blobs=true"
    )
    response = transport.get(
        url, headers={"Accept": "application/json"}, max_bytes=_API_LIMIT,
        timeout_seconds=_TIMEOUT_SECONDS,
    )
    record = _decode_api(response.body)
    if record.get("id") != OFFICIAL_REPOSITORY:
        raise K3XError("OFFICIAL_REPOSITORY_DRIFT")
    if record.get("private") is not False or record.get("gated") is not False:
        raise K3XError("OFFICIAL_REPOSITORY_ACCESS")
    resolved = record.get("sha")
    if (
        not isinstance(resolved, str)
        or _HEX40.fullmatch(resolved) is None
        or resolved != OFFICIAL_RESOLVED_REVISION
    ):
        raise K3XError("OFFICIAL_REVISION_DRIFT")
    siblings = record.get("siblings")
    if not isinstance(siblings, list) or not siblings:
        raise K3XError("INVALID_OFFICIAL_API")
    parsed: dict[str, OfficialFile] = {}
    for value in siblings:
        item = _parse_file(value)
        if item.path in parsed:
            raise K3XError("INVALID_OFFICIAL_FILE")
        parsed[item.path] = item
    if "config.json" not in parsed or "model.safetensors.index.json" not in parsed:
        raise K3XError("INVALID_OFFICIAL_FILE")
    canonical = _canonical_snapshot(resolved, parsed)
    timestamp = observed_at or datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return OfficialSnapshot(
        OFFICIAL_REPOSITORY,
        OFFICIAL_REQUESTED_REVISION,
        resolved,
        timestamp,
        MappingProxyType(parsed),
        len(parsed),
        sum(item.size for item in parsed.values()),
        hashlib.sha256(canonical).hexdigest(),
    )


def _resolve_url(snapshot: OfficialSnapshot, path: str) -> str:
    return (
        f"https://huggingface.co/{snapshot.repository}/resolve/"
        f"{snapshot.resolved_revision}/{path}"
    )


def _fetch_file(
    snapshot: OfficialSnapshot,
    transport: Transport,
    path: str,
    limit: int,
) -> tuple[OfficialFile, bytes]:
    item = snapshot.files.get(path)
    if item is None or item.size > limit:
        raise K3XError("INVALID_OFFICIAL_FILE")
    response = transport.get(
        _resolve_url(snapshot, path),
        headers={"Accept": "application/octet-stream"},
        max_bytes=limit,
        timeout_seconds=_TIMEOUT_SECONDS,
    )
    if len(response.body) != item.size:
        raise K3XError("OFFICIAL_FILE_SIZE_MISMATCH", path)
    return item, response.body


def load_official_index(
    snapshot: OfficialSnapshot,
    transport: Transport,
) -> OfficialIndex:
    item, body = _fetch_file(
        snapshot, transport, "model.safetensors.index.json", _INDEX_LIMIT
    )
    digest = hashlib.sha256(body).hexdigest()
    if item.lfs_sha256 is None or digest != item.lfs_sha256:
        raise K3XError("OFFICIAL_INDEX_SHA256_MISMATCH")
    value = _decode_document(body, "INVALID_OFFICIAL_INDEX")
    if set(value) != {"metadata", "weight_map"}:
        raise K3XError("INVALID_OFFICIAL_INDEX")
    metadata = value["metadata"]
    weight_map = value["weight_map"]
    if (
        not isinstance(metadata, dict)
        or set(metadata) != {"total_size"}
        or not _is_int(metadata["total_size"])
        or metadata["total_size"] <= 0
        or not isinstance(weight_map, dict)
        or not weight_map
    ):
        raise K3XError("INVALID_OFFICIAL_INDEX")
    parsed: dict[str, str] = {}
    for name, shard_path in weight_map.items():
        if (
            not isinstance(name, str)
            or not name
            or not _valid_path(shard_path)
            or not isinstance(shard_path, str)
            or not shard_path.endswith(".safetensors")
        ):
            raise K3XError("INVALID_OFFICIAL_INDEX")
        parsed[name] = shard_path
    declared = tuple(
        sorted(path for path in snapshot.files if path.endswith(".safetensors"))
    )
    if (
        len(declared) != 96
        or set(parsed.values()) != set(declared)
        or any(snapshot.files[path].lfs_sha256 is None for path in declared)
    ):
        raise K3XError("OFFICIAL_SHARD_SET_MISMATCH")
    return OfficialIndex(
        metadata["total_size"],
        MappingProxyType(parsed),
        declared,
        len(parsed),
        digest,
    )


def _git_blob_id(body: bytes) -> str:
    prefix = b"blob " + str(len(body)).encode("ascii") + b"\0"
    return hashlib.sha1(prefix + body).hexdigest()


def _matches_number(value: object, expected: int | float) -> bool:
    return (
        isinstance(value, (int, float))
        and not isinstance(value, bool)
        and value == expected
    )


def load_official_config(
    snapshot: OfficialSnapshot,
    transport: Transport,
) -> OfficialConfig:
    item, body = _fetch_file(snapshot, transport, "config.json", _CONFIG_LIMIT)
    blob_id = _git_blob_id(body)
    if blob_id != item.blob_id:
        raise K3XError("OFFICIAL_CONFIG_BLOB_MISMATCH")
    value = _decode_document(body, "INVALID_OFFICIAL_CONFIG")
    text = value.get("text_config")
    expected: dict[str, int | float | str] = {
        "model_type": "kimi_linear",
        "vocab_size": 163_840,
        "num_hidden_layers": 93,
        "first_k_dense_replace": 1,
        "moe_layer_freq": 1,
        "num_experts": 896,
        "num_experts_per_token": 16,
        "num_shared_experts": 2,
        "hidden_size": 7_168,
        "routed_expert_hidden_size": 3_584,
        "moe_intermediate_size": 3_072,
        "activation_situ_beta": 4.0,
        "activation_situ_linear_beta": 25.0,
        "routed_scaling_factor": 1.0,
    }
    if value.get("model_type") != "kimi_k3" or not isinstance(text, dict):
        raise K3XError("OFFICIAL_CONFIG_MISMATCH")
    for key, expected_value in expected.items():
        actual = text.get(key)
        if isinstance(expected_value, str):
            valid = actual == expected_value
        else:
            valid = _matches_number(actual, expected_value)
        if not valid:
            raise K3XError("OFFICIAL_CONFIG_MISMATCH", key)
    return OfficialConfig(
        hashlib.sha256(body).hexdigest(),
        blob_id,
        text["hidden_size"],
        text["num_experts"],
        text["num_experts_per_token"],
        text["routed_expert_hidden_size"],
        text["moe_intermediate_size"],
    )


def _fetch_exact_range(
    snapshot: OfficialSnapshot,
    shard: OfficialFile,
    transport: Transport,
    start: int,
    end: int,
) -> bytes:
    if start < 0 or end < start or end >= shard.size:
        raise K3XError("INVALID_OFFICIAL_RANGE")
    length = end - start + 1
    if length > _HEADER_LIMIT:
        raise K3XError("OFFICIAL_BODY_LIMIT")
    response = transport.get(
        _resolve_url(snapshot, shard.path),
        headers={"Range": f"bytes={start}-{end}"},
        max_bytes=length,
        timeout_seconds=_TIMEOUT_SECONDS,
        expected_status=206,
    )
    if response.status != 206:
        raise K3XError("OFFICIAL_HTTP_STATUS", str(response.status))
    expected_range = f"bytes {start}-{end}/{shard.size}"
    if response.headers.get("content-range") != expected_range:
        raise K3XError("OFFICIAL_CONTENT_RANGE_MISMATCH")
    if len(response.body) != length:
        raise K3XError("OFFICIAL_RANGE_LENGTH_MISMATCH")
    return response.body


def inspect_official_shard_header(
    snapshot: OfficialSnapshot,
    shard_path: str,
    transport: Transport,
) -> OfficialShardHeader:
    shard = snapshot.files.get(shard_path)
    if (
        shard is None
        or not shard.path.endswith(".safetensors")
        or shard.lfs_sha256 is None
    ):
        raise K3XError("INVALID_OFFICIAL_FILE")
    raw_length = _fetch_exact_range(snapshot, shard, transport, 0, 7)
    header_length = struct.unpack("<Q", raw_length)[0]
    if (
        header_length == 0
        or header_length > _HEADER_LIMIT
        or header_length > shard.size - 8
    ):
        raise K3XError("INVALID_SOURCE_HEADER")
    header_bytes = _fetch_exact_range(
        snapshot, shard, transport, 8, 7 + header_length
    )
    data_start = 8 + header_length
    tensors = parse_safetensors_header(
        header_bytes, data_start=data_start, file_size=shard.size
    )
    return OfficialShardHeader(
        shard.path,
        shard.size,
        header_length,
        data_start,
        MappingProxyType(tensors),
    )


def plan_official_expert(
    index: OfficialIndex,
    header: OfficialShardHeader,
    *,
    layer_id: int,
    expert_id: int,
) -> ExpertPlan:
    if layer_id < 1 or expert_id < 0:
        raise K3XError("INVALID_OFFICIAL_EXPERT")
    official_base = (
        f"language_model.model.layers.{layer_id}.block_sparse_moe."
        f"experts.{expert_id}"
    )
    canonical_base = (
        f"model.layers.{layer_id}.feed_forward.experts.{expert_id}"
    )
    specifications = (
        ("w1", "weight_packed", "gate", (3072, 1792), 5_505_024),
        ("w1", "weight_scale", "gate", (3072, 112), 344_064),
        ("w2", "weight_packed", "down", (3584, 1536), 5_505_024),
        ("w2", "weight_scale", "down", (3584, 96), 344_064),
        ("w3", "weight_packed", "up", (3072, 1792), 5_505_024),
        ("w3", "weight_scale", "up", (3072, 112), 344_064),
    )
    planned: list[PlannedTensor] = []
    for matrix, kind, role, shape, length in specifications:
        official_name = f"{official_base}.{matrix}.{kind}"
        metadata = header.tensors.get(official_name)
        if (
            index.weight_map.get(official_name) != header.shard_path
            or metadata is None
            or metadata.dtype != "U8"
            or metadata.shape != shape
            or metadata.length != length
        ):
            raise K3XError("INVALID_OFFICIAL_EXPERT", official_name)
        planned.append(
            PlannedTensor(
                official_name,
                f"{canonical_base}.{role}.{kind}",
                role,
                metadata.dtype,
                metadata.shape,
                metadata.offset,
                metadata.length,
            )
        )
    for left, right in zip(planned, planned[1:]):
        if left.offset + left.length != right.offset:
            raise K3XError("INVALID_OFFICIAL_EXPERT", "noncontiguous")
    payload_start = planned[0].offset
    payload_end = planned[-1].offset + planned[-1].length
    payload_bytes = payload_end - payload_start
    if payload_bytes != 17_547_264 or payload_bytes > _PAYLOAD_LIMIT:
        raise K3XError("INVALID_OFFICIAL_EXPERT", "payload length")
    return ExpertPlan(
        layer_id,
        expert_id,
        header.shard_path,
        payload_start,
        payload_end,
        payload_bytes,
        tuple(planned),
    )
