# 공식 Kimi K3 snapshot의 고정 파일 신원과 canonical digest를 생성합니다.
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import PurePosixPath
from types import MappingProxyType
from typing import Mapping, Protocol

from .format import K3XError
from .official_transport import HttpResponse


OFFICIAL_REPOSITORY = "moonshotai/Kimi-K3"
OFFICIAL_REQUESTED_REVISION = "main"
OFFICIAL_RESOLVED_REVISION = "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"
_API_LIMIT = 4 * 1024 * 1024
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

