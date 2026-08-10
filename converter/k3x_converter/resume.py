# 중단 가능한 변환의 검증 완료 extent ledger를 원자적으로 관리합니다.
from __future__ import annotations

import json
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from .format import K3XError

_MANIFEST_KEYS = {
    "source_fingerprint",
    "converter_version",
    "configuration_fingerprint",
    "file_uuid",
    "completed",
}
_EXTENT_KEYS = {"extent_id", "offset", "length", "crc32c"}
_FINGERPRINT_RE = re.compile(r"[0-9a-f]{64}")
_UUID_RE = re.compile(r"[0-9a-f]{32}")
_EXTENT_ID_RE = re.compile(r"[0-9a-f]{16}:(data|auxiliary)")


@dataclass(frozen=True)
class CompletedExtent:
    extent_id: str
    offset: int
    length: int
    crc32c: int


@dataclass(frozen=True)
class ResumeManifest:
    source_fingerprint: str
    converter_version: str
    configuration_fingerprint: str
    file_uuid: str
    completed: tuple[CompletedExtent, ...]


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _reject_non_standard_constant(_: str) -> object:
    raise ValueError("non-standard JSON constant")


def _is_uint(value: object, maximum: int) -> bool:
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value <= maximum


def read_resume_manifest(path: Path) -> ResumeManifest:
    try:
        data = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=_reject_non_standard_constant,
        )
        if not isinstance(data, dict) or set(data) != _MANIFEST_KEYS:
            raise ValueError("invalid manifest keys")
        source_fingerprint = data["source_fingerprint"]
        converter_version = data["converter_version"]
        configuration_fingerprint = data["configuration_fingerprint"]
        file_uuid = data["file_uuid"]
        completed = data["completed"]
        if (
            not isinstance(source_fingerprint, str)
            or not _FINGERPRINT_RE.fullmatch(source_fingerprint)
            or not isinstance(converter_version, str)
            or not converter_version
            or not isinstance(configuration_fingerprint, str)
            or not _FINGERPRINT_RE.fullmatch(configuration_fingerprint)
            or not isinstance(file_uuid, str)
            or not _UUID_RE.fullmatch(file_uuid)
            or not isinstance(completed, list)
        ):
            raise ValueError("invalid manifest value")
        extents: list[CompletedExtent] = []
        for item in completed:
            if not isinstance(item, dict) or set(item) != _EXTENT_KEYS:
                raise ValueError("invalid extent keys")
            extent_id = item["extent_id"]
            offset = item["offset"]
            length = item["length"]
            crc32c = item["crc32c"]
            if (
                not isinstance(extent_id, str)
                or not _EXTENT_ID_RE.fullmatch(extent_id)
                or not _is_uint(offset, 2**64 - 1)
                or not _is_uint(length, 2**64 - 1)
                or not _is_uint(crc32c, 2**32 - 1)
            ):
                raise ValueError("invalid extent value")
            extents.append(CompletedExtent(extent_id, offset, length, crc32c))
        return ResumeManifest(
            source_fingerprint,
            converter_version,
            configuration_fingerprint,
            file_uuid,
            tuple(extents),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, TypeError, KeyError, ValueError) as error:
        raise K3XError("INVALID_RESUME_MANIFEST") from error


def write_resume_manifest(path: Path, manifest: ResumeManifest) -> None:
    temporary = path.with_name(path.name + ".tmp")
    data = asdict(manifest)
    data["completed"] = [asdict(item) for item in manifest.completed]
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(data, stream, sort_keys=True, separators=(",", ":"))
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
