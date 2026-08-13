# 로컬 K3X 제조 계획과 재시작 가능한 IMMORTAL ledger를 관리합니다.
from __future__ import annotations

import hashlib
import fcntl
import json
import os
import re
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

from .format import K3XError

OUTPUT_BUDGET_BYTES = 1_280_000_000_000
QUALITY_OUTPUT_BUDGET_BYTES = 1_510_500_000_000
DESTINATION_RESERVE_BYTES = 200 * 2**30
STAGING_RESERVE_BYTES = 100 * 2**30
_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_LEDGER_KEYS = {
    "format",
    "plan_sha256",
    "completed_units",
    "completed_output_bytes",
    "record_sha256",
}
_COMPLETED_KEYS = {
    "unit_id",
    "source_sha256",
    "output_sha256",
    "output_bytes",
}


@contextmanager
def _ledger_lock(path: Path):
    lock_path = path.with_name(path.name + ".lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+b") as stream:
        fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


@dataclass(frozen=True)
class ShardUnit:
    unit_id: str
    filename: str
    source_bytes: int
    source_sha256: str
    slot: int


@dataclass(frozen=True)
class LocalFoundryPlan:
    repository: str
    revision: str
    output_budget_bytes: int
    destination_reserve_bytes: int
    staging_reserve_bytes: int
    units: tuple[ShardUnit, ...]
    plan_sha256: str


def _canonical_bytes(value: object) -> bytes:
    return json.dumps(value, sort_keys=True, separators=(",", ":")).encode()


def _digest(value: object) -> str:
    return hashlib.sha256(_canonical_bytes(value)).hexdigest()


def build_local_plan(
    repository: str,
    revision: str,
    shards: Sequence[tuple[str, int, str]],
    *,
    output_budget_bytes: int = OUTPUT_BUDGET_BYTES,
) -> LocalFoundryPlan:
    if not repository or not revision or not shards or output_budget_bytes <= 0:
        raise K3XError("INVALID_LOCAL_PLAN")
    ordered = sorted(shards, key=lambda item: item[0])
    if len({item[0] for item in ordered}) != len(ordered):
        raise K3XError("INVALID_LOCAL_PLAN", "duplicate shard")
    units = []
    for index, (filename, source_bytes, source_sha256) in enumerate(ordered):
        if (
            not filename.endswith(".safetensors")
            or source_bytes <= 0
            or not _SHA256_RE.fullmatch(source_sha256)
        ):
            raise K3XError("INVALID_LOCAL_PLAN", filename)
        identity = {
            "repository": repository,
            "revision": revision,
            "filename": filename,
            "source_bytes": source_bytes,
            "source_sha256": source_sha256,
        }
        units.append(
            ShardUnit(
                _digest(identity),
                filename,
                source_bytes,
                source_sha256,
                index % 2,
            )
        )
    contract = {
        "repository": repository,
        "revision": revision,
        "output_budget_bytes": output_budget_bytes,
        "destination_reserve_bytes": DESTINATION_RESERVE_BYTES,
        "staging_reserve_bytes": STAGING_RESERVE_BYTES,
        "units": [asdict(item) for item in units],
    }
    return LocalFoundryPlan(
        repository,
        revision,
        output_budget_bytes,
        DESTINATION_RESERVE_BYTES,
        STAGING_RESERVE_BYTES,
        tuple(units),
        _digest(contract),
    )


def check_disk_budget(
    plan: LocalFoundryPlan,
    *,
    destination_free_bytes: int,
    staging_free_bytes: int,
    completed_output_bytes: int = 0,
) -> None:
    if (
        completed_output_bytes < 0
        or completed_output_bytes > plan.output_budget_bytes
        or destination_free_bytes + completed_output_bytes
        < plan.output_budget_bytes + plan.destination_reserve_bytes
    ):
        raise K3XError("LOCAL_DESTINATION_SPACE")
    required_staging = 2 * max(unit.source_bytes for unit in plan.units)
    required_staging += plan.staging_reserve_bytes
    if staging_free_bytes < required_staging:
        raise K3XError("LOCAL_STAGING_SPACE")


def xet_environment() -> dict[str, str]:
    return {
        "HF_XET_HIGH_PERFORMANCE": "1",
        "HF_HUB_DISABLE_XET": "0",
    }


def staged_source_path(unit: ShardUnit, staging_root: Path) -> Path:
    return Path(staging_root) / f"slot-{unit.slot}" / unit.filename


def build_xet_command(
    plan: LocalFoundryPlan, unit: ShardUnit, staging_root: Path
) -> tuple[str, ...]:
    if unit not in plan.units:
        raise K3XError("INVALID_LOCAL_UNIT")
    return (
        "hf",
        "download",
        plan.repository,
        unit.filename,
        "--revision",
        plan.revision,
        "--local-dir",
        str(staged_source_path(unit, staging_root).parent),
        "--quiet",
    )


def verify_staged_unit(unit: ShardUnit, source_path: Path) -> None:
    try:
        if source_path.stat().st_size != unit.source_bytes:
            raise K3XError("LOCAL_SOURCE_LENGTH", unit.filename)
        digest = hashlib.sha256()
        with source_path.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
    except OSError as error:
        raise K3XError("LOCAL_SOURCE_IO", unit.filename) from error
    if digest.hexdigest() != unit.source_sha256:
        raise K3XError("LOCAL_SOURCE_SHA256", unit.filename)


def source_deletion_allowed(
    ledger_path: Path,
    plan: LocalFoundryPlan,
    unit: ShardUnit,
    source_path: Path,
) -> bool:
    verify_staged_unit(unit, source_path)
    ledger = load_ledger(ledger_path, plan)
    return any(
        item["unit_id"] == unit.unit_id for item in ledger["completed_units"]
    )


def _reject_duplicate_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result: dict[str, object] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("duplicate JSON key")
        result[key] = value
    return result


def _read_json(path: Path) -> dict[str, object]:
    try:
        value = json.loads(
            path.read_text(encoding="utf-8"),
            object_pairs_hook=_reject_duplicate_pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(ValueError(value)),
        )
    except (OSError, UnicodeError, json.JSONDecodeError, ValueError) as error:
        raise K3XError("INVALID_LOCAL_LEDGER") from error
    if not isinstance(value, dict):
        raise K3XError("INVALID_LOCAL_LEDGER")
    return value


def _write_ledger(path: Path, value: dict[str, object]) -> None:
    value = dict(value)
    value["record_sha256"] = _digest(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(_canonical_bytes(value).decode() + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def write_source_manifest(path: Path, plan: LocalFoundryPlan) -> None:
    value = {
        "repository": plan.repository,
        "revision": plan.revision,
        "shards": [
            {
                "filename": unit.filename,
                "bytes": unit.source_bytes,
                "sha256": unit.source_sha256,
            }
            for unit in plan.units
        ],
    }
    value["record_sha256"] = _digest(value)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(_canonical_bytes(value).decode() + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def load_source_manifest(
    path: Path,
) -> tuple[str, str, tuple[tuple[str, int, str], ...]]:
    value = _read_json(path)
    if set(value) != {"repository", "revision", "shards", "record_sha256"}:
        raise K3XError("INVALID_LOCAL_SOURCE_MANIFEST")
    embedded = value.pop("record_sha256")
    if embedded != _digest(value):
        raise K3XError("LOCAL_SOURCE_MANIFEST_DIGEST")
    shards = value.get("shards")
    if not isinstance(shards, list):
        raise K3XError("INVALID_LOCAL_SOURCE_MANIFEST")
    normalized = []
    for item in shards:
        if not isinstance(item, dict) or set(item) != {
            "filename",
            "bytes",
            "sha256",
        }:
            raise K3XError("INVALID_LOCAL_SOURCE_MANIFEST")
        normalized.append((item["filename"], item["bytes"], item["sha256"]))
    return value["repository"], value["revision"], tuple(normalized)


def create_ledger(path: Path, plan: LocalFoundryPlan) -> None:
    with _ledger_lock(path):
        if path.exists():
            load_ledger(path, plan)
            return
        _write_ledger(
            path,
            {
                "format": "k3x-local-foundry-ledger-v1",
                "plan_sha256": plan.plan_sha256,
                "completed_units": [],
                "completed_output_bytes": 0,
            },
        )


def load_ledger(path: Path, plan: LocalFoundryPlan) -> dict[str, object]:
    value = _read_json(path)
    if set(value) != _LEDGER_KEYS:
        raise K3XError("INVALID_LOCAL_LEDGER")
    embedded = value.pop("record_sha256")
    if embedded != _digest(value):
        raise K3XError("LOCAL_LEDGER_DIGEST")
    value["record_sha256"] = embedded
    completed = value.get("completed_units")
    by_id = {unit.unit_id: unit for unit in plan.units}
    if (
        value.get("format") != "k3x-local-foundry-ledger-v1"
        or value.get("plan_sha256") != plan.plan_sha256
        or not isinstance(completed, list)
    ):
        raise K3XError("INVALID_LOCAL_LEDGER")
    total = 0
    seen = set()
    for item in completed:
        if not isinstance(item, dict) or set(item) != _COMPLETED_KEYS:
            raise K3XError("INVALID_LOCAL_LEDGER")
        unit = by_id.get(item["unit_id"])
        if (
            unit is None
            or unit.unit_id in seen
            or item["source_sha256"] != unit.source_sha256
            or not _SHA256_RE.fullmatch(str(item["output_sha256"]))
            or not isinstance(item["output_bytes"], int)
            or isinstance(item["output_bytes"], bool)
            or item["output_bytes"] <= 0
        ):
            raise K3XError("INVALID_LOCAL_LEDGER")
        seen.add(unit.unit_id)
        total += item["output_bytes"]
    if total > plan.output_budget_bytes or value.get("completed_output_bytes") != total:
        raise K3XError("LOCAL_OUTPUT_BUDGET")
    return value


def record_completed_unit(
    path: Path,
    plan: LocalFoundryPlan,
    *,
    unit_id: str,
    source_sha256: str,
    output_sha256: str,
    output_bytes: int,
) -> None:
    with _ledger_lock(path):
        value = load_ledger(path, plan)
        by_id = {unit.unit_id: unit for unit in plan.units}
        unit = by_id.get(unit_id)
        if (
            unit is None
            or source_sha256 != unit.source_sha256
            or not _SHA256_RE.fullmatch(output_sha256)
            or not isinstance(output_bytes, int)
            or isinstance(output_bytes, bool)
            or output_bytes <= 0
            or any(item["unit_id"] == unit_id for item in value["completed_units"])
        ):
            raise K3XError("INVALID_LOCAL_COMPLETION")
        new_total = value["completed_output_bytes"] + output_bytes
        if new_total > plan.output_budget_bytes:
            raise K3XError("LOCAL_OUTPUT_BUDGET")
        completed = list(value["completed_units"])
        completed.append(
            {
                "unit_id": unit_id,
                "source_sha256": source_sha256,
                "output_sha256": output_sha256,
                "output_bytes": output_bytes,
            }
        )
        _write_ledger(
            path,
            {
                "format": value["format"],
                "plan_sha256": value["plan_sha256"],
                "completed_units": completed,
                "completed_output_bytes": new_total,
            },
        )
