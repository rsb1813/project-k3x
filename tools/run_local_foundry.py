# 로컬 K3X 제조 계획의 디스크·소스 경계를 실행 전에 검증합니다.
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from k3x_converter.format import K3XError
from k3x_converter.local_foundry import build_local_plan, check_disk_budget


def _load_manifest(path: Path) -> tuple[str, str, tuple[tuple[str, int, str], ...]]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(value, dict) or set(value) != {
            "repository",
            "revision",
            "shards",
        }:
            raise ValueError("manifest keys")
        shards = value["shards"]
        if not isinstance(shards, list):
            raise ValueError("shards")
        normalized = []
        for item in shards:
            if not isinstance(item, dict) or set(item) != {
                "filename",
                "bytes",
                "sha256",
            }:
                raise ValueError("shard")
            normalized.append((item["filename"], item["bytes"], item["sha256"]))
        return value["repository"], value["revision"], tuple(normalized)
    except (OSError, UnicodeError, json.JSONDecodeError, KeyError, ValueError) as error:
        raise K3XError("INVALID_LOCAL_SOURCE_MANIFEST") from error


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--staging", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.dry_run:
        raise K3XError("LOCAL_OFFICIAL_GATE_CLOSED")
    repository, revision, shards = _load_manifest(args.manifest.resolve())
    plan = build_local_plan(repository, revision, shards)
    destination = args.destination.resolve(strict=True)
    staging = args.staging.resolve(strict=True)
    destination_free = shutil.disk_usage(destination).free
    staging_free = shutil.disk_usage(staging).free
    check_disk_budget(
        plan,
        destination_free_bytes=destination_free,
        staging_free_bytes=staging_free,
    )
    report = {
        "format": "k3x-local-foundry-dry-run-v1",
        "repository": plan.repository,
        "revision": plan.revision,
        "plan_sha256": plan.plan_sha256,
        "unit_count": len(plan.units),
        "output_budget_bytes": plan.output_budget_bytes,
        "destination_free_bytes": destination_free,
        "destination_reserve_bytes": plan.destination_reserve_bytes,
        "staging_free_bytes": staging_free,
        "staging_reserve_bytes": plan.staging_reserve_bytes,
        "maximum_shard_bytes": max(unit.source_bytes for unit in plan.units),
        "ledger_path": str(args.ledger.resolve()),
        "official_launch_enabled": False,
    }
    print(json.dumps(report, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
