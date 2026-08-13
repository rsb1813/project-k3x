# 검증된 공식 shard 하나를 K3X로 제조하고 ledger 확정 후 원본을 정리합니다.
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

from k3x_converter.format import K3XError
from k3x_converter.local_foundry import (
    QUALITY_OUTPUT_BUDGET_BYTES,
    build_local_plan,
    check_disk_budget,
    create_ledger,
    load_ledger,
    load_source_manifest,
    record_completed_unit,
    source_deletion_allowed,
)
from k3x_converter.local_shard import convert_local_official_shard


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--config-manifest", type=Path, required=True)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--temporary-directory", type=Path)
    parser.add_argument(
        "--output-budget-bytes", type=int, default=QUALITY_OUTPUT_BUDGET_BYTES
    )
    parser.add_argument("--delete-source", action="store_true")
    args = parser.parse_args(argv)

    repository, revision, shards = load_source_manifest(args.manifest.resolve())
    plan = build_local_plan(
        repository,
        revision,
        shards,
        output_budget_bytes=args.output_budget_bytes,
    )
    source = args.source.resolve(strict=True)
    unit = next((item for item in plan.units if item.filename == source.name), None)
    if unit is None:
        raise K3XError("INVALID_LOCAL_UNIT", source.name)
    destination = args.destination.resolve(strict=True)
    ledger_path = args.ledger.resolve()
    create_ledger(ledger_path, plan)
    ledger = load_ledger(ledger_path, plan)
    check_disk_budget(
        plan,
        destination_free_bytes=shutil.disk_usage(destination).free,
        staging_free_bytes=shutil.disk_usage(source.parent).free,
        completed_output_bytes=ledger["completed_output_bytes"],
    )
    config_document = json.loads(
        args.config_manifest.read_text(encoding="utf-8")
    )
    config = config_document.get("config")
    if not isinstance(config, dict):
        raise K3XError("INVALID_LOCAL_CONFIG")
    conversion_start = time.perf_counter()
    report = convert_local_official_shard(
        source,
        destination,
        config=config,
        expected_sha256=unit.source_sha256,
        temporary_directory=(
            args.temporary_directory
            if args.temporary_directory is not None
            else source.parent / ".foundry-work"
        ),
    )
    conversion_seconds = time.perf_counter() - conversion_start
    record_completed_unit(
        ledger_path,
        plan,
        unit_id=unit.unit_id,
        source_sha256=unit.source_sha256,
        output_sha256=report.output_sha256,
        output_bytes=report.output_bytes,
    )
    deleted = False
    if args.delete_source:
        if not source_deletion_allowed(
            ledger_path,
            plan,
            unit,
            source,
            verified_source_sha256=report.source_sha256,
            verified_source_identity=report.source_identity,
        ):
            raise K3XError("LOCAL_SOURCE_DELETE_NOT_ALLOWED", source.name)
        source.unlink()
        deleted = True
    print(
        json.dumps(
            {
                "format": "k3x-local-shard-result-v1",
                "unit_id": unit.unit_id,
                "source": str(source),
                "source_sha256": report.source_sha256,
                "output": str(report.output_path),
                "output_sha256": report.output_sha256,
                "output_bytes": report.output_bytes,
                "tensor_count": report.tensor_count,
                "quant8_tensor_count": report.quant8_tensor_count,
                "native_expert_tensor_count": report.native_expert_tensor_count,
                "conversion_seconds": conversion_seconds,
                "source_deleted": deleted,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
