# 완료된 Local Foundry ledger를 실행 가능한 K3X set manifest로 봉인합니다.
from __future__ import annotations

import argparse
import json
from pathlib import Path

from k3x_converter.format import K3XError
from k3x_converter.fragment_set import write_fragment_set_manifest
from k3x_converter.local_foundry import (
    QUALITY_OUTPUT_BUDGET_BYTES,
    build_local_plan,
    load_ledger,
    load_source_manifest,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--ledger", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--output-budget-bytes", type=int, default=QUALITY_OUTPUT_BUDGET_BYTES
    )
    args = parser.parse_args(argv)

    repository, revision, shards = load_source_manifest(args.manifest.resolve())
    plan = build_local_plan(
        repository,
        revision,
        shards,
        output_budget_bytes=args.output_budget_bytes,
    )
    ledger = load_ledger(args.ledger.resolve(), plan)
    completed = {item["unit_id"]: item for item in ledger["completed_units"]}
    if set(completed) != {unit.unit_id for unit in plan.units}:
        raise K3XError("INCOMPLETE_FRAGMENT_SET")
    destination = args.destination.resolve(strict=True)
    output = args.output.resolve()
    if output.parent != destination:
        raise K3XError("INVALID_FRAGMENT_SET_OUTPUT")
    fragments = []
    for unit in plan.units:
        fragment = destination / Path(unit.filename).with_suffix(".k3x").name
        record = completed[unit.unit_id]
        if not fragment.is_file() or fragment.stat().st_size != record["output_bytes"]:
            raise K3XError("INVALID_FRAGMENT_SET_ARTIFACT", fragment.name)
        fragments.append(fragment)
    record_sha256 = write_fragment_set_manifest(
        output, fragments, plan_sha256=plan.plan_sha256
    )
    print(
        json.dumps(
            {
                "format": "k3x-fragment-set-result-v1",
                "fragment_count": len(fragments),
                "manifest": str(output),
                "manifest_sha256": record_sha256,
                "payload_bytes": ledger["completed_output_bytes"],
                "plan_sha256": plan.plan_sha256,
            },
            sort_keys=True,
            separators=(",", ":"),
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
