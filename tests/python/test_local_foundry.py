# 로컬 K3X 제조 계획과 영속 ledger의 실패 폐쇄 경계를 검증합니다.
from __future__ import annotations

import json

import pytest

from k3x_converter.format import K3XError
from k3x_converter.local_foundry import (
    DESTINATION_RESERVE_BYTES,
    OUTPUT_BUDGET_BYTES,
    STAGING_RESERVE_BYTES,
    build_local_plan,
    check_disk_budget,
    create_ledger,
    load_ledger,
    record_completed_unit,
)


SHARDS = (
    ("model-00001-of-000002.safetensors", 17_000_000_000, "11" * 32),
    ("model-00002-of-000002.safetensors", 4_700_000_000, "22" * 32),
)


def test_plan_assigns_two_slots_and_enforces_disk_reserves():
    plan = build_local_plan("moonshotai/Kimi-K3", "9f62e4e", SHARDS)

    assert [unit.slot for unit in plan.units] == [0, 1]
    assert plan.output_budget_bytes == OUTPUT_BUDGET_BYTES
    check_disk_budget(
        plan,
        destination_free_bytes=OUTPUT_BUDGET_BYTES + DESTINATION_RESERVE_BYTES,
        staging_free_bytes=2 * SHARDS[0][1] + STAGING_RESERVE_BYTES,
    )
    with pytest.raises(K3XError, match="LOCAL_DESTINATION_SPACE"):
        check_disk_budget(
            plan,
            destination_free_bytes=(
                OUTPUT_BUDGET_BYTES + DESTINATION_RESERVE_BYTES - 1
            ),
            staging_free_bytes=2 * SHARDS[0][1] + STAGING_RESERVE_BYTES,
        )


def test_ledger_resumes_only_checksum_bound_completed_units(tmp_path):
    plan = build_local_plan("moonshotai/Kimi-K3", "9f62e4e", SHARDS)
    path = tmp_path / "immortal.json"
    create_ledger(path, plan)
    record_completed_unit(
        path,
        plan,
        unit_id=plan.units[0].unit_id,
        source_sha256=SHARDS[0][2],
        output_sha256="aa" * 32,
        output_bytes=9_000_000_000,
    )

    resumed = load_ledger(path, plan)
    assert resumed["completed_units"][0]["unit_id"] == plan.units[0].unit_id
    assert resumed["completed_output_bytes"] == 9_000_000_000

    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["completed_units"][0]["output_bytes"] += 1
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(K3XError, match="LOCAL_LEDGER_DIGEST"):
        load_ledger(path, plan)


def test_ledger_rejects_output_budget_overflow(tmp_path):
    plan = build_local_plan("moonshotai/Kimi-K3", "9f62e4e", SHARDS)
    path = tmp_path / "immortal.json"
    create_ledger(path, plan)

    with pytest.raises(K3XError, match="LOCAL_OUTPUT_BUDGET"):
        record_completed_unit(
            path,
            plan,
            unit_id=plan.units[0].unit_id,
            source_sha256=SHARDS[0][2],
            output_sha256="aa" * 32,
            output_bytes=OUTPUT_BUDGET_BYTES + 1,
        )
