# 로컬 K3X 제조 계획과 영속 ledger의 실패 폐쇄 경계를 검증합니다.
from __future__ import annotations

import json

import pytest
import torch

from k3x_converter.format import K3XError
from k3x_converter.local_foundry import (
    DESTINATION_RESERVE_BYTES,
    OUTPUT_BUDGET_BYTES,
    STAGING_RESERVE_BYTES,
    build_local_plan,
    build_xet_command,
    check_disk_budget,
    create_ledger,
    load_ledger,
    record_completed_unit,
    source_deletion_allowed,
    staged_source_path,
    verify_staged_unit,
    xet_environment,
)
from k3x_ref.quant3 import Quant3Tensor, decode_groupwise_3bit, quantize_groupwise_3bit
from tools.run_local_foundry import main as local_foundry_main


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


def test_quant3_round_trip_is_deterministic_and_budget_closed():
    source = torch.linspace(-3.0, 3.0, 64, dtype=torch.float32).reshape(8, 8)

    first = quantize_groupwise_3bit(source)
    second = quantize_groupwise_3bit(source)
    decoded = decode_groupwise_3bit(first)

    assert first == second
    assert source.shape == decoded.shape
    assert len(first.packed) == 24
    assert len(first.scales_bf16) == 4
    assert torch.max(torch.abs(source - decoded)).item() < 0.55


def test_quant3_decode_rejects_reserved_code():
    encoded = Quant3Tensor(
        shape=(8,),
        values=8,
        group_size=32,
        packed=b"\xff\xff\xff" + bytes(9),
        scales_bf16=torch.tensor([1.0], dtype=torch.bfloat16)
        .view(torch.uint8)
        .numpy()
        .tobytes(),
    )

    with pytest.raises(K3XError, match="QUANT3_RESERVED_CODE"):
        decode_groupwise_3bit(encoded)


def test_xet_staging_is_token_free_and_deletion_is_ledger_gated(tmp_path):
    payload = b"checksum-bound-source"
    source_sha256 = __import__("hashlib").sha256(payload).hexdigest()
    plan = build_local_plan(
        "moonshotai/Kimi-K3",
        "9f62e4e",
        (("model-00001-of-000001.safetensors", len(payload), source_sha256),),
    )
    unit = plan.units[0]

    command = build_xet_command(plan, unit, tmp_path / "staging")
    assert command[:3] == ("hf", "download", "moonshotai/Kimi-K3")
    assert "--token" not in command
    assert xet_environment() == {
        "HF_XET_HIGH_PERFORMANCE": "1",
        "HF_HUB_DISABLE_XET": "0",
    }
    source_path = staged_source_path(unit, tmp_path / "staging")
    source_path.parent.mkdir(parents=True)
    source_path.write_bytes(payload)
    verify_staged_unit(unit, source_path)

    ledger_path = tmp_path / "immortal.json"
    create_ledger(ledger_path, plan)
    assert not source_deletion_allowed(ledger_path, plan, unit, source_path)
    record_completed_unit(
        ledger_path,
        plan,
        unit_id=unit.unit_id,
        source_sha256=source_sha256,
        output_sha256="aa" * 32,
        output_bytes=10,
    )
    assert source_deletion_allowed(ledger_path, plan, unit, source_path)


def test_local_foundry_dry_run_publishes_no_ledger(tmp_path, capsys):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "repository": "moonshotai/Kimi-K3",
                "revision": "9f62e4e",
                "shards": [
                    {"filename": name, "bytes": size, "sha256": digest}
                    for name, size, digest in SHARDS
                ],
            }
        ),
        encoding="utf-8",
    )
    ledger = tmp_path / "immortal.json"

    assert (
        local_foundry_main(
            [
                "--manifest",
                str(manifest),
                "--destination",
                str(__import__("pathlib").Path.cwd()),
                "--staging",
                str(__import__("pathlib").Path.cwd()),
                "--ledger",
                str(ledger),
                "--dry-run",
            ]
        )
        == 0
    )
    report = json.loads(capsys.readouterr().out)
    assert report["format"] == "k3x-local-foundry-dry-run-v1"
    assert report["unit_count"] == 2
    assert report["official_launch_enabled"] is False
    assert not ledger.exists()
