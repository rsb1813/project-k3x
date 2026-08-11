# B-0033 공식 MoE device-routing 행렬과 원자적 증거 계약을 검증합니다.
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.python.test_official_kda_device_state_ablation import _state_record
from tests.python.test_official_layer_ablation import _manifest
from tools.ablate_official_moe_device_routing import (
    CASES,
    run_ablation,
    verify_summary,
)


ROUTE_WEIGHT_BYTES = 12_888_064
ROUTER_LOGIT_BYTES = 896 * 4
PREPARED_SLOT_BYTES = 2 * 7_168 * 4


def _route_record(
    route_preparation: str,
    warmups: int,
    iterations: int,
    artifact_bytes: int,
) -> dict[str, object]:
    record = _state_record(
        "ab-incremental", "device", warmups, iterations, artifact_bytes
    )
    device = route_preparation == "device"
    if device:
        record["cold_weight_h2d_bytes"] += ROUTE_WEIGHT_BYTES  # type: ignore[operator]
        record["cold_bf16_weight_h2d_bytes"] += ROUTE_WEIGHT_BYTES  # type: ignore[operator]
        record["resident_weight_bytes"] += ROUTE_WEIGHT_BYTES  # type: ignore[operator]
        record["peak_resident_weight_bytes"] += ROUTE_WEIGHT_BYTES  # type: ignore[operator]
        record["device_to_host_bytes"] += ROUTER_LOGIT_BYTES * 2 * iterations  # type: ignore[operator]
        record["stream_synchronization_count"] += 2 * iterations  # type: ignore[operator]
        record["weight_cache_hits"] += 8 * iterations  # type: ignore[operator]
        record["cold_immutable_validation_scans"] += 4  # type: ignore[operator]
        record["cold_immutable_validation_hits"] += 4  # type: ignore[operator]
        record["cold_immutable_validation_bytes"] += ROUTE_WEIGHT_BYTES  # type: ignore[operator]
        record["immutable_validation_hits"] += 8 * iterations  # type: ignore[operator]
    calls = 2 if device else 0
    record.update(
        {
            "route_preparation": route_preparation,
            "cold_official_moe_route_prepare_calls": calls,
            "cold_official_moe_route_prepare_kernel_launches": calls * 2,
            "cold_official_moe_router_logit_d2h_bytes": (
                ROUTER_LOGIT_BYTES * calls
            ),
            "cold_official_moe_prepared_seeds": calls,
            "cold_official_moe_prepared_consumes": calls,
            "cold_official_moe_prepared_discards": 0,
            "cold_official_moe_prepared_invalidations": 0,
            "official_moe_route_prepare_calls": calls * iterations,
            "official_moe_route_prepare_kernel_launches": calls * 2 * iterations,
            "official_moe_router_logit_d2h_bytes": (
                ROUTER_LOGIT_BYTES * calls * iterations
            ),
            "official_moe_prepared_seeds": calls * iterations,
            "official_moe_prepared_consumes": calls * iterations,
            "official_moe_prepared_discards": 0,
            "official_moe_prepared_invalidations": 0,
            "official_moe_prepared_slot_bytes": (
                PREPARED_SLOT_BYTES if device else 0
            ),
        }
    )
    return record


def _generate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: tuple[str, object] | None = None,
):
    artifact = tmp_path / "model.k3x"
    manifest = tmp_path / "routes.json"
    runner = tmp_path / "runner"
    artifact.write_bytes(b"artifact")
    manifest.write_text(json.dumps(_manifest()), encoding="utf-8")
    runner.write_bytes(b"runner")
    calls: list[list[str]] = []

    def fake_run(
        command: list[str], **_: object
    ) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        route = command[command.index("--route-preparation") + 1]
        warmups = int(command[command.index("--warmups") + 1])
        iterations = int(command[command.index("--iterations") + 1])
        record = _route_record(
            route, warmups, iterations, artifact.stat().st_size
        )
        device_only = {
            "output_sha256",
            "weight_h2d_bytes",
            "resident_weight_bytes",
            "device_to_host_bytes",
        }
        if mutation and (mutation[0] not in device_only or route == "device"):
            record[mutation[0]] = mutation[1]
        return subprocess.CompletedProcess(command, 0, json.dumps(record), "")

    monkeypatch.setattr(
        "tools.ablate_official_moe_device_routing.subprocess.run", fake_run
    )
    output = tmp_path / "out"
    summary = run_ablation(
        artifact,
        manifest,
        runner,
        output_dir=output,
        warmups=2,
        iterations=5,
    )
    return artifact, manifest, runner, output, summary, calls


def test_case_order_and_controls_are_fixed() -> None:
    assert CASES == (
        ("ab-incremental-device-state-host-route", "host"),
        ("ab-incremental-device-state-device-route", "device"),
    )


def test_runner_is_directly_executable_from_repository_root() -> None:
    root = Path(__file__).resolve().parents[2]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = "converter:reference"
    result = subprocess.run(
        [
            sys.executable,
            "tools/ablate_official_moe_device_routing.py",
            "--help",
        ],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--verify-existing" in result.stdout


def test_run_writes_digest_backed_lf_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact, manifest, runner, output, summary, calls = _generate(
        tmp_path, monkeypatch
    )
    assert [
        (
            call[call.index("--case") + 1],
            call[call.index("--weight-mode") + 1],
            call[call.index("--validation") + 1],
            call[call.index("--state-transfer") + 1],
            call[call.index("--route-preparation") + 1],
        )
        for call in calls
    ] == [
        ("ab-incremental", "resident", "admission", "device", route)
        for _, route in CASES
    ]
    assert b"\r\n" not in (output / "summary.csv").read_bytes()
    assert summary["artifact_sha256"] == hashlib.sha256(
        artifact.read_bytes()
    ).hexdigest()
    assert (
        verify_summary(
            output / "summary.json",
            output / "summary.csv",
            artifact=artifact,
            manifest=manifest,
            runner=runner,
            strict_official=False,
        )
        == summary
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (("route_preparation", "cached"), "route-preparation identity"),
        (("cold_official_moe_route_prepare_calls", 0), "route formula"),
        (("official_moe_route_prepare_kernel_launches", 0), "route formula"),
        (("official_moe_router_logit_d2h_bytes", 0), "route formula"),
        (("official_moe_prepared_seeds", 0), "route formula"),
        (("official_moe_prepared_consumes", 0), "route formula"),
        (("official_moe_prepared_discards", 1), "route formula"),
        (("official_moe_prepared_invalidations", 1), "route formula"),
        (("official_moe_prepared_slot_bytes", 0), "route formula"),
        (("resident_weight_bytes", 1), "route formula"),
        (("device_to_host_bytes", 0), "route formula"),
        (("weight_h2d_bytes", 1), "route formula"),
        (("decode_tok_s", 5.0), "schema"),
    ],
)
def test_run_rejects_schema_and_route_formula_mutations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: tuple[str, object],
    message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        _generate(tmp_path, monkeypatch, mutation)
    assert not (tmp_path / "out").exists()
    assert not (tmp_path / ".out.partial").exists()


def test_cross_row_output_parity_is_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(RuntimeError, match="cross-row output_sha256 parity"):
        _generate(tmp_path, monkeypatch, ("output_sha256", "f" * 64))


def test_verify_rejects_raw_and_csv_mutations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact, manifest, runner, output, _, _ = _generate(tmp_path, monkeypatch)
    raw = output / "ab-incremental-device-state-device-route.json"
    raw.write_bytes(raw.read_bytes() + b" ")
    with pytest.raises(RuntimeError, match="raw JSON digest"):
        verify_summary(
            output / "summary.json",
            output / "summary.csv",
            artifact=artifact,
            manifest=manifest,
            runner=runner,
            strict_official=False,
        )
    raw.write_bytes(raw.read_bytes()[:-1])
    csv_path = output / "summary.csv"
    with csv_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows, fields = list(reader), tuple(reader.fieldnames or ())
    rows[0]["official_moe_prepared_invalidations"] = "1"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(RuntimeError, match="CSV"):
        verify_summary(
            output / "summary.json",
            csv_path,
            artifact=artifact,
            manifest=manifest,
            runner=runner,
            strict_official=False,
        )


def test_strict_verification_requires_fixed_iteration_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact, manifest, runner, output, _, _ = _generate(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="official iteration gate"):
        verify_summary(
            output / "summary.json",
            output / "summary.csv",
            artifact=artifact,
            manifest=manifest,
            runner=runner,
        )
