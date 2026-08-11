# B-0032 공식 KDA device-state handoff 행렬과 원자적 증거 계약을 검증합니다.
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tests.python.test_official_kda_validation_ablation import (
    _validation_record,
)
from tests.python.test_official_layer_ablation import _manifest
from tools.ablate_official_kda_device_state import (
    CASES,
    run_ablation,
    verify_summary,
)


STATE_BYTES = 6_512_640


def _state_record(
    case: str,
    state_transfer: str,
    warmups: int,
    iterations: int,
    artifact_bytes: int,
) -> dict[str, object]:
    record = _validation_record(
        case, "admission", warmups, iterations, artifact_bytes
    )
    device = state_transfer == "device"
    if device:
        reduction = STATE_BYTES * iterations
        record["activation_h2d_bytes"] -= reduction  # type: ignore[operator]
        record["device_to_host_bytes"] -= reduction  # type: ignore[operator]
        record["official_kda_state_h2d_bytes"] -= reduction  # type: ignore[operator]
        record["official_kda_state_d2h_bytes"] -= reduction  # type: ignore[operator]
    record.update(
        {
            "state_transfer": state_transfer,
            "cold_official_kda_device_state_seeds": 1 if device else 0,
            "cold_official_kda_device_state_continuations": 1 if device else 0,
            "cold_official_kda_device_state_publications": 1 if device else 0,
            "cold_official_kda_device_state_invalidations": 0,
            "official_kda_device_state_seeds": iterations if device else 0,
            "official_kda_device_state_continuations": (
                iterations if device else 0
            ),
            "official_kda_device_state_publications": (
                iterations if device else 0
            ),
            "official_kda_device_state_invalidations": 0,
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
        case = command[command.index("--case") + 1]
        state_transfer = command[command.index("--state-transfer") + 1]
        warmups = int(command[command.index("--warmups") + 1])
        iterations = int(command[command.index("--iterations") + 1])
        record = _state_record(
            case,
            state_transfer,
            warmups,
            iterations,
            artifact.stat().st_size,
        )
        if mutation and (
            mutation[0] != "output_sha256" or state_transfer == "device"
        ):
            record[mutation[0]] = mutation[1]
        return subprocess.CompletedProcess(command, 0, json.dumps(record), "")

    monkeypatch.setattr(
        "tools.ablate_official_kda_device_state.subprocess.run", fake_run
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


def test_case_order_is_fixed() -> None:
    assert CASES == (
        ("ab-incremental-resident-admission-host", "ab-incremental", "host"),
        (
            "ab-incremental-resident-admission-device",
            "ab-incremental",
            "device",
        ),
        ("ab-full-resident-admission-host", "ab-full", "host"),
    )


def test_runner_is_directly_executable_from_repository_root() -> None:
    root = Path(__file__).resolve().parents[2]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = "converter:reference"
    result = subprocess.run(
        [sys.executable, "tools/ablate_official_kda_device_state.py", "--help"],
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
        )
        for call in calls
    ] == [
        (case, "resident", "admission", state_transfer)
        for _, case, state_transfer in CASES
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
        (("state_transfer", "cached"), "state-transfer identity"),
        (("cold_official_kda_device_state_seeds", 0), "state formula"),
        (("cold_official_kda_device_state_continuations", 0), "state formula"),
        (("cold_official_kda_device_state_publications", 0), "state formula"),
        (("cold_official_kda_device_state_invalidations", 1), "state formula"),
        (("official_kda_device_state_seeds", 0), "state formula"),
        (("official_kda_device_state_continuations", 0), "state formula"),
        (("official_kda_device_state_publications", 0), "state formula"),
        (("official_kda_device_state_invalidations", 1), "state formula"),
        (("official_kda_state_h2d_bytes", 0), "state formula"),
        (("official_kda_state_d2h_bytes", 0), "state formula"),
        (("activation_h2d_bytes", 0), "state formula"),
        (("device_to_host_bytes", 0), "state formula"),
        (("decode_tok_s", 5.0), "schema"),
    ],
)
def test_run_rejects_schema_and_state_formula_mutations(
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
    raw = output / "ab-incremental-resident-admission-device.json"
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
    rows[0]["official_kda_device_state_invalidations"] = "1"
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
