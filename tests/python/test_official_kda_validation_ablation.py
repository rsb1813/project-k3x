# B-0031 공식 KDA validation 행렬과 원자적 증거 계약을 검증합니다.
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from tests.python.test_official_layer_ablation import _manifest, _record
from tools.ablate_official_kda_validation import CASES, run_ablation, verify_summary


KDA_WEIGHT_BYTES = 887_800_832


def _validation_record(
    case: str, validation: str, warmups: int, iterations: int,
    artifact_bytes: int,
) -> dict[str, object]:
    record = _record(case, "resident", warmups, iterations, artifact_bytes)
    calls = 2 if case == "ab-incremental" else 1
    admission = validation == "admission"
    measured_calls = calls * iterations
    record.update({
        "validation": validation,
        "cold_immutable_validation_scans": 14 if admission else 14 * calls,
        "cold_immutable_validation_hits": 14 * (calls - 1) if admission else 0,
        "cold_immutable_validation_bytes": (
            KDA_WEIGHT_BYTES if admission else KDA_WEIGHT_BYTES * calls
        ),
        "cold_immutable_validation_nanoseconds": 11,
        "immutable_validation_scans": 0 if admission else 14 * measured_calls,
        "immutable_validation_hits": 14 * measured_calls if admission else 0,
        "immutable_validation_bytes": (
            0 if admission else KDA_WEIGHT_BYTES * measured_calls
        ),
        "immutable_validation_nanoseconds": 0 if admission else 17,
    })
    return record


def _generate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    mutation: tuple[str, object] | None = None,
):
    artifact = tmp_path / "model.k3x"
    manifest = tmp_path / "routes.json"
    runner = tmp_path / "runner"
    artifact.write_bytes(b"artifact")
    manifest.write_text(json.dumps(_manifest()), encoding="utf-8")
    runner.write_bytes(b"runner")
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        case = command[command.index("--case") + 1]
        validation = command[command.index("--validation") + 1]
        warmups = int(command[command.index("--warmups") + 1])
        iterations = int(command[command.index("--iterations") + 1])
        record = _validation_record(
            case, validation, warmups, iterations, artifact.stat().st_size
        )
        if mutation and (
            mutation[0] != "output_sha256" or validation == "admission"
        ):
            record[mutation[0]] = mutation[1]
        return subprocess.CompletedProcess(command, 0, json.dumps(record), "")

    monkeypatch.setattr(
        "tools.ablate_official_kda_validation.subprocess.run", fake_run
    )
    output = tmp_path / "out"
    summary = run_ablation(
        artifact, manifest, runner, output_dir=output, warmups=2, iterations=5
    )
    return artifact, manifest, runner, output, summary, calls


def test_case_order_is_fixed() -> None:
    assert CASES == (
        ("ab-incremental-resident-per-call", "ab-incremental", "per-call"),
        ("ab-incremental-resident-admission", "ab-incremental", "admission"),
        ("ab-full-resident-per-call", "ab-full", "per-call"),
        ("ab-full-resident-admission", "ab-full", "admission"),
    )


def test_run_writes_digest_backed_lf_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact, manifest, runner, output, summary, calls = _generate(
        tmp_path, monkeypatch
    )
    assert [
        (
            call[call.index("--case") + 1],
            call[call.index("--weight-mode") + 1],
            call[call.index("--validation") + 1],
        )
        for call in calls
    ] == [(case, "resident", validation) for _, case, validation in CASES]
    assert b"\r\n" not in (output / "summary.csv").read_bytes()
    assert summary["artifact_sha256"] == hashlib.sha256(
        artifact.read_bytes()
    ).hexdigest()
    assert verify_summary(
        output / "summary.json", output / "summary.csv", artifact=artifact,
        manifest=manifest, runner=runner, strict_official=False,
    ) == summary


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (("cold_immutable_validation_scans", 0), "validation formula"),
        (("cold_immutable_validation_hits", 1), "validation formula"),
        (("cold_immutable_validation_bytes", 0), "validation formula"),
        (("cold_immutable_validation_nanoseconds", 0), "cold validation time"),
        (("immutable_validation_scans", 1), "validation formula"),
        (("immutable_validation_hits", 1), "validation formula"),
        (("immutable_validation_bytes", 1), "validation formula"),
        (("immutable_validation_nanoseconds", 1), "admission validation time"),
        (("validation", "cached"), "validation identity"),
        (("decode_tok_s", 5.0), "schema"),
    ],
)
def test_run_rejects_schema_formula_and_time_mutations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    mutation: tuple[str, object], message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        _generate(tmp_path, monkeypatch, mutation)
    assert not (tmp_path / "out").exists()
    assert not (tmp_path / ".out.partial").exists()


def test_cross_row_output_parity_is_required(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(RuntimeError, match="cross-row output_sha256 parity"):
        _generate(tmp_path, monkeypatch, ("output_sha256", "f" * 64))


def test_verify_rejects_raw_and_csv_mutations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact, manifest, runner, output, _, _ = _generate(tmp_path, monkeypatch)
    raw = output / "ab-full-resident-admission.json"
    raw.write_bytes(raw.read_bytes() + b" ")
    with pytest.raises(RuntimeError, match="raw JSON digest"):
        verify_summary(
            output / "summary.json", output / "summary.csv", artifact=artifact,
            manifest=manifest, runner=runner, strict_official=False,
        )
    raw.write_bytes(raw.read_bytes()[:-1])
    csv_path = output / "summary.csv"
    with csv_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows, fields = list(reader), tuple(reader.fieldnames or ())
    rows[0]["immutable_validation_hits"] = "1"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(RuntimeError, match="CSV"):
        verify_summary(
            output / "summary.json", csv_path, artifact=artifact,
            manifest=manifest, runner=runner, strict_official=False,
        )


def test_strict_verification_requires_fixed_iteration_identity(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    artifact, manifest, runner, output, _, _ = _generate(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="official iteration gate"):
        verify_summary(
            output / "summary.json", output / "summary.csv", artifact=artifact,
            manifest=manifest, runner=runner,
        )
