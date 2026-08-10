# B-0026 변환 무결성 감사 증거와 검증 경계를 확인합니다.
from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from tools.audit_converter_integrity import (
    main,
    run_converter_integrity_audit,
    verify_evidence,
)


def _csv_records(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return [
            {key: json.loads(value) for key, value in row.items()}
            for row in csv.DictReader(stream)
        ]


def _keys(value: object) -> set[str]:
    if isinstance(value, dict):
        return set(value).union(*(_keys(item) for item in value.values()))
    if isinstance(value, list):
        return set().union(*(_keys(item) for item in value))
    return set()


def test_audit_runs_three_real_converter_scenarios_and_writes_parity_evidence(
    tmp_path: Path
) -> None:
    evidence_dir = tmp_path / "evidence"
    summary = run_converter_integrity_audit(
        evidence_dir,
        environment_label="pytest-wsl",
        chunk_bytes=257,
        stop_after_extents=2,
        orphan_suffix_bytes=8192,
    )
    json_summary = json.loads((evidence_dir / "summary.json").read_text(encoding="utf-8"))
    csv_bytes = (evidence_dir / "summary.csv").read_bytes()
    records = summary["records"]

    assert summary == json_summary
    assert summary["schema"] == "k3x-converter-integrity-audit-v1"
    assert summary["benchmark_id"] == "B-0026"
    assert summary["evidence"] == "measured"
    assert [record["scenario"] for record in records] == [
        "fresh",
        "resume-clean",
        "resume-orphan",
    ]
    assert all(record["maximum_source_read_bytes"] <= 257 for record in records)
    assert {record["output_bytes"] for record in records}
    assert len({record["output_bytes"] for record in records}) == 1
    assert records[0]["reused_extent_count"] == 0
    assert all(record["reused_extent_count"] > 0 for record in records[1:])
    assert all(record["artifact_valid"] is True for record in records)
    assert all(len(record["root_sha256"]) == 64 for record in records)
    assert all(len(record["artifact_sha256"]) == 64 for record in records)
    assert records[0]["committed_prefix_bytes"] == 0
    assert all(record["committed_prefix_bytes"] > 0 for record in records[1:])
    assert all(record["rss_measurement"] == "not-measured" for record in records)
    assert all(record["peak_rss_bytes"] is None for record in records)
    assert _csv_records(evidence_dir / "summary.csv") == records
    assert b"\r" not in csv_bytes
    assert len(summary["runner_sha256"]) == 64
    assert len(summary["aggregate_sha256"]) == 64
    assert len(summary["summary_csv_sha256"]) == 64
    forbidden = ("decode", "prefill", "tok", "ttft", "top_k", "top-k", "specul", "gpu", "vram", "nvme", "h2d", "quality")
    assert not any(token in key.lower() for key in _keys(summary) for token in forbidden)


def test_verify_evidence_and_cli_verify_only_return_the_recorded_summary(
    tmp_path: Path, capsys
) -> None:
    evidence_dir = tmp_path / "evidence"
    expected = run_converter_integrity_audit(evidence_dir, environment_label="pytest-wsl")

    assert verify_evidence(evidence_dir) == expected
    assert main([str(evidence_dir), "--verify-only"]) == 0
    assert json.loads(capsys.readouterr().out) == expected


@pytest.mark.parametrize(
    "mutate",
    [
        lambda contents: contents.replace(b"\n", b"\r\n"),
        lambda contents: contents[:-1],
    ],
    ids=["crlf", "missing-final-lf"],
)
def test_verify_evidence_rejects_non_lf_terminated_summary_json(
    tmp_path: Path, mutate
) -> None:
    evidence_dir = tmp_path / "evidence"
    run_converter_integrity_audit(evidence_dir, environment_label="pytest-wsl")
    summary_json = evidence_dir / "summary.json"
    summary_json.write_bytes(mutate(summary_json.read_bytes()))

    with pytest.raises(ValueError, match="LF"):
        verify_evidence(evidence_dir)


def test_committed_b0026_evidence_matches_current_runner() -> None:
    root = Path(__file__).resolve().parents[2]

    summary = verify_evidence(
        root / "results" / "b0026-converter-integrity-wsl",
        runner=root / "tools" / "audit_converter_integrity.py",
    )

    assert summary["benchmark_id"] == "B-0026"
    assert summary["environment_label"] == (
        "wsl2-ubuntu-24.04.4-synthetic-converter-integrity"
    )
