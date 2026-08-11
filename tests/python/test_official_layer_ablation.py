# B-0030 complete-layer runner의 고정 순서와 증거 무결성을 검증합니다.
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from tools.ablate_official_layer import CASES, run_ablation, verify_summary


ROUTE_A = list(range(16))
ROUTE_B = list(range(16, 32))
CONTRIBUTIONS = [1.0 / 16.0] * 16
STATE_BYTES = 6_512_640
KDA_WEIGHT_BYTES = 887_800_832
KDA_F32_BYTES = 640_000
KDA_BF16_BYTES = KDA_WEIGHT_BYTES - KDA_F32_BYTES
MOE_COMMON_BYTES = 367_008_768
EXPERT_BYTES = 17_547_264
RESIDENT_BYTES = 1_816_322_048


def _manifest() -> dict[str, object]:
    return {
        "repository": "moonshotai/Kimi-K3",
        "resolved_revision": "9f62e4e9fffbd0a83ddd60e1c209d828994b3569",
        "selected_experts": [*ROUTE_A, *ROUTE_B],
        "steps": [
            {"expert_ids": ROUTE_A, "contributions": CONTRIBUTIONS},
            {"expert_ids": ROUTE_B, "contributions": CONTRIBUTIONS},
        ],
        "artifact": {"k3x_root_sha256": "a" * 64},
    }


def _record(case: str, mode: str, warmups: int, iterations: int,
            artifact_bytes: int) -> dict[str, object]:
    tokens = 1 if case == "a" else 2
    kda_calls = 2 if case == "ab-incremental" else 1
    state_bytes = STATE_BYTES * kda_calls * iterations
    kda_output = 28_672 * tokens * iterations
    moe_output = 28_672 * tokens * iterations
    activation = (
        (STATE_BYTES * kda_calls + 28_672 * tokens + 58_176 * tokens)
        * iterations
    )
    d2h = state_bytes + kda_output + moe_output
    resident = mode == "resident"
    weight = 0 if resident else (
        KDA_WEIGHT_BYTES + MOE_COMMON_BYTES + 16 * EXPERT_BYTES
    ) * iterations
    bf16_weight = 0 if resident else (
        KDA_BF16_BYTES + MOE_COMMON_BYTES
    ) * iterations
    f32_weight = 0 if resident else KDA_F32_BYTES * iterations
    mxfp4_weight = 0 if resident else 16 * EXPERT_BYTES * iterations
    cold_experts = 16 if case == "a" else 32
    return {
        "artifact_kind": "official_kimi_k3_kda_layer",
        "repository": "moonshotai/Kimi-K3",
        "resolved_revision": "9f62e4e9fffbd0a83ddd60e1c209d828994b3569",
        "case": case,
        "weight_mode": mode,
        "token_semantics": False,
        "routing_semantics": True,
        "full_transformer_layer": True,
        "quality_measured": False,
        "k3x_root_sha256": "a" * 64,
        "warmups": warmups,
        "iterations": iterations,
        "selected_union": [*ROUTE_A, *ROUTE_B],
        "route_a": ROUTE_A,
        "route_b": ROUTE_B,
        "route_a_contributions": CONTRIBUTIONS,
        "route_b_contributions": CONTRIBUTIONS,
        "output_sha256": "b" * 64 if case == "a" else "c" * 64,
        "state_sha256": "d" * 64 if case == "a" else "e" * 64,
        "source_bytes": 1_829_256_704,
        "k3x_bytes": artifact_bytes,
        "cold_latency_nanoseconds": 20,
        "cold_kernel_nanoseconds": 10,
        "cold_weight_h2d_bytes": (
            KDA_WEIGHT_BYTES + MOE_COMMON_BYTES +
            cold_experts * EXPERT_BYTES
        ),
        "cold_bf16_weight_h2d_bytes": KDA_BF16_BYTES + MOE_COMMON_BYTES,
        "cold_f32_weight_h2d_bytes": KDA_F32_BYTES,
        "cold_mxfp4_weight_h2d_bytes": cold_experts * EXPERT_BYTES,
        "latency_nanoseconds_p05": 10,
        "latency_nanoseconds_median": 12,
        "latency_nanoseconds_p95": 15,
        "kernel_nanoseconds": 7 * iterations,
        "orchestration_nanoseconds": 5 * iterations,
        "weight_h2d_bytes": weight,
        "bf16_weight_h2d_bytes": bf16_weight,
        "f32_weight_h2d_bytes": f32_weight,
        "mxfp4_weight_h2d_bytes": mxfp4_weight,
        "activation_h2d_bytes": activation,
        "device_to_host_bytes": d2h,
        "official_kda_calls": kda_calls * iterations,
        "official_kda_kernel_launches": (
            24 if case == "ab-full" else 16 * kda_calls
        ) * iterations,
        "official_kda_state_h2d_bytes": state_bytes,
        "official_kda_state_d2h_bytes": state_bytes,
        "official_kda_output_d2h_bytes": kda_output,
        "resident_weight_bytes": RESIDENT_BYTES if resident else 0,
        "peak_resident_weight_bytes": RESIDENT_BYTES if resident else 0,
        "weight_cache_hits": (
            (136 if case == "ab-incremental" else 122) * iterations
            if resident else 0
        ),
        "weight_cache_misses": 0,
        "weight_cache_bypasses": 0,
        "device_allocation_count": 0 if resident else 40 * iterations,
        "stream_synchronization_count": (kda_calls + tokens) * iterations,
        "peak_vram_bytes": 1_900_000_000 if resident else 800_000_000,
        "process_peak_rss_bytes": 2_000_000_000,
        "reader_read_calls": 60,
        "reader_requested_bytes": 1_829_256_704,
        "reader_completed_bytes": 1_829_256_704,
        "reader_storage_submitted_bytes": 1_829_256_704,
        "reader_storage_completed_bytes": 1_829_256_704,
        "maximum_absolute_error": 0.00048828125,
        "all_finite": True,
    }


def _generate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
              mutation: tuple[str, object] | None = None):
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
        mode = command[command.index("--weight-mode") + 1]
        warmups = int(command[command.index("--warmups") + 1])
        iterations = int(command[command.index("--iterations") + 1])
        record = _record(case, mode, warmups, iterations, artifact.stat().st_size)
        if mutation and (mutation[0] != "state_sha256" or case == "ab-full"):
            record[mutation[0]] = mutation[1]
        return subprocess.CompletedProcess(command, 0, json.dumps(record), "")

    monkeypatch.setattr("tools.ablate_official_layer.subprocess.run", fake_run)
    output = tmp_path / "out"
    summary = run_ablation(
        artifact, manifest, runner, output_dir=output, warmups=2, iterations=5
    )
    return artifact, manifest, runner, output, summary, calls


def test_case_order_is_fixed() -> None:
    assert CASES == (
        ("a-transient", "a", "transient"),
        ("ab-incremental-resident", "ab-incremental", "resident"),
        ("ab-full-resident", "ab-full", "resident"),
    )


def test_run_writes_digest_backed_lf_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact, manifest, runner, output, summary, calls = _generate(
        tmp_path, monkeypatch
    )
    assert [(call[call.index("--case") + 1],
             call[call.index("--weight-mode") + 1]) for call in calls] == [
        (case, mode) for _, case, mode in CASES
    ]
    assert b"\r\n" not in (output / "summary.csv").read_bytes()
    assert summary["artifact_sha256"] == hashlib.sha256(
        artifact.read_bytes()
    ).hexdigest()
    assert verify_summary(
        output / "summary.json", output / "summary.csv", artifact=artifact,
        manifest=manifest, runner=runner, strict_official=False
    ) == summary


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (("decode_tok_s", 5.0), "forbidden metric"),
        (("weight_h2d_bytes", 1), "traffic"),
        (("maximum_absolute_error", 0.021), "numerical"),
        (("all_finite", False), "identity"),
    ],
)
def test_run_rejects_schema_formula_and_finite_mutations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    mutation: tuple[str, object], message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        _generate(tmp_path, monkeypatch, mutation)
    assert not (tmp_path / "out").exists()
    assert not (tmp_path / ".out.partial").exists()


def test_verify_rejects_raw_and_csv_mutations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact, manifest, runner, output, summary, _ = _generate(
        tmp_path, monkeypatch
    )
    raw = output / "a-transient.json"
    raw.write_bytes(raw.read_bytes() + b" ")
    with pytest.raises(RuntimeError, match="raw JSON digest"):
        verify_summary(
            output / "summary.json", output / "summary.csv", artifact=artifact,
            manifest=manifest, runner=runner, strict_official=False
        )
    raw.write_bytes(raw.read_bytes()[:-1])
    csv_path = output / "summary.csv"
    with csv_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fields = tuple(reader.fieldnames or ())
    rows[0]["kernel_nanoseconds"] = "1"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(RuntimeError, match="CSV"):
        verify_summary(
            output / "summary.json", csv_path, artifact=artifact,
            manifest=manifest, runner=runner, strict_official=False
        )


def test_full_and_incremental_rows_require_same_output_and_state(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with pytest.raises(RuntimeError, match="full/incremental parity"):
        _generate(tmp_path, monkeypatch, ("state_sha256", "f" * 64))


def test_committed_b0030_evidence_is_self_consistent() -> None:
    root = Path(__file__).resolve().parents[2]
    output = root / "results" / "b0030-official-layer-wsl"
    summary = verify_summary(
        output / "summary.json", output / "summary.csv", strict_official=False
    )
    assert summary["warmups"] == 3
    assert summary["iterations"] == 20
    assert summary["artifact_sha256"] == (
        "9f0c29fcb18b8cdab5aeeec67d8e5e0113b8dffb7352a2dcdac1ae41ae5198c6"
    )
    assert summary["manifest_sha256"] == (
        "cf0dd554d5dfc7db640cb3313f7527e6c354a6fd74f9011cd747348b247168d4"
    )
    assert summary["runner_sha256"] == (
        "253af0dfa411b771913997f9685c3bb4c5d5877ae68d7fe263eaff6e67f2b1b9"
    )
    assert summary["aggregate_sha256"] == (
        "86f0007af7da007d6646dec6fa8fba4008c1bf7bedff53971d5d31926c9f6452"
    )
    assert summary["summary_csv_sha256"] == (
        "1e5af9bb7d5b9abb16f62962bbce3584b62014873b12ce7642868e919770a635"
    )
    records = {record["name"]: record for record in summary["records"]}
    assert records["a-transient"]["latency_nanoseconds_median"] == 262_801_334
    assert records["ab-incremental-resident"]["latency_nanoseconds_median"] == 168_577_563
    assert records["ab-full-resident"]["latency_nanoseconds_median"] == 114_804_882
    assert records["ab-incremental-resident"]["weight_h2d_bytes"] == 0
    assert records["ab-full-resident"]["weight_h2d_bytes"] == 0
    assert records["ab-incremental-resident"]["output_sha256"] == (
        records["ab-full-resident"]["output_sha256"]
    )
    assert records["ab-incremental-resident"]["state_sha256"] == (
        records["ab-full-resident"]["state_sha256"]
    )
    assert all(record["maximum_absolute_error"] == 0.00048828125
               for record in records.values())
