# B-0029 runner의 고정 실행 순서, 수식, digest, CSV와 금지 metric을 검증합니다.
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from tools.ablate_official_moe import CASES, run_ablation, verify_summary


HIDDEN = 7_168
TOP_K = 16
EXPERT_BYTES = 17_547_264
COMMON_BYTES = (
    3_584 * 7_168 * 2 + 3_584 * 2 + 7_168 * 3_584 * 2
    + 6_144 * 7_168 * 2 + 6_144 * 7_168 * 2 + 7_168 * 6_144 * 2
)
ACTIVATION_BYTES = 2 * HIDDEN * 4 + TOP_K * 4 + 3 * TOP_K * 16
D2H_BYTES = HIDDEN * 4


def _manifest() -> dict[str, object]:
    route_a = list(range(16))
    route_b = list(range(8, 24))
    contribution = [1.0 / 16.0] * 16
    return {
        "repository": "moonshotai/Kimi-K3",
        "resolved_revision": "9f62e4e9fffbd0a83ddd60e1c209d828994b3569",
        "artifact": {"k3x_root_sha256": "a" * 64},
        "selected_experts": list(range(24)),
        "routes": [
            {"expert_ids": route_a, "contributions": contribution},
            {"expert_ids": route_b, "contributions": contribution},
        ],
    }


def _record(case: str, mode: str, warmup: int, iterations: int,
            artifact_bytes: int) -> dict[str, object]:
    alternating = case == "alternating"
    calls = 2 if alternating else 1
    total_calls = calls * iterations
    resident = mode == "resident"
    selected = 24 if alternating else 16
    cold_bf16 = COMMON_BYTES if resident else calls * COMMON_BYTES
    cold_mxfp4 = selected * EXPERT_BYTES
    resident_bytes = COMMON_BYTES + selected * EXPERT_BYTES if resident else 0
    route_a = list(range(16))
    route_b = list(range(8, 24))
    contribution = [1.0 / 16.0] * 16
    return {
        "artifact_kind": "official_kimi_k3_moe_ffn",
        "repository": "moonshotai/Kimi-K3",
        "resolved_revision": "9f62e4e9fffbd0a83ddd60e1c209d828994b3569",
        "case": case, "weight_mode": mode, "token_semantics": False,
        "routing_semantics": True, "full_moe_ffn": True,
        "full_transformer_layer": False, "quality_measured": False,
        "k3x_root_sha256": "a" * 64, "warmup": warmup,
        "iterations": iterations, "input_elements": HIDDEN,
        "output_elements": HIDDEN, "selected_union": list(range(24)),
        "route_a": route_a, "route_b": route_b,
        "route_a_contributions": contribution,
        "route_b_contributions": contribution,
        "source_bytes": 379_900_416 + 24 * EXPERT_BYTES,
        "k3x_bytes": artifact_bytes,
        "cpu_oracle_nanoseconds": 10, "attention_residual_nanoseconds": 2,
        "router_nanoseconds": 3, "cold_latency_nanoseconds": 20,
        "cold_kernel_nanoseconds": 10,
        "cold_weight_h2d_bytes": cold_bf16 + cold_mxfp4,
        "cold_bf16_weight_h2d_bytes": cold_bf16,
        "cold_mxfp4_weight_h2d_bytes": cold_mxfp4,
        "latency_nanoseconds_p05": 10,
        "latency_nanoseconds_median": 12,
        "latency_nanoseconds_p95": 15,
        "kernel_nanoseconds": 7 * total_calls,
        "orchestration_nanoseconds": 5 * total_calls,
        "weight_h2d_bytes": 0 if resident else total_calls * (COMMON_BYTES + TOP_K * EXPERT_BYTES),
        "bf16_weight_h2d_bytes": 0 if resident else total_calls * COMMON_BYTES,
        "mxfp4_weight_h2d_bytes": 0 if resident else total_calls * TOP_K * EXPERT_BYTES,
        "activation_h2d_bytes": total_calls * ACTIVATION_BYTES,
        "device_to_host_bytes": total_calls * D2H_BYTES,
        "resident_weight_bytes": resident_bytes,
        "peak_resident_weight_bytes": resident_bytes,
        "weight_cache_hits": total_calls * 54 if resident else 0,
        "weight_cache_misses": 0, "weight_cache_bypasses": 0,
        "device_allocation_count": 0 if resident else total_calls * 102,
        "stream_synchronization_count": total_calls,
        "peak_vram_bytes": 1_000, "maximum_absolute_error": 1.0e-3,
        "all_finite": True,
    }


def _generate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
              mutation: tuple[str, object] | None = None):
    artifact, manifest, runner = (tmp_path / name for name in ("model.k3x", "routes.json", "runner"))
    artifact.write_bytes(b"artifact")
    manifest.write_text(json.dumps(_manifest()), encoding="utf-8")
    runner.write_bytes(b"runner")
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        case = command[command.index("--case") + 1]
        mode = command[command.index("--weight-mode") + 1]
        warmup = int(command[command.index("--warmup") + 1])
        iterations = int(command[command.index("--iterations") + 1])
        record = _record(case, mode, warmup, iterations, artifact.stat().st_size)
        if mutation is not None:
            record[mutation[0]] = mutation[1]
        return subprocess.CompletedProcess(command, 0, json.dumps(record), "")

    monkeypatch.setattr("tools.ablate_official_moe.subprocess.run", fake_run)
    output = tmp_path / "out"
    summary = run_ablation(artifact, manifest, runner, output_dir=output,
                           warmup=2, iterations=5)
    return artifact, manifest, runner, output, summary, calls


def _rewrite(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n",
                    encoding="utf-8", newline="\n")


def test_case_order_is_fixed() -> None:
    assert CASES == (
        ("a-transient", "a", "transient"),
        ("a-resident", "a", "resident"),
        ("alternating-resident", "alternating", "resident"),
    )


def test_run_ablation_writes_fixed_digest_backed_lf_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact, manifest, runner, output, summary, calls = _generate(tmp_path, monkeypatch)
    assert [(call[call.index("--case") + 1], call[call.index("--weight-mode") + 1])
            for call in calls] == [(case, mode) for _, case, mode in CASES]
    assert all(call[call.index("--warmup") + 1] == "2" for call in calls)
    assert all(call[call.index("--iterations") + 1] == "5" for call in calls)
    assert summary["artifact_sha256"] == hashlib.sha256(artifact.read_bytes()).hexdigest()
    assert summary["manifest_sha256"] == hashlib.sha256(manifest.read_bytes()).hexdigest()
    assert summary["runner_sha256"] == hashlib.sha256(runner.read_bytes()).hexdigest()
    assert b"\r\n" not in (output / "summary.csv").read_bytes()
    assert verify_summary(output / "summary.json", output / "summary.csv",
                          artifact=artifact, manifest=manifest, runner=runner,
                          strict_official=False) == summary
    assert verify_summary(output / "summary.json", output / "summary.csv",
                          strict_official=False) == summary
    with pytest.raises(RuntimeError, match="requires artifact"):
        verify_summary(output / "summary.json", output / "summary.csv")
    with pytest.raises(RuntimeError, match="iteration gate"):
        verify_summary(output / "summary.json", output / "summary.csv",
                       artifact=artifact, manifest=manifest, runner=runner)


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (("decode_tok_s", 5.0), "forbidden metric"),
        (("k3x_root_sha256", "b" * 64), "identity field"),
        (("maximum_absolute_error", 0.021), "numerical divergence"),
        (("device_to_host_bytes", 1), "common traffic"),
        (("all_finite", False), "identity field"),
    ],
)
def test_run_ablation_rejects_schema_identity_formula_and_finite_mutations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
    mutation: tuple[str, object], message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        _generate(tmp_path, monkeypatch, mutation)


def test_run_ablation_rejects_resident_warm_weight_h2d(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact, manifest, runner = (tmp_path / name for name in ("m", "j", "r"))
    artifact.write_bytes(b"artifact")
    manifest.write_text(json.dumps(_manifest()), encoding="utf-8")
    runner.write_bytes(b"runner")

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        case = command[command.index("--case") + 1]
        mode = command[command.index("--weight-mode") + 1]
        record = _record(case, mode, 2, 5, artifact.stat().st_size)
        if mode == "resident":
            record["weight_h2d_bytes"] = 1
        return subprocess.CompletedProcess(command, 0, json.dumps(record), "")

    monkeypatch.setattr("tools.ablate_official_moe.subprocess.run", fake_run)
    with pytest.raises(RuntimeError, match="resident traffic"):
        run_ablation(artifact, manifest, runner, output_dir=tmp_path / "out",
                     warmup=2, iterations=5)


@pytest.mark.parametrize(("drift", "accepted"), [(5.0e-7, True), (1.1e-6, False)])
def test_route_contribution_uses_the_harness_tolerance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, drift: float, accepted: bool
) -> None:
    artifact, manifest, runner = (tmp_path / name for name in ("m", "j", "r"))
    artifact.write_bytes(b"artifact")
    manifest.write_text(json.dumps(_manifest()), encoding="utf-8")
    runner.write_bytes(b"runner")

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        case = command[command.index("--case") + 1]
        mode = command[command.index("--weight-mode") + 1]
        record = _record(case, mode, 2, 5, artifact.stat().st_size)
        values = list(record["route_a_contributions"])
        values[0] += drift
        record["route_a_contributions"] = values
        return subprocess.CompletedProcess(command, 0, json.dumps(record), "")

    monkeypatch.setattr("tools.ablate_official_moe.subprocess.run", fake_run)
    if accepted:
        run_ablation(artifact, manifest, runner, output_dir=tmp_path / "out",
                     warmup=2, iterations=5)
    else:
        with pytest.raises(RuntimeError, match="route contribution"):
            run_ablation(artifact, manifest, runner, output_dir=tmp_path / "out",
                         warmup=2, iterations=5)


def test_verify_rejects_raw_csv_aggregate_and_case_order_mutations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact, manifest, runner, output, summary, _ = _generate(tmp_path, monkeypatch)
    raw = output / "a-transient.json"
    raw.write_bytes(raw.read_bytes() + b" ")
    with pytest.raises(RuntimeError, match="raw JSON digest"):
        verify_summary(output / "summary.json", output / "summary.csv",
                       artifact=artifact, manifest=manifest, runner=runner,
                       strict_official=False)
    raw.write_bytes(raw.read_bytes()[:-1])

    records = summary["records"]
    assert isinstance(records, list)
    records.reverse()
    summary["aggregate_sha256"] = hashlib.sha256(
        json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    _rewrite(output / "summary.json", summary)
    with pytest.raises(RuntimeError, match="case order"):
        verify_summary(output / "summary.json", output / "summary.csv",
                       artifact=artifact, manifest=manifest, runner=runner,
                       strict_official=False)


def test_verify_rejects_csv_parity_even_after_digest_rehash(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact, manifest, runner, output, summary, _ = _generate(tmp_path, monkeypatch)
    csv_path = output / "summary.csv"
    with csv_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows, fields = list(reader), tuple(reader.fieldnames or ())
    rows[0]["kernel_nanoseconds"] = "1"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary["summary_csv_sha256"] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    _rewrite(output / "summary.json", summary)
    with pytest.raises(RuntimeError, match="CSV parity"):
        verify_summary(output / "summary.json", csv_path, artifact=artifact,
                       manifest=manifest, runner=runner, strict_official=False)
