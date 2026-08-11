# B-0028 공식 expert CUDA ablation의 실행 순서, 수식, digest, CSV 증거를 검증합니다.
from __future__ import annotations

import csv
import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from tools.ablate_official_expert_cuda import CASES, run_ablation, verify_summary


PAYLOAD_BYTES = 17_547_264
ACTIVATION_BYTES = 14_336
B0027_SUMMARY_SHA256 = (
    "57ebd9d85ed3ae55a4e2ab01f023bc451faf02cd7b6e69f478d11e3ea73e982a"
)


def _record(mode: str, warmup: int, iterations: int) -> dict[str, object]:
    resident = mode == "resident"
    return {
        "artifact_kind": "official_kimi_k3_expert",
        "repository": "moonshotai/Kimi-K3",
        "resolved_revision": "9f62e4e9fffbd0a83ddd60e1c209d828994b3569",
        "token_semantics": False,
        "routing_semantics": False,
        "full_moe_layer": False,
        "layer_id": 1,
        "expert_id": 0,
        "weight_mode": mode,
        "k3x_root_sha256": (
            "d585d283325e13e1316a0194c2d6274dd89ef75a28b96b02f02733290b7658be"
        ),
        "ordered_sha256": (
            "4e23bd960dfb5e8b10def10e12a94bac1119500f72918698986bd332d56d33ff"
        ),
        "expert_payload_bytes": PAYLOAD_BYTES,
        "input_elements": 3_584,
        "output_elements": 3_584,
        "warmup": warmup,
        "iterations": iterations,
        "cpu_oracle_nanoseconds": 180_000_000,
        "cold_latency_nanoseconds": 7_000_000,
        "cold_kernel_nanoseconds": 1_900_000,
        "cold_weight_h2d_bytes": PAYLOAD_BYTES,
        "cold_activation_h2d_bytes": ACTIVATION_BYTES,
        "cold_device_to_host_bytes": ACTIVATION_BYTES,
        "latency_nanoseconds_median": 500_000 if resident else 1_800_000,
        "latency_nanoseconds_p05": 450_000 if resident else 1_700_000,
        "latency_nanoseconds_p95": 600_000 if resident else 2_000_000,
        "kernel_nanoseconds": 220_000 * iterations,
        "weight_h2d_bytes": 0 if resident else PAYLOAD_BYTES * iterations,
        "activation_h2d_bytes": ACTIVATION_BYTES * iterations,
        "device_to_host_bytes": ACTIVATION_BYTES * iterations,
        "device_allocation_count": 0,
        "stream_synchronization_count": iterations,
        "weight_cache_hits": 3 * iterations if resident else 0,
        "weight_cache_misses": 0,
        "weight_cache_bypasses": 0,
        "resident_weight_bytes": PAYLOAD_BYTES if resident else 0,
        "peak_resident_weight_bytes": PAYLOAD_BYTES if resident else 0,
        "peak_vram_bytes": 24_000_000 if resident else 6_000_000,
        "maximum_absolute_error": 3.1e-9,
        "all_finite": True,
    }


def _fake_subprocess(
    calls: list[list[str]],
    *,
    mutation: tuple[str, object] | None = None,
):
    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        mode = command[command.index("--weight-mode") + 1]
        warmup = int(command[command.index("--warmup") + 1])
        iterations = int(command[command.index("--iterations") + 1])
        record = _record(mode, warmup, iterations)
        if mutation is not None:
            record[mutation[0]] = mutation[1]
        return subprocess.CompletedProcess(
            command, 0, json.dumps(record, separators=(",", ":")), ""
        )

    return fake_run


def _generate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    mutation: tuple[str, object] | None = None,
) -> tuple[Path, Path, Path, dict[str, object], list[list[str]]]:
    artifact = tmp_path / "expert.k3x"
    runner = tmp_path / "runner"
    output = tmp_path / "b0028"
    artifact.write_bytes(b"official-expert-fixture")
    runner.write_bytes(b"official-expert-runner")
    calls: list[list[str]] = []
    monkeypatch.setattr(
        "tools.ablate_official_expert_cuda.subprocess.run",
        _fake_subprocess(calls, mutation=mutation),
    )
    summary = run_ablation(
        artifact,
        runner,
        output_dir=output,
        warmup=2,
        iterations=5,
    )
    return artifact, runner, output, summary, calls


def _rewrite_summary(path: Path, summary: dict[str, object]) -> None:
    path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def test_case_order_is_fixed() -> None:
    assert CASES == (("transient", "transient"), ("resident", "resident"))


def test_run_ablation_forwards_cases_and_writes_digest_backed_lf_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact, runner, output, summary, calls = _generate(tmp_path, monkeypatch)

    expected_prefix = [str(runner.resolve()), "--model", str(artifact.resolve())]
    assert calls == [
        [
            *expected_prefix,
            "--weight-mode",
            mode,
            "--warmup",
            "2",
            "--iterations",
            "5",
        ]
        for _, mode in CASES
    ]
    assert [record["name"] for record in summary["records"]] == [
        name for name, _ in CASES
    ]
    assert summary["source_b0027_summary_sha256"] == B0027_SUMMARY_SHA256
    assert summary["artifact_sha256"] == hashlib.sha256(
        artifact.read_bytes()
    ).hexdigest()
    assert summary["runner_sha256"] == hashlib.sha256(
        runner.read_bytes()
    ).hexdigest()
    aggregate = json.dumps(
        summary["records"], sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    assert summary["aggregate_sha256"] == hashlib.sha256(aggregate).hexdigest()
    assert summary["summary_csv_sha256"] == hashlib.sha256(
        (output / "summary.csv").read_bytes()
    ).hexdigest()
    assert b"\r\n" not in (output / "summary.csv").read_bytes()

    for record in summary["records"]:
        raw = output / f"{record['name']}.json"
        assert raw.read_bytes().endswith(b"\n")
        assert b"\r\n" not in raw.read_bytes()
        assert record["raw_json_sha256"] == hashlib.sha256(
            raw.read_bytes()
        ).hexdigest()
        payload = json.loads(raw.read_text(encoding="utf-8"))
        assert payload == {
            key: value
            for key, value in record.items()
            if key not in {"name", "raw_json_sha256"}
        }

    assert verify_summary(
        output / "summary.json",
        output / "summary.csv",
        artifact=artifact,
        runner=runner,
        strict_official=False,
    ) == summary
    with pytest.raises(RuntimeError, match="requires artifact and runner"):
        verify_summary(
            output / "summary.json",
            output / "summary.csv",
        )
    with pytest.raises(RuntimeError, match="official artifact identity"):
        verify_summary(
            output / "summary.json",
            output / "summary.csv",
            artifact=artifact,
            runner=runner,
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (("ordered_sha256", "0" * 64), "identity field ordered_sha256"),
        (("decode_tok_s", 5.0), "forbidden metric"),
        (("maximum_absolute_error", 1.1e-6), "numerical divergence"),
    ],
)
def test_run_ablation_rejects_identity_forbidden_and_parity_mutations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: tuple[str, object],
    message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        _generate(tmp_path, monkeypatch, mutation=mutation)


@pytest.mark.parametrize(
    ("mode", "field", "value", "message"),
    [
        ("transient", "weight_h2d_bytes", 1, "transient traffic"),
        ("resident", "weight_h2d_bytes", 1, "resident traffic"),
    ],
)
def test_run_ablation_rejects_mode_specific_weight_traffic(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
    field: str,
    value: object,
    message: str,
) -> None:
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        selected = command[command.index("--weight-mode") + 1]
        record = _record(selected, 2, 5)
        if selected == mode:
            record[field] = value
        return subprocess.CompletedProcess(command, 0, json.dumps(record), "")

    artifact = tmp_path / "expert.k3x"
    runner = tmp_path / "runner"
    artifact.write_bytes(b"artifact")
    runner.write_bytes(b"runner")
    monkeypatch.setattr(
        "tools.ablate_official_expert_cuda.subprocess.run", fake_run
    )
    with pytest.raises(RuntimeError, match=message):
        run_ablation(
            artifact,
            runner,
            output_dir=tmp_path / "bad",
            warmup=2,
            iterations=5,
        )


def test_verify_summary_rejects_raw_digest_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact, runner, output, _, _ = _generate(tmp_path, monkeypatch)
    raw = output / "transient.json"
    raw.write_bytes(raw.read_bytes() + b" ")
    with pytest.raises(RuntimeError, match="raw JSON digest"):
        verify_summary(
            output / "summary.json",
            output / "summary.csv",
            artifact=artifact,
            runner=runner,
            strict_official=False,
        )


def test_verify_summary_rejects_csv_parity_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact, runner, output, summary, _ = _generate(tmp_path, monkeypatch)
    csv_path = output / "summary.csv"
    with csv_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fieldnames = tuple(reader.fieldnames or ())
    rows[0]["kernel_nanoseconds"] = "1"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    summary["summary_csv_sha256"] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    _rewrite_summary(output / "summary.json", summary)
    with pytest.raises(RuntimeError, match="summary CSV parity"):
        verify_summary(
            output / "summary.json",
            csv_path,
            artifact=artifact,
            runner=runner,
            strict_official=False,
        )


def test_verify_summary_rejects_case_order_even_with_rehashed_aggregate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact, runner, output, summary, _ = _generate(tmp_path, monkeypatch)
    records = summary["records"]
    assert isinstance(records, list)
    records.reverse()
    aggregate = json.dumps(
        records, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    summary["aggregate_sha256"] = hashlib.sha256(aggregate).hexdigest()
    _rewrite_summary(output / "summary.json", summary)
    with pytest.raises(RuntimeError, match="case order"):
        verify_summary(
            output / "summary.json",
            output / "summary.csv",
            artifact=artifact,
            runner=runner,
            strict_official=False,
        )


def test_committed_b0028_evidence_is_self_consistent() -> None:
    root = Path(__file__).resolve().parents[2]
    output = root / "results" / "b0028-official-expert-cuda-wsl"
    summary = verify_summary(
        output / "summary.json",
        output / "summary.csv",
        strict_official=False,
    )

    assert summary["warmup"] == 3
    assert summary["iterations"] == 20
    assert summary["artifact_sha256"] == (
        "e08293cd854ed11913bd8f1bc3a51d1eb577202fd5fd9b5b7e3c96ef1bccecc7"
    )
    assert summary["runner_sha256"] == (
        "48f0f295ab7299af07f261522ffd2999814bd5967e12bfcc3e7b0b3d21b201fa"
    )
    assert summary["aggregate_sha256"] == (
        "eb4580b74481855d04fdf9d3f7ed5921ea25b0e5b56408d561cd645a3ea99172"
    )
    assert summary["summary_csv_sha256"] == (
        "d339a8774283e49608393172ffd551d46692a076e00cb4d63e1e2a347ae42a91"
    )
    assert [record["raw_json_sha256"] for record in summary["records"]] == [
        "3b39610b5f5b6f4cfd5ec1da1bc3588e00c0af62f58438c81a7f9b3357093518",
        "79c935869226108431f391bb61402e10b61a616493720bac75fc545512cc30bf",
    ]
