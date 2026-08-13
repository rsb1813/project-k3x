# B-0036 두 레이어 연산별 계측 증거의 트랜잭션과 공식을 검증합니다.
from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path

import pytest

from tests.python.test_official_two_layer_ablation import _manifest
from tests.python.test_official_two_layer_attribution import _attribution_record
from tools.ablate_official_two_layer_operation_attribution import (
    CASES,
    run_ablation,
    verify_summary,
)


def _operation_record(mode: str, warmups: int, iterations: int) -> dict[str, object]:
    record = _attribution_record(mode, warmups, iterations)
    record["schema"] = "k3x-official-two-layer-operation-attribution-v1"
    if mode == "host-round-trip":
        values = (0, 0, 0, 0, 0)
    else:
        values = (800_000, 200_000, 0, 1_500_000, 0)
    (
        record["front_kda_device_nanoseconds"],
        record["front_route_device_nanoseconds"],
        record["front_unclassified_device_nanoseconds"],
        record["tail_ffn_device_nanoseconds"],
        record["tail_unclassified_device_nanoseconds"],
    ) = values
    return record


def _generate(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: tuple[str, object] | None = None,
):
    artifact = tmp_path / "official-two-layer.k3x"
    manifest = tmp_path / "two-layer-route-state-manifest.json"
    oracle = tmp_path / "official-two-layer-oracle-v1.bin"
    runner = tmp_path / "runner"
    artifact.write_bytes(b"artifact")
    oracle.write_bytes(b"oracle" + bytes(13_053_992 - len(b"oracle")))
    manifest_value = _manifest()
    manifest_value["oracle"]["sha256"] = hashlib.sha256(  # type: ignore[index]
        oracle.read_bytes()
    ).hexdigest()
    manifest.write_text(json.dumps(manifest_value), encoding="utf-8")
    runner.write_bytes(b"runner")
    calls: list[list[str]] = []

    def fake_run(command: list[str], **_: object) -> subprocess.CompletedProcess[str]:
        calls.append(command)
        mode = command[command.index("--mode") + 1]
        warmups = int(command[command.index("--warmup") + 1])
        iterations = int(command[command.index("--iterations") + 1])
        record = _operation_record(mode, warmups, iterations)
        if mutation and mode == "device-closure":
            record[mutation[0]] = mutation[1]
        return subprocess.CompletedProcess(command, 0, json.dumps(record), "")

    monkeypatch.setattr(
        "tools.ablate_official_two_layer_operation_attribution.subprocess.run",
        fake_run,
    )
    output = tmp_path / "out"
    summary = run_ablation(
        artifact,
        manifest,
        oracle,
        runner,
        output_dir=output,
        warmups=2,
        iterations=5,
        resident_bytes=4_294_967_296,
    )
    return artifact, manifest, oracle, runner, output, summary, calls


def test_case_order_and_operation_control_are_fixed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    *_, calls = _generate(tmp_path, monkeypatch)

    assert CASES == (
        ("host-round-trip", "host-round-trip"),
        ("device-closure", "device-closure"),
    )
    assert all(
        call[call.index("--operation-attribution") + 1] == "true" for call in calls
    )


def test_run_writes_digest_backed_closed_operation_attribution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact, manifest, oracle, runner, output, summary, _ = _generate(
        tmp_path, monkeypatch
    )

    assert b"\r\n" not in (output / "summary.csv").read_bytes()
    assert summary["benchmark"] == "B-0036"
    assert verify_summary(
        output / "summary.json",
        output / "summary.csv",
        artifact=artifact,
        manifest=manifest,
        oracle=oracle,
        runner=runner,
        strict_official=False,
    ) == summary


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (("front_kda_device_nanoseconds", 800_001), "operation formula"),
        (("tail_ffn_device_nanoseconds", -1), "operation timing"),
        (("decode_tok_s", 5.0), "forbidden metric"),
    ],
)
def test_run_rejects_operation_and_forbidden_mutations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: tuple[str, object],
    message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        _generate(tmp_path, monkeypatch, mutation)


def test_verify_rejects_raw_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact, manifest, oracle, runner, output, _, _ = _generate(
        tmp_path, monkeypatch
    )
    raw = output / "device-closure.json"
    payload = json.loads(raw.read_text(encoding="utf-8"))
    payload["front_route_device_nanoseconds"] += 1
    raw.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="raw JSON digest"):
        verify_summary(
            output / "summary.json",
            output / "summary.csv",
            artifact=artifact,
            manifest=manifest,
            oracle=oracle,
            runner=runner,
            strict_official=False,
        )
