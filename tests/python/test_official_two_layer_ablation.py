# B-0034 공식 2레이어 closure 행렬과 원자적 증거 계약을 검증합니다.
from __future__ import annotations

import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from tools.ablate_official_two_layer_closure import (
    CASES,
    run_ablation,
    verify_summary,
)


HIDDEN_BYTES = 2 * 7_168 * 4
STATE_BYTES = 2 * 6_512_640
KDA_OUTPUT_BYTES = 4 * 7_168 * 4
ROUTER_BYTES = 4 * 896 * 4
RESIDENT_BYTES = 2 * 1_816_322_048
ROUTE_PREPARATION_BYTES = 2 * 12_888_064
DEVICE_FRONT_BYTES = 2 * 3 * 7_168 * 2


def _manifest() -> dict[str, object]:
    routes = [list(range(offset, offset + 16)) for offset in (0, 16, 32, 48)]
    contributions = [[1.0 / 16] * 16 for _ in range(4)]
    states = [f"{value:x}" * 64 for value in (3, 4, 5, 6)]
    steps = []
    for index, (position, layer) in enumerate(
        (("a", 1), ("a", 2), ("b", 1), ("b", 2))
    ):
        steps.append(
            {
                "position": position,
                "layer_id": layer,
                "hidden_input_sha256": (
                    f"{index + 7:x}" * 64
                    if layer == 1
                    else f"{index + 8:x}" * 64
                ),
                "block_sha256": "b" * 64,
                "consumes_state_sha256": (
                    "1" * 64 if position == "a" else states[layer - 1]
                ),
                "state_sha256": states[index],
                "kda_output_sha256": "c" * 64,
                "expert_ids": routes[index],
                "contributions": contributions[index],
                "contribution_sha256": f"{index + 8:x}" * 64,
                "output_sha256": f"{index + 9:x}" * 64,
            }
        )
    return {
        "format": "k3x-official-two-layer-v1",
        "repository": "moonshotai/Kimi-K3",
        "resolved_revision": "9f62e4e9fffbd0a83ddd60e1c209d828994b3569",
        "snapshot_sha256": "deaa6394b80afe12976ce8efbbf2463f6808c291d83b029e6b0cfb98de90a4e5",
        "index_sha256": "a1c5210650ce71d2d3ae9ec5a101ac4afd3cf4b10091be589853437eb967febd",
        "config_sha256": "9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213",
        "source_blob_id": "b8c41e8bfce768d74d8da3a37e693f5ee43876a0",
        "shard_paths": [
            "model-00002-of-000096.safetensors",
            "model-00003-of-000096.safetensors",
        ],
        "layer_ids": [1, 2],
        "step_order": ["a:1", "a:2", "b:1", "b:2"],
        "steps": steps,
        "selected_experts": [routes[0] + routes[2], routes[1] + routes[3]],
        "final_state_sha256": [states[2], states[3]],
        "oracle": {
            "format": "k3x-official-two-layer-oracle-v1",
            "filename": "official-two-layer-oracle-v1.bin",
            "sha256": "d" * 64,
            "bytes": 13_053_992,
        },
        "objects": [],
        "traffic": {
            "requested_payload_bytes": 1,
            "downloaded_payload_bytes": 1,
            "reused_objects": 0,
            "requests": 1,
            "maximum_response_bytes": 1,
        },
        "artifact": {
            "filename": "official-two-layer.k3x",
            "k3x_root_sha256": "e" * 64,
            "source_sha256": "f" * 64,
            "tensor_sha256": {},
        },
    }


def _runner_record(mode: str, warmups: int, iterations: int) -> dict[str, object]:
    host = mode == "host-round-trip"
    runs = warmups + iterations
    kda_output = KDA_OUTPUT_BYTES if host else 0
    inter = HIDDEN_BYTES if host else 0
    return {
        "schema": "k3x-official-two-layer-bench-v1",
        "mode": mode,
        "warmup": warmups,
        "iterations": iterations,
        "wall_nanoseconds": [1_000_000 + index for index in range(iterations)],
        "maximum_absolute_error": 0.0001,
        "weight_h2d_bytes": 0,
        "activation_h2d_bytes": STATE_BYTES + 229_376,
        "device_to_host_bytes": (
            STATE_BYTES + kda_output + ROUTER_BYTES + inter + HIDDEN_BYTES
        ),
        "state_h2d_bytes": STATE_BYTES,
        "state_d2h_bytes": STATE_BYTES,
        "kda_output_d2h_bytes": kda_output,
        "router_logit_d2h_bytes": ROUTER_BYTES,
        "inter_layer_hidden_h2d_bytes": inter,
        "inter_layer_hidden_d2h_bytes": inter,
        "final_hidden_d2h_bytes": HIDDEN_BYTES,
        "layer_front_calls": 0 if host else 4,
        "layer_tail_calls": 0 if host else 4,
        "state_seeds": 2 * runs,
        "state_continuations": 2 * runs,
        "state_publications": 2 * runs,
        "state_invalidations": 0,
        "prepared_seeds": 4 * runs,
        "prepared_consumes": 4 * runs,
        "prepared_discards": 0,
        "prepared_invalidations": 0,
        "resident_weight_bytes": (
            RESIDENT_BYTES
            + ROUTE_PREPARATION_BYTES
            + (0 if host else DEVICE_FRONT_BYTES)
        ),
        "peak_device_bytes": (
            RESIDENT_BYTES
            + ROUTE_PREPARATION_BYTES
            + (0 if host else DEVICE_FRONT_BYTES)
            + 1_048_576
        ),
        "k3x_root_sha256": "e" * 64,
        "route_expert_ids": [
            list(range(offset, offset + 16)) for offset in (0, 16, 32, 48)
        ],
        "route_contribution_sha256": [f"{value:x}" * 64 for value in (8, 9, 10, 11)],
        "final_output_sha256": [
            ("1" if host else "2") * 64,
            ("3" if host else "4") * 64,
        ],
        "final_state_sha256": [
            ("5" if host else "6") * 64,
            ("7" if host else "8") * 64,
        ],
    }


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
        record = _runner_record(mode, warmups, iterations)
        if mutation and (mutation[0] != "mode" or mode == "device-closure"):
            record[mutation[0]] = mutation[1]
        return subprocess.CompletedProcess(command, 0, json.dumps(record), "")

    monkeypatch.setattr(
        "tools.ablate_official_two_layer_closure.subprocess.run", fake_run
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


def test_case_order_and_controls_are_fixed() -> None:
    assert CASES == (
        ("host-round-trip", "host-round-trip"),
        ("device-closure", "device-closure"),
    )


def test_runner_is_directly_executable_from_repository_root() -> None:
    root = Path(__file__).resolve().parents[2]
    environment = dict(os.environ)
    environment["PYTHONPATH"] = "."
    result = subprocess.run(
        [sys.executable, "tools/ablate_official_two_layer_closure.py", "--help"],
        cwd=root,
        env=environment,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert "--oracle" in result.stdout
    assert "--verify-existing" in result.stdout


def test_run_writes_fixed_digest_backed_lf_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact, manifest, oracle, runner, output, summary, calls = _generate(
        tmp_path, monkeypatch
    )
    assert [call[call.index("--mode") + 1] for call in calls] == [
        mode for _, mode in CASES
    ]
    for call in calls:
        assert call[call.index("--resident-bytes") + 1] == "4294967296"
        assert call[call.index("--warmup") + 1] == "2"
        assert call[call.index("--iterations") + 1] == "5"
    assert b"\r\n" not in (output / "summary.csv").read_bytes()
    assert summary["oracle_sha256"] == hashlib.sha256(oracle.read_bytes()).hexdigest()
    assert (
        verify_summary(
            output / "summary.json",
            output / "summary.csv",
            artifact=artifact,
            manifest=manifest,
            oracle=oracle,
            runner=runner,
            strict_official=False,
        )
        == summary
    )


def test_run_accepts_observed_two_layer_bf16_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "tools.ablate_official_layer._fsync_directory", lambda _: None
    )
    _, _, _, _, _, summary, _ = _generate(
        tmp_path,
        monkeypatch,
        ("maximum_absolute_error", 0.001953125),
    )

    assert all(
        record["maximum_absolute_error"] == 0.001953125
        for record in summary["records"]
    )


def test_run_preserves_measured_output_and_state_digest_divergence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "tools.ablate_official_layer._fsync_directory", lambda _: None
    )
    _, _, _, _, _, summary, _ = _generate(tmp_path, monkeypatch)

    host, device = summary["records"]
    assert host["final_output_sha256"] != device["final_output_sha256"]
    assert host["final_state_sha256"] != device["final_state_sha256"]
    assert device["resident_weight_bytes"] - host["resident_weight_bytes"] == (
        DEVICE_FRONT_BYTES
    )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (("mode", "host-round-trip"), "mode identity"),
        (("weight_h2d_bytes", 1), "warm weight"),
        (("state_h2d_bytes", 1), "state transfer"),
        (("state_d2h_bytes", 1), "state transfer"),
        (("kda_output_d2h_bytes", 1), "KDA output transfer"),
        (("router_logit_d2h_bytes", 1), "router transfer"),
        (("inter_layer_hidden_h2d_bytes", 1), "inter-layer transfer"),
        (("inter_layer_hidden_d2h_bytes", 1), "inter-layer transfer"),
        (("final_hidden_d2h_bytes", 1), "final hidden transfer"),
        (("layer_front_calls", 1), "lifetime counter"),
        (("state_publications", 1), "lifetime counter"),
        (("prepared_consumes", 1), "lifetime counter"),
        (("resident_weight_bytes", 1), "resident weight"),
        (("maximum_absolute_error", 0.0020001), "numerical divergence"),
        (("route_expert_ids", [[0] * 16] * 4), "measured identity"),
        (("route_contribution_sha256", ["g" * 64] * 4), "measured identity"),
        (("final_output_sha256", ["g" * 64] * 2), "measured identity"),
        (("final_state_sha256", ["g" * 64] * 2), "measured identity"),
        (("decode_tok_s", 5.0), "forbidden metric"),
    ],
)
def test_run_rejects_one_field_mutations(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    mutation: tuple[str, object],
    message: str,
) -> None:
    with pytest.raises(RuntimeError, match=message):
        _generate(tmp_path, monkeypatch, mutation)
    assert not (tmp_path / "out").exists()
    assert not (tmp_path / ".out.partial").exists()


def test_run_rejects_manifest_route_and_state_chain_mutations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = _manifest()
    manifest["final_state_sha256"] = ["0" * 64, "0" * 64]
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(RuntimeError, match="final state identity"):
        from tools.ablate_official_two_layer_closure import manifest_identity

        manifest_identity(path)


def test_verify_rejects_raw_csv_and_identity_mutations(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact, manifest, oracle, runner, output, _, _ = _generate(tmp_path, monkeypatch)
    raw = output / "device-closure.json"
    raw.write_bytes(raw.read_bytes() + b" ")
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
    raw.write_bytes(raw.read_bytes()[:-1])
    csv_path = output / "summary.csv"
    with csv_path.open(newline="", encoding="utf-8") as stream:
        reader_value = csv.DictReader(stream)
        rows, fields = list(reader_value), tuple(reader_value.fieldnames or ())
    rows[0]["weight_h2d_bytes"] = "1"
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer_value = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer_value.writeheader()
        writer_value.writerows(rows)
    with pytest.raises(RuntimeError, match="CSV"):
        verify_summary(
            output / "summary.json",
            csv_path,
            artifact=artifact,
            manifest=manifest,
            oracle=oracle,
            runner=runner,
            strict_official=False,
        )


def test_strict_verification_requires_fixed_transaction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact, manifest, oracle, runner, output, _, _ = _generate(tmp_path, monkeypatch)
    with pytest.raises(RuntimeError, match="official transaction gate"):
        verify_summary(
            output / "summary.json",
            output / "summary.csv",
            artifact=artifact,
            manifest=manifest,
            oracle=oracle,
            runner=runner,
        )


def test_verify_rejects_summary_oracle_identity_mutation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact, manifest, oracle, runner, output, _, _ = _generate(tmp_path, monkeypatch)
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["oracle_sha256"] = "0" * 64
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with pytest.raises(RuntimeError, match="summary input size"):
        verify_summary(
            summary_path,
            output / "summary.csv",
            artifact=artifact,
            manifest=manifest,
            oracle=None,
            runner=runner,
            strict_official=False,
        )
