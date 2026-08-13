# 공식 두 레이어 CUDA harness의 입력 신뢰 경계와 출력 스키마를 검증합니다.
from __future__ import annotations

import json
import hashlib
import os
import struct
import subprocess
from pathlib import Path

import pytest

from conftest import cpp_binary


def _runner() -> Path:
    if Path(os.environ.get("K3X_BUILD_DIR", "build")).name != "build-cuda":
        pytest.skip("official two-layer benchmark requires build-cuda")
    return cpp_binary("k3x_cuda_official_two_layer_bench")


def _command(tmp_path: Path, *, mode: str = "device-closure") -> list[str]:
    return [
        str(_runner()),
        "--artifact",
        str(tmp_path / "fixture.k3x"),
        "--manifest",
        str(tmp_path / "manifest.json"),
        "--oracle",
        str(tmp_path / "official-two-layer-oracle-v1.bin"),
        "--mode",
        mode,
        "--resident-bytes",
        "8388608",
        "--warmup",
        "0",
        "--iterations",
        "1",
    ]


def _valid_manifest() -> dict:
    ids = list(range(16))
    masses = [1.0 / 16] * 16
    states = {
        ("a", 1): "3" * 64,
        ("a", 2): "4" * 64,
        ("b", 1): "5" * 64,
        ("b", 2): "6" * 64,
    }
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
        "steps": [
            {
                "position": position,
                "layer_id": layer,
                "expert_ids": ids,
                "contributions": masses,
                "consumes_state_sha256": (
                    "7" * 64 if position == "a" else states[("a", layer)]
                ),
                "state_sha256": states[(position, layer)],
                "kda_output_sha256": "8" * 64,
                "contribution_sha256": "9" * 64,
                "output_sha256": "a" * 64,
            }
            for position in ("a", "b")
            for layer in (1, 2)
        ],
        "selected_experts": [ids, ids],
        "final_state_sha256": [states[("b", 1)], states[("b", 2)]],
        "oracle": {
            "format": "k3x-official-two-layer-oracle-v1",
            "filename": "official-two-layer-oracle-v1.bin",
            "sha256": "0" * 64,
            "bytes": 13_053_992,
        },
        "artifact": {
            "filename": "official-two-layer.k3x",
            "k3x_root_sha256": "1" * 64,
            "source_sha256": "2" * 64,
            "tensor_sha256": {},
        },
    }


def test_two_layer_harness_rejects_invalid_mode_before_files(tmp_path: Path) -> None:
    completed = subprocess.run(
        _command(tmp_path, mode="unknown"), capture_output=True, text=True
    )

    assert completed.returncode == 2
    assert "invalid mode" in completed.stderr


def test_two_layer_harness_rejects_duplicate_manifest_keys(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(
        '{"format":"k3x-official-two-layer-v1",'
        '"format":"k3x-official-two-layer-v1"}',
        encoding="utf-8",
    )

    completed = subprocess.run(
        _command(tmp_path), capture_output=True, text=True
    )

    assert completed.returncode == 4
    assert "invalid two-layer manifest" in completed.stderr


def test_two_layer_harness_rejects_noncanonical_layer_order(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(
        json.dumps(
            {
                "format": "k3x-official-two-layer-v1",
                "layer_ids": [2, 1],
                "step_order": ["a:1", "a:2", "b:1", "b:2"],
            }
        ),
        encoding="utf-8",
    )

    completed = subprocess.run(
        _command(tmp_path), capture_output=True, text=True
    )

    assert completed.returncode == 4
    assert "two-layer manifest identity mismatch" in completed.stderr


def test_two_layer_harness_rejects_oracle_before_artifact(tmp_path: Path) -> None:
    (tmp_path / "manifest.json").write_text(
        json.dumps(_valid_manifest()), encoding="utf-8"
    )
    (tmp_path / "official-two-layer-oracle-v1.bin").write_bytes(b"K3XORC2\0")

    completed = subprocess.run(
        _command(tmp_path), capture_output=True, text=True
    )

    assert completed.returncode == 4
    assert "invalid two-layer oracle" in completed.stderr


def test_two_layer_harness_rejects_artifact_after_verified_oracle(
    tmp_path: Path,
) -> None:
    oracle = struct.pack("<8sQQQQ", b"K3XORC2\0", 14_336, 2, 36_864, 1_572_864)
    oracle += bytes(13_053_992 - len(oracle))
    manifest = _valid_manifest()
    manifest["oracle"]["sha256"] = hashlib.sha256(oracle).hexdigest()
    (tmp_path / "manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )
    (tmp_path / "official-two-layer-oracle-v1.bin").write_bytes(oracle)
    (tmp_path / "fixture.k3x").write_bytes(b"not-a-k3x")

    completed = subprocess.run(
        _command(tmp_path), capture_output=True, text=True
    )

    assert completed.returncode == 4
    assert "TRUNCATED_FILE" in completed.stderr


def test_two_layer_harness_executes_bounded_fixture_on_cuda() -> None:
    root = Path(__file__).resolve().parents[2]
    fixture = root / "artifacts" / "m33-official-two-layer"
    artifact = fixture / "official-two-layer.k3x"
    manifest = fixture / "two-layer-route-state-manifest.json"
    oracle = fixture / "official-two-layer-oracle-v1.bin"
    if not artifact.is_file() or not manifest.is_file() or not oracle.is_file():
        pytest.skip("bounded official two-layer fixture is not materialized")

    completed = subprocess.run(
        [
            str(_runner()),
            "--artifact",
            str(artifact),
            "--manifest",
            str(manifest),
            "--oracle",
            str(oracle),
            "--mode",
            "host-round-trip",
            "--resident-bytes",
            "4294967296",
            "--warmup",
            "0",
            "--iterations",
            "1",
        ],
        capture_output=True,
        text=True,
        timeout=900,
    )

    assert completed.returncode == 0, completed.stderr
    payload = json.loads(completed.stdout)
    assert payload["schema"] == "k3x-official-two-layer-bench-v1"
    assert payload["mode"] == "host-round-trip"
    assert payload["warmup"] == 0
    assert payload["iterations"] == 1
    assert payload["weight_h2d_bytes"] > 0
