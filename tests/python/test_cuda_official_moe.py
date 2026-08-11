# 공식 MoE CUDA harness의 CLI, 고정 identity, 실행 schema와 fail-closed 경계를 검증합니다.
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from conftest import cpp_binary
from k3x_converter.reader import K3XReader
from k3x_converter.writer import convert
from k3x_ref.storage_fixture import write_bounded_expert_source


def _runner() -> Path:
    if Path(os.environ.get("K3X_BUILD_DIR", "build")).name != "build-cuda":
        pytest.skip("official MoE benchmark requires build-cuda")
    return cpp_binary("k3x_cuda_official_moe_bench")


def _manifest(root: str = "0" * 64) -> dict:
    ids = list(range(16))
    contributions = [1.0 / 16.0] * 16
    base = "model.layers.1."
    tensor_names = [
        base + "mlp_res_norm.weight",
        base + "mlp_res_proj.weight",
        base + "post_attention_layernorm.weight",
        base + "block_sparse_moe.gate.weight",
        base + "block_sparse_moe.gate.e_score_correction_bias",
        base + "block_sparse_moe.routed_expert_down_proj.weight",
        base + "block_sparse_moe.routed_expert_norm.weight",
        base + "block_sparse_moe.routed_expert_up_proj.weight",
        base + "block_sparse_moe.shared_experts.gate_proj.weight",
        base + "block_sparse_moe.shared_experts.up_proj.weight",
        base + "block_sparse_moe.shared_experts.down_proj.weight",
    ]
    for expert_id in ids:
        expert = base + f"feed_forward.experts.{expert_id}."
        for role in ("gate", "up", "down"):
            tensor_names.extend(
                [expert + role + ".weight_packed", expert + role + ".weight_scale"]
            )
    return {
        "format": "k3x-official-moe-routes-v1",
        "converter_version": "k3x-converter-0.1.0",
        "repository": "moonshotai/Kimi-K3",
        "requested_revision": "main",
        "resolved_revision": "9f62e4e9fffbd0a83ddd60e1c209d828994b3569",
        "snapshot_sha256": "deaa6394b80afe12976ce8efbbf2463f6808c291d83b029e6b0cfb98de90a4e5",
        "index_sha256": "a1c5210650ce71d2d3ae9ec5a101ac4afd3cf4b10091be589853437eb967febd",
        "config_sha256": "9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213",
        "config_git_blob_id": "d7f26ead420b1d967f2759679dbebc65edfcff93",
        "shard_path": "model-00002-of-000096.safetensors",
        "shard_lfs_sha256": "26a3284e1d2cb567934ebef002e6a1813551d646739e8bcb1e9e3fe7f878e0f5",
        "inputs": [
            {
                "name": "a",
                "prefix_sha256": "acc7746e19fcb6bb17d09ce08d387ca91d3a742c4f671046aaa0184a290d2cc3",
                "block_sha256": "c7d98135ee7f46f4d82822d2e267d368dcdee51411575e578e63385a12e9bc3e",
            },
            {
                "name": "b",
                "prefix_sha256": "9b8f886591586999d0fb6a9661c938e24f2ade01cfdfbe352ea57961a642d566",
                "block_sha256": "323b027923f323953dc12c6bc16618672e84d264891c6ed0a9aa3383b0045046",
            },
        ],
        "routes": [
            {"name": "a", "expert_ids": ids, "contributions": contributions},
            {"name": "b", "expert_ids": ids, "contributions": contributions},
        ],
        "selected_experts": ids,
        "provenance": "transport-pinned-ranges",
        "always_active_objects": [],
        "artifact": {
            "filename": "official-moe-l1.k3x",
            "k3x_root_sha256": root,
            "source_sha256": "1" * 64,
            "tensor_sha256": {name: "2" * 64 for name in tensor_names},
        },
    }


def _write_manifest(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ((), "model path is required"),
        (("--model", "x"), "manifest path is required"),
        (("--case", "other"), "unknown case: other"),
        (("--weight-mode", "other"), "unknown weight mode: other"),
        (("--iterations", "0"), "iterations must be positive"),
        (("--unknown", "x"), "invalid option: --unknown"),
        (("--model", "x", "trailing"), "missing option value"),
    ],
)
def test_official_moe_bench_rejects_invalid_arguments(
    arguments: tuple[str, ...], message: str
) -> None:
    result = subprocess.run(
        [str(_runner()), *arguments], capture_output=True, text=True
    )
    assert result.returncode == 2
    assert result.stderr.strip() == message


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("repository",), "other/repo"),
        (("resolved_revision",), "f" * 40),
        (("index_sha256",), "f" * 64),
        (("config_sha256",), "f" * 64),
        (("inputs", 0, "prefix_sha256"), "f" * 64),
        (("inputs", 1, "block_sha256"), "f" * 64),
        (("provenance",), "full-object"),
    ],
)
def test_official_moe_bench_rejects_fixed_manifest_identity_before_model(
    tmp_path: Path, path: tuple[str | int, ...], value: object
) -> None:
    manifest = _manifest()
    target: object = manifest
    for key in path[:-1]:
        target = target[key]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, manifest)
    model = tmp_path / "not-a-model.k3x"
    model.write_bytes(b"x")

    result = subprocess.run(
        [
            str(_runner()), "--model", str(model), "--manifest",
            str(manifest_path), "--case", "a", "--weight-mode", "transient",
            "--warmup", "0", "--iterations", "1",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 4
    assert result.stderr.strip() == "INVALID_EXTENT: official MoE manifest identity mismatch"


@pytest.mark.parametrize(
    "payload",
    [
        '{"repository":"a","repository":"b"}\n',
        '{"routes":[{"contributions":[NaN]}]}\n',
        '{"routes":[{"contributions":[Infinity]}]}\n',
    ],
)
def test_official_moe_bench_rejects_noncanonical_json(
    tmp_path: Path, payload: str
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(payload, encoding="utf-8")
    model = tmp_path / "not-a-model.k3x"
    model.write_bytes(b"x")
    result = subprocess.run(
        [
            str(_runner()), "--model", str(model), "--manifest",
            str(manifest_path), "--case", "a", "--weight-mode", "transient",
            "--warmup", "0", "--iterations", "1",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 4
    assert result.stderr.strip() == "INVALID_EXTENT: invalid official MoE manifest"


def test_official_moe_bench_rejects_generic_storage_fixture_before_cuda(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    write_bounded_expert_source(source, chunk_bytes=257 * 1024)
    artifact = tmp_path / "generic.k3x"
    convert(source, artifact, chunk_bytes=193 * 1024)
    root = K3XReader.open(artifact).superblock.root_sha256.hex()
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, _manifest(root))

    result = subprocess.run(
        [
            str(_runner()), "--model", str(artifact), "--manifest",
            str(manifest_path), "--case", "a", "--weight-mode", "transient",
            "--warmup", "0", "--iterations", "1",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 4
    assert result.stderr.strip() == "INVALID_EXTENT: artifact is not official MoE fixture"


@pytest.mark.parametrize(
    ("case_name", "weight_mode"),
    [("a", "transient"), ("b", "transient"), ("alternating", "resident")],
)
def test_official_moe_bench_executes_ignored_fixture(
    case_name: str, weight_mode: str
) -> None:
    artifact_value = os.environ.get("K3X_TEST_OFFICIAL_MOE")
    manifest_value = os.environ.get("K3X_TEST_OFFICIAL_MOE_MANIFEST")
    if artifact_value is None or manifest_value is None:
        pytest.skip("set K3X_TEST_OFFICIAL_MOE and its manifest for the ignored fixture")
    result = subprocess.run(
        [
            str(_runner()), "--model", artifact_value, "--manifest", manifest_value,
            "--case", case_name, "--weight-mode", weight_mode,
            "--warmup", "0", "--iterations", "1",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    record = json.loads(result.stdout)
    assert record["artifact_kind"] == "official_kimi_k3_moe_ffn"
    assert record["case"] == case_name
    assert record["weight_mode"] == weight_mode
    assert record["token_semantics"] is False
    assert record["full_transformer_layer"] is False
    assert record["quality_measured"] is False
    assert record["maximum_absolute_error"] <= 2.0e-2
    assert record["all_finite"] is True
    assert "decode_tok_s" not in record
    assert "nvme_gb_per_token" not in record
