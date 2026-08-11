# 공식 complete-layer CUDA harness의 CLI와 backend 이전 preflight를 검증합니다.
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


_KDA_SUFFIXES = (
    "self_attention_res_norm.weight",
    "self_attention_res_proj.weight",
    "input_layernorm.weight",
    "self_attn.q_proj.weight",
    "self_attn.q_conv1d.weight",
    "self_attn.k_proj.weight",
    "self_attn.k_conv1d.weight",
    "self_attn.v_proj.weight",
    "self_attn.v_conv1d.weight",
    "self_attn.f_a_proj.weight",
    "self_attn.f_b_proj.weight",
    "self_attn.A_log",
    "self_attn.dt_bias",
    "self_attn.b_proj.weight",
    "self_attn.g_proj.weight",
    "self_attn.o_norm.weight",
    "self_attn.o_proj.weight",
)
_MOE_SUFFIXES = (
    "mlp_res_norm.weight",
    "mlp_res_proj.weight",
    "post_attention_layernorm.weight",
    "block_sparse_moe.gate.weight",
    "block_sparse_moe.gate.e_score_correction_bias",
    "block_sparse_moe.routed_expert_down_proj.weight",
    "block_sparse_moe.routed_expert_norm.weight",
    "block_sparse_moe.routed_expert_up_proj.weight",
    "block_sparse_moe.shared_experts.gate_proj.weight",
    "block_sparse_moe.shared_experts.up_proj.weight",
    "block_sparse_moe.shared_experts.down_proj.weight",
)
_KDA_RANGES = (
    (381_373_456, 14_336), (381_387_792, 14_336),
    (381_316_112, 14_336), (916_241_424, 176_160_768),
    (1_069_072, 196_608), (563_919_888, 176_160_768),
    (871_952, 196_608), (1_092_402_192, 176_160_768),
    (1_265_680, 196_608), (382_778_384, 1_835_008),
    (384_613_392, 3_145_728), (822_288, 512), (822_800, 49_152),
    (381_402_128, 1_376_256), (387_759_120, 176_160_768),
    (1_068_560, 512), (740_080_656, 176_160_768),
)
_MOE_RANGES = (
    (381_330_448, 14_336), (381_344_784, 14_336),
    (381_359_120, 14_336), (1_462_288, 12_845_056),
    (818_704, 3_584), (14_307_344, 51_380_224),
    (65_687_568, 7_168), (65_694_736, 51_380_224),
    (205_155_344, 88_080_384), (293_235_728, 88_080_384),
    (117_074_960, 88_080_384),
)


def _runner() -> Path:
    if Path(os.environ.get("K3X_BUILD_DIR", "build")).name != "build-cuda":
        pytest.skip("official layer benchmark requires build-cuda")
    return cpp_binary("k3x_cuda_official_layer_bench")


def _manifest(root: str = "0" * 64) -> dict:
    ids = list(range(16))
    contributions = [1.0 / 16.0] * 16
    official_prefix = "language_model.model.layers.1."
    canonical_prefix = "model.layers.1."
    tensor_names = [canonical_prefix + suffix for suffix in (*_KDA_SUFFIXES, *_MOE_SUFFIXES)]
    for expert_id in ids:
        expert = canonical_prefix + f"feed_forward.experts.{expert_id}."
        for role in ("gate", "up", "down"):
            tensor_names.extend(
                [expert + role + ".weight_packed", expert + role + ".weight_scale"]
            )
    initial = "3" * 64
    first = "4" * 64
    final = "5" * 64
    return {
        "format": "k3x-official-kda-layer-routes-v1",
        "converter_version": "k3x-converter-0.1.0",
        "repository": "moonshotai/Kimi-K3",
        "requested_revision": "main",
        "resolved_revision": "9f62e4e9fffbd0a83ddd60e1c209d828994b3569",
        "snapshot_sha256": "deaa6394b80afe12976ce8efbbf2463f6808c291d83b029e6b0cfb98de90a4e5",
        "index_sha256": "a1c5210650ce71d2d3ae9ec5a101ac4afd3cf4b10091be589853437eb967febd",
        "config_sha256": "9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213",
        "config_git_blob_id": "d7f26ead420b1d967f2759679dbebc65edfcff93",
        "source_blob_id": "b8c41e8bfce768d74d8da3a37e693f5ee43876a0",
        "shard_path": "model-00002-of-000096.safetensors",
        "shard_lfs_sha256": "26a3284e1d2cb567934ebef002e6a1813551d646739e8bcb1e9e3fe7f878e0f5",
        "header": {
            "file_size": 16_990_911_504,
            "header_length": 818_696,
            "data_start": 818_704,
        },
        "state_layout": "v-first-fp32",
        "initial_state_sha256": initial,
        "final_state_sha256": final,
        "inputs": [
            {
                "name": "a",
                "hidden_sha256": "acc7746e19fcb6bb17d09ce08d387ca91d3a742c4f671046aaa0184a290d2cc3",
                "block_sha256": "c7d98135ee7f46f4d82822d2e267d368dcdee51411575e578e63385a12e9bc3e",
            },
            {
                "name": "b",
                "hidden_sha256": "9b8f886591586999d0fb6a9661c938e24f2ade01cfdfbe352ea57961a642d566",
                "block_sha256": "323b027923f323953dc12c6bc16618672e84d264891c6ed0a9aa3383b0045046",
            },
        ],
        "steps": [
            {
                "name": "a",
                "consumes_state_sha256": initial,
                "state_sha256": first,
                "kda_output_sha256": "6" * 64,
                "expert_ids": ids,
                "contributions": contributions,
            },
            {
                "name": "b",
                "consumes_state_sha256": first,
                "state_sha256": final,
                "kda_output_sha256": "7" * 64,
                "expert_ids": ids,
                "contributions": contributions,
            },
        ],
        "selected_experts": ids,
        "kda_objects": [
            {
                "name": official_prefix + suffix,
                "range": [offset, offset + length],
                "sha256": "8" * 64,
            }
            for suffix, (offset, length) in zip(_KDA_SUFFIXES, _KDA_RANGES)
        ],
        "always_active_objects": [
            {
                "name": official_prefix + suffix,
                "range": [offset, offset + length],
                "sha256": "9" * 64,
            }
            for suffix, (offset, length) in zip(_MOE_SUFFIXES, _MOE_RANGES)
        ],
        "expert_objects": [
            {
                "expert_id": expert_id,
                "range": [
                    1_268_562_960 + expert_id * 17_547_264,
                    1_286_110_224 + expert_id * 17_547_264,
                ],
                "sha256": "a" * 64,
            }
            for expert_id in ids
        ],
        "provenance": "transport-pinned-ranges",
        "artifact": {
            "filename": "official-kda-layer-l1.k3x",
            "k3x_root_sha256": root,
            "k3x_source_fingerprint_sha256": "b" * 64,
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
        ((), "artifact path is required"),
        (("--artifact", "x"), "manifest path is required"),
        (("--case", "other"), "unknown case: other"),
        (("--weight-mode", "other"), "unknown weight mode: other"),
        (("--iterations", "0"), "iterations must be positive"),
        (("--unknown", "x"), "invalid option: --unknown"),
        (("--artifact", "x", "trailing"), "missing option value"),
    ],
)
def test_official_layer_bench_rejects_invalid_arguments(
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
        (("source_blob_id",), "f" * 40),
        (("state_layout",), "head-first-fp32"),
        (("inputs", 0, "hidden_sha256"), "f" * 64),
        (("steps", 1, "consumes_state_sha256"), "f" * 64),
        (("steps", 0, "expert_ids", 1), 0),
        (("selected_experts",), list(reversed(range(16)))),
        (("kda_objects", 0, "name"), "wrong"),
        (("provenance",), "full-object"),
    ],
)
def test_official_layer_bench_rejects_manifest_before_artifact(
    tmp_path: Path, path: tuple[str | int, ...], value: object
) -> None:
    manifest = _manifest()
    target: object = manifest
    for key in path[:-1]:
        target = target[key]  # type: ignore[index]
    target[path[-1]] = value  # type: ignore[index]
    manifest_path = tmp_path / "manifest.json"
    _write_manifest(manifest_path, manifest)
    artifact = tmp_path / "not-a-model.k3x"
    artifact.write_bytes(b"x")

    result = subprocess.run(
        [
            str(_runner()), "--artifact", str(artifact), "--manifest",
            str(manifest_path), "--case", "a", "--weight-mode", "transient",
            "--warmups", "0", "--iterations", "1",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 4
    assert result.stderr.strip() == "INVALID_EXTENT: official layer manifest identity mismatch"


@pytest.mark.parametrize(
    "payload",
    [
        '{"repository":"a","repository":"b"}\n',
        '{"steps":[{"contributions":[NaN]}]}\n',
        '{"steps":[{"contributions":[Infinity]}]}\n',
    ],
)
def test_official_layer_bench_rejects_noncanonical_json(
    tmp_path: Path, payload: str
) -> None:
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(payload, encoding="utf-8")
    artifact = tmp_path / "not-a-model.k3x"
    artifact.write_bytes(b"x")
    result = subprocess.run(
        [
            str(_runner()), "--artifact", str(artifact), "--manifest",
            str(manifest_path), "--case", "a", "--weight-mode", "transient",
            "--warmups", "0", "--iterations", "1",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 4
    assert result.stderr.strip() == "INVALID_EXTENT: invalid official layer manifest"


def test_official_layer_bench_rejects_generic_artifact_before_cuda(
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
            str(_runner()), "--artifact", str(artifact), "--manifest",
            str(manifest_path), "--case", "a", "--weight-mode", "transient",
            "--warmups", "0", "--iterations", "1",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 4
    assert result.stderr.strip() == "INVALID_EXTENT: artifact is not official layer fixture"
