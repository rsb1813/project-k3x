# 공식 가중치 discovery CLI의 dry-run과 증거 출력을 검증합니다.
from __future__ import annotations

import csv
import hashlib
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from k3x_converter.format import K3XError, OPTIONAL_STORAGE_FIXTURE
from k3x_converter.official_moe import MaterializedRangeObject, OfficialMoeRoute
from k3x_converter.official_two_layer import (
    OfficialLayerSourceBytes,
    OfficialPreparedSourceStep,
    OfficialTwoLayerMaterializationReport,
    OfficialTwoLayerState,
    OfficialTwoLayerStepExecution,
    parse_official_two_layer_oracle,
)
from k3x_converter.official_transport import HttpResponse, TransportStats
from tools.discover_official_kimi_k3 import main
from tools.verify_official_discovery import (
    CSV_FIELDS,
    canonical_record_sha256,
    summary_csv_row,
    verify_summary,
)


_COMMIT = "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"
_SHARD = "model-00002-of-000096.safetensors"
_SHARD_2 = "model-00003-of-000096.safetensors"
_SHARD_SIZE = 16_990_911_504
_HEADER_LENGTH = 818_696
_DATA_START = 818_704
_PAYLOAD_START = 1_268_562_960
_PAYLOAD_END = 1_286_110_224
_BASE = "language_model.model.layers.1.block_sparse_moe.experts.0"
_KDA_SOURCE_BLOB = "b8c41e8bfce768d74d8da3a37e693f5ee43876a0"
_KDA_SPECIFICATIONS = (
    ("self_attention_res_norm.weight", "BF16", [7_168]),
    ("self_attention_res_proj.weight", "BF16", [1, 7_168]),
    ("input_layernorm.weight", "BF16", [7_168]),
    ("self_attn.q_proj.weight", "BF16", [12_288, 7_168]),
    ("self_attn.q_conv1d.weight", "F32", [12_288, 1, 4]),
    ("self_attn.k_proj.weight", "BF16", [12_288, 7_168]),
    ("self_attn.k_conv1d.weight", "F32", [12_288, 1, 4]),
    ("self_attn.v_proj.weight", "BF16", [12_288, 7_168]),
    ("self_attn.v_conv1d.weight", "F32", [12_288, 1, 4]),
    ("self_attn.f_a_proj.weight", "BF16", [128, 7_168]),
    ("self_attn.f_b_proj.weight", "BF16", [12_288, 128]),
    ("self_attn.A_log", "F32", [128]),
    ("self_attn.dt_bias", "F32", [12_288]),
    ("self_attn.b_proj.weight", "BF16", [96, 7_168]),
    ("self_attn.g_proj.weight", "BF16", [12_288, 7_168]),
    ("self_attn.o_norm.weight", "F32", [128]),
    ("self_attn.o_proj.weight", "BF16", [7_168, 12_288]),
)


def _git_blob_id(body: bytes) -> str:
    prefix = b"blob " + str(len(body)).encode() + b"\0"
    return hashlib.sha1(prefix + body).hexdigest()


def _config_body() -> bytes:
    value = {
        "model_type": "kimi_k3",
        "text_config": {
            "model_type": "kimi_linear",
            "vocab_size": 163_840,
            "num_hidden_layers": 93,
            "first_k_dense_replace": 1,
            "moe_layer_freq": 1,
            "num_experts": 896,
            "num_experts_per_token": 16,
            "num_shared_experts": 2,
            "hidden_size": 7_168,
            "routed_expert_hidden_size": 3_584,
            "moe_intermediate_size": 3_072,
            "moe_renormalize": True,
            "moe_router_activation_func": "sigmoid",
            "num_expert_group": 1,
            "topk_group": 1,
            "activation_situ_beta": 4.0,
            "activation_situ_linear_beta": 25.0,
            "routed_scaling_factor": 1.0,
            "latent_moe_use_norm": True,
            "rms_norm_eps": 1.0e-5,
            "attn_res_block_size": 12,
            "linear_attn_config": {
                "full_attn_layers": [*range(4, 94, 4), 93],
                "gate_lower_bound": -5.0,
                "head_dim": 128,
                "kda_layers": [
                    index for index in range(1, 92) if index % 4 != 0
                ],
                "num_heads": 96,
                "short_conv_kernel_size": 4,
                "use_full_rank_gate": True,
            },
        },
    }
    return json.dumps(value, separators=(",", ":")).encode()


def _index_body(shards: tuple[str, ...]) -> bytes:
    weight_map = {
        f"{_BASE}.{matrix}.{kind}": _SHARD
        for matrix in ("w1", "w2", "w3")
        for kind in ("weight_packed", "weight_scale")
    }
    weight_map.update(
        {f"unused.{index}": path for index, path in enumerate(shards) if path != _SHARD}
    )
    weight_map.update(
        {
            f"language_model.model.layers.1.{name}": _SHARD
            for name, _, _ in _KDA_SPECIFICATIONS
        }
    )
    always_active = (
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
    weight_map.update({f"language_model.model.layers.1.{name}": _SHARD for name in always_active})
    layer_2_base = "language_model.model.layers.2.block_sparse_moe.experts.0"
    weight_map.update(
        {
            f"{layer_2_base}.{matrix}.{kind}": _SHARD_2
            for matrix in ("w1", "w2", "w3")
            for kind in ("weight_packed", "weight_scale")
        }
    )
    weight_map.update(
        {
            f"language_model.model.layers.2.{name}": _SHARD_2
            for name, _, _ in _KDA_SPECIFICATIONS
        }
    )
    weight_map.update(
        {
            f"language_model.model.layers.2.{name}": _SHARD_2
            for name in always_active
        }
    )
    value = {
        "metadata": {"total_size": 1_560_860_324_864},
        "weight_map": weight_map,
    }
    return json.dumps(value, separators=(",", ":")).encode()


def _header_body(layer_id: int = 1) -> bytes:
    selected = [
        ("w1.weight_packed", [3072, 1792], [1_267_744_256, 1_273_249_280]),
        ("w1.weight_scale", [3072, 112], [1_273_249_280, 1_273_593_344]),
        ("w2.weight_packed", [3584, 1536], [1_273_593_344, 1_279_098_368]),
        ("w2.weight_scale", [3584, 96], [1_279_098_368, 1_279_442_432]),
        ("w3.weight_packed", [3072, 1792], [1_279_442_432, 1_284_947_456]),
        ("w3.weight_scale", [3072, 112], [1_284_947_456, 1_285_291_520]),
    ]
    always_specifications = (
        ("mlp_res_norm.weight", "BF16", [7168]),
        ("mlp_res_proj.weight", "BF16", [1, 7168]),
        ("post_attention_layernorm.weight", "BF16", [7168]),
        ("block_sparse_moe.gate.weight", "BF16", [896, 7168]),
        ("block_sparse_moe.gate.e_score_correction_bias", "F32", [896]),
        ("block_sparse_moe.routed_expert_down_proj.weight", "BF16", [3584, 7168]),
        ("block_sparse_moe.routed_expert_norm.weight", "BF16", [3584]),
        ("block_sparse_moe.routed_expert_up_proj.weight", "BF16", [7168, 3584]),
        ("block_sparse_moe.shared_experts.gate_proj.weight", "BF16", [6144, 7168]),
        ("block_sparse_moe.shared_experts.up_proj.weight", "BF16", [6144, 7168]),
        ("block_sparse_moe.shared_experts.down_proj.weight", "BF16", [7168, 6144]),
    )
    cursor = 0
    kda: dict[str, object] = {}
    for suffix, dtype, shape in _KDA_SPECIFICATIONS:
        values = 1
        for dimension in shape:
            values *= dimension
        length = values * (4 if dtype == "F32" else 2)
        kda[f"language_model.model.layers.{layer_id}.{suffix}"] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [cursor, cursor + length],
        }
        cursor += length
    always: dict[str, object] = {}
    for suffix, dtype, shape in always_specifications:
        values = 1
        for dimension in shape:
            values *= dimension
        length = values * (4 if dtype == "F32" else 2)
        always[f"language_model.model.layers.{layer_id}.{suffix}"] = {
            "dtype": dtype,
            "shape": shape,
            "data_offsets": [cursor, cursor + length],
        }
        cursor += length
    assert cursor == selected[0][2][0]
    value: dict[str, object] = {
        **kda,
        **always,
        **{
            f"language_model.model.layers.{layer_id}.block_sparse_moe."
            f"experts.0.{suffix}": {
                "dtype": "U8",
                "shape": shape,
                "data_offsets": offsets,
            }
            for suffix, shape, offsets in selected
        },
        "after": {
            "dtype": "I16",
            "shape": [1],
            "data_offsets": [selected[-1][2][1], _SHARD_SIZE - _DATA_START],
        },
    }
    encoded = json.dumps(value, separators=(",", ":")).encode()
    return encoded + b" " * (_HEADER_LENGTH - len(encoded))


class _DiscoveryTransport:
    def __init__(self) -> None:
        self.config = _config_body()
        self.shards = tuple(
            f"model-{index:05d}-of-000096.safetensors" for index in range(1, 97)
        )
        self.index = _index_body(self.shards)
        self.headers = {_SHARD: _header_body(1), _SHARD_2: _header_body(2)}
        cycle = bytes(range(256))
        length = _PAYLOAD_END - _PAYLOAD_START
        self.payload = (cycle * ((length + 255) // 256))[:length]
        siblings: list[dict[str, object]] = [
            {
                "rfilename": "modeling_kimi_linear.py",
                "size": 51_506,
                "blobId": _KDA_SOURCE_BLOB,
            },
            {
                "rfilename": "config.json",
                "size": len(self.config),
                "blobId": _git_blob_id(self.config),
            },
            {
                "rfilename": "model.safetensors.index.json",
                "size": len(self.index),
                "blobId": "2" * 40,
                "lfs": {
                    "size": len(self.index),
                    "sha256": hashlib.sha256(self.index).hexdigest(),
                },
            },
        ]
        for index, path in enumerate(self.shards):
            size = _SHARD_SIZE if path in {_SHARD, _SHARD_2} else 10_000 + index
            siblings.append(
                {
                    "rfilename": path,
                    "size": size,
                    "blobId": f"{index + 3:040x}",
                    "lfs": {"size": size, "sha256": f"{index + 3:064x}"},
                }
            )
        self.api = json.dumps(
            {
                "id": "moonshotai/Kimi-K3",
                "sha": _COMMIT,
                "private": False,
                "gated": False,
                "siblings": siblings,
            },
            separators=(",", ":"),
        ).encode()
        self.calls: list[str] = []
        self.response_bytes = 0
        self.maximum_response_bytes = 0
        self.payload_requested = False

    @property
    def stats(self) -> TransportStats:
        return TransportStats(
            len(self.calls), self.response_bytes, self.maximum_response_bytes
        )

    def get(
        self,
        url: str,
        *,
        headers: dict[str, str],
        max_bytes: int,
        timeout_seconds: float,
        expected_status: int = 200,
    ) -> HttpResponse:
        self.calls.append(url)
        response_headers: dict[str, str] = {}
        if "/api/models/" in url:
            body = self.api
        elif "model.safetensors.index.json" in url:
            body = self.index
        elif url.endswith("/config.json"):
            body = self.config
        else:
            requested = headers["Range"]
            if requested == "bytes=0-7":
                body = struct.pack("<Q", _HEADER_LENGTH)
                start, end = 0, 7
            elif requested == f"bytes=8-{_HEADER_LENGTH + 7}":
                shard_path = _SHARD_2 if _SHARD_2 in url else _SHARD
                body = self.headers[shard_path]
                start, end = 8, _HEADER_LENGTH + 7
            elif requested == f"bytes={_PAYLOAD_START}-{_PAYLOAD_END - 1}":
                self.payload_requested = True
                body = self.payload
                start, end = _PAYLOAD_START, _PAYLOAD_END - 1
            else:
                raise AssertionError(requested)
            response_headers["content-range"] = f"bytes {start}-{end}/{_SHARD_SIZE}"
        assert len(body) <= max_bytes
        self.response_bytes += len(body)
        self.maximum_response_bytes = max(self.maximum_response_bytes, len(body))
        return HttpResponse(expected_status, url, response_headers, body)


def test_cli_dry_run_plans_real_shape_without_payload_access(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    summary_path = tmp_path / "dry-run.json"
    transport = _DiscoveryTransport()

    assert main(["--summary-json", str(summary_path)], transport=transport) == 0

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    stdout = json.loads(capsys.readouterr().out)
    assert stdout == summary
    assert summary["mode"] == "dry-run"
    assert summary["resolved_revision"] == _COMMIT
    assert summary["expert"]["payload_bytes"] == 17_547_264
    assert summary["traffic"]["header_bytes"] == 818_704
    assert summary["traffic"]["tensor_payload_bytes"] == 0
    assert summary["reader_valid"] is False
    assert transport.payload_requested is False


def test_cli_moe_scope_dry_run_plans_dependency_closed_bytes_without_payload(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    summary_path = tmp_path / "moe-dry-run.json"
    transport = _DiscoveryTransport()

    assert main(
        ["--scope", "moe-ffn", "--summary-json", str(summary_path)],
        transport=transport,
    ) == 0

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert json.loads(capsys.readouterr().out) == summary
    assert summary["format"] == "k3x-official-moe-discovery-v1"
    assert summary["scope"] == "moe-ffn"
    assert summary["mode"] == "dry-run"
    assert summary["always_active_tensor_count"] == 11
    assert summary["always_active_bytes"] == 379_900_416
    assert summary["maximum_two_case_bytes"] == 941_412_864
    assert summary["selected_experts"] == []
    assert summary["traffic"]["tensor_payload_bytes"] == 0
    assert transport.payload_requested is False


def test_cli_kda_layer_scope_dry_run_plans_complete_layer_without_payload(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    summary_path = tmp_path / "layer-dry-run.json"
    transport = _DiscoveryTransport()

    assert main(
        ["--scope", "kda-layer", "--summary-json", str(summary_path)],
        transport=transport,
    ) == 0

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert json.loads(capsys.readouterr().out) == summary
    assert summary["format"] == "k3x-official-kda-layer-discovery-v1"
    assert summary["scope"] == "kda-layer"
    assert summary["mode"] == "dry-run"
    assert summary["kda_tensor_count"] == 17
    assert summary["kda_payload_bytes"] == 887_843_840
    assert summary["base_payload_bytes"] == 1_267_744_256
    assert summary["maximum_two_token_bytes"] == 1_829_256_704
    assert summary["traffic"]["tensor_payload_bytes"] == 0
    assert transport.payload_requested is False


def test_cli_two_layer_scope_dry_run_plans_both_shards_without_payload(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    summary_path = tmp_path / "two-layer-dry-run.json"
    transport = _DiscoveryTransport()

    assert main(
        ["--scope", "two-layer", "--summary-json", str(summary_path)],
        transport=transport,
    ) == 0

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert json.loads(capsys.readouterr().out) == summary
    assert summary["format"] == "k3x-official-two-layer-discovery-v1"
    assert summary["scope"] == "two-layer"
    assert summary["mode"] == "dry-run"
    assert summary["layer_ids"] == [1, 2]
    assert summary["shard_paths"] == [_SHARD, _SHARD_2]
    assert summary["base_payload_bytes"] == 2_535_488_512
    assert summary["maximum_two_position_bytes"] == 3_658_513_408
    assert summary["traffic"]["tensor_payload_bytes"] == 0
    assert transport.payload_requested is False


def test_cli_two_layer_materialization_prints_canonical_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "artifacts"
    summary_path = tmp_path / "two-layer-summary.json"
    captured: dict[str, object] = {}

    def fake_materialize(
        snapshot,
        index,
        config,
        headers,
        plan,
        transport,
        output_dir,
        *,
        chunk_bytes,
    ):
        captured.update(
            headers=tuple(header.shard_path for header in headers),
            output_dir=output_dir,
            chunk_bytes=chunk_bytes,
        )
        return SimpleNamespace(
            selected_experts=((7, 8), (9, 10)),
            requested_payload_bytes=2_600_000_000,
            downloaded_payload_bytes=123_456,
            reused_objects=4,
            requests=12,
            maximum_response_bytes=8 * 1024 * 1024,
            microshard_sha256="4" * 64,
            tensor_sha256={"tensor": "5" * 64},
            k3x_root_sha256="6" * 64,
            route_manifest_path=output / "two-layer-route-state-manifest.json",
            oracle_path=output / "official-two-layer-oracle-v1.bin",
            oracle_sha256="7" * 64,
            oracle_bytes=12_000_000,
        )

    monkeypatch.setattr(
        "tools.discover_official_kimi_k3.materialize_official_two_layer",
        fake_materialize,
    )

    assert main(
        [
            "--scope", "two-layer", "--materialize",
            "--output-dir", str(output),
            "--summary-json", str(summary_path),
        ],
        transport=_DiscoveryTransport(),
    ) == 0

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert json.loads(capsys.readouterr().out) == summary
    assert summary["format"] == "k3x-official-two-layer-materialization-v1"
    assert summary["mode"] == "materialize"
    assert summary["selected_experts"] == [[7, 8], [9, 10]]
    assert summary["traffic"]["tensor_payload_bytes"] == 123_456
    assert summary["traffic"]["source_object_bytes"] == 2_600_000_000
    assert summary["artifacts"]["k3x_root_sha256"] == "6" * 64
    assert captured == {
        "headers": (_SHARD, _SHARD_2),
        "output_dir": output.resolve(),
        "chunk_bytes": 257 * 1024,
    }


def test_cli_two_layer_real_orchestrator_fetches_experts_after_all_trunks(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, int]] = []
    output = tmp_path / "orchestrated"

    def fake_range(
        snapshot,
        shard_path,
        offset,
        length,
        transport,
        object_directory,
        *,
        chunk_bytes,
    ):
        calls.append((shard_path, length))
        return MaterializedRangeObject(
            tmp_path / f"object-{len(calls)}.bin",
            f"{len(calls):064x}",
            length,
            False,
            1,
            min(length, chunk_bytes),
            length,
        )

    def fake_source(plan, kda_objects, always_objects, expert_plans, expert_objects):
        return OfficialLayerSourceBytes(
            plan.layer_id,
            SimpleNamespace(),
            (),
            (),
            16,
            1.0e-5,
            4.0,
            25.0,
        )

    def fake_prepare(layer, item, state, source):
        expert_id = layer.layer_id * 10 + (0 if item.name == "a" else 1)
        next_state = OfficialTwoLayerState(
            state.value,
            hashlib.sha256(
                f"{layer.layer_id}:{item.name}".encode()
            ).hexdigest(),
        )
        return OfficialPreparedSourceStep(
            layer.layer_id,
            torch.zeros(1),
            torch.zeros(1),
            torch.zeros(1),
            torch.zeros(1),
            torch.zeros((1, 1), dtype=torch.bfloat16),
            next_state,
            hashlib.sha256(f"kda:{layer.layer_id}:{item.name}".encode()).hexdigest(),
            OfficialMoeRoute((expert_id,), (1.0,)),
        )

    def fake_finish(prepared, source):
        width = 7_168
        output = tuple(float(prepared.layer_id) for _ in range(width))
        return OfficialTwoLayerStepExecution(
            output,
            prepared.state,
            prepared.kda_output_sha256,
            prepared.route,
        )

    def fake_expert(index, header, *, layer_id, expert_id):
        return SimpleNamespace(
            layer_id=layer_id,
            expert_id=expert_id,
            shard_path=header.shard_path,
            payload_start=expert_id,
            payload_end=expert_id + 1,
            payload_bytes=1,
            index_sha256=index.sha256,
            tensors=(),
        )

    def fake_manufacture(
        output_directory,
        tensors,
        config,
        metadata,
        *,
        chunk_bytes,
        stop_after_extents=None,
    ):
        return OfficialTwoLayerMaterializationReport(
            output_directory / "source",
            output_directory / "source" / "source-manifest.json",
            output_directory / "source" / "model.safetensors",
            output_directory / "official-two-layer.k3x",
            True,
            chunk_bytes,
            microshard_sha256="4" * 64,
            tensor_sha256={"tensor": "5" * 64},
            k3x_root_sha256="6" * 64,
            oracle_path=output_directory / "official-two-layer-oracle-v1.bin",
            oracle_sha256="7" * 64,
            oracle_bytes=12_000_000,
        )

    monkeypatch.setattr(
        "k3x_converter.official_two_layer.materialize_official_range_object",
        fake_range,
    )
    monkeypatch.setattr(
        "k3x_converter.official_two_layer.load_official_layer_source_bytes",
        fake_source,
    )
    monkeypatch.setattr(
        "k3x_converter.official_two_layer.prepare_official_source_step",
        fake_prepare,
    )
    monkeypatch.setattr(
        "k3x_converter.official_two_layer.finish_official_source_step",
        fake_finish,
    )
    monkeypatch.setattr(
        "k3x_converter.official_two_layer.plan_official_expert",
        fake_expert,
    )
    monkeypatch.setattr(
        "k3x_converter.official_two_layer._load_official_expert_bytes",
        lambda layer, expert_plans, expert_objects: (),
    )
    monkeypatch.setattr(
        "k3x_converter.official_two_layer.build_official_layer_source_tensors",
        lambda *args: (),
    )
    monkeypatch.setattr(
        "k3x_converter.official_two_layer.manufacture_official_two_layer_fixture",
        fake_manufacture,
    )

    assert main(
        [
            "--scope", "two-layer", "--materialize",
            "--output-dir", str(output),
        ],
        transport=_DiscoveryTransport(),
    ) == 0

    summary = json.loads(capsys.readouterr().out)
    expert_call_indices = [
        index for index, (_, length) in enumerate(calls) if length == 1
    ]
    assert summary["selected_experts"] == [[10, 11], [20, 21]]
    assert len(expert_call_indices) == 4
    assert expert_call_indices[0] == len(calls) - 4
    assert (output / "two-layer-route-state-manifest.json").is_file()
    oracle_payload = (output / "official-two-layer-oracle-v1.bin").read_bytes()
    oracle = parse_official_two_layer_oracle(oracle_payload)
    assert oracle.output.shape == (2, 7_168)
    assert torch.equal(oracle.output, torch.full_like(oracle.output, 2.0))
    assert len(oracle.states) == 2
    with pytest.raises(K3XError, match="INVALID_OFFICIAL_TWO_LAYER_ORACLE"):
        parse_official_two_layer_oracle(oracle_payload[:-1])


def test_cli_kda_layer_materialization_prints_canonical_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "artifacts"
    summary_path = tmp_path / "summary.json"
    captured: dict[str, object] = {}

    def fake_materialize(
        snapshot,
        index,
        config,
        header,
        plan,
        transport,
        output_dir,
        *,
        chunk_bytes,
    ):
        captured.update(output_dir=output_dir, chunk_bytes=chunk_bytes)
        return SimpleNamespace(
            selected_experts=(7, 8),
            requested_payload_bytes=1_267_744_268,
            downloaded_payload_bytes=654_321,
            reused_objects=4,
            requests=9,
            maximum_response_bytes=8 * 1024 * 1024,
            microshard_sha256="4" * 64,
            tensor_sha256={"tensor": "5" * 64},
            k3x_root_sha256="6" * 64,
            oracle_path=output / "official-layer-oracle-v1.bin",
            oracle_sha256="7" * 64,
            oracle_bytes=6_541_344,
        )

    monkeypatch.setattr(
        "tools.discover_official_kimi_k3.materialize_official_kda_layer",
        fake_materialize,
    )

    assert main(
        [
            "--scope", "kda-layer", "--materialize",
            "--output-dir", str(output),
            "--summary-json", str(summary_path),
        ],
        transport=_DiscoveryTransport(),
    ) == 0

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert json.loads(capsys.readouterr().out) == summary
    assert summary["format"] == "k3x-official-kda-layer-materialization-v1"
    assert summary["mode"] == "materialize"
    assert summary["selected_experts"] == [7, 8]
    assert summary["traffic"]["tensor_payload_bytes"] == 654_321
    assert summary["traffic"]["source_object_bytes"] == 1_267_744_268
    assert summary["artifacts"]["k3x_root_sha256"] == "6" * 64
    assert summary["artifacts"]["oracle_sha256"] == "7" * 64
    assert summary["artifacts"]["oracle_bytes"] == 6_541_344
    assert captured == {"output_dir": output.resolve(), "chunk_bytes": 257 * 1024}


def test_cli_kda_layer_rejects_source_blob_drift_before_payload() -> None:
    transport = _DiscoveryTransport()
    api = json.loads(transport.api)
    source = next(
        item
        for item in api["siblings"]
        if item["rfilename"] == "modeling_kimi_linear.py"
    )
    source["blobId"] = "0" * 40
    transport.api = json.dumps(api, separators=(",", ":")).encode()

    with pytest.raises(K3XError, match="INVALID_OFFICIAL_LAYER_SOURCE"):
        main(["--scope", "kda-layer"], transport=transport)

    assert transport.payload_requested is False


def test_cli_moe_materialization_requires_scope_and_output_directory() -> None:
    with pytest.raises(SystemExit):
        main(["--scope", "moe-ffn", "--materialize"], transport=_DiscoveryTransport())
    with pytest.raises(SystemExit):
        main(["--materialize", "--output-dir", "elsewhere"], transport=_DiscoveryTransport())


def test_cli_moe_materialization_prints_canonical_report(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    output = tmp_path / "artifacts"
    summary_path = tmp_path / "summary.json"
    captured: dict[str, object] = {}

    def fake_materialize(
        snapshot,
        index,
        config,
        header,
        plan,
        transport,
        output_dir,
        *,
        chunk_bytes,
    ):
        captured.update(output_dir=output_dir, chunk_bytes=chunk_bytes)
        return SimpleNamespace(
            selected_experts=(7, 8),
            requested_payload_bytes=379_900_428,
            downloaded_payload_bytes=123_456,
            reused_objects=0,
            requests=13,
            maximum_response_bytes=8 * 1024 * 1024,
            microshard_sha256="4" * 64,
            tensor_sha256={"tensor": "5" * 64},
            k3x_root_sha256="6" * 64,
        )

    monkeypatch.setattr(
        "tools.discover_official_kimi_k3.materialize_official_moe_slice",
        fake_materialize,
    )

    assert main(
        [
            "--scope", "moe-ffn", "--materialize",
            "--output-dir", str(output),
            "--summary-json", str(summary_path),
        ],
        transport=_DiscoveryTransport(),
    ) == 0

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    assert json.loads(capsys.readouterr().out) == summary
    assert summary["format"] == "k3x-official-moe-materialization-v1"
    assert summary["mode"] == "materialize"
    assert summary["selected_experts"] == [7, 8]
    assert summary["traffic"]["tensor_payload_bytes"] == 123_456
    assert summary["traffic"]["source_object_bytes"] == 379_900_428
    assert summary["artifacts"]["k3x_root_sha256"] == "6" * 64
    assert captured == {"output_dir": output.resolve(), "chunk_bytes": 257 * 1024}


def test_cli_requires_explicit_live_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("K3X_TEST_OFFICIAL_DISCOVERY", raising=False)

    with pytest.raises(K3XError, match="OFFICIAL_LIVE_OPT_IN_REQUIRED"):
        main([])


def test_cli_materialization_requires_untracked_output_directory(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["--materialize-expert"], transport=_DiscoveryTransport())

    repository_results = Path(__file__).resolve().parents[2] / "results" / "forbidden"
    with pytest.raises(K3XError, match="OFFICIAL_OUTPUT_LOCATION"):
        main(
            ["--materialize-expert", "--output-dir", str(repository_results)],
            transport=_DiscoveryTransport(),
        )


def test_cli_materializes_and_writes_verifiable_json_csv(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "artifacts"
    summary_json = tmp_path / "summary.json"
    summary_csv = tmp_path / "summary.csv"
    transport = _DiscoveryTransport()

    assert main(
        [
            "--materialize-expert",
            "--output-dir",
            str(output),
            "--summary-json",
            str(summary_json),
            "--summary-csv",
            str(summary_csv),
        ],
        transport=transport,
    ) == 0
    capsys.readouterr()

    summary = verify_summary(summary_json, summary_csv, strict_official=False)
    assert summary["mode"] == "materialize-expert"
    assert summary["traffic"]["tensor_payload_bytes"] == 17_547_264
    assert summary["reader_valid"] is True
    assert summary["optional_features"] == OPTIONAL_STORAGE_FIXTURE
    assert transport.payload_requested is True
    with summary_csv.open(newline="", encoding="utf-8") as stream:
        assert len(list(csv.DictReader(stream))) == 1


def test_verifier_rejects_consistently_rehashed_invalid_artifact_digest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    summary_json = tmp_path / "summary.json"
    summary_csv = tmp_path / "summary.csv"
    main(
        [
            "--materialize-expert",
            "--output-dir",
            str(tmp_path / "artifacts"),
            "--summary-json",
            str(summary_json),
            "--summary-csv",
            str(summary_csv),
        ],
        transport=_DiscoveryTransport(),
    )
    capsys.readouterr()
    record = json.loads(summary_json.read_text(encoding="utf-8"))
    record["artifacts"]["payload_sha256"] = "not-a-digest"
    record.pop("record_sha256")
    record.pop("summary_csv_sha256")
    record["record_sha256"] = canonical_record_sha256(record)
    with summary_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerow(summary_csv_row(record))
    record["summary_csv_sha256"] = hashlib.sha256(summary_csv.read_bytes()).hexdigest()
    summary_json.write_text(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(K3XError, match="INVALID_OFFICIAL_EVIDENCE"):
        verify_summary(summary_json, summary_csv, strict_official=False)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(repository="other/model"),
        lambda value: value.update(requested_revision="latest"),
        lambda value: value.update(repository_bytes=1),
        lambda value: value.update(snapshot_sha256="0" * 64),
        lambda value: value["config"].update(git_blob_id="0" * 40),
        lambda value: value["index"].update(tensor_count=1),
        lambda value: value["expert"].update(shard_path="other.safetensors"),
        lambda value: value["artifacts"].update(payload_sha256="0" * 64),
        lambda value: value["artifacts"]["tensor_sha256"].update(
            {"model.layers.1.feed_forward.experts.0.down.weight_packed": "0" * 64}
        ),
    ],
)
def test_strict_verifier_binds_official_snapshot_and_layout_identity(
    tmp_path: Path, mutation
) -> None:
    root = Path(__file__).resolve().parents[2]
    record = json.loads(
        (root / "results/b0027-official-range/summary.json").read_text(
            encoding="utf-8"
        )
    )
    mutation(record)
    record.pop("record_sha256")
    record.pop("summary_csv_sha256")
    record["record_sha256"] = canonical_record_sha256(record)
    summary_csv = tmp_path / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerow(summary_csv_row(record))
    record["summary_csv_sha256"] = hashlib.sha256(summary_csv.read_bytes()).hexdigest()
    summary_json = tmp_path / "summary.json"
    summary_json.write_text(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(K3XError, match="OFFICIAL_IDENTITY_MISMATCH"):
        verify_summary(summary_json, summary_csv)
