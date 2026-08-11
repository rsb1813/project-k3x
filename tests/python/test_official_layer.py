# 공식 Kimi K3 layer-1 KDA 텐서 계획의 고정 계약을 검증합니다.
from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import replace
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import pytest
import torch

from k3x_converter.format import K3XError
from k3x_converter.official_layer import (
    OfficialLayerRouteStep,
    OfficialLayerRoutes,
    build_official_layer_source_tensors,
    derive_official_layer_routes,
    materialize_official_kda_layer,
    official_layer_inputs,
    plan_official_kda_layer,
)
from k3x_converter.official_moe import (
    AssembledOfficialMoeSource,
    MaterializedRangeObject,
    OfficialMoeRoute,
)
from k3x_converter.official_source import (
    OfficialConfig,
    OfficialFile,
    OfficialIndex,
    OfficialShardHeader,
    OfficialSnapshot,
    PlannedTensor,
)
from k3x_converter.safetensors_reader import TensorMetadata
from k3x_converter.writer import ConversionReport
from k3x_ref.official_kda import OfficialKdaResult, OfficialKdaState


_SHARD = "model-00002-of-000096.safetensors"
_PREFIX = "language_model.model.layers.1"
_SOURCE_BLOB = "b8c41e8bfce768d74d8da3a37e693f5ee43876a0"
_COMMIT = "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"
_KDA_LAYERS = tuple(index for index in range(1, 92) if index % 4 != 0)
_KDA_SPECS = (
    ("self_attention_res_norm.weight", "BF16", (7_168,), "self_res_norm"),
    ("self_attention_res_proj.weight", "BF16", (1, 7_168), "self_res_proj"),
    ("input_layernorm.weight", "BF16", (7_168,), "input_norm"),
    ("self_attn.q_proj.weight", "BF16", (12_288, 7_168), "kda_q_proj"),
    ("self_attn.q_conv1d.weight", "F32", (12_288, 1, 4), "kda_q_conv"),
    ("self_attn.k_proj.weight", "BF16", (12_288, 7_168), "kda_k_proj"),
    ("self_attn.k_conv1d.weight", "F32", (12_288, 1, 4), "kda_k_conv"),
    ("self_attn.v_proj.weight", "BF16", (12_288, 7_168), "kda_v_proj"),
    ("self_attn.v_conv1d.weight", "F32", (12_288, 1, 4), "kda_v_conv"),
    ("self_attn.f_a_proj.weight", "BF16", (128, 7_168), "kda_f_a"),
    ("self_attn.f_b_proj.weight", "BF16", (12_288, 128), "kda_f_b"),
    ("self_attn.A_log", "F32", (128,), "kda_a_log"),
    ("self_attn.dt_bias", "F32", (12_288,), "kda_dt_bias"),
    ("self_attn.b_proj.weight", "BF16", (96, 7_168), "kda_beta"),
    ("self_attn.g_proj.weight", "BF16", (12_288, 7_168), "kda_output_gate"),
    ("self_attn.o_norm.weight", "F32", (128,), "kda_output_norm"),
    ("self_attn.o_proj.weight", "BF16", (7_168, 12_288), "kda_output_proj"),
)
_MOE_SPECS = (
    ("mlp_res_norm.weight", "BF16", (7_168,)),
    ("mlp_res_proj.weight", "BF16", (1, 7_168)),
    ("post_attention_layernorm.weight", "BF16", (7_168,)),
    ("block_sparse_moe.gate.weight", "BF16", (896, 7_168)),
    ("block_sparse_moe.gate.e_score_correction_bias", "F32", (896,)),
    ("block_sparse_moe.routed_expert_down_proj.weight", "BF16", (3_584, 7_168)),
    ("block_sparse_moe.routed_expert_norm.weight", "BF16", (3_584,)),
    ("block_sparse_moe.routed_expert_up_proj.weight", "BF16", (7_168, 3_584)),
    ("block_sparse_moe.shared_experts.gate_proj.weight", "BF16", (6_144, 7_168)),
    ("block_sparse_moe.shared_experts.up_proj.weight", "BF16", (6_144, 7_168)),
    ("block_sparse_moe.shared_experts.down_proj.weight", "BF16", (7_168, 6_144)),
)


def _config() -> OfficialConfig:
    return OfficialConfig(
        "5" * 64,
        "6" * 40,
        7_168,
        896,
        16,
        3_584,
        3_072,
        2,
        4.0,
        25.0,
        True,
        1.0e-5,
        True,
        "sigmoid",
        1,
        1,
        1.0,
        93,
        _KDA_LAYERS,
        96,
        128,
        4,
        -5.0,
        True,
        12,
    )


def _metadata() -> dict[str, TensorMetadata]:
    result: dict[str, TensorMetadata] = {}
    offset = 818_704
    for suffix, dtype, shape, *_ in (*_KDA_SPECS, *_MOE_SPECS):
        name = f"{_PREFIX}.{suffix}"
        values = 1
        for dimension in shape:
            values *= dimension
        length = values * (4 if dtype == "F32" else 2)
        result[name] = TensorMetadata(name, dtype, shape, offset, length)
        offset += length
    return result


def _plan_inputs() -> tuple[OfficialIndex, OfficialShardHeader]:
    metadata = _metadata()
    weight_map = {name: _SHARD for name in metadata}
    index = OfficialIndex(
        sum(item.length for item in metadata.values()),
        MappingProxyType(weight_map),
        (_SHARD,),
        len(weight_map),
        "7" * 64,
    )
    header = OfficialShardHeader(
        _SHARD,
        16_990_911_504,
        818_696,
        818_704,
        MappingProxyType(metadata),
    )
    return index, header


def test_official_kda_layer_plan_binds_exact_execution_order_and_bytes() -> None:
    index, header = _plan_inputs()

    plan = plan_official_kda_layer(
        index,
        header,
        _config(),
        source_blob_id=_SOURCE_BLOB,
        layer_id=1,
    )

    assert tuple(item.role for item in plan.kda_tensors) == tuple(
        role for *_, role in _KDA_SPECS
    )
    assert plan.layer_id == 1
    assert plan.shard_path == _SHARD
    assert plan.index_sha256 == "7" * 64
    assert plan.source_blob_id == _SOURCE_BLOB
    assert plan.kda_payload_bytes == 887_843_840
    assert plan.base_payload_bytes == 1_267_744_256
    assert plan.maximum_two_token_bytes == 1_829_256_704
    assert next(
        item for item in plan.kda_tensors if item.role == "kda_a_log"
    ).shape == (128,)


def test_official_kda_layer_plan_rejects_head_shaped_a_log() -> None:
    index, header = _plan_inputs()
    tensors = dict(header.tensors)
    name = f"{_PREFIX}.self_attn.A_log"
    current = tensors[name]
    tensors[name] = TensorMetadata(name, "F32", (96,), current.offset, 96 * 4)
    bad_header = OfficialShardHeader(
        header.shard_path,
        header.file_size,
        header.header_length,
        header.data_start,
        MappingProxyType(tensors),
    )

    with pytest.raises(K3XError, match="INVALID_OFFICIAL_LAYER"):
        plan_official_kda_layer(
            index,
            bad_header,
            _config(),
            source_blob_id=_SOURCE_BLOB,
            layer_id=1,
        )


def test_official_layer_inputs_bind_fixed_formulas_and_fp32_hashes() -> None:
    inputs = official_layer_inputs()

    assert tuple(item.name for item in inputs) == ("a", "b")
    assert inputs[0].hidden_input[:3] == pytest.approx(
        (-125 / 1024, -108 / 1024, -91 / 1024)
    )
    assert inputs[0].block_source[:3] == pytest.approx(
        (-114 / 1024, -85 / 1024, -56 / 1024)
    )
    for item in inputs:
        assert len(item.hidden_input) == 7_168
        assert len(item.block_source) == 7_168
        assert item.hidden_sha256 == hashlib.sha256(
            struct.pack("<7168f", *item.hidden_input)
        ).hexdigest()
        assert item.block_sha256 == hashlib.sha256(
            struct.pack("<7168f", *item.block_source)
        ).hexdigest()


def test_official_layer_route_derivation_links_incremental_state_and_natural_router(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index, header = _plan_inputs()
    plan = plan_official_kda_layer(
        index, header, _config(), source_blob_id=_SOURCE_BLOB, layer_id=1
    )
    objects = {
        item.official_name: MaterializedRangeObject(
            tmp_path / f"{position}.blob", "0" * 64, item.length, False, 0, 0
        )
        for position, item in enumerate(
            (*plan.kda_tensors, *plan.moe_plan.always_active)
        )
    }
    by_role: dict[str, torch.Tensor] = {}
    for item in (*plan.kda_tensors, *plan.moe_plan.always_active):
        if item.role in {
            "self_res_norm", "input_norm", "mlp_res_norm", "post_attention_norm"
        }:
            value = torch.ones(7_168, dtype=torch.bfloat16)
        elif item.role in {"self_res_proj", "mlp_res_proj"}:
            value = torch.linspace(-0.1, 0.1, 7_168, dtype=torch.bfloat16).reshape(
                1, 7_168
            )
        elif item.role == "router":
            value = torch.zeros((896, 7_168), dtype=torch.bfloat16)
        elif item.role == "router_correction":
            value = torch.arange(896, dtype=torch.float32) / 896.0
        elif item.role in {"kda_q_conv", "kda_k_conv", "kda_v_conv"}:
            value = torch.zeros((12_288, 1, 4), dtype=torch.float32)
        else:
            value = torch.zeros(1, dtype=torch.float32)
        by_role[item.role] = value

    monkeypatch.setattr(
        "k3x_converter.official_layer._load_object_tensor",
        lambda item, object_set: by_role[item.role],
    )

    def make_state(count: int) -> OfficialKdaState:
        return OfficialKdaState(
            *(torch.full((1, 1, 1), count, dtype=torch.bfloat16) for _ in range(3)),
            torch.full((1, 1, 1, 1), count, dtype=torch.float32),
        )

    monkeypatch.setattr(
        "k3x_converter.official_layer.zero_official_kda_state",
        lambda *args, **kwargs: make_state(0),
    )

    def fake_kda(hidden, weights, state, config):
        count = int(state.recurrent_v_first.item())
        outputs = []
        for index in range(hidden.shape[1]):
            outputs.append((hidden[:, index].float() * 0.25).to(torch.bfloat16))
        output = torch.stack(outputs, dim=1)
        return OfficialKdaResult(
            output,
            make_state(count + hidden.shape[1]),
            SimpleNamespace(recurrent_output=output.float()),
        )

    monkeypatch.setattr("k3x_converter.official_layer.official_kda", fake_kda)

    routes = derive_official_layer_routes(plan, objects, official_layer_inputs())

    assert tuple(step.name for step in routes.steps) == ("a", "b")
    assert routes.steps[1].consumes_state_sha256 == routes.steps[0].state_sha256
    assert routes.final_state_sha256 == routes.steps[1].state_sha256
    assert len(routes.selected_experts) == 16
    assert routes.selected_experts == tuple(range(895, 879, -1))


def _small_expert_plan(expert_id: int, index_sha256: str):
    from k3x_converter.official_source import ExpertPlan

    tensors = tuple(
        PlannedTensor(
            f"official.{expert_id}.{role}.{kind}",
            f"model.layers.1.feed_forward.experts.{expert_id}.{role}.{kind}",
            role,
            "U8",
            (1,),
            100 + position,
            1,
        )
        for position, (role, kind) in enumerate(
            (
                ("gate", "weight_packed"),
                ("gate", "weight_scale"),
                ("down", "weight_packed"),
                ("down", "weight_scale"),
                ("up", "weight_packed"),
                ("up", "weight_scale"),
            )
        )
    )
    return ExpertPlan(1, expert_id, _SHARD, 100, 106, 6, index_sha256, tensors)


def test_official_layer_source_tensors_preserve_complete_execution_order(
    tmp_path: Path,
) -> None:
    index, header = _plan_inputs()
    plan = plan_official_kda_layer(
        index, header, _config(), source_blob_id=_SOURCE_BLOB, layer_id=1
    )
    routes = OfficialLayerRoutes(
        (
            OfficialLayerRouteStep(
                "a", "0" * 64, "1" * 64, "2" * 64,
                OfficialMoeRoute((7,), (1.0,)),
            ),
            OfficialLayerRouteStep(
                "b", "1" * 64, "3" * 64, "4" * 64,
                OfficialMoeRoute((8,), (1.0,)),
            ),
        ),
        (7, 8),
        "0" * 64,
        "3" * 64,
    )
    kda_objects = {
        item.official_name: MaterializedRangeObject(
            tmp_path / f"kda-{position}.blob", "a" * 64, item.length, False, 0, 0
        )
        for position, item in enumerate(plan.kda_tensors)
    }
    always_objects = {
        item.official_name: MaterializedRangeObject(
            tmp_path / f"always-{position}.blob", "b" * 64, item.length, False, 0, 0
        )
        for position, item in enumerate(plan.moe_plan.always_active)
    }
    expert_plans = {
        expert_id: _small_expert_plan(expert_id, index.sha256)
        for expert_id in routes.selected_experts
    }
    expert_objects = {
        expert_id: MaterializedRangeObject(
            tmp_path / f"expert-{expert_id}.blob", "c" * 64, 6, False, 0, 0
        )
        for expert_id in routes.selected_experts
    }

    tensors = build_official_layer_source_tensors(
        plan, routes, expert_plans, kda_objects, always_objects, expert_objects
    )

    assert tuple(item.name for item in tensors[:17]) == tuple(
        item.canonical_name for item in plan.kda_tensors
    )
    assert tensors[17].name.endswith("mlp_res_norm.weight")
    expert_weights = [
        item.name
        for item in tensors
        if ".feed_forward.experts." in item.name
        and item.name.endswith("weight_packed")
    ]
    assert [name.split(".experts.", 1)[1].split(".", 1)[0] for name in expert_weights] == [
        "7", "7", "7", "8", "8", "8"
    ]
    assert tensors[-1].name.endswith("shared_experts.down_proj.weight")


def test_official_layer_materializer_publishes_state_routes_before_experts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    index, header = _plan_inputs()
    config = _config()
    plan = plan_official_kda_layer(
        index, header, config, source_blob_id=_SOURCE_BLOB, layer_id=1
    )
    snapshot = OfficialSnapshot(
        "moonshotai/Kimi-K3",
        "main",
        _COMMIT,
        "2026-08-11T00:00:00Z",
        MappingProxyType(
            {
                _SHARD: OfficialFile(_SHARD, header.file_size, "1" * 40, "2" * 64),
                "modeling_kimi_linear.py": OfficialFile(
                    "modeling_kimi_linear.py", 51_506, _SOURCE_BLOB, None
                ),
            }
        ),
        2,
        header.file_size + 51_506,
        "3" * 64,
    )
    routes = OfficialLayerRoutes(
        (
            OfficialLayerRouteStep(
                "a", "0" * 64, "1" * 64, "2" * 64,
                OfficialMoeRoute((7,), (1.0,)),
            ),
            OfficialLayerRouteStep(
                "b", "1" * 64, "3" * 64, "4" * 64,
                OfficialMoeRoute((8,), (1.0,)),
            ),
        ),
        (7, 8),
        "0" * 64,
        "3" * 64,
    )
    events: list[str] = []
    counter = 0

    def fake_object(*args, **kwargs):
        nonlocal counter
        counter += 1
        path = tmp_path / f"object-{counter}.blob"
        path.write_bytes(b"x")
        return MaterializedRangeObject(
            path, f"{counter:064x}", args[3], False, 1, 1, 1
        )

    def fake_routes(*args, **kwargs):
        assert counter == 28
        events.append("derive")
        return routes

    def fake_expert(*args, expert_id: int, **kwargs):
        manifest_path = tmp_path / "out" / "route-state-manifest.json"
        assert manifest_path.is_file()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        assert manifest["state_layout"] == "v-first-fp32"
        events.append(f"expert-{expert_id}")
        return _small_expert_plan(expert_id, index.sha256)

    assembled_dir = tmp_path / "assembled"
    assembled_dir.mkdir()
    source_manifest = assembled_dir / "source-manifest.json"
    microshard = assembled_dir / "model.safetensors"
    source_manifest.write_text("{}\n", encoding="utf-8")
    microshard.write_bytes(b"source")

    def fake_assemble(*args, **kwargs):
        assert kwargs["official_metadata_key"] == "official_layer"
        assert tuple(item.name for item in args[1][:17]) == tuple(
            item.canonical_name for item in plan.kda_tensors
        )
        events.append("assemble")
        return AssembledOfficialMoeSource(
            assembled_dir, source_manifest, microshard, "5" * 64, {"t": "6" * 64}
        )

    def fake_convert(source, output, **kwargs):
        events.append("convert")
        Path(output).write_bytes(b"k3x")
        return ConversionReport(True, (), 1, Path(output))

    class _Reader:
        class _Superblock:
            root_sha256 = bytes.fromhex("7" * 64)
            source_sha256 = bytes.fromhex("8" * 64)
            optional_features = 3

        superblock = _Superblock()

    import k3x_converter.official_layer as module

    monkeypatch.setattr(module, "materialize_official_range_object", fake_object)
    monkeypatch.setattr(module, "derive_official_layer_routes", fake_routes)
    monkeypatch.setattr(module, "plan_official_expert", fake_expert)
    monkeypatch.setattr(module, "assemble_official_moe_source", fake_assemble)
    monkeypatch.setattr(module, "convert", fake_convert)
    monkeypatch.setattr(module.K3XReader, "open", lambda path: _Reader())

    report = materialize_official_kda_layer(
        snapshot,
        index,
        config,
        header,
        plan,
        object(),
        tmp_path / "out",
        chunk_bytes=17,
    )

    assert events == ["derive", "expert-7", "expert-8", "assemble", "convert"]
    assert report.selected_experts == (7, 8)
    assert report.requested_payload_bytes == plan.base_payload_bytes + 12
    assert report.downloaded_payload_bytes == 28 + 2
    assert report.maximum_response_bytes == 1
    final_manifest = json.loads(report.route_manifest_path.read_text(encoding="utf-8"))
    assert final_manifest["steps"][1]["consumes_state_sha256"] == "1" * 64
    assert final_manifest["artifact"]["k3x_root_sha256"] == "7" * 64
    assert final_manifest["artifact"]["k3x_source_fingerprint_sha256"] == "8" * 64


def test_official_layer_materializer_rejects_plan_drift_before_payload(
    tmp_path: Path,
) -> None:
    index, header = _plan_inputs()
    config = _config()
    plan = plan_official_kda_layer(
        index, header, config, source_blob_id=_SOURCE_BLOB, layer_id=1
    )
    snapshot = OfficialSnapshot(
        "moonshotai/Kimi-K3",
        "main",
        _COMMIT,
        "2026-08-11T00:00:00Z",
        MappingProxyType(
            {
                _SHARD: OfficialFile(_SHARD, header.file_size, "1" * 40, "2" * 64),
                "modeling_kimi_linear.py": OfficialFile(
                    "modeling_kimi_linear.py", 51_506, _SOURCE_BLOB, None
                ),
            }
        ),
        2,
        header.file_size + 51_506,
        "3" * 64,
    )

    with pytest.raises(K3XError, match="INVALID_OFFICIAL_LAYER_MATERIALIZATION"):
        materialize_official_kda_layer(
            snapshot,
            index,
            config,
            header,
            replace(plan, base_payload_bytes=1),
            object(),
            tmp_path / "out",
        )

    assert not (tmp_path / "out").exists()
