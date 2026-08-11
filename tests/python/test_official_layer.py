# 공식 Kimi K3 layer-1 KDA 텐서 계획의 고정 계약을 검증합니다.
from __future__ import annotations

from types import MappingProxyType

import pytest

from k3x_converter.format import K3XError
from k3x_converter.official_layer import plan_official_kda_layer
from k3x_converter.official_source import (
    OfficialConfig,
    OfficialIndex,
    OfficialShardHeader,
)
from k3x_converter.safetensors_reader import TensorMetadata


_SHARD = "model-00002-of-000096.safetensors"
_PREFIX = "language_model.model.layers.1"
_SOURCE_BLOB = "b8c41e8bfce768d74d8da3a37e693f5ee43876a0"
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
