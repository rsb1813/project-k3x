# 공식 Kimi K3 layer-1 KDA 전체 경계의 tensor 계획을 검증합니다.
from __future__ import annotations

from dataclasses import dataclass

from .format import K3XError
from .official_moe import OfficialMoePlan, plan_official_moe_slice
from .official_source import (
    OfficialConfig,
    OfficialIndex,
    OfficialShardHeader,
    PlannedTensor,
)


_SOURCE_BLOB_ID = "b8c41e8bfce768d74d8da3a37e693f5ee43876a0"
_KDA_PAYLOAD_BYTES = 887_843_840
_BASE_PAYLOAD_BYTES = 1_267_744_256
_KDA_TENSORS = (
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


@dataclass(frozen=True)
class OfficialLayerPlan:
    layer_id: int
    shard_path: str
    index_sha256: str
    source_blob_id: str
    kda_tensors: tuple[PlannedTensor, ...]
    kda_payload_bytes: int
    moe_plan: OfficialMoePlan
    base_payload_bytes: int
    maximum_two_token_bytes: int


def plan_official_kda_layer(
    index: OfficialIndex,
    header: OfficialShardHeader,
    config: OfficialConfig,
    *,
    source_blob_id: str,
    layer_id: int = 1,
) -> OfficialLayerPlan:
    if (
        layer_id != 1
        or source_blob_id != _SOURCE_BLOB_ID
        or config.num_hidden_layers != 93
        or layer_id not in config.kda_layers
        or config.kda_heads != 96
        or config.kda_head_dim != 128
        or config.short_conv_kernel_size != 4
        or config.kda_gate_lower_bound != -5.0
        or config.kda_use_full_rank_gate is not True
        or config.attn_res_block_size != 12
    ):
        raise K3XError("INVALID_OFFICIAL_LAYER")

    prefix = f"language_model.model.layers.{layer_id}"
    planned: list[PlannedTensor] = []
    for suffix, dtype, shape, role in _KDA_TENSORS:
        official_name = f"{prefix}.{suffix}"
        metadata = header.tensors.get(official_name)
        values = 1
        for dimension in shape:
            values *= dimension
        expected_length = values * (4 if dtype == "F32" else 2)
        if (
            index.weight_map.get(official_name) != header.shard_path
            or metadata is None
            or metadata.dtype != dtype
            or metadata.shape != shape
            or metadata.length != expected_length
            or metadata.offset < header.data_start
            or metadata.offset + metadata.length > header.file_size
        ):
            raise K3XError("INVALID_OFFICIAL_LAYER", official_name)
        planned.append(
            PlannedTensor(
                official_name,
                f"model.layers.{layer_id}.{suffix}",
                role,
                dtype,
                shape,
                metadata.offset,
                metadata.length,
            )
        )
    total = sum(item.length for item in planned)
    if total != _KDA_PAYLOAD_BYTES:
        raise K3XError("INVALID_OFFICIAL_LAYER")
    moe_plan = plan_official_moe_slice(index, header, config, layer_id=layer_id)
    if total + moe_plan.always_active_bytes != _BASE_PAYLOAD_BYTES:
        raise K3XError("INVALID_OFFICIAL_LAYER")
    return OfficialLayerPlan(
        layer_id,
        header.shard_path,
        index.sha256,
        source_blob_id,
        tuple(planned),
        total,
        moe_plan,
        _BASE_PAYLOAD_BYTES,
        _BASE_PAYLOAD_BYTES + 32 * moe_plan.expert_payload_bytes,
    )
