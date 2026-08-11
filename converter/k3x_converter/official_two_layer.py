# 공식 Kimi K3 레이어 1·2의 의존형 제조 계획을 구성합니다.
from __future__ import annotations

from dataclasses import dataclass

from .format import K3XError
from .official_layer import OFFICIAL_KDA_SOURCE_BLOB_ID, OfficialLayerPlan


_LAYER_IDS = (1, 2)
_KDA_PAYLOAD_BYTES = 887_843_840
_ALWAYS_ACTIVE_BYTES = 379_900_416
_BASE_PAYLOAD_BYTES = 1_267_744_256
_EXPERT_PAYLOAD_BYTES = 17_547_264
_MAXIMUM_TWO_POSITION_BYTES = 1_829_256_704


@dataclass(frozen=True)
class OfficialTwoLayerPlan:
    layers: tuple[OfficialLayerPlan, OfficialLayerPlan]
    layer_ids: tuple[int, int]
    shard_paths: tuple[str, str]
    base_payload_bytes: int
    maximum_two_position_bytes: int


def plan_official_two_layer(
    first: OfficialLayerPlan,
    second: OfficialLayerPlan,
) -> OfficialTwoLayerPlan:
    layers = (first, second)
    if (
        tuple(layer.layer_id for layer in layers) != _LAYER_IDS
        or first.index_sha256 != second.index_sha256
        or first.source_blob_id != second.source_blob_id
        or first.source_blob_id != OFFICIAL_KDA_SOURCE_BLOB_ID
        or any(
            layer.kda_payload_bytes != _KDA_PAYLOAD_BYTES
            or layer.base_payload_bytes != _BASE_PAYLOAD_BYTES
            or layer.maximum_two_token_bytes != _MAXIMUM_TWO_POSITION_BYTES
            or layer.moe_plan.layer_id != layer.layer_id
            or layer.moe_plan.shard_path != layer.shard_path
            or layer.moe_plan.index_sha256 != layer.index_sha256
            or layer.moe_plan.always_active_bytes != _ALWAYS_ACTIVE_BYTES
            or layer.moe_plan.expert_payload_bytes != _EXPERT_PAYLOAD_BYTES
            for layer in layers
        )
    ):
        raise K3XError("INVALID_OFFICIAL_TWO_LAYER_PLAN")
    return OfficialTwoLayerPlan(
        layers,
        _LAYER_IDS,
        (first.shard_path, second.shard_path),
        sum(layer.base_payload_bytes for layer in layers),
        sum(layer.maximum_two_token_bytes for layer in layers),
    )
