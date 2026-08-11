# 공식 Kimi K3 두 레이어 제조 계획과 의존 실행 순서를 검증합니다.
from __future__ import annotations

from dataclasses import replace

import pytest

from k3x_converter.format import K3XError
from k3x_converter.official_layer import OfficialLayerPlan
from k3x_converter.official_moe import OfficialMoePlan
from k3x_converter.official_two_layer import plan_official_two_layer


_SOURCE_BLOB = "b8c41e8bfce768d74d8da3a37e693f5ee43876a0"


def _layer_plan(layer_id: int) -> OfficialLayerPlan:
    shard = f"model-{layer_id + 1:05d}-of-000096.safetensors"
    moe = OfficialMoePlan(
        layer_id,
        shard,
        "7" * 64,
        (),
        379_900_416,
        17_547_264,
        941_412_864,
    )
    return OfficialLayerPlan(
        layer_id,
        shard,
        "7" * 64,
        _SOURCE_BLOB,
        (),
        887_843_840,
        moe,
        1_267_744_256,
        1_829_256_704,
    )


def test_official_two_layer_plan_binds_exact_order_and_byte_bounds() -> None:
    first = _layer_plan(1)
    second = _layer_plan(2)

    plan = plan_official_two_layer(first, second)

    assert plan.layers == (first, second)
    assert plan.layer_ids == (1, 2)
    assert plan.base_payload_bytes == 2_535_488_512
    assert plan.maximum_two_position_bytes == 3_658_513_408
    assert plan.shard_paths == (
        "model-00002-of-000096.safetensors",
        "model-00003-of-000096.safetensors",
    )


@pytest.mark.parametrize("layer_ids", [(2, 1), (1, 1), (2, 2)])
def test_official_two_layer_plan_rejects_noncanonical_layer_order(
    layer_ids: tuple[int, int],
) -> None:
    with pytest.raises(K3XError, match="INVALID_OFFICIAL_TWO_LAYER_PLAN"):
        plan_official_two_layer(*(_layer_plan(layer_id) for layer_id in layer_ids))


@pytest.mark.parametrize("field", ["index_sha256", "source_blob_id"])
def test_official_two_layer_plan_rejects_cross_layer_source_drift(field: str) -> None:
    first = _layer_plan(1)
    second = replace(_layer_plan(2), **{field: "8" * 64})

    with pytest.raises(K3XError, match="INVALID_OFFICIAL_TWO_LAYER_PLAN"):
        plan_official_two_layer(first, second)


def test_official_two_layer_plan_rejects_matching_unpinned_source_blobs() -> None:
    first = replace(_layer_plan(1), source_blob_id="8" * 40)
    second = replace(_layer_plan(2), source_blob_id="8" * 40)

    with pytest.raises(K3XError, match="INVALID_OFFICIAL_TWO_LAYER_PLAN"):
        plan_official_two_layer(first, second)
