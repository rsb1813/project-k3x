# 공식 Kimi K3 두 레이어 제조 계획과 의존 실행 순서를 검증합니다.
from __future__ import annotations

from dataclasses import replace
import hashlib
import struct

import pytest

from k3x_converter.format import K3XError
from k3x_converter.official_layer import OfficialLayerPlan
from k3x_converter.official_layer import OfficialLayerInput
from k3x_converter.official_moe import OfficialMoePlan, OfficialMoeRoute
from k3x_converter.official_two_layer import (
    OfficialTwoLayerState,
    OfficialTwoLayerStepExecution,
    derive_official_two_layer_trace,
    plan_official_two_layer,
)


_SOURCE_BLOB = "b8c41e8bfce768d74d8da3a37e693f5ee43876a0"


def _text_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _float_digest(values: tuple[float, ...]) -> str:
    return hashlib.sha256(struct.pack(f"<{len(values)}f", *values)).hexdigest()


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


def test_official_two_layer_trace_interleaves_positions_and_layer_states() -> None:
    plan = plan_official_two_layer(_layer_plan(1), _layer_plan(2))
    inputs = (
        OfficialLayerInput(
            "a",
            (1.0, 2.0),
            (10.0, 20.0),
            _float_digest((1.0, 2.0)),
            _float_digest((10.0, 20.0)),
        ),
        OfficialLayerInput(
            "b",
            (3.0, 4.0),
            (30.0, 40.0),
            _float_digest((3.0, 4.0)),
            _float_digest((30.0, 40.0)),
        ),
    )
    layer_1_initial = _text_digest("layer-1-initial")
    layer_2_initial = _text_digest("layer-2-initial")
    states = (
        OfficialTwoLayerState(1, layer_1_initial),
        OfficialTwoLayerState(2, layer_2_initial),
    )
    calls: list[tuple[str, int, tuple[float, ...], tuple[float, ...], str]] = []

    def execute(layer, item, state):
        calls.append(
            (
                item.name,
                layer.layer_id,
                item.hidden_input,
                item.block_source,
                state.sha256,
            )
        )
        output = tuple(value + layer.layer_id for value in item.hidden_input)
        return OfficialTwoLayerStepExecution(
            output,
            OfficialTwoLayerState(
                state.value + 10,
                _text_digest(f"layer-{layer.layer_id}-{item.name}"),
            ),
            _text_digest(f"kda-{layer.layer_id}-{item.name}"),
            OfficialMoeRoute((layer.layer_id, layer.layer_id + 10), (0.75, 0.25)),
        )

    trace = derive_official_two_layer_trace(plan, inputs, states, execute)

    assert tuple((step.position, step.layer_id) for step in trace.steps) == (
        ("a", 1),
        ("a", 2),
        ("b", 1),
        ("b", 2),
    )
    assert calls == [
        ("a", 1, (1.0, 2.0), (10.0, 20.0), layer_1_initial),
        ("a", 2, (2.0, 3.0), (10.0, 20.0), layer_2_initial),
        ("b", 1, (3.0, 4.0), (30.0, 40.0), _text_digest("layer-1-a")),
        ("b", 2, (4.0, 5.0), (30.0, 40.0), _text_digest("layer-2-a")),
    ]
    assert trace.steps[1].hidden_input_sha256 == trace.steps[0].output_sha256
    assert trace.steps[3].hidden_input_sha256 == trace.steps[2].output_sha256
    assert trace.steps[2].consumes_state_sha256 == trace.steps[0].state_sha256
    assert trace.steps[3].consumes_state_sha256 == trace.steps[1].state_sha256
    assert trace.selected_experts == ((1, 11), (2, 12))
    assert trace.final_state_sha256 == (
        _text_digest("layer-1-b"),
        _text_digest("layer-2-b"),
    )
    assert trace.outputs == ((4.0, 5.0), (6.0, 7.0))
