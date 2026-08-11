# 공식 Kimi K3 레이어 1·2의 의존형 제조 계획을 구성합니다.
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
import struct
from typing import Callable

from .format import K3XError
from .official_layer import (
    OFFICIAL_KDA_SOURCE_BLOB_ID,
    OfficialLayerInput,
    OfficialLayerPlan,
)
from .official_moe import OfficialMoeRoute


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


@dataclass(frozen=True)
class OfficialTwoLayerState:
    value: object
    sha256: str


@dataclass(frozen=True)
class OfficialTwoLayerStepExecution:
    output: tuple[float, ...]
    state: OfficialTwoLayerState
    kda_output_sha256: str
    route: OfficialMoeRoute


@dataclass(frozen=True)
class OfficialTwoLayerTraceStep:
    position: str
    layer_id: int
    hidden_input_sha256: str
    block_sha256: str
    consumes_state_sha256: str
    state_sha256: str
    kda_output_sha256: str
    route: OfficialMoeRoute
    output_sha256: str


@dataclass(frozen=True)
class OfficialTwoLayerTrace:
    steps: tuple[
        OfficialTwoLayerTraceStep,
        OfficialTwoLayerTraceStep,
        OfficialTwoLayerTraceStep,
        OfficialTwoLayerTraceStep,
    ]
    selected_experts: tuple[tuple[int, ...], tuple[int, ...]]
    initial_state_sha256: tuple[str, str]
    final_state_sha256: tuple[str, str]
    outputs: tuple[tuple[float, ...], tuple[float, ...]]


OfficialTwoLayerExecutor = Callable[
    [OfficialLayerPlan, OfficialLayerInput, OfficialTwoLayerState],
    OfficialTwoLayerStepExecution,
]


def _float_digest(values: tuple[float, ...]) -> str:
    return hashlib.sha256(struct.pack(f"<{len(values)}f", *values)).hexdigest()


def _valid_digest(value: str) -> bool:
    if len(value) != 64:
        return False
    try:
        return bytes.fromhex(value).hex() == value
    except ValueError:
        return False


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


def derive_official_two_layer_trace(
    plan: OfficialTwoLayerPlan,
    inputs: tuple[OfficialLayerInput, OfficialLayerInput],
    initial_states: tuple[OfficialTwoLayerState, OfficialTwoLayerState],
    execute: OfficialTwoLayerExecutor,
) -> OfficialTwoLayerTrace:
    if (
        plan.layer_ids != _LAYER_IDS
        or tuple(item.name for item in inputs) != ("a", "b")
        or len(initial_states) != len(plan.layers)
        or not all(_valid_digest(state.sha256) for state in initial_states)
        or not callable(execute)
    ):
        raise K3XError("INVALID_OFFICIAL_TWO_LAYER_TRACE")
    for item in inputs:
        if (
            not item.hidden_input
            or len(item.hidden_input) != len(item.block_source)
            or not all(math.isfinite(value) for value in item.hidden_input)
            or not all(math.isfinite(value) for value in item.block_source)
            or item.hidden_sha256 != _float_digest(item.hidden_input)
            or item.block_sha256 != _float_digest(item.block_source)
        ):
            raise K3XError("INVALID_OFFICIAL_TWO_LAYER_TRACE")

    states = list(initial_states)
    steps: list[OfficialTwoLayerTraceStep] = []
    selected: list[list[int]] = [[], []]
    outputs: list[tuple[float, ...]] = []
    for item in inputs:
        current = item
        for layer_index, layer in enumerate(plan.layers):
            consumed = states[layer_index]
            result = execute(layer, current, consumed)
            if (
                len(result.output) != len(item.hidden_input)
                or not all(math.isfinite(value) for value in result.output)
                or not _valid_digest(result.state.sha256)
                or not _valid_digest(result.kda_output_sha256)
                or not result.route.expert_ids
                or len(result.route.expert_ids) != len(result.route.contributions)
                or len(set(result.route.expert_ids)) != len(result.route.expert_ids)
                or any(not 0 <= expert_id < 896 for expert_id in result.route.expert_ids)
                or any(
                    not math.isfinite(value) or value < 0.0
                    for value in result.route.contributions
                )
                or not math.isclose(
                    sum(result.route.contributions), 1.0, rel_tol=0.0, abs_tol=1.0e-6
                )
            ):
                raise K3XError("INVALID_OFFICIAL_TWO_LAYER_TRACE")
            output_sha256 = _float_digest(result.output)
            steps.append(
                OfficialTwoLayerTraceStep(
                    item.name,
                    layer.layer_id,
                    current.hidden_sha256,
                    current.block_sha256,
                    consumed.sha256,
                    result.state.sha256,
                    result.kda_output_sha256,
                    result.route,
                    output_sha256,
                )
            )
            states[layer_index] = result.state
            selected[layer_index].extend(
                expert_id
                for expert_id in result.route.expert_ids
                if expert_id not in selected[layer_index]
            )
            current = OfficialLayerInput(
                item.name,
                result.output,
                item.block_source,
                output_sha256,
                item.block_sha256,
            )
        outputs.append(current.hidden_input)
    return OfficialTwoLayerTrace(
        (steps[0], steps[1], steps[2], steps[3]),
        (tuple(selected[0]), tuple(selected[1])),
        (initial_states[0].sha256, initial_states[1].sha256),
        (states[0].sha256, states[1].sha256),
        (outputs[0], outputs[1]),
    )
