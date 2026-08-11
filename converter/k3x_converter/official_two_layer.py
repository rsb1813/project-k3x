# 공식 Kimi K3 레이어 1·2의 의존형 제조 계획을 구성합니다.
from __future__ import annotations

from dataclasses import dataclass, replace
import hashlib
import json
import math
import os
from pathlib import Path
import re
import struct
from typing import Callable

import torch

from k3x_ref.mxfp4 import mxfp4_matmul
from k3x_ref.official_kda import (
    OfficialKdaConfig,
    OfficialKdaState,
    OfficialKdaWeights,
    official_kda,
    zero_official_kda_state,
)

from .format import K3XError
from .official_layer import (
    OFFICIAL_KDA_SOURCE_BLOB_ID,
    OfficialLayerInput,
    OfficialLayerPlan,
    OfficialLayerRouteStep,
    OfficialLayerRoutes,
    _attention_residual,
    _rms_norm,
    _state_digest,
    _tensor_digest,
    build_official_layer_source_tensors,
    official_layer_inputs,
    plan_official_kda_layer,
)
from .official_moe import (
    MaterializedRangeObject,
    OfficialMoeSourceTensor,
    OfficialMoeRoute,
    assemble_official_moe_source,
    materialize_official_range_object,
    prepare_official_moe_hidden,
    route_official_hidden,
)
from .official_source import (
    ExpertPlan,
    OfficialConfig,
    OfficialIndex,
    OfficialShardHeader,
    OfficialSnapshot,
    Transport,
    _released_storage_config,
    plan_official_expert,
)
from .reader import K3XReader
from .writer import convert


_LAYER_IDS = (1, 2)
_KDA_PAYLOAD_BYTES = 887_843_840
_ALWAYS_ACTIVE_BYTES = 379_900_416
_BASE_PAYLOAD_BYTES = 1_267_744_256
_EXPERT_PAYLOAD_BYTES = 17_547_264
_MAXIMUM_TWO_POSITION_BYTES = 1_829_256_704
_LAYER_NAME = re.compile(r"^model\.layers\.(1|2)\.")
_EXPERT_SOURCE_NAME = re.compile(
    r"^model\.layers\.(1|2)\.feed_forward\.experts\.(\d+)\."
    r"(gate|up|down)\.weight_(packed|scale)$"
)


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
class OfficialSourceTensorBytes:
    dtype: str
    shape: tuple[int, ...]
    payload: bytes


@dataclass(frozen=True)
class OfficialMxfp4MatrixBytes:
    packed: bytes
    scales: bytes
    rows: int
    cols: int
    group_size: int = 32


@dataclass(frozen=True)
class OfficialMxfp4ExpertBytes:
    expert_id: int
    gate: OfficialMxfp4MatrixBytes
    up: OfficialMxfp4MatrixBytes
    down: OfficialMxfp4MatrixBytes


@dataclass(frozen=True)
class OfficialLayerSourceBytes:
    layer_id: int
    kda_config: OfficialKdaConfig
    tensors: tuple[tuple[str, OfficialSourceTensorBytes], ...]
    experts: tuple[OfficialMxfp4ExpertBytes, ...]
    top_k: int
    rms_norm_epsilon: float
    situ_beta: float
    situ_linear_beta: float | None


@dataclass(frozen=True)
class OfficialTwoLayerMaterializationReport:
    source_directory: Path
    manifest_path: Path
    microshard_path: Path
    k3x_path: Path
    completed: bool
    maximum_source_read_bytes: int
    selected_experts: tuple[tuple[int, ...], tuple[int, ...]] = ((), ())
    requested_payload_bytes: int = 0
    downloaded_payload_bytes: int = 0
    reused_objects: int = 0
    requests: int = 0
    maximum_response_bytes: int = 0
    microshard_sha256: str = ""
    tensor_sha256: dict[str, str] | None = None
    k3x_root_sha256: str = ""
    trace: OfficialTwoLayerTrace | None = None
    route_manifest_path: Path | None = None
    oracle_path: Path | None = None
    oracle_sha256: str = ""
    oracle_bytes: int = 0


@dataclass(frozen=True)
class OfficialTwoLayerOracle:
    output: torch.Tensor
    states: tuple[OfficialKdaState, OfficialKdaState]


@dataclass(frozen=True)
class OfficialPreparedSourceStep:
    layer_id: int
    prefix: torch.Tensor
    latent: torch.Tensor
    shared: torch.Tensor
    routed_norm: torch.Tensor
    routed_up: torch.Tensor
    state: OfficialTwoLayerState
    kda_output_sha256: str
    route: OfficialMoeRoute


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
    contribution_sha256: str
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


def _contribution_digest(route: OfficialMoeRoute) -> str:
    payload = struct.pack(f"<{len(route.expert_ids)}I", *route.expert_ids)
    payload += struct.pack(f"<{len(route.contributions)}f", *route.contributions)
    return hashlib.sha256(payload).hexdigest()


def _valid_digest(value: str) -> bool:
    if len(value) != 64:
        return False
    try:
        return bytes.fromhex(value).hex() == value
    except ValueError:
        return False


_TWO_LAYER_ORACLE_HEADER = struct.Struct("<8sQQQQ")
_TWO_LAYER_ORACLE_MAGIC = b"K3XORC2\0"
_TWO_LAYER_OUTPUT_VALUES = 2 * 7_168
_TWO_LAYER_CONV_VALUES = 3 * 12_288
_TWO_LAYER_RECURRENT_VALUES = 96 * 128 * 128


def _tensor_bytes(value: torch.Tensor) -> bytes:
    return value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def _official_two_layer_oracle_payload(
    trace: OfficialTwoLayerTrace,
    states: tuple[OfficialTwoLayerState, OfficialTwoLayerState],
) -> bytes:
    output = torch.tensor(trace.outputs, dtype=torch.bfloat16)
    values = tuple(state.value for state in states)
    if (
        output.shape != (2, 7_168)
        or any(not isinstance(value, OfficialKdaState) for value in values)
        or any(
            value.conv_q.shape != (1, 3, 12_288)
            or value.conv_k.shape != (1, 3, 12_288)
            or value.conv_v.shape != (1, 3, 12_288)
            or value.recurrent_v_first.shape != (1, 96, 128, 128)
            or value.conv_q.dtype != torch.bfloat16
            or value.conv_k.dtype != torch.bfloat16
            or value.conv_v.dtype != torch.bfloat16
            or value.recurrent_v_first.dtype != torch.float32
            for value in values
        )
    ):
        raise K3XError("INVALID_OFFICIAL_TWO_LAYER_ORACLE")
    return b"".join(
        (
            _TWO_LAYER_ORACLE_HEADER.pack(
                _TWO_LAYER_ORACLE_MAGIC,
                _TWO_LAYER_OUTPUT_VALUES,
                len(values),
                _TWO_LAYER_CONV_VALUES,
                _TWO_LAYER_RECURRENT_VALUES,
            ),
            _tensor_bytes(output),
            *(
                payload
                for value in values
                for payload in (
                    _tensor_bytes(value.conv_q),
                    _tensor_bytes(value.conv_k),
                    _tensor_bytes(value.conv_v),
                    _tensor_bytes(value.recurrent_v_first),
                )
            ),
        )
    )


def parse_official_two_layer_oracle(payload: bytes) -> OfficialTwoLayerOracle:
    expected_bytes = (
        _TWO_LAYER_ORACLE_HEADER.size
        + _TWO_LAYER_OUTPUT_VALUES * 2
        + 2
        * (
            3 * _TWO_LAYER_CONV_VALUES * 2
            + _TWO_LAYER_RECURRENT_VALUES * 4
        )
    )
    if len(payload) != expected_bytes:
        raise K3XError("INVALID_OFFICIAL_TWO_LAYER_ORACLE")
    magic, output_values, layers, conv_values, recurrent_values = (
        _TWO_LAYER_ORACLE_HEADER.unpack_from(payload)
    )
    if (
        magic != _TWO_LAYER_ORACLE_MAGIC
        or output_values != _TWO_LAYER_OUTPUT_VALUES
        or layers != 2
        or conv_values != _TWO_LAYER_CONV_VALUES
        or recurrent_values != _TWO_LAYER_RECURRENT_VALUES
    ):
        raise K3XError("INVALID_OFFICIAL_TWO_LAYER_ORACLE")
    storage = bytearray(payload[_TWO_LAYER_ORACLE_HEADER.size :])
    words = torch.frombuffer(storage, dtype=torch.uint16)
    cursor = _TWO_LAYER_OUTPUT_VALUES
    output = words[:cursor].view(torch.bfloat16).reshape(2, 7_168).clone()
    states: list[OfficialKdaState] = []
    for _ in range(2):
        conv_values_by_kind = []
        for _ in range(3):
            end = cursor + _TWO_LAYER_CONV_VALUES
            conv_values_by_kind.append(
                words[cursor:end]
                .view(torch.bfloat16)
                .reshape(1, 3, 12_288)
                .clone()
            )
            cursor = end
        recurrent_words = _TWO_LAYER_RECURRENT_VALUES * 2
        end = cursor + recurrent_words
        recurrent = (
            words[cursor:end]
            .view(torch.float32)
            .reshape(1, 96, 128, 128)
            .clone()
        )
        cursor = end
        states.append(
            OfficialKdaState(
                conv_values_by_kind[0],
                conv_values_by_kind[1],
                conv_values_by_kind[2],
                recurrent,
            )
        )
    if cursor != words.numel():
        raise K3XError("INVALID_OFFICIAL_TWO_LAYER_ORACLE")
    return OfficialTwoLayerOracle(output, (states[0], states[1]))


def official_two_layer_state(value: OfficialKdaState) -> OfficialTwoLayerState:
    return OfficialTwoLayerState(value, _state_digest(value))


def _decode_source_tensor(source: OfficialSourceTensorBytes) -> torch.Tensor:
    dtype = {"BF16": torch.bfloat16, "F32": torch.float32}.get(source.dtype)
    if (
        dtype is None
        or not source.shape
        or any(dimension <= 0 for dimension in source.shape)
    ):
        raise K3XError("INVALID_OFFICIAL_TWO_LAYER_SOURCE")
    expected = math.prod(source.shape) * (2 if dtype == torch.bfloat16 else 4)
    if len(source.payload) != expected:
        raise K3XError("INVALID_OFFICIAL_TWO_LAYER_SOURCE")
    return torch.frombuffer(bytearray(source.payload), dtype=dtype).reshape(
        source.shape
    ).clone()


def _bf16_matvec(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    if (
        value.ndim != 1
        or weight.ndim != 2
        or weight.dtype != torch.bfloat16
        or weight.shape[1] != value.shape[0]
        or not torch.isfinite(value).all()
        or not torch.isfinite(weight).all()
    ):
        raise K3XError("INVALID_OFFICIAL_TWO_LAYER_SOURCE")
    return (weight.float() @ value.float()).to(torch.bfloat16)


def _situ(
    gate: torch.Tensor,
    up: torch.Tensor,
    beta: float,
    linear_beta: float | None,
) -> torch.Tensor:
    gate_float = gate.float()
    up_float = up.float()
    bounded_gate = (
        beta * torch.tanh(gate_float / beta) * torch.sigmoid(gate_float)
    )
    if linear_beta is not None:
        up_float = linear_beta * torch.tanh(up_float / linear_beta)
    return bounded_gate * up_float


def _dense_ffn(
    value: torch.Tensor,
    gate: torch.Tensor,
    up: torch.Tensor,
    down: torch.Tensor,
    beta: float,
    linear_beta: float | None,
) -> torch.Tensor:
    return _bf16_matvec(
        _situ(
            _bf16_matvec(value, gate),
            _bf16_matvec(value, up),
            beta,
            linear_beta,
        ),
        down,
    )


def _mxfp4_forward(
    value: torch.Tensor,
    expert: OfficialMxfp4ExpertBytes,
    beta: float,
    linear_beta: float | None,
) -> torch.Tensor:
    def project(
        source: torch.Tensor, matrix: OfficialMxfp4MatrixBytes
    ) -> torch.Tensor:
        try:
            return mxfp4_matmul(
                source,
                matrix.packed,
                matrix.scales,
                matrix.rows,
                matrix.cols,
                matrix.group_size,
            )
        except ValueError as error:
            raise K3XError("INVALID_OFFICIAL_TWO_LAYER_SOURCE") from error

    gate = project(value, expert.gate)
    up = project(value, expert.up)
    if gate.shape != up.shape:
        raise K3XError("INVALID_OFFICIAL_TWO_LAYER_SOURCE")
    return project(_situ(gate, up, beta, linear_beta), expert.down).to(
        torch.bfloat16
    )


def prepare_official_source_step(
    layer: OfficialLayerPlan,
    item: OfficialLayerInput,
    state: OfficialTwoLayerState,
    source: OfficialLayerSourceBytes,
) -> OfficialPreparedSourceStep:
    if (
        layer.layer_id != source.layer_id
        or not isinstance(state.value, OfficialKdaState)
        or state.sha256 != _state_digest(state.value)
        or isinstance(source.top_k, bool)
        or source.top_k <= 0
        or not math.isfinite(source.rms_norm_epsilon)
        or source.rms_norm_epsilon <= 0.0
        or not math.isfinite(source.situ_beta)
        or source.situ_beta <= 0.0
        or (
            source.situ_linear_beta is not None
            and (
                not math.isfinite(source.situ_linear_beta)
                or source.situ_linear_beta <= 0.0
            )
        )
    ):
        raise K3XError("INVALID_OFFICIAL_TWO_LAYER_SOURCE")
    roles = dict(source.tensors)
    if len(roles) != len(source.tensors):
        raise K3XError("INVALID_OFFICIAL_TWO_LAYER_SOURCE")
    required = {
        "self_res_norm",
        "self_res_proj",
        "input_norm",
        "kda_q_proj",
        "kda_k_proj",
        "kda_v_proj",
        "kda_q_conv",
        "kda_k_conv",
        "kda_v_conv",
        "kda_f_a",
        "kda_f_b",
        "kda_a_log",
        "kda_dt_bias",
        "kda_beta",
        "kda_output_gate",
        "kda_output_norm",
        "kda_output_proj",
        "mlp_res_norm",
        "mlp_res_proj",
        "post_attention_norm",
        "router",
        "router_correction",
        "routed_down",
        "routed_norm",
        "routed_up",
        "shared_gate",
        "shared_up",
        "shared_down",
    }
    if set(roles) != required:
        raise K3XError("INVALID_OFFICIAL_TWO_LAYER_SOURCE")
    fp32_roles = {
        "kda_q_conv",
        "kda_k_conv",
        "kda_v_conv",
        "kda_a_log",
        "kda_dt_bias",
        "kda_output_norm",
        "router_correction",
    }
    if any(
        value.dtype != ("F32" if role in fp32_roles else "BF16")
        for role, value in roles.items()
    ):
        raise K3XError("INVALID_OFFICIAL_TWO_LAYER_SOURCE")
    tensors = {role: _decode_source_tensor(value) for role, value in roles.items()}
    config = source.kda_config
    projection = config.heads * config.head_dim
    for role in ("kda_q_conv", "kda_k_conv", "kda_v_conv"):
        convolution = tensors[role]
        if tuple(convolution.shape) == (projection, 1, config.conv_width):
            tensors[role] = convolution.reshape(projection, config.conv_width)
        elif tuple(convolution.shape) != (projection, config.conv_width):
            raise K3XError("INVALID_OFFICIAL_TWO_LAYER_SOURCE")
    hidden = torch.tensor(item.hidden_input, dtype=torch.bfloat16)
    block = torch.tensor(item.block_source, dtype=torch.bfloat16)
    residual = _attention_residual(
        hidden, block, tensors["self_res_norm"], tensors["self_res_proj"]
    )
    kda_input = _rms_norm(residual, tensors["input_norm"])
    weights = OfficialKdaWeights(
        tensors["kda_q_proj"], tensors["kda_k_proj"], tensors["kda_v_proj"],
        tensors["kda_q_conv"], tensors["kda_k_conv"], tensors["kda_v_conv"],
        tensors["kda_f_a"], tensors["kda_f_b"], tensors["kda_a_log"],
        tensors["kda_dt_bias"], tensors["kda_beta"],
        tensors["kda_output_gate"], tensors["kda_output_norm"],
        tensors["kda_output_proj"],
    )
    try:
        kda = official_kda(
            kda_input.reshape(1, 1, -1), weights, state.value, config
        )
    except ValueError as error:
        raise K3XError("INVALID_OFFICIAL_TWO_LAYER_SOURCE") from error
    kda_output = kda.output.reshape(-1)
    prefix = (hidden.float() + kda_output.float()).to(torch.bfloat16)
    prepared = prepare_official_moe_hidden(
        prefix,
        block,
        tensors["mlp_res_norm"],
        tensors["mlp_res_proj"].reshape(-1),
        tensors["post_attention_norm"],
        rms_norm_eps=source.rms_norm_epsilon,
    )
    route = route_official_hidden(
        prepared,
        tensors["router"],
        tensors["router_correction"],
        top_k=source.top_k,
    )
    latent = _bf16_matvec(prepared, tensors["routed_down"])
    shared = _dense_ffn(
        prepared,
        tensors["shared_gate"],
        tensors["shared_up"],
        tensors["shared_down"],
        source.situ_beta,
        source.situ_linear_beta,
    )
    return OfficialPreparedSourceStep(
        layer.layer_id,
        prefix,
        latent,
        shared,
        tensors["routed_norm"],
        tensors["routed_up"],
        official_two_layer_state(kda.state),
        _tensor_digest(kda_output, b"kda-output-bf16\0"),
        route,
    )


def finish_official_source_step(
    prepared: OfficialPreparedSourceStep,
    source: OfficialLayerSourceBytes,
) -> OfficialTwoLayerStepExecution:
    if prepared.layer_id != source.layer_id:
        raise K3XError("INVALID_OFFICIAL_TWO_LAYER_SOURCE")
    expert_by_id = {expert.expert_id: expert for expert in source.experts}
    if (
        len(expert_by_id) != len(source.experts)
        or any(not 0 <= expert_id < 896 for expert_id in expert_by_id)
        or any(
            expert_id not in expert_by_id
            for expert_id in prepared.route.expert_ids
        )
    ):
        raise K3XError("INVALID_OFFICIAL_TWO_LAYER_SOURCE")
    mixed = torch.zeros_like(prepared.latent, dtype=torch.float32)
    for expert_id, contribution in zip(
        prepared.route.expert_ids, prepared.route.contributions
    ):
        mixed += contribution * _mxfp4_forward(
            prepared.latent,
            expert_by_id[expert_id],
            source.situ_beta,
            source.situ_linear_beta,
        ).float()
    mixed = mixed.to(torch.bfloat16)
    routed_norm = _rms_norm(mixed, prepared.routed_norm)
    routed = _bf16_matvec(routed_norm, prepared.routed_up)
    combined = (routed.float() + prepared.shared.float()).to(torch.bfloat16)
    output = (prepared.prefix.float() + combined.float()).to(torch.bfloat16)
    return OfficialTwoLayerStepExecution(
        tuple(float(value) for value in output.float()),
        prepared.state,
        prepared.kda_output_sha256,
        prepared.route,
    )


def _verified_object_payload(value: MaterializedRangeObject) -> bytes:
    if (
        value.length <= 0
        or not value.path.is_file()
        or value.path.stat().st_size != value.length
    ):
        raise K3XError("INVALID_OFFICIAL_TWO_LAYER_OBJECT")
    payload = value.path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != value.sha256:
        raise K3XError("INVALID_OFFICIAL_TWO_LAYER_OBJECT")
    return payload


def load_official_layer_source_bytes(
    plan: OfficialLayerPlan,
    kda_objects: dict[str, MaterializedRangeObject],
    always_objects: dict[str, MaterializedRangeObject],
    expert_plans: dict[int, ExpertPlan],
    expert_objects: dict[int, MaterializedRangeObject],
) -> OfficialLayerSourceBytes:
    expected_kda = {item.official_name for item in plan.kda_tensors}
    expected_always = {item.official_name for item in plan.moe_plan.always_active}
    if (
        set(kda_objects) != expected_kda
        or set(always_objects) != expected_always
        or set(expert_plans) != set(expert_objects)
    ):
        raise K3XError("INVALID_OFFICIAL_TWO_LAYER_OBJECT_SET")
    tensors: list[tuple[str, OfficialSourceTensorBytes]] = []
    dense_objects = kda_objects | always_objects
    for item in (*plan.kda_tensors, *plan.moe_plan.always_active):
        value = dense_objects[item.official_name]
        payload = _verified_object_payload(value)
        if len(payload) != item.length:
            raise K3XError("INVALID_OFFICIAL_TWO_LAYER_OBJECT")
        tensors.append(
            (item.role, OfficialSourceTensorBytes(item.dtype, item.shape, payload))
        )

    experts = _load_official_expert_bytes(
        plan, expert_plans, expert_objects
    )
    return OfficialLayerSourceBytes(
        plan.layer_id,
        OfficialKdaConfig(7_168, 96, 128, 4, 1.0e-5, -5.0),
        tuple(tensors),
        experts,
        16,
        1.0e-5,
        4.0,
        25.0,
    )


def _load_official_expert_bytes(
    plan: OfficialLayerPlan,
    expert_plans: dict[int, ExpertPlan],
    expert_objects: dict[int, MaterializedRangeObject],
) -> tuple[OfficialMxfp4ExpertBytes, ...]:
    if set(expert_plans) != set(expert_objects):
        raise K3XError("INVALID_OFFICIAL_TWO_LAYER_OBJECT_SET")
    shapes = {
        "gate": (3_072, 3_584),
        "up": (3_072, 3_584),
        "down": (3_584, 3_072),
    }
    experts: list[OfficialMxfp4ExpertBytes] = []
    for expert_id in sorted(expert_plans):
        expert_plan = expert_plans[expert_id]
        value = expert_objects[expert_id]
        if (
            expert_plan.layer_id != plan.layer_id
            or expert_plan.expert_id != expert_id
            or expert_plan.shard_path != plan.shard_path
            or expert_plan.index_sha256 != plan.index_sha256
            or value.length != expert_plan.payload_bytes
        ):
            raise K3XError("INVALID_OFFICIAL_TWO_LAYER_OBJECT")
        payload = _verified_object_payload(value)
        parts: dict[str, dict[str, bytes]] = {}
        for item in expert_plan.tensors:
            start = item.offset - expert_plan.payload_start
            end = start + item.length
            if start < 0 or end > len(payload):
                raise K3XError("INVALID_OFFICIAL_TWO_LAYER_OBJECT")
            kind = (
                "packed"
                if item.canonical_name.endswith(".weight_packed")
                else "scale"
            )
            parts.setdefault(item.role, {})[kind] = payload[start:end]
        if set(parts) != {"gate", "up", "down"} or any(
            set(value) != {"packed", "scale"} for value in parts.values()
        ):
            raise K3XError("INVALID_OFFICIAL_TWO_LAYER_OBJECT")
        matrices = {
            role: OfficialMxfp4MatrixBytes(
                parts[role]["packed"],
                parts[role]["scale"],
                shapes[role][0],
                shapes[role][1],
                32,
            )
            for role in ("gate", "up", "down")
        }
        experts.append(
            OfficialMxfp4ExpertBytes(
                expert_id,
                matrices["gate"],
                matrices["up"],
                matrices["down"],
            )
        )
    return tuple(experts)


def _execute_official_source_bytes(
    layer: OfficialLayerPlan,
    item: OfficialLayerInput,
    state: OfficialTwoLayerState,
    source: OfficialLayerSourceBytes,
) -> OfficialTwoLayerStepExecution:
    return finish_official_source_step(
        prepare_official_source_step(layer, item, state, source), source
    )


def make_official_source_byte_executor(
    sources: tuple[OfficialLayerSourceBytes, OfficialLayerSourceBytes],
) -> OfficialTwoLayerExecutor:
    by_layer = {source.layer_id: source for source in sources}
    if len(by_layer) != 2 or set(by_layer) != set(_LAYER_IDS):
        raise K3XError("INVALID_OFFICIAL_TWO_LAYER_SOURCE")

    def execute(
        layer: OfficialLayerPlan,
        item: OfficialLayerInput,
        state: OfficialTwoLayerState,
    ) -> OfficialTwoLayerStepExecution:
        source = by_layer.get(layer.layer_id)
        if source is None:
            raise K3XError("INVALID_OFFICIAL_TWO_LAYER_SOURCE")
        return _execute_official_source_bytes(layer, item, state, source)

    return execute


def manufacture_official_two_layer_fixture(
    output_directory: Path,
    tensors: tuple[OfficialMoeSourceTensor, ...],
    config: dict[str, object],
    metadata: dict[str, object],
    *,
    chunk_bytes: int = 8 * 1024 * 1024,
    stop_after_extents: int | None = None,
) -> OfficialTwoLayerMaterializationReport:
    if (
        not tensors
        or not isinstance(config, dict)
        or not isinstance(metadata, dict)
        or chunk_bytes <= 0
        or metadata.get("format") != "k3x-official-two-layer-v1"
        or metadata.get("layer_ids") != [1, 2]
        or metadata.get("step_order") != ["a:1", "a:2", "b:1", "b:2"]
    ):
        raise K3XError("INVALID_OFFICIAL_TWO_LAYER_MATERIALIZATION")
    layer_order: list[int] = []
    expert_parts: dict[tuple[int, int], set[tuple[str, str]]] = {}
    for tensor in tensors:
        match = _LAYER_NAME.match(tensor.name)
        if match is None:
            raise K3XError("INVALID_OFFICIAL_TWO_LAYER_MATERIALIZATION")
        layer_id = int(match.group(1))
        if not layer_order or layer_order[-1] != layer_id:
            layer_order.append(layer_id)
        expert_match = _EXPERT_SOURCE_NAME.match(tensor.name)
        if expert_match is not None:
            identity = (int(expert_match.group(1)), int(expert_match.group(2)))
            expert_parts.setdefault(identity, set()).add(
                (expert_match.group(3), expert_match.group(4))
            )
    complete_expert = {
        (role, kind)
        for role in ("gate", "up", "down")
        for kind in ("packed", "scale")
    }
    if (
        layer_order != [1, 2]
        or {layer_id for layer_id, _ in expert_parts} != {1, 2}
        or any(parts != complete_expert for parts in expert_parts.values())
    ):
        raise K3XError("INVALID_OFFICIAL_TWO_LAYER_MATERIALIZATION")

    output_directory = Path(output_directory)
    assembled = assemble_official_moe_source(
        output_directory,
        tensors,
        config,
        chunk_bytes=chunk_bytes,
        official_metadata=metadata,
        official_metadata_key="official_two_layer",
    )
    k3x_path = output_directory / "official-two-layer.k3x"
    converted = convert(
        assembled.source_directory,
        k3x_path,
        chunk_bytes=chunk_bytes,
        stop_after_extents=stop_after_extents,
    )
    reader = None
    if converted.completed:
        reader = K3XReader.open(k3x_path)
        active_layers = {
            record.layer_index
            for record in reader.layer_records
            if record.tensor_count
        }
        expert_layers = {record.layer_index for record in reader.expert_records}
        if active_layers != {1, 2} or expert_layers != {1, 2}:
            raise K3XError("INVALID_OFFICIAL_TWO_LAYER_ARTIFACT")
    return OfficialTwoLayerMaterializationReport(
        assembled.source_directory,
        assembled.manifest_path,
        assembled.microshard_path,
        k3x_path,
        converted.completed,
        converted.maximum_source_read_bytes,
        microshard_sha256=assembled.microshard_sha256,
        tensor_sha256=assembled.tensor_sha256,
        k3x_root_sha256=(
            reader.superblock.root_sha256.hex() if reader is not None else ""
        ),
    )


def _write_json_atomic(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    with partial.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)


def materialize_official_two_layer(
    snapshot: OfficialSnapshot,
    index: OfficialIndex,
    config: OfficialConfig,
    headers: tuple[OfficialShardHeader, OfficialShardHeader],
    plan: OfficialTwoLayerPlan,
    transport: Transport,
    output_directory: Path,
    *,
    chunk_bytes: int = 8 * 1024 * 1024,
) -> OfficialTwoLayerMaterializationReport:
    source_file = snapshot.files.get("modeling_kimi_linear.py")
    if len(headers) != 2:
        raise K3XError("INVALID_OFFICIAL_TWO_LAYER_MATERIALIZATION")
    expected_layers = tuple(
        plan_official_kda_layer(
            index,
            header,
            config,
            source_blob_id=OFFICIAL_KDA_SOURCE_BLOB_ID,
            layer_id=layer_id,
        )
        for layer_id, header in zip(_LAYER_IDS, headers)
    )
    expected = plan_official_two_layer(expected_layers[0], expected_layers[1])
    if (
        chunk_bytes <= 0
        or snapshot.repository != "moonshotai/Kimi-K3"
        or snapshot.resolved_revision
        != "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"
        or source_file is None
        or source_file.size != 51_506
        or source_file.blob_id != OFFICIAL_KDA_SOURCE_BLOB_ID
        or source_file.lfs_sha256 is not None
        or plan != expected
        or tuple(header.shard_path for header in headers) != plan.shard_paths
        or any(
            (shard := snapshot.files.get(header.shard_path)) is None
            or shard.size != header.file_size
            or shard.lfs_sha256 is None
            for header in headers
        )
    ):
        raise K3XError("INVALID_OFFICIAL_TWO_LAYER_MATERIALIZATION")

    bounded_chunk = min(chunk_bytes, 8 * 1024 * 1024)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    object_directory = output_directory / "objects"
    materialized: list[MaterializedRangeObject] = []
    kda_objects: list[dict[str, MaterializedRangeObject]] = [{}, {}]
    always_objects: list[dict[str, MaterializedRangeObject]] = [{}, {}]
    expert_plans: list[dict[int, ExpertPlan]] = [{}, {}]
    expert_objects: list[dict[int, MaterializedRangeObject]] = [{}, {}]

    def fetch_range(
        layer_index: int, offset: int, length: int
    ) -> MaterializedRangeObject:
        value = materialize_official_range_object(
            snapshot,
            plan.layers[layer_index].shard_path,
            offset,
            length,
            transport,
            object_directory,
            chunk_bytes=bounded_chunk,
        )
        materialized.append(value)
        return value

    for layer_index, layer in enumerate(plan.layers):
        kda_objects[layer_index] = {
            item.official_name: fetch_range(layer_index, item.offset, item.length)
            for item in layer.kda_tensors
        }
        always_objects[layer_index] = {
            item.official_name: fetch_range(layer_index, item.offset, item.length)
            for item in layer.moe_plan.always_active
        }

    sources = [
        load_official_layer_source_bytes(
            layer,
            kda_objects[layer_index],
            always_objects[layer_index],
            {},
            {},
        )
        for layer_index, layer in enumerate(plan.layers)
    ]
    kda_config = OfficialKdaConfig(7_168, 96, 128, 4, 1.0e-5, -5.0)
    states = [
        official_two_layer_state(
            zero_official_kda_state(kda_config, 1, torch.device("cpu"))
        )
        for _ in plan.layers
    ]
    inputs = official_layer_inputs()
    executions: list[OfficialTwoLayerStepExecution] = []
    for item in inputs:
        current = item
        for layer_index, layer in enumerate(plan.layers):
            prepared = prepare_official_source_step(
                layer, current, states[layer_index], sources[layer_index]
            )
            for expert_id in prepared.route.expert_ids:
                if expert_id in expert_plans[layer_index]:
                    continue
                expert_plan = plan_official_expert(
                    index,
                    headers[layer_index],
                    layer_id=layer.layer_id,
                    expert_id=expert_id,
                )
                expert_value = fetch_range(
                    layer_index,
                    expert_plan.payload_start,
                    expert_plan.payload_bytes,
                )
                expert_plans[layer_index][expert_id] = expert_plan
                expert_objects[layer_index][expert_id] = expert_value
            source = replace(
                sources[layer_index],
                experts=_load_official_expert_bytes(
                    layer,
                    expert_plans[layer_index],
                    expert_objects[layer_index],
                ),
            )
            result = finish_official_source_step(prepared, source)
            executions.append(result)
            states[layer_index] = result.state
            output_sha256 = _float_digest(result.output)
            current = OfficialLayerInput(
                item.name,
                result.output,
                item.block_source,
                output_sha256,
                item.block_sha256,
            )

    execution_iterator = iter(executions)
    trace = derive_official_two_layer_trace(
        plan,
        inputs,
        (
            official_two_layer_state(
                zero_official_kda_state(kda_config, 1, torch.device("cpu"))
            ),
            official_two_layer_state(
                zero_official_kda_state(kda_config, 1, torch.device("cpu"))
            ),
        ),
        lambda _layer, _item, _state: next(execution_iterator),
    )
    try:
        next(execution_iterator)
    except StopIteration:
        pass
    else:
        raise K3XError("INVALID_OFFICIAL_TWO_LAYER_TRACE")

    oracle_payload = _official_two_layer_oracle_payload(
        trace, (states[0], states[1])
    )
    parse_official_two_layer_oracle(oracle_payload)
    oracle_path = output_directory / "official-two-layer-oracle-v1.bin"
    oracle_partial = oracle_path.with_suffix(oracle_path.suffix + ".partial")
    with oracle_partial.open("wb") as stream:
        stream.write(oracle_payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(oracle_partial, oracle_path)
    oracle_sha256 = hashlib.sha256(oracle_payload).hexdigest()

    layer_routes: list[OfficialLayerRoutes] = []
    for layer_index, layer_id in enumerate(_LAYER_IDS):
        steps = tuple(step for step in trace.steps if step.layer_id == layer_id)
        layer_routes.append(
            OfficialLayerRoutes(
                tuple(
                    OfficialLayerRouteStep(
                        step.position,
                        step.consumes_state_sha256,
                        step.state_sha256,
                        step.kda_output_sha256,
                        step.route,
                    )
                    for step in steps
                ),
                trace.selected_experts[layer_index],
                trace.initial_state_sha256[layer_index],
                trace.final_state_sha256[layer_index],
            )
        )
    tensors = tuple(
        tensor
        for layer_index, layer in enumerate(plan.layers)
        for tensor in build_official_layer_source_tensors(
            layer,
            layer_routes[layer_index],
            expert_plans[layer_index],
            kda_objects[layer_index],
            always_objects[layer_index],
            expert_objects[layer_index],
        )
    )
    metadata: dict[str, object] = {
        "format": "k3x-official-two-layer-v1",
        "layer_ids": [1, 2],
        "step_order": ["a:1", "a:2", "b:1", "b:2"],
        "repository": snapshot.repository,
        "resolved_revision": snapshot.resolved_revision,
        "snapshot_sha256": snapshot.canonical_sha256,
        "index_sha256": index.sha256,
        "config_sha256": config.sha256,
        "source_blob_id": OFFICIAL_KDA_SOURCE_BLOB_ID,
        "shard_paths": list(plan.shard_paths),
        "steps": [
            {
                "position": step.position,
                "layer_id": step.layer_id,
                "hidden_input_sha256": step.hidden_input_sha256,
                "block_sha256": step.block_sha256,
                "consumes_state_sha256": step.consumes_state_sha256,
                "state_sha256": step.state_sha256,
                "kda_output_sha256": step.kda_output_sha256,
                "expert_ids": list(step.route.expert_ids),
                "contributions": list(step.route.contributions),
                "contribution_sha256": step.contribution_sha256,
                "output_sha256": step.output_sha256,
            }
            for step in trace.steps
        ],
        "selected_experts": [list(value) for value in trace.selected_experts],
        "final_state_sha256": list(trace.final_state_sha256),
        "oracle": {
            "format": "k3x-official-two-layer-oracle-v1",
            "filename": oracle_path.name,
            "sha256": oracle_sha256,
            "bytes": len(oracle_payload),
        },
        "objects": [
            {
                "layer_id": layer.layer_id,
                "trunks": [
                    {
                        "name": item.official_name,
                        "range": [item.offset, item.offset + item.length],
                        "sha256": (
                            (kda_objects[layer_index] | always_objects[layer_index])[
                                item.official_name
                            ].sha256
                        ),
                    }
                    for item in (*layer.kda_tensors, *layer.moe_plan.always_active)
                ],
                "experts": [
                    {
                        "expert_id": expert_id,
                        "range": [
                            expert_plans[layer_index][expert_id].payload_start,
                            expert_plans[layer_index][expert_id].payload_end,
                        ],
                        "sha256": expert_objects[layer_index][expert_id].sha256,
                    }
                    for expert_id in trace.selected_experts[layer_index]
                ],
            }
            for layer_index, layer in enumerate(plan.layers)
        ],
    }
    report = manufacture_official_two_layer_fixture(
        output_directory,
        tensors,
        _released_storage_config(),
        metadata,
        chunk_bytes=bounded_chunk,
    )
    route_manifest_path = output_directory / "two-layer-route-state-manifest.json"
    requested_payload_bytes = sum(item.length for item in materialized)
    downloaded_payload_bytes = sum(item.response_bytes for item in materialized)
    metadata["traffic"] = {
        "requested_payload_bytes": requested_payload_bytes,
        "downloaded_payload_bytes": downloaded_payload_bytes,
        "reused_objects": sum(1 for item in materialized if item.reused),
        "requests": sum(item.requests for item in materialized),
        "maximum_response_bytes": max(
            (item.maximum_response_bytes for item in materialized), default=0
        ),
    }
    metadata["artifact"] = {
        "filename": report.k3x_path.name,
        "k3x_root_sha256": report.k3x_root_sha256,
        "source_sha256": report.microshard_sha256,
        "tensor_sha256": report.tensor_sha256,
    }
    _write_json_atomic(route_manifest_path, metadata)
    return replace(
        report,
        selected_experts=trace.selected_experts,
        requested_payload_bytes=requested_payload_bytes,
        downloaded_payload_bytes=downloaded_payload_bytes,
        reused_objects=metadata["traffic"]["reused_objects"],
        requests=metadata["traffic"]["requests"],
        maximum_response_bytes=metadata["traffic"]["maximum_response_bytes"],
        trace=trace,
        route_manifest_path=route_manifest_path,
        oracle_path=oracle_path,
        oracle_sha256=oracle_sha256,
        oracle_bytes=len(oracle_payload),
    )


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
                    _contribution_digest(result.route),
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
