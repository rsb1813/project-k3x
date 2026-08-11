# 공식 Kimi K3 레이어 1·2의 의존형 제조 계획을 구성합니다.
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
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
)

from .format import K3XError
from .official_layer import (
    OFFICIAL_KDA_SOURCE_BLOB_ID,
    OfficialLayerInput,
    OfficialLayerPlan,
    _attention_residual,
    _rms_norm,
    _state_digest,
    _tensor_digest,
)
from .official_moe import (
    OfficialMoeSourceTensor,
    OfficialMoeRoute,
    assemble_official_moe_source,
    prepare_official_moe_hidden,
    route_official_hidden,
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


def _execute_official_source_bytes(
    layer: OfficialLayerPlan,
    item: OfficialLayerInput,
    state: OfficialTwoLayerState,
    source: OfficialLayerSourceBytes,
) -> OfficialTwoLayerStepExecution:
    if (
        layer.layer_id != source.layer_id
        or not isinstance(state.value, OfficialKdaState)
        or state.sha256 != _state_digest(state.value)
        or isinstance(source.top_k, bool)
        or not 0 < source.top_k <= len(source.experts)
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
    expert_by_id = {expert.expert_id: expert for expert in source.experts}
    if (
        len(expert_by_id) != len(source.experts)
        or any(not 0 <= expert_id < 896 for expert_id in expert_by_id)
        or any(expert_id not in expert_by_id for expert_id in route.expert_ids)
    ):
        raise K3XError("INVALID_OFFICIAL_TWO_LAYER_SOURCE")
    latent = _bf16_matvec(prepared, tensors["routed_down"])
    shared = _dense_ffn(
        prepared,
        tensors["shared_gate"],
        tensors["shared_up"],
        tensors["shared_down"],
        source.situ_beta,
        source.situ_linear_beta,
    )
    mixed = torch.zeros_like(latent, dtype=torch.float32)
    for expert_id, contribution in zip(route.expert_ids, route.contributions):
        mixed += contribution * _mxfp4_forward(
            latent,
            expert_by_id[expert_id],
            source.situ_beta,
            source.situ_linear_beta,
        ).float()
    mixed = mixed.to(torch.bfloat16)
    routed_norm = _rms_norm(mixed, tensors["routed_norm"])
    routed = _bf16_matvec(routed_norm, tensors["routed_up"])
    combined = (routed.float() + shared.float()).to(torch.bfloat16)
    output = (prefix.float() + combined.float()).to(torch.bfloat16)
    return OfficialTwoLayerStepExecution(
        tuple(float(value) for value in output.float()),
        official_two_layer_state(kda.state),
        _tensor_digest(kda_output, b"kda-output-bf16\0"),
        route,
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
