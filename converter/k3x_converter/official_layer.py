# 공식 Kimi K3 layer-1 KDA 전체 경계의 tensor 계획을 검증합니다.
from __future__ import annotations

import hashlib
import math
import struct
from dataclasses import dataclass
from pathlib import Path

import torch

from .format import (
    K3XError,
    OPTIONAL_OFFICIAL_MOE_FIXTURE,
    OPTIONAL_STORAGE_FIXTURE,
)
from .official_moe import (
    AssembledOfficialMoeSource,
    MaterializedRangeObject,
    OfficialMoePlan,
    OfficialMoeRoute,
    OfficialMoeRoutes,
    OfficialMoeRouteCase,
    OfficialMoeSourceTensor,
    assemble_official_moe_source,
    build_official_moe_source_tensors,
    materialize_official_range_object,
    plan_official_moe_slice,
    prepare_official_moe_hidden,
    route_official_hidden,
    _sha256_path,
    _write_json_atomic,
)
from .official_source import (
    ExpertPlan,
    OfficialConfig,
    OfficialIndex,
    OfficialShardHeader,
    OfficialSnapshot,
    PlannedTensor,
    Transport,
    _released_storage_config,
    plan_official_expert,
)
from .reader import K3XReader
from .writer import CONVERTER_VERSION, convert

from k3x_ref.official_kda import (
    OfficialKdaConfig,
    OfficialKdaResult,
    OfficialKdaState,
    OfficialKdaWeights,
    official_kda,
    zero_official_kda_state,
)


OFFICIAL_KDA_SOURCE_BLOB_ID = "b8c41e8bfce768d74d8da3a37e693f5ee43876a0"
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


@dataclass(frozen=True)
class OfficialLayerInput:
    name: str
    hidden_input: tuple[float, ...]
    block_source: tuple[float, ...]
    hidden_sha256: str
    block_sha256: str


@dataclass(frozen=True)
class OfficialLayerRouteStep:
    name: str
    consumes_state_sha256: str
    state_sha256: str
    kda_output_sha256: str
    route: OfficialMoeRoute


@dataclass(frozen=True)
class OfficialLayerRoutes:
    steps: tuple[OfficialLayerRouteStep, OfficialLayerRouteStep]
    selected_experts: tuple[int, ...]
    initial_state_sha256: str
    final_state_sha256: str


@dataclass(frozen=True)
class OfficialLayerMaterializationReport:
    source_directory: Path
    route_manifest_path: Path
    manifest_path: Path
    microshard_path: Path
    k3x_path: Path
    selected_experts: tuple[int, ...]
    requested_payload_bytes: int
    downloaded_payload_bytes: int
    reused_objects: int
    requests: int
    maximum_response_bytes: int
    microshard_sha256: str
    tensor_sha256: dict[str, str]
    k3x_root_sha256: str


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
        or source_blob_id != OFFICIAL_KDA_SOURCE_BLOB_ID
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


def _input_values(
    multiplier: int,
    increment: int,
    modulus: int,
    offset: int,
) -> tuple[float, ...]:
    return tuple(
        (((multiplier * index + increment) % modulus) - offset) / 1024.0
        for index in range(7_168)
    )


def _float_digest(values: tuple[float, ...]) -> str:
    return hashlib.sha256(struct.pack(f"<{len(values)}f", *values)).hexdigest()


def official_layer_inputs() -> tuple[OfficialLayerInput, OfficialLayerInput]:
    specifications = (
        ("a", (17, 3, 257, 128), (29, 11, 251, 125)),
        ("b", (31, 7, 263, 131), (43, 19, 269, 134)),
    )
    result: list[OfficialLayerInput] = []
    for name, hidden_spec, block_spec in specifications:
        hidden = _input_values(*hidden_spec)
        block = _input_values(*block_spec)
        result.append(
            OfficialLayerInput(
                name,
                hidden,
                block,
                _float_digest(hidden),
                _float_digest(block),
            )
        )
    return result[0], result[1]


def _tensor_bytes(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def _tensor_digest(tensor: torch.Tensor, identity: bytes) -> str:
    digest = hashlib.sha256(identity)
    digest.update(_tensor_bytes(tensor))
    return digest.hexdigest()


def _state_digest(state: OfficialKdaState) -> str:
    digest = hashlib.sha256(b"k3x-official-kda-state-v1\0v-first-fp32\0")
    for name, tensor in (
        ("conv_q", state.conv_q),
        ("conv_k", state.conv_k),
        ("conv_v", state.conv_v),
        ("recurrent_v_first", state.recurrent_v_first),
    ):
        digest.update(name.encode("ascii") + b"\0")
        digest.update(struct.pack(f"<{tensor.ndim}Q", *tensor.shape))
        digest.update(_tensor_bytes(tensor))
    return digest.hexdigest()


def _load_object_tensor(
    item: PlannedTensor,
    objects: dict[str, MaterializedRangeObject],
) -> torch.Tensor:
    materialized = objects.get(item.official_name)
    if (
        materialized is None
        or materialized.length != item.length
        or not materialized.path.is_file()
        or materialized.path.stat().st_size != item.length
        or _sha256_path(materialized.path, 8 * 1024 * 1024)
        != materialized.sha256
    ):
        raise K3XError("INVALID_OFFICIAL_LAYER_OBJECT", item.official_name)
    dtype = torch.float32 if item.dtype == "F32" else torch.bfloat16
    return torch.from_file(
        str(materialized.path),
        shared=False,
        size=math.prod(item.shape),
        dtype=dtype,
    ).reshape(item.shape)


def _rms_norm(value: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    source = value.float()
    return (
        source
        * torch.rsqrt(source.square().mean(dim=-1, keepdim=True) + 1.0e-5)
        * weight.float()
    ).to(torch.bfloat16)


def _attention_residual(
    prefix_sum: torch.Tensor,
    block_source: torch.Tensor,
    norm: torch.Tensor,
    projection: torch.Tensor,
) -> torch.Tensor:
    values = torch.stack((block_source, prefix_sum)).float()
    normalized = values * torch.rsqrt(
        values.square().mean(dim=-1, keepdim=True) + 1.0e-5
    )
    score_weight = norm.float() * projection.reshape(-1).float()
    scores = (normalized * score_weight).sum(dim=-1)
    probabilities = scores.softmax(dim=-1)
    return (probabilities.unsqueeze(-1) * values).sum(dim=0).to(torch.bfloat16)


def _require_incremental_parity(
    full: OfficialKdaResult,
    first: OfficialKdaResult,
    second: OfficialKdaResult,
) -> None:
    incremental_output = torch.cat((first.output, second.output), dim=1)
    if not torch.equal(full.output, incremental_output):
        raise K3XError("OFFICIAL_KDA_INCREMENTAL_MISMATCH", "output")
    for name in ("conv_q", "conv_k", "conv_v"):
        if not torch.equal(getattr(full.state, name), getattr(second.state, name)):
            raise K3XError("OFFICIAL_KDA_INCREMENTAL_MISMATCH", name)
    if not torch.allclose(
        full.state.recurrent_v_first,
        second.state.recurrent_v_first,
        atol=1.0e-6,
        rtol=1.0e-6,
    ):
        raise K3XError("OFFICIAL_KDA_INCREMENTAL_MISMATCH", "recurrent_v_first")
    if full.boundaries is None or first.boundaries is None or second.boundaries is None:
        raise K3XError("OFFICIAL_KDA_INCREMENTAL_MISMATCH", "boundaries")
    for name, full_value in full.boundaries.__dict__.items():
        incremental_value = torch.cat(
            (getattr(first.boundaries, name), getattr(second.boundaries, name)),
            dim=1,
        )
        if full_value.dtype in {torch.float32, torch.float64}:
            matches = torch.allclose(
                full_value, incremental_value, atol=1.0e-6, rtol=1.0e-6
            )
        else:
            matches = torch.equal(full_value, incremental_value)
        if not matches:
            raise K3XError("OFFICIAL_KDA_INCREMENTAL_MISMATCH", name)


def derive_official_layer_routes(
    plan: OfficialLayerPlan,
    objects: dict[str, MaterializedRangeObject],
    inputs: tuple[OfficialLayerInput, OfficialLayerInput],
) -> OfficialLayerRoutes:
    if (
        plan.layer_id != 1
        or plan.source_blob_id != OFFICIAL_KDA_SOURCE_BLOB_ID
        or inputs != official_layer_inputs()
    ):
        raise K3XError("INVALID_OFFICIAL_LAYER_ROUTE_INPUT")
    expected_names = {
        item.official_name
        for item in (*plan.kda_tensors, *plan.moe_plan.always_active)
    }
    if set(objects) != expected_names:
        raise K3XError("INVALID_OFFICIAL_LAYER_OBJECT_SET")
    route_roles = {
        "self_res_norm",
        "self_res_proj",
        "input_norm",
        "mlp_res_norm",
        "mlp_res_proj",
        "post_attention_norm",
        "router",
        "router_correction",
    }
    route_roles.update(item.role for item in plan.kda_tensors)
    by_role = {
        item.role: (item, _load_object_tensor(item, objects))
        for item in (*plan.kda_tensors, *plan.moe_plan.always_active)
        if item.role in route_roles
    }

    kda_config = OfficialKdaConfig(7_168, 96, 128, 4, 1.0e-5, -5.0)
    weights = OfficialKdaWeights(
        q_proj=by_role["kda_q_proj"][1],
        k_proj=by_role["kda_k_proj"][1],
        v_proj=by_role["kda_v_proj"][1],
        q_conv=by_role["kda_q_conv"][1].reshape(12_288, 4),
        k_conv=by_role["kda_k_conv"][1].reshape(12_288, 4),
        v_conv=by_role["kda_v_conv"][1].reshape(12_288, 4),
        f_a_proj=by_role["kda_f_a"][1],
        f_b_proj=by_role["kda_f_b"][1],
        a_log=by_role["kda_a_log"][1],
        dt_bias=by_role["kda_dt_bias"][1],
        b_proj=by_role["kda_beta"][1],
        g_proj=by_role["kda_output_gate"][1],
        o_norm=by_role["kda_output_norm"][1],
        o_proj=by_role["kda_output_proj"][1],
    )
    hidden_values: list[torch.Tensor] = []
    block_values: list[torch.Tensor] = []
    kda_inputs: list[torch.Tensor] = []
    for item in inputs:
        hidden = torch.tensor(item.hidden_input, dtype=torch.bfloat16)
        block = torch.tensor(item.block_source, dtype=torch.bfloat16)
        residual = _attention_residual(
            hidden,
            block,
            by_role["self_res_norm"][1],
            by_role["self_res_proj"][1],
        )
        kda_input = _rms_norm(residual, by_role["input_norm"][1])
        hidden_values.append(hidden)
        block_values.append(block)
        kda_inputs.append(kda_input)

    sequence = torch.stack(kda_inputs, dim=0).unsqueeze(0)
    zero = zero_official_kda_state(kda_config, 1, sequence.device)
    full = official_kda(sequence, weights, zero, kda_config)
    first = official_kda(sequence[:, :1], weights, zero, kda_config)
    second = official_kda(sequence[:, 1:], weights, first.state, kda_config)
    _require_incremental_parity(full, first, second)

    state_hashes = (_state_digest(first.state), _state_digest(second.state))
    initial_hash = _state_digest(zero)
    routes: list[OfficialMoeRoute] = []
    output_hashes: list[str] = []
    for index, (hidden, block) in enumerate(zip(hidden_values, block_values)):
        kda_output = full.output[0, index]
        output_hashes.append(_tensor_digest(kda_output, b"kda-output-bf16\0"))
        prefix_sum = (hidden.float() + kda_output.float()).to(torch.bfloat16)
        moe_hidden = prepare_official_moe_hidden(
            prefix_sum,
            block,
            by_role["mlp_res_norm"][1],
            by_role["mlp_res_proj"][1].reshape(-1),
            by_role["post_attention_norm"][1],
            rms_norm_eps=1.0e-5,
        )
        routes.append(
            route_official_hidden(
                moe_hidden,
                by_role["router"][1],
                by_role["router_correction"][1],
                top_k=16,
            )
        )
    steps = (
        OfficialLayerRouteStep(
            "a", initial_hash, state_hashes[0], output_hashes[0], routes[0]
        ),
        OfficialLayerRouteStep(
            "b", state_hashes[0], state_hashes[1], output_hashes[1], routes[1]
        ),
    )
    selected = tuple(
        dict.fromkeys((*routes[0].expert_ids, *routes[1].expert_ids))
    )
    return OfficialLayerRoutes(steps, selected, initial_hash, state_hashes[1])


def _layer_route_manifest(
    snapshot: OfficialSnapshot,
    index: OfficialIndex,
    config: OfficialConfig,
    header: OfficialShardHeader,
    plan: OfficialLayerPlan,
    inputs: tuple[OfficialLayerInput, OfficialLayerInput],
    routes: OfficialLayerRoutes,
    kda_objects: dict[str, MaterializedRangeObject],
    always_objects: dict[str, MaterializedRangeObject],
) -> dict[str, object]:
    return {
        "format": "k3x-official-kda-layer-routes-v1",
        "converter_version": CONVERTER_VERSION,
        "repository": snapshot.repository,
        "requested_revision": snapshot.requested_revision,
        "resolved_revision": snapshot.resolved_revision,
        "snapshot_sha256": snapshot.canonical_sha256,
        "index_sha256": index.sha256,
        "config_sha256": config.sha256,
        "config_git_blob_id": config.git_blob_id,
        "source_blob_id": plan.source_blob_id,
        "shard_path": plan.shard_path,
        "shard_lfs_sha256": snapshot.files[plan.shard_path].lfs_sha256,
        "header": {
            "file_size": header.file_size,
            "header_length": header.header_length,
            "data_start": header.data_start,
        },
        "state_layout": "v-first-fp32",
        "initial_state_sha256": routes.initial_state_sha256,
        "final_state_sha256": routes.final_state_sha256,
        "inputs": [
            {
                "name": item.name,
                "hidden_sha256": item.hidden_sha256,
                "block_sha256": item.block_sha256,
            }
            for item in inputs
        ],
        "steps": [
            {
                "name": step.name,
                "consumes_state_sha256": step.consumes_state_sha256,
                "state_sha256": step.state_sha256,
                "kda_output_sha256": step.kda_output_sha256,
                "expert_ids": list(step.route.expert_ids),
                "contributions": list(step.route.contributions),
            }
            for step in routes.steps
        ],
        "selected_experts": list(routes.selected_experts),
        "kda_objects": [
            {
                "name": item.official_name,
                "range": [item.offset, item.offset + item.length],
                "sha256": kda_objects[item.official_name].sha256,
            }
            for item in plan.kda_tensors
        ],
        "always_active_objects": [
            {
                "name": item.official_name,
                "range": [item.offset, item.offset + item.length],
                "sha256": always_objects[item.official_name].sha256,
            }
            for item in plan.moe_plan.always_active
        ],
        "provenance": "transport-pinned-ranges",
    }


def build_official_layer_source_tensors(
    plan: OfficialLayerPlan,
    routes: OfficialLayerRoutes,
    expert_plans: dict[int, ExpertPlan],
    kda_objects: dict[str, MaterializedRangeObject],
    always_objects: dict[str, MaterializedRangeObject],
    expert_objects: dict[int, MaterializedRangeObject],
) -> tuple[OfficialMoeSourceTensor, ...]:
    if set(kda_objects) != {item.official_name for item in plan.kda_tensors}:
        raise K3XError("INVALID_OFFICIAL_LAYER_OBJECT_SET")
    kda_tensors: list[OfficialMoeSourceTensor] = []
    for item in plan.kda_tensors:
        materialized = kda_objects[item.official_name]
        if materialized.length != item.length:
            raise K3XError("INVALID_OFFICIAL_LAYER_OBJECT", item.official_name)
        kda_tensors.append(
            OfficialMoeSourceTensor(
                item.canonical_name,
                item.dtype,
                item.shape,
                materialized.path,
                0,
                item.length,
            )
        )
    moe_routes = OfficialMoeRoutes(
        (
            OfficialMoeRouteCase(routes.steps[0].name, routes.steps[0].route),
            OfficialMoeRouteCase(routes.steps[1].name, routes.steps[1].route),
        ),
        routes.selected_experts,
    )
    return tuple(kda_tensors) + build_official_moe_source_tensors(
        plan.moe_plan,
        moe_routes,
        expert_plans,
        always_objects,
        expert_objects,
    )


def materialize_official_kda_layer(
    snapshot: OfficialSnapshot,
    index: OfficialIndex,
    config: OfficialConfig,
    header: OfficialShardHeader,
    plan: OfficialLayerPlan,
    transport: Transport,
    output_directory: Path,
    *,
    chunk_bytes: int = 8 * 1024 * 1024,
) -> OfficialLayerMaterializationReport:
    source_file = snapshot.files.get("modeling_kimi_linear.py")
    shard_file = snapshot.files.get(header.shard_path)
    expected_plan = plan_official_kda_layer(
        index,
        header,
        config,
        source_blob_id=OFFICIAL_KDA_SOURCE_BLOB_ID,
        layer_id=1,
    )
    if (
        chunk_bytes <= 0
        or snapshot.repository != "moonshotai/Kimi-K3"
        or snapshot.resolved_revision
        != "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"
        or source_file is None
        or source_file.size != 51_506
        or source_file.blob_id != OFFICIAL_KDA_SOURCE_BLOB_ID
        or source_file.lfs_sha256 is not None
        or shard_file is None
        or shard_file.size != header.file_size
        or shard_file.lfs_sha256 is None
        or plan != expected_plan
        or plan.layer_id != 1
        or plan.shard_path != header.shard_path
        or plan.index_sha256 != index.sha256
        or plan.source_blob_id != OFFICIAL_KDA_SOURCE_BLOB_ID
    ):
        raise K3XError("INVALID_OFFICIAL_LAYER_MATERIALIZATION")
    bounded_chunk = min(chunk_bytes, 8 * 1024 * 1024)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    object_directory = output_directory / "objects"
    materialized: list[MaterializedRangeObject] = []

    def fetch(item: PlannedTensor) -> MaterializedRangeObject:
        value = materialize_official_range_object(
            snapshot,
            plan.shard_path,
            item.offset,
            item.length,
            transport,
            object_directory,
            chunk_bytes=bounded_chunk,
        )
        materialized.append(value)
        return value

    kda_objects = {item.official_name: fetch(item) for item in plan.kda_tensors}
    always_objects = {
        item.official_name: fetch(item) for item in plan.moe_plan.always_active
    }
    inputs = official_layer_inputs()
    route_objects = {**kda_objects, **always_objects}
    routes = derive_official_layer_routes(plan, route_objects, inputs)
    route_manifest = _layer_route_manifest(
        snapshot,
        index,
        config,
        header,
        plan,
        inputs,
        routes,
        kda_objects,
        always_objects,
    )
    route_manifest_path = output_directory / "route-state-manifest.json"
    _write_json_atomic(route_manifest_path, route_manifest)

    expert_plans: dict[int, ExpertPlan] = {}
    expert_objects: dict[int, MaterializedRangeObject] = {}
    for expert_id in routes.selected_experts:
        expert_plan = plan_official_expert(
            index, header, layer_id=plan.layer_id, expert_id=expert_id
        )
        value = materialize_official_range_object(
            snapshot,
            expert_plan.shard_path,
            expert_plan.payload_start,
            expert_plan.payload_bytes,
            transport,
            object_directory,
            chunk_bytes=bounded_chunk,
        )
        expert_plans[expert_id] = expert_plan
        expert_objects[expert_id] = value
        materialized.append(value)

    tensors = build_official_layer_source_tensors(
        plan,
        routes,
        expert_plans,
        kda_objects,
        always_objects,
        expert_objects,
    )
    official_metadata = {
        **route_manifest,
        "expert_objects": [
            {
                "expert_id": expert_id,
                "range": [
                    expert_plans[expert_id].payload_start,
                    expert_plans[expert_id].payload_end,
                ],
                "sha256": expert_objects[expert_id].sha256,
            }
            for expert_id in routes.selected_experts
        ],
    }
    assembled: AssembledOfficialMoeSource = assemble_official_moe_source(
        output_directory,
        tensors,
        _released_storage_config(),
        chunk_bytes=bounded_chunk,
        official_metadata=official_metadata,
        official_metadata_key="official_layer",
    )
    k3x_path = output_directory / "official-kda-layer-l1.k3x"
    convert(assembled.source_directory, k3x_path, chunk_bytes=bounded_chunk)
    reader = K3XReader.open(k3x_path)
    expected_optional = OPTIONAL_STORAGE_FIXTURE | OPTIONAL_OFFICIAL_MOE_FIXTURE
    if reader.superblock.optional_features != expected_optional:
        raise K3XError("INVALID_OFFICIAL_LAYER_ARTIFACT")
    route_manifest["expert_objects"] = official_metadata["expert_objects"]
    route_manifest["artifact"] = {
        "filename": k3x_path.name,
        "k3x_root_sha256": reader.superblock.root_sha256.hex(),
        "k3x_source_fingerprint_sha256": reader.superblock.source_sha256.hex(),
        "source_sha256": assembled.microshard_sha256,
        "tensor_sha256": assembled.tensor_sha256,
    }
    _write_json_atomic(route_manifest_path, route_manifest)
    return OfficialLayerMaterializationReport(
        assembled.source_directory,
        route_manifest_path,
        assembled.manifest_path,
        assembled.microshard_path,
        k3x_path,
        routes.selected_experts,
        sum(item.length for item in materialized),
        sum(item.response_bytes for item in materialized),
        sum(1 for item in materialized if item.reused),
        sum(item.requests for item in materialized),
        max((item.maximum_response_bytes for item in materialized), default=0),
        assembled.microshard_sha256,
        assembled.tensor_sha256,
        reader.superblock.root_sha256.hex(),
    )
