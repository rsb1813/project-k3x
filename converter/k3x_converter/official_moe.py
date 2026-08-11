# 공식 Kimi K3 MoE FFN의 bounded 계획과 결정적 입력을 정의합니다.
from __future__ import annotations

import hashlib
import json
import math
import os
import struct
from dataclasses import dataclass
from pathlib import Path

import torch

from .format import (
    K3XError,
    OPTIONAL_OFFICIAL_MOE_FIXTURE,
    OPTIONAL_STORAGE_FIXTURE,
)
from .official_source import (
    ExpertPlan,
    OfficialConfig,
    OfficialIndex,
    OfficialShardHeader,
    PlannedTensor,
    OfficialSnapshot,
    Transport,
    _released_storage_config,
    _fetch_exact_range,
    plan_official_expert,
)
from .reader import K3XReader
from .writer import CONVERTER_VERSION, convert


_HIDDEN_SIZE = 7_168
_EXPERT_PAYLOAD_BYTES = 17_547_264
_ALWAYS_ACTIVE_BYTES = 379_900_416

_ALWAYS_ACTIVE_TENSORS = (
    ("mlp_res_norm.weight", "BF16", (7_168,), "mlp_res_norm"),
    ("mlp_res_proj.weight", "BF16", (1, 7_168), "mlp_res_proj"),
    (
        "post_attention_layernorm.weight",
        "BF16",
        (7_168,),
        "post_attention_norm",
    ),
    ("block_sparse_moe.gate.weight", "BF16", (896, 7_168), "router"),
    (
        "block_sparse_moe.gate.e_score_correction_bias",
        "F32",
        (896,),
        "router_correction",
    ),
    (
        "block_sparse_moe.routed_expert_down_proj.weight",
        "BF16",
        (3_584, 7_168),
        "routed_down",
    ),
    (
        "block_sparse_moe.routed_expert_norm.weight",
        "BF16",
        (3_584,),
        "routed_norm",
    ),
    (
        "block_sparse_moe.routed_expert_up_proj.weight",
        "BF16",
        (7_168, 3_584),
        "routed_up",
    ),
    (
        "block_sparse_moe.shared_experts.gate_proj.weight",
        "BF16",
        (6_144, 7_168),
        "shared_gate",
    ),
    (
        "block_sparse_moe.shared_experts.up_proj.weight",
        "BF16",
        (6_144, 7_168),
        "shared_up",
    ),
    (
        "block_sparse_moe.shared_experts.down_proj.weight",
        "BF16",
        (7_168, 6_144),
        "shared_down",
    ),
)


@dataclass(frozen=True)
class OfficialMoeInput:
    name: str
    prefix_sum: tuple[float, ...]
    block_residual: tuple[float, ...]
    prefix_sha256: str
    block_sha256: str


@dataclass(frozen=True)
class OfficialMoePlan:
    layer_id: int
    shard_path: str
    index_sha256: str
    always_active: tuple[PlannedTensor, ...]
    always_active_bytes: int
    expert_payload_bytes: int
    maximum_two_case_bytes: int
    selected_experts: tuple[int, ...] = ()


@dataclass(frozen=True)
class OfficialMoeRoute:
    expert_ids: tuple[int, ...]
    contributions: tuple[float, ...]


@dataclass(frozen=True)
class OfficialMoeRouteCase:
    name: str
    route: OfficialMoeRoute


@dataclass(frozen=True)
class OfficialMoeRoutes:
    cases: tuple[OfficialMoeRouteCase, OfficialMoeRouteCase]
    selected_experts: tuple[int, ...]


@dataclass(frozen=True)
class MaterializedRangeObject:
    path: Path
    sha256: str
    length: int
    reused: bool
    requests: int
    maximum_response_bytes: int
    response_bytes: int = 0


@dataclass(frozen=True)
class OfficialMoeSourceTensor:
    name: str
    dtype: str
    shape: tuple[int, ...]
    object_path: Path
    object_offset: int
    length: int
    packed_shape: tuple[int, ...] | None = None


@dataclass(frozen=True)
class AssembledOfficialMoeSource:
    source_directory: Path
    manifest_path: Path
    microshard_path: Path
    microshard_sha256: str
    tensor_sha256: dict[str, str]


@dataclass(frozen=True)
class OfficialMoeMaterializationReport:
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


def _sha256_path(path: Path, chunk_bytes: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(chunk_bytes):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json_atomic(path: Path, value: dict[str, object]) -> None:
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)


def _load_json(path: Path) -> dict[str, object] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def materialize_official_range_object(
    snapshot: OfficialSnapshot,
    shard_path: str,
    start: int,
    length: int,
    transport: Transport,
    object_directory: Path,
    *,
    chunk_bytes: int = 8 * 1024 * 1024,
) -> MaterializedRangeObject:
    shard = snapshot.files.get(shard_path)
    if (
        shard is None
        or shard.lfs_sha256 is None
        or start < 0
        or length <= 0
        or start + length < start
        or start + length > shard.size
        or chunk_bytes <= 0
    ):
        raise K3XError("INVALID_OFFICIAL_RANGE_OBJECT")
    request_bytes = min(chunk_bytes, 8 * 1024 * 1024)
    object_directory = Path(object_directory)
    object_directory.mkdir(parents=True, exist_ok=True)
    identity = {
        "resolved_revision": snapshot.resolved_revision,
        "shard_path": shard_path,
        "shard_lfs_sha256": shard.lfs_sha256,
        "start": start,
        "length": length,
    }
    encoded_identity = json.dumps(
        identity, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    range_id = hashlib.sha256(encoded_identity).hexdigest()
    partial_path = object_directory / f"{range_id}.partial"
    progress_path = object_directory / f"{range_id}.progress.json"
    record_path = object_directory / f"{range_id}.object.json"

    record = _load_json(record_path) if record_path.exists() else None
    if record is not None and all(record.get(key) == value for key, value in identity.items()):
        digest = record.get("sha256")
        object_name = record.get("object")
        if isinstance(digest, str) and isinstance(object_name, str):
            object_path = object_directory / object_name
            if (
                object_path.name == f"{digest}.blob"
                and object_path.is_file()
                and object_path.stat().st_size == length
                and _sha256_path(object_path, request_bytes) == digest
            ):
                return MaterializedRangeObject(
                    object_path, digest, length, True, 0, 0
                )
            if object_path.is_file():
                object_path.unlink()
        record_path.unlink(missing_ok=True)

    completed = 0
    progress = _load_json(progress_path) if progress_path.exists() else None
    if partial_path.exists() and progress is not None:
        completed_value = progress.get("completed")
        prefix_digest = progress.get("partial_sha256")
        valid_progress = (
            all(progress.get(key) == value for key, value in identity.items())
            and isinstance(completed_value, int)
            and not isinstance(completed_value, bool)
            and 0 <= completed_value <= length
            and partial_path.stat().st_size == completed_value
            and isinstance(prefix_digest, str)
            and _sha256_path(partial_path, request_bytes) == prefix_digest
        )
        if valid_progress:
            completed = completed_value
        else:
            partial_path.unlink(missing_ok=True)
            progress_path.unlink(missing_ok=True)
    elif partial_path.exists() or progress_path.exists():
        partial_path.unlink(missing_ok=True)
        progress_path.unlink(missing_ok=True)

    requests = 0
    maximum_response = 0
    response_bytes = 0
    mode = "ab" if completed else "wb"
    with partial_path.open(mode) as stream:
        position = start + completed
        end = start + length
        while position < end:
            next_end = min(position + request_bytes, end) - 1
            body = _fetch_exact_range(
                snapshot, shard, transport, position, next_end
            )
            stream.write(body)
            stream.flush()
            os.fsync(stream.fileno())
            position += len(body)
            completed += len(body)
            requests += 1
            response_bytes += len(body)
            maximum_response = max(maximum_response, len(body))
            _write_json_atomic(
                progress_path,
                {
                    **identity,
                    "completed": completed,
                    "partial_sha256": _sha256_path(partial_path, request_bytes),
                },
            )

    if partial_path.stat().st_size != length:
        raise K3XError("OFFICIAL_RANGE_LENGTH_MISMATCH")
    digest = _sha256_path(partial_path, request_bytes)
    object_path = object_directory / f"{digest}.blob"
    if object_path.exists():
        if (
            object_path.stat().st_size != length
            or _sha256_path(object_path, request_bytes) != digest
        ):
            object_path.unlink()
            os.replace(partial_path, object_path)
        else:
            partial_path.unlink()
    else:
        os.replace(partial_path, object_path)
    _write_json_atomic(
        record_path,
        {**identity, "sha256": digest, "object": object_path.name},
    )
    progress_path.unlink(missing_ok=True)
    return MaterializedRangeObject(
        object_path,
        digest,
        length,
        False,
        requests,
        maximum_response,
        response_bytes,
    )


def build_official_moe_source_tensors(
    plan: OfficialMoePlan,
    routes: OfficialMoeRoutes,
    expert_plans: dict[int, ExpertPlan],
    always_objects: dict[str, MaterializedRangeObject],
    expert_objects: dict[int, MaterializedRangeObject],
) -> tuple[OfficialMoeSourceTensor, ...]:
    if (
        tuple(expert_plans) != routes.selected_experts
        or tuple(expert_objects) != routes.selected_experts
        or set(always_objects) != {item.official_name for item in plan.always_active}
    ):
        raise K3XError("INVALID_OFFICIAL_MOE_OBJECT_SET")

    result: list[OfficialMoeSourceTensor] = []

    def add_always(item: PlannedTensor) -> None:
        materialized = always_objects[item.official_name]
        if materialized.length != item.length:
            raise K3XError("INVALID_OFFICIAL_MOE_OBJECT_SET", item.official_name)
        result.append(
            OfficialMoeSourceTensor(
                item.canonical_name,
                item.dtype,
                item.shape,
                materialized.path,
                0,
                item.length,
            )
        )

    for item in plan.always_active[:6]:
        add_always(item)

    packed_shapes = {
        "gate": (3_072, 3_584),
        "up": (3_072, 3_584),
        "down": (3_584, 3_072),
    }
    for expert_id in routes.selected_experts:
        expert_plan = expert_plans[expert_id]
        materialized = expert_objects[expert_id]
        if (
            expert_plan.layer_id != plan.layer_id
            or expert_plan.expert_id != expert_id
            or expert_plan.shard_path != plan.shard_path
            or expert_plan.index_sha256 != plan.index_sha256
            or materialized.length != expert_plan.payload_bytes
        ):
            raise K3XError("INVALID_OFFICIAL_MOE_EXPERT_OBJECT", str(expert_id))
        by_role = {
            role: tuple(item for item in expert_plan.tensors if item.role == role)
            for role in ("gate", "up", "down")
        }
        if any(len(items) != 2 for items in by_role.values()):
            raise K3XError("INVALID_OFFICIAL_MOE_EXPERT_OBJECT", str(expert_id))
        for role in ("gate", "up", "down"):
            for item in by_role[role]:
                result.append(
                    OfficialMoeSourceTensor(
                        item.canonical_name,
                        item.dtype,
                        item.shape,
                        materialized.path,
                        item.offset - expert_plan.payload_start,
                        item.length,
                        packed_shapes[role]
                        if item.canonical_name.endswith(".weight_packed")
                        else None,
                    )
                )

    for item in plan.always_active[6:]:
        add_always(item)
    return tuple(result)


def assemble_official_moe_source(
    output_directory: Path,
    tensors: tuple[OfficialMoeSourceTensor, ...],
    config: dict[str, object],
    *,
    chunk_bytes: int = 8 * 1024 * 1024,
    official_metadata: dict[str, object] | None = None,
    official_metadata_key: str = "official_moe",
) -> AssembledOfficialMoeSource:
    if (
        chunk_bytes <= 0
        or not tensors
        or not isinstance(config, dict)
        or (official_metadata is not None and not isinstance(official_metadata, dict))
        or official_metadata_key not in {"official_moe", "official_layer"}
    ):
        raise K3XError("INVALID_OFFICIAL_MOE_SOURCE")
    names = tuple(item.name for item in tensors)
    if len(set(names)) != len(names):
        raise K3XError("INVALID_OFFICIAL_MOE_SOURCE")
    metadata: dict[str, dict[str, object]] = {}
    offset = 0
    for item in tensors:
        if (
            not item.name
            or item.dtype not in {"F32", "BF16", "U8"}
            or not item.shape
            or any(dimension <= 0 for dimension in item.shape)
            or item.object_offset < 0
            or item.length <= 0
            or not item.object_path.is_file()
            or item.object_offset + item.length > item.object_path.stat().st_size
        ):
            raise K3XError("INVALID_OFFICIAL_MOE_SOURCE", item.name)
        values = math.prod(item.shape)
        expected = values * {"F32": 4, "BF16": 2, "U8": 1}[item.dtype]
        if expected != item.length:
            raise K3XError("INVALID_OFFICIAL_MOE_SOURCE", item.name)
        metadata[item.name] = {
            "dtype": item.dtype,
            "shape": list(item.shape),
            "data_offsets": [offset, offset + item.length],
        }
        offset += item.length

    output_directory = Path(output_directory)
    source_directory = output_directory / "source"
    source_directory.mkdir(parents=True, exist_ok=True)
    microshard_path = source_directory / "model.safetensors"
    partial_path = microshard_path.with_suffix(".safetensors.partial")
    header = json.dumps(metadata, separators=(",", ":")).encode("utf-8")
    tensor_digests: dict[str, str] = {}
    with partial_path.open("wb") as output:
        output.write(struct.pack("<Q", len(header)))
        output.write(header)
        for item in tensors:
            digest = hashlib.sha256()
            remaining = item.length
            with item.object_path.open("rb") as source:
                source.seek(item.object_offset)
                while remaining:
                    chunk = source.read(min(chunk_bytes, remaining))
                    if not chunk:
                        raise K3XError("TRUNCATED_OFFICIAL_MOE_OBJECT", item.name)
                    output.write(chunk)
                    digest.update(chunk)
                    remaining -= len(chunk)
            tensor_digests[item.name] = digest.hexdigest()
        output.flush()
        os.fsync(output.fileno())
    os.replace(partial_path, microshard_path)

    packed_shapes: dict[str, list[int]] = {}
    tensor_order: list[str] = []
    for item in tensors:
        if item.name.endswith(".weight_scale"):
            continue
        if item.name.endswith(".weight_packed"):
            base = item.name.removesuffix(".weight_packed")
            if item.packed_shape is None:
                raise K3XError("INVALID_OFFICIAL_MOE_SOURCE", item.name)
            packed_shapes[base] = list(item.packed_shape)
            tensor_order.append(base)
        else:
            if item.packed_shape is not None:
                raise K3XError("INVALID_OFFICIAL_MOE_SOURCE", item.name)
            tensor_order.append(item.name)
    manifest = {
        "format": "k3-official-moe-slice-v1",
        "artifact_kind": "official_moe_fixture",
        "provenance": "transport-pinned-ranges",
        "config": config,
        "packed_shapes": packed_shapes,
        "weight_map": {name: microshard_path.name for name in names},
        "tensor_order": tensor_order,
        "source_sha256": _sha256_path(microshard_path, chunk_bytes),
        "tensor_sha256": tensor_digests,
    }
    if official_metadata is not None:
        manifest[official_metadata_key] = official_metadata
    manifest_path = source_directory / "source-manifest.json"
    _write_json_atomic(manifest_path, manifest)
    return AssembledOfficialMoeSource(
        source_directory,
        manifest_path,
        microshard_path,
        manifest["source_sha256"],
        tensor_digests,
    )


def prepare_official_moe_hidden(
    prefix_sum: torch.Tensor,
    block_residual: torch.Tensor,
    residual_norm: torch.Tensor,
    residual_proj: torch.Tensor,
    post_norm: torch.Tensor,
    *,
    rms_norm_eps: float,
) -> torch.Tensor:
    if (
        prefix_sum.dtype != torch.bfloat16
        or block_residual.dtype != torch.bfloat16
        or residual_norm.dtype != torch.bfloat16
        or residual_proj.dtype != torch.bfloat16
        or post_norm.dtype != torch.bfloat16
        or prefix_sum.ndim != 1
        or block_residual.shape != prefix_sum.shape
        or residual_norm.shape != prefix_sum.shape
        or residual_proj.shape != prefix_sum.shape
        or post_norm.shape != prefix_sum.shape
        or not math.isfinite(rms_norm_eps)
        or rms_norm_eps <= 0.0
    ):
        raise K3XError("INVALID_OFFICIAL_MOE_INPUT")
    values = torch.stack((block_residual, prefix_sum)).float()
    variance = values.pow(2).mean(dim=-1, keepdim=True)
    normalized = values * torch.rsqrt(variance + rms_norm_eps)
    score_weight = residual_norm.float() * residual_proj.float()
    scores = (normalized * score_weight).sum(dim=-1)
    probabilities = scores.softmax(dim=-1)
    hidden = (probabilities.unsqueeze(-1) * values).sum(dim=0).to(torch.bfloat16)
    hidden_float = hidden.float()
    result = (
        hidden_float
        * torch.rsqrt(hidden_float.pow(2).mean() + rms_norm_eps)
        * post_norm.float()
    )
    return result.to(torch.bfloat16)


def route_official_hidden(
    hidden: torch.Tensor,
    router_weight: torch.Tensor,
    correction_bias: torch.Tensor,
    *,
    top_k: int,
) -> OfficialMoeRoute:
    if (
        hidden.dtype != torch.bfloat16
        or router_weight.dtype != torch.bfloat16
        or correction_bias.dtype != torch.float32
        or hidden.ndim != 1
        or router_weight.ndim != 2
        or router_weight.shape[1] != hidden.shape[0]
        or correction_bias.shape != (router_weight.shape[0],)
        or isinstance(top_k, bool)
        or not 0 < top_k <= router_weight.shape[0]
    ):
        raise K3XError("INVALID_OFFICIAL_MOE_ROUTER")
    scores = torch.sigmoid(router_weight.float() @ hidden.float())
    adjusted = scores + correction_bias
    selected = torch.topk(adjusted, top_k, sorted=False).indices.tolist()
    canonical = tuple(
        sorted(selected, key=lambda expert: (-float(adjusted[expert]), expert))
    )
    weights = scores[list(canonical)]
    weights = weights / (weights.sum() + 1.0e-20)
    return OfficialMoeRoute(
        canonical,
        tuple(float(value) for value in weights),
    )


def derive_official_moe_routes(
    plan: OfficialMoePlan,
    objects: dict[str, MaterializedRangeObject],
) -> OfficialMoeRoutes:
    by_role = {item.role: item for item in plan.always_active}
    required_roles = {
        "mlp_res_norm",
        "mlp_res_proj",
        "post_attention_norm",
        "router",
        "router_correction",
    }
    if set(by_role) < required_roles:
        raise K3XError("INCOMPLETE_OFFICIAL_MOE_ROUTE_SOURCE")

    def load(role: str) -> torch.Tensor:
        item = by_role[role]
        materialized = objects.get(item.official_name)
        if (
            materialized is None
            or materialized.length != item.length
            or not materialized.path.is_file()
            or materialized.path.stat().st_size != item.length
            or _sha256_path(materialized.path, 8 * 1024 * 1024)
            != materialized.sha256
        ):
            raise K3XError("INVALID_OFFICIAL_MOE_ROUTE_OBJECT", item.official_name)
        dtype = torch.float32 if item.dtype == "F32" else torch.bfloat16
        values = math.prod(item.shape)
        return torch.from_file(
            str(materialized.path), shared=False, size=values, dtype=dtype
        ).reshape(item.shape)

    residual_norm = load("mlp_res_norm")
    residual_proj = load("mlp_res_proj").reshape(-1)
    post_norm = load("post_attention_norm")
    router_weight = load("router")
    correction = load("router_correction")
    cases: list[OfficialMoeRouteCase] = []
    for case in official_moe_inputs():
        hidden = prepare_official_moe_hidden(
            torch.tensor(case.prefix_sum, dtype=torch.bfloat16),
            torch.tensor(case.block_residual, dtype=torch.bfloat16),
            residual_norm,
            residual_proj,
            post_norm,
            rms_norm_eps=1.0e-5,
        )
        cases.append(
            OfficialMoeRouteCase(
                case.name,
                route_official_hidden(
                    hidden,
                    router_weight,
                    correction,
                    top_k=16,
                ),
            )
        )
    if set(cases[0].route.expert_ids) == set(cases[1].route.expert_ids):
        raise K3XError("OFFICIAL_MOE_ROUTES_NOT_DISTINCT")
    selected = tuple(
        dict.fromkeys(
            (*cases[0].route.expert_ids, *cases[1].route.expert_ids)
        )
    )
    return OfficialMoeRoutes((cases[0], cases[1]), selected)


def _input_values(
    multiplier: int,
    increment: int,
    modulus: int,
    offset: int,
) -> tuple[float, ...]:
    return tuple(
        (((multiplier * index + increment) % modulus) - offset) / 1024.0
        for index in range(_HIDDEN_SIZE)
    )


def _digest(values: tuple[float, ...]) -> str:
    encoded = struct.pack(f"<{len(values)}f", *values)
    return hashlib.sha256(encoded).hexdigest()


def official_moe_inputs() -> tuple[OfficialMoeInput, OfficialMoeInput]:
    specifications = (
        ("a", (17, 3, 257, 128), (29, 11, 251, 125)),
        ("b", (31, 7, 263, 131), (43, 19, 269, 134)),
    )
    result: list[OfficialMoeInput] = []
    for name, prefix_spec, block_spec in specifications:
        prefix = _input_values(*prefix_spec)
        block = _input_values(*block_spec)
        result.append(
            OfficialMoeInput(
                name,
                prefix,
                block,
                _digest(prefix),
                _digest(block),
            )
        )
    return result[0], result[1]


def plan_official_moe_slice(
    index: OfficialIndex,
    header: OfficialShardHeader,
    config: OfficialConfig,
    *,
    layer_id: int,
) -> OfficialMoePlan:
    if (
        layer_id not in (1, 2)
        or config.hidden_size != 7_168
        or config.num_experts != 896
        or config.top_k != 16
        or config.routed_latent_size != 3_584
        or config.expert_intermediate_size != 3_072
        or config.num_shared_experts != 2
        or config.activation_situ_beta != 4.0
        or config.activation_situ_linear_beta != 25.0
        or config.latent_moe_use_norm is not True
        or config.rms_norm_eps != 1.0e-5
        or config.moe_renormalize is not True
        or config.moe_router_activation_func != "sigmoid"
        or config.num_expert_group != 1
        or config.topk_group != 1
        or config.routed_scaling_factor != 1.0
    ):
        raise K3XError("INVALID_OFFICIAL_MOE_CONFIG")
    prefix = f"language_model.model.layers.{layer_id}"
    planned: list[PlannedTensor] = []
    for suffix, dtype, shape, role in _ALWAYS_ACTIVE_TENSORS:
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
            raise K3XError("INVALID_OFFICIAL_MOE_TENSOR", official_name)
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
    if total != _ALWAYS_ACTIVE_BYTES:
        raise K3XError("INVALID_OFFICIAL_MOE_LENGTH")
    return OfficialMoePlan(
        layer_id,
        header.shard_path,
        index.sha256,
        tuple(planned),
        total,
        _EXPERT_PAYLOAD_BYTES,
        total + 32 * _EXPERT_PAYLOAD_BYTES,
    )


def materialize_official_moe_slice(
    snapshot: OfficialSnapshot,
    index: OfficialIndex,
    config: OfficialConfig,
    header: OfficialShardHeader,
    plan: OfficialMoePlan,
    transport: Transport,
    output_directory: Path,
    *,
    chunk_bytes: int = 8 * 1024 * 1024,
) -> OfficialMoeMaterializationReport:
    if (
        chunk_bytes <= 0
        or plan.layer_id != 1
        or plan.shard_path != header.shard_path
        or plan.index_sha256 != index.sha256
    ):
        raise K3XError("INVALID_OFFICIAL_MOE_MATERIALIZATION")
    bounded_chunk = min(chunk_bytes, 8 * 1024 * 1024)
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    object_directory = output_directory / "objects"

    always_objects: dict[str, MaterializedRangeObject] = {}
    materialized: list[MaterializedRangeObject] = []
    for item in plan.always_active:
        value = materialize_official_range_object(
            snapshot,
            plan.shard_path,
            item.offset,
            item.length,
            transport,
            object_directory,
            chunk_bytes=bounded_chunk,
        )
        always_objects[item.official_name] = value
        materialized.append(value)

    routes = derive_official_moe_routes(plan, always_objects)
    inputs = official_moe_inputs()
    route_manifest = {
        "format": "k3x-official-moe-routes-v1",
        "converter_version": CONVERTER_VERSION,
        "repository": snapshot.repository,
        "requested_revision": snapshot.requested_revision,
        "resolved_revision": snapshot.resolved_revision,
        "snapshot_sha256": snapshot.canonical_sha256,
        "index_sha256": index.sha256,
        "config_sha256": config.sha256,
        "config_git_blob_id": config.git_blob_id,
        "shard_path": plan.shard_path,
        "shard_lfs_sha256": snapshot.files[plan.shard_path].lfs_sha256,
        "inputs": [
            {
                "name": item.name,
                "prefix_sha256": item.prefix_sha256,
                "block_sha256": item.block_sha256,
            }
            for item in inputs
        ],
        "routes": [
            {
                "name": case.name,
                "expert_ids": list(case.route.expert_ids),
                "contributions": list(case.route.contributions),
            }
            for case in routes.cases
        ],
        "selected_experts": list(routes.selected_experts),
        "always_active_objects": [
            {
                "name": item.official_name,
                "range": [item.offset, item.offset + item.length],
                "sha256": always_objects[item.official_name].sha256,
            }
            for item in plan.always_active
        ],
        "provenance": "transport-pinned-ranges",
    }
    route_manifest_path = output_directory / "route-manifest.json"
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

    tensors = build_official_moe_source_tensors(
        plan, routes, expert_plans, always_objects, expert_objects
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
    assembled = assemble_official_moe_source(
        output_directory,
        tensors,
        _released_storage_config(),
        chunk_bytes=bounded_chunk,
        official_metadata=official_metadata,
    )
    k3x_path = output_directory / "official-moe-l1.k3x"
    convert(
        assembled.source_directory,
        k3x_path,
        chunk_bytes=bounded_chunk,
    )
    reader = K3XReader.open(k3x_path)
    expected_optional = OPTIONAL_STORAGE_FIXTURE | OPTIONAL_OFFICIAL_MOE_FIXTURE
    if reader.superblock.optional_features != expected_optional:
        raise K3XError("INVALID_OFFICIAL_MOE_ARTIFACT")
    route_manifest["artifact"] = {
        "filename": k3x_path.name,
        "k3x_root_sha256": reader.superblock.root_sha256.hex(),
        "source_sha256": assembled.microshard_sha256,
        "tensor_sha256": assembled.tensor_sha256,
    }
    _write_json_atomic(route_manifest_path, route_manifest)
    return OfficialMoeMaterializationReport(
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
