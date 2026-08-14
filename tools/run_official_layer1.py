# 공식 Kimi K3의 실제 KDA-MoE 레이어를 저장된 prefix state에서 이어 실행합니다.
from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch

from k3x_converter.format import K3XError
from k3x_converter.official_layer import (
    OFFICIAL_KDA_SOURCE_BLOB_ID,
    _rms_norm,
    _state_digest,
    _tensor_digest,
    plan_official_kda_layer,
)
from k3x_converter.official_moe import (
    materialize_official_range_object,
    route_official_hidden,
)
from k3x_converter.official_source import (
    discover_official_snapshot,
    inspect_official_shard_header,
    load_official_config,
    load_official_index,
    plan_official_expert,
)
from k3x_converter.official_transport import UrllibTransport
from k3x_converter.official_two_layer import _bf16_matvec, _dense_ffn, _situ
from k3x_ref.official_kda import (
    OfficialKdaConfig,
    OfficialKdaWeights,
    official_kda,
    zero_official_kda_state,
)
from k3x_ref.attn_res import apply_attn_res
from k3x_ref.ops import rms_norm
from tools.run_official_layer0 import (
    _load_tensor,
    _load_topology,
    _tensor_payload,
    _write_bytes_atomic,
    _write_json_atomic,
)
from tools.official_k3x_source import (
    expert_matvec,
    k3x_set_identity,
    load_planned_tensors,
    open_official_fragment,
    require_k3x_state_identity,
)


_E2M1 = (
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0,
)


def _load_state(path: Path, device: torch.device, layer_id: int):
    record = json.loads(path.read_text(encoding="utf-8"))
    digest = record.pop("record_sha256", None)
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    if digest != hashlib.sha256(encoded).hexdigest() or record.get(
        "completed_layer"
    ) != layer_id - 1:
        raise K3XError("OFFICIAL_PREFIX_STATE")
    record["record_sha256"] = digest

    def load(name: str):
        item = record["tensors"][name]
        tensor_path = path.parent / item["path"]
        if (
            tensor_path.stat().st_size != item["bytes"]
            or hashlib.sha256(tensor_path.read_bytes()).hexdigest() != item["sha256"]
        ):
            raise K3XError("OFFICIAL_PREFIX_STATE_TENSOR", name)
        return _load_tensor(
            tensor_path,
            "BF16" if item["dtype"] == "bfloat16" else "F32",
            item["shape"],
            device,
        )

    blocks = tuple(
        load(name)
        for name in sorted(
            (
                name
                for name in record["tensors"]
                if name.startswith("block_source_")
            ),
            key=lambda name: int(name.rsplit("_", 1)[1]),
        )
    )
    if not blocks:
        raise K3XError("OFFICIAL_PREFIX_STATE_BLOCKS")
    return record, load(f"hidden_after_layer_{layer_id - 1}"), blocks


def _residual_input(
    prefix: torch.Tensor,
    blocks: tuple[torch.Tensor, ...],
    norm: torch.Tensor,
    projection: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    bank = torch.stack(blocks)
    return apply_attn_res(
        prefix,
        bank,
        norm,
        projection.reshape(-1),
        epsilon,
    ).to(torch.bfloat16)


def _prepare_ffn_hidden(
    prefix: torch.Tensor,
    blocks: tuple[torch.Tensor, ...],
    residual_norm: torch.Tensor,
    residual_proj: torch.Tensor,
    post_norm: torch.Tensor,
    epsilon: float,
) -> torch.Tensor:
    mixed = _residual_input(
        prefix, blocks, residual_norm, residual_proj, epsilon
    )
    return rms_norm(mixed, post_norm, epsilon).to(torch.bfloat16)


def _fetch(snapshot, shard: str, item, object_dir: Path):
    return item, materialize_official_range_object(
        snapshot,
        shard,
        item.offset,
        item.length,
        UrllibTransport(),
        object_dir,
    )


def _expert_matrix(
    path: Path,
    plan,
    role: str,
    value: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    items = [item for item in plan.tensors if item.role == role]
    packed_item = next(item for item in items if item.canonical_name.endswith("weight_packed"))
    scale_item = next(item for item in items if item.canonical_name.endswith("weight_scale"))
    with path.open("rb") as stream:
        stream.seek(packed_item.offset - plan.payload_start)
        packed_bytes = stream.read(packed_item.length)
        stream.seek(scale_item.offset - plan.payload_start)
        scale_bytes = stream.read(scale_item.length)
    packed = torch.frombuffer(bytearray(packed_bytes), dtype=torch.uint8).to(device)
    scales = torch.frombuffer(bytearray(scale_bytes), dtype=torch.uint8).to(device)
    if bool((scales == 0xFF).any()):
        raise K3XError("INVALID_MXFP4")
    lookup = torch.tensor(_E2M1, dtype=torch.float32, device=device)
    nibbles = torch.stack(
        (packed.bitwise_and(0x0F), packed.bitwise_right_shift(4)), dim=1
    ).reshape(-1)
    decoded = lookup[nibbles.long()]
    exponents = scales.to(torch.int32) - 127
    scale_values = torch.ldexp(torch.ones_like(exponents, dtype=torch.float32), exponents)
    weight = (decoded * scale_values.repeat_interleave(32)).reshape(
        packed_item.shape[0], packed_item.shape[1] * 2
    )
    return weight @ value.float()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--object-dir", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--layer-id", type=int, required=True)
    parser.add_argument("--k3x-set", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise K3XError("CUDA_UNAVAILABLE")

    topology = _load_topology(args.topology.resolve())
    object_dir = args.object_dir.resolve()
    transport = UrllibTransport(cache_directory=object_dir / "official-metadata-cache")
    snapshot = discover_official_snapshot(transport)
    index = load_official_index(snapshot, transport)
    config = load_official_config(snapshot, transport)
    if (
        topology["resolved_revision"] != snapshot.resolved_revision
        or topology["index_sha256"] != index.sha256
        or topology["config_sha256"] != config.sha256
    ):
        raise K3XError("OFFICIAL_TOPOLOGY_SOURCE_DRIFT")
    layer_id = args.layer_id
    if (
        not 1 <= layer_id < topology["layer_count"]
        or topology["layers"][layer_id]["attention"] != "kda"
    ):
        raise K3XError("OFFICIAL_KDA_LAYER_SEQUENCE")
    prefix = f"language_model.model.layers.{layer_id}."
    shard = topology["layers"][layer_id]["shards"][0]
    header = inspect_official_shard_header(snapshot, shard, transport)
    plan = plan_official_kda_layer(
        index,
        header,
        config,
        source_blob_id=OFFICIAL_KDA_SOURCE_BLOB_ID,
        layer_id=layer_id,
    )

    download_start = time.perf_counter()
    objects = {}
    requests = 0
    downloaded_bytes = 0
    reused_objects = 0
    planned = (*plan.kda_tensors, *plan.moe_plan.always_active)
    store = (
        open_official_fragment(args.k3x_set, shard)
        if args.k3x_set is not None
        else None
    )
    set_identity = (
        k3x_set_identity(args.k3x_set) if args.k3x_set is not None else None
    )
    if store is None:
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(_fetch, snapshot, shard, item, object_dir): item
                for item in planned
            }
            for position, future in enumerate(as_completed(futures), 1):
                item, result = future.result()
                objects[item.official_name] = result
                requests += result.requests
                downloaded_bytes += result.response_bytes
                reused_objects += int(result.reused)
                print(
                    f"layer{layer_id}_trunk={position}/{len(planned)} "
                    f"downloaded_bytes={downloaded_bytes}",
                    flush=True,
                )
    else:
        reused_objects += len(planned)

    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    prior_state, hidden, block_sources = _load_state(
        args.state_dir.resolve() / "state.json", device, layer_id
    )
    if set_identity is not None:
        require_k3x_state_identity(prior_state, set_identity)
    roles = (
        load_planned_tensors(store, planned, device)
        if store is not None
        else {
            item.role: _load_tensor(
                objects[item.official_name].path,
                item.dtype,
                list(item.shape),
                device,
            )
            for item in planned
        }
    )
    residual = _residual_input(
        hidden,
        block_sources,
        roles["self_res_norm"],
        roles["self_res_proj"],
        config.rms_norm_eps,
    )
    kda_input = _rms_norm(residual, roles["input_norm"])
    kda_config = OfficialKdaConfig(7_168, 96, 128, 4, 1.0e-5, -5.0)
    kda_weights = OfficialKdaWeights(
        roles["kda_q_proj"], roles["kda_k_proj"], roles["kda_v_proj"],
        roles["kda_q_conv"].reshape(12_288, 4),
        roles["kda_k_conv"].reshape(12_288, 4),
        roles["kda_v_conv"].reshape(12_288, 4),
        roles["kda_f_a"], roles["kda_f_b"], roles["kda_a_log"],
        roles["kda_dt_bias"], roles["kda_beta"], roles["kda_output_gate"],
        roles["kda_output_norm"], roles["kda_output_proj"],
    )
    zero = zero_official_kda_state(kda_config, 1, device)
    kda = official_kda(kda_input.reshape(1, 1, 7_168), kda_weights, zero, kda_config)
    block_write = layer_id % config.attn_res_block_size == 0
    if block_write:
        prefix_sum = kda.output.reshape(-1)
        block_sources = (*block_sources, hidden)
    else:
        prefix_sum = (hidden.float() + kda.output.reshape(-1).float()).to(
            torch.bfloat16
        )
    ffn_hidden = _prepare_ffn_hidden(
        prefix_sum,
        block_sources,
        roles["mlp_res_norm"],
        roles["mlp_res_proj"].reshape(-1),
        roles["post_attention_norm"],
        config.rms_norm_eps,
    )
    route = route_official_hidden(
        ffn_hidden,
        roles["router"],
        roles["router_correction"],
        top_k=config.top_k,
    )
    print(
        f"layer{layer_id}_route={','.join(map(str, route.expert_ids))}",
        flush=True,
    )

    expert_objects = {}
    expert_plans = {
        expert_id: plan_official_expert(
            index, header, layer_id=layer_id, expert_id=expert_id
        )
        for expert_id in route.expert_ids
    }
    if store is None:
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(
                    materialize_official_range_object,
                    snapshot,
                    shard,
                    expert_plan.payload_start,
                    expert_plan.payload_bytes,
                    UrllibTransport(),
                    object_dir,
                ): expert_id
                for expert_id, expert_plan in expert_plans.items()
            }
            for position, future in enumerate(as_completed(futures), 1):
                expert_id = futures[future]
                result = future.result()
                expert_objects[expert_id] = result
                requests += result.requests
                downloaded_bytes += result.response_bytes
                reused_objects += int(result.reused)
                print(
                    f"layer{layer_id}_experts={position}/{len(route.expert_ids)} "
                    f"downloaded_bytes={downloaded_bytes}",
                    flush=True,
                )
    else:
        reused_objects += len(route.expert_ids)
    download_seconds = time.perf_counter() - download_start

    torch.cuda.synchronize(device)
    compute_start = time.perf_counter()
    latent = _bf16_matvec(ffn_hidden, roles["routed_down"])
    shared = _dense_ffn(
        ffn_hidden,
        roles["shared_gate"],
        roles["shared_up"],
        roles["shared_down"],
        config.activation_situ_beta,
        config.activation_situ_linear_beta,
    )
    mixed = torch.zeros_like(latent, dtype=torch.float32)
    for expert_id, contribution in zip(route.expert_ids, route.contributions):
        expert_plan = expert_plans[expert_id]
        gate = (
            expert_matvec(store, expert_plan, "gate", latent)
            if store is not None
            else _expert_matrix(
                expert_objects[expert_id].path, expert_plan, "gate", latent, device
            )
        )
        up = (
            expert_matvec(store, expert_plan, "up", latent)
            if store is not None
            else _expert_matrix(
                expert_objects[expert_id].path, expert_plan, "up", latent, device
            )
        )
        activated = _situ(
            gate,
            up,
            config.activation_situ_beta,
            config.activation_situ_linear_beta,
        )
        down = (
            expert_matvec(store, expert_plan, "down", activated)
            if store is not None
            else _expert_matrix(
                expert_objects[expert_id].path,
                expert_plan,
                "down",
                activated,
                device,
            )
        )
        mixed += contribution * down
    routed_norm = _rms_norm(mixed.to(torch.bfloat16), roles["routed_norm"])
    routed = _bf16_matvec(routed_norm, roles["routed_up"])
    combined = (routed.float() + shared.float()).to(torch.bfloat16)
    output = (prefix_sum.float() + combined.float()).to(torch.bfloat16)
    torch.cuda.synchronize(device)
    compute_seconds = time.perf_counter() - compute_start
    if not torch.isfinite(output).all():
        raise K3XError("OFFICIAL_LAYER1_NONFINITE")

    state_dir = args.state_dir.resolve()
    state_records = dict(prior_state["tensors"])
    new_tensors = {
        f"hidden_after_layer_{layer_id}": output,
        f"kda_{layer_id}_conv_q": kda.state.conv_q,
        f"kda_{layer_id}_conv_k": kda.state.conv_k,
        f"kda_{layer_id}_conv_v": kda.state.conv_v,
        f"kda_{layer_id}_recurrent_v_first": kda.state.recurrent_v_first,
    }
    if block_write:
        new_tensors[f"block_source_{layer_id}"] = hidden
    for name, tensor in new_tensors.items():
        path = state_dir / f"{name}.bin"
        payload = _tensor_payload(tensor)
        state_records[name] = {
            "path": path.name,
            "dtype": str(tensor.dtype).removeprefix("torch."),
            "shape": list(tensor.shape),
            "bytes": len(payload),
            "sha256": _write_bytes_atomic(path, payload),
        }
    state_manifest = {
        "format": "k3x-official-prefix-state-v1",
        "resolved_revision": snapshot.resolved_revision,
        "index_sha256": index.sha256,
        "config_sha256": config.sha256,
        "topology_record_sha256": topology["record_sha256"],
        "token_id": prior_state["token_id"],
        "completed_layer": layer_id,
        "tensors": state_records,
    }
    if set_identity is not None:
        state_manifest["k3x_set_manifest_sha256"] = set_identity
    state_encoded = json.dumps(
        state_manifest, sort_keys=True, separators=(",", ":")
    ).encode()
    state_manifest["record_sha256"] = hashlib.sha256(state_encoded).hexdigest()
    _write_json_atomic(state_dir / "state.json", state_manifest)

    result = {
        "format": "k3x-official-kda-moe-layer-execution-v1",
        "repository": snapshot.repository,
        "resolved_revision": snapshot.resolved_revision,
        "snapshot_sha256": snapshot.canonical_sha256,
        "index_sha256": index.sha256,
        "config_sha256": config.sha256,
        "topology_record_sha256": topology["record_sha256"],
        "token_id": prior_state["token_id"],
        "layer_id": layer_id,
        "completed_layers": list(range(layer_id + 1)),
        "route_expert_ids": list(route.expert_ids),
        "route_contributions": list(route.contributions),
        "layer_output_sha256": _tensor_digest(
            output, f"k3x-official-layer{layer_id}-output-bf16\0".encode()
        ),
        "layer_kda_state_sha256": _state_digest(kda.state),
        "state_manifest_sha256": state_manifest["record_sha256"],
        "downloaded_payload_bytes": downloaded_bytes,
        "requested_payload_bytes": plan.base_payload_bytes
        + config.top_k * plan.moe_plan.expert_payload_bytes,
        "range_requests": requests,
        "reused_objects": reused_objects,
        "download_seconds": download_seconds,
        "compute_seconds": compute_seconds,
        "cuda_device": torch.cuda.get_device_name(device),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "token_generated": False,
    }
    if set_identity is not None:
        result["weight_source"] = "k3x-set"
        result["k3x_set_manifest_sha256"] = set_identity
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["record_sha256"] = hashlib.sha256(encoded).hexdigest()
    _write_json_atomic(args.output.resolve(), result)
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
