# 공식 Kimi K3의 실제 3번 Gated MLA-MoE 레이어를 저장된 prefix state에서 이어 실행합니다.
from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch

from k3x_converter.format import K3XError
from k3x_converter.official_layer import _rms_norm, _tensor_digest
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
from k3x_ref.config import SyntheticK3Config
from k3x_ref.mla import MLAWeights, empty_mla_state, mla_decode
from tools.run_official_layer0 import (
    _load_tensor,
    _load_topology,
    _tensor_payload,
    _write_bytes_atomic,
    _write_json_atomic,
)
from tools.run_official_layer1 import (
    _expert_matrix,
    _load_state,
    _prepare_ffn_hidden,
    _residual_input,
)
from tools.official_k3x_source import (
    expert_matvec,
    k3x_set_identity,
    open_official_fragment,
    require_k3x_state_identity,
)


_ROLE_BY_SUFFIX = {
    "self_attention_res_norm.weight": "self_res_norm",
    "self_attention_res_proj.weight": "self_res_proj",
    "input_layernorm.weight": "input_norm",
    "self_attn.q_a_proj.weight": "q_a_proj",
    "self_attn.q_a_layernorm.weight": "q_a_norm",
    "self_attn.q_b_proj.weight": "q_b_proj",
    "self_attn.kv_a_proj_with_mqa.weight": "kv_a_proj",
    "self_attn.kv_a_layernorm.weight": "kv_a_norm",
    "self_attn.kv_b_proj.weight": "kv_b_proj",
    "self_attn.g_proj.weight": "g_proj",
    "self_attn.o_proj.weight": "o_proj",
    "mlp_res_norm.weight": "mlp_res_norm",
    "mlp_res_proj.weight": "mlp_res_proj",
    "post_attention_layernorm.weight": "post_attention_norm",
    "block_sparse_moe.gate.weight": "router",
    "block_sparse_moe.gate.e_score_correction_bias": "router_correction",
    "block_sparse_moe.routed_expert_down_proj.weight": "routed_down",
    "block_sparse_moe.routed_expert_norm.weight": "routed_norm",
    "block_sparse_moe.routed_expert_up_proj.weight": "routed_up",
    "block_sparse_moe.shared_experts.gate_proj.weight": "shared_gate",
    "block_sparse_moe.shared_experts.up_proj.weight": "shared_up",
    "block_sparse_moe.shared_experts.down_proj.weight": "shared_down",
}


def _fetch(snapshot, shard: str, item, object_dir: Path):
    return item, materialize_official_range_object(
        snapshot,
        shard,
        item["offset"],
        item["length"],
        UrllibTransport(),
        object_dir,
    )


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
    transport = UrllibTransport()
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
        or topology["layers"][layer_id]["attention"] != "mla"
    ):
        raise K3XError("OFFICIAL_MLA_LAYER_SEQUENCE")
    prefix = f"language_model.model.layers.{layer_id}."
    shard = topology["layers"][layer_id]["shards"][0]
    header = inspect_official_shard_header(snapshot, shard, UrllibTransport())
    trunk = []
    for name in sorted(header.tensors):
        if not name.startswith(prefix) or ".experts." in name:
            continue
        tensor = header.tensors[name]
        trunk.append(
            {
                "name": name,
                "dtype": tensor.dtype,
                "shape": list(tensor.shape),
                "shard": shard,
                "offset": tensor.offset,
                "length": tensor.length,
            }
        )
    if {item["name"][len(prefix) :] for item in trunk} != set(_ROLE_BY_SUFFIX):
        raise K3XError("OFFICIAL_MLA_CONTRACT")
    object_dir = args.object_dir.resolve()
    download_start = time.perf_counter()
    objects = {}
    requests = downloaded_bytes = reused_objects = 0
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
                for item in trunk
            }
            for position, future in enumerate(as_completed(futures), 1):
                item, result = future.result()
                objects[item["name"]] = result
                requests += result.requests
                downloaded_bytes += result.response_bytes
                reused_objects += int(result.reused)
                print(
                    f"layer{layer_id}_trunk={position}/{len(trunk)} "
                    f"downloaded_bytes={downloaded_bytes}",
                    flush=True,
                )
    else:
        reused_objects += len(trunk)

    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    prior_state, hidden, block_sources = _load_state(
        args.state_dir.resolve() / "state.json", device, layer_id
    )
    if set_identity is not None:
        require_k3x_state_identity(prior_state, set_identity)
    roles = (
        {
            _ROLE_BY_SUFFIX[item["name"][len(prefix) :]]: store.load(
                item["name"].removeprefix("language_model."), device=device
            )
            for item in trunk
        }
        if store is not None
        else {
            _ROLE_BY_SUFFIX[item["name"][len(prefix) :]]: _load_tensor(
                objects[item["name"]].path,
                item["dtype"],
                item["shape"],
                device,
            )
            for item in trunk
        }
    )
    residual = _residual_input(
        hidden,
        block_sources,
        roles["self_res_norm"],
        roles["self_res_proj"],
        config.rms_norm_eps,
    )
    attention_input = _rms_norm(residual, roles["input_norm"])
    mla_config = SyntheticK3Config(
        hidden_size=7_168,
        mla_heads=96,
        q_lora_rank=1_536,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        rms_norm_eps=config.rms_norm_eps,
    )
    mla_weights = MLAWeights(
        roles["q_a_proj"],
        roles["q_a_norm"],
        roles["q_b_proj"],
        roles["kv_a_proj"],
        roles["kv_a_norm"],
        roles["kv_b_proj"],
        roles["g_proj"],
        roles["o_proj"],
    )
    empty = empty_mla_state(1, mla_config, torch.bfloat16, device)
    attention_output, mla_state = mla_decode(
        attention_input.reshape(1, 1, 7_168), mla_weights, empty, mla_config
    )
    prefix_sum = (hidden.float() + attention_output.reshape(-1).float()).to(
        torch.bfloat16
    )
    if layer_id % config.attn_res_block_size == 0:
        block_sources = (*block_sources, hidden)
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

    expert_plans = {
        expert_id: plan_official_expert(
            index, header, layer_id=layer_id, expert_id=expert_id
        )
        for expert_id in route.expert_ids
    }
    expert_objects = {}
    if store is None:
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {
                executor.submit(
                    materialize_official_range_object,
                    snapshot,
                    shard,
                    plan.payload_start,
                    plan.payload_bytes,
                    UrllibTransport(),
                    object_dir,
                ): expert_id
                for expert_id, plan in expert_plans.items()
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
        plan = expert_plans[expert_id]
        gate = (
            expert_matvec(store, plan, "gate", latent)
            if store is not None
            else _expert_matrix(
                expert_objects[expert_id].path, plan, "gate", latent, device
            )
        )
        up = (
            expert_matvec(store, plan, "up", latent)
            if store is not None
            else _expert_matrix(
                expert_objects[expert_id].path, plan, "up", latent, device
            )
        )
        activated = _situ(
            gate,
            up,
            config.activation_situ_beta,
            config.activation_situ_linear_beta,
        )
        mixed += contribution * (
            expert_matvec(store, plan, "down", activated)
            if store is not None
            else _expert_matrix(
                expert_objects[expert_id].path, plan, "down", activated, device
            )
        )
    routed_norm = _rms_norm(mixed.to(torch.bfloat16), roles["routed_norm"])
    routed = _bf16_matvec(routed_norm, roles["routed_up"])
    combined = (routed.float() + shared.float()).to(torch.bfloat16)
    output = (prefix_sum.float() + combined.float()).to(torch.bfloat16)
    torch.cuda.synchronize(device)
    compute_seconds = time.perf_counter() - compute_start
    if not torch.isfinite(output).all():
        raise K3XError("OFFICIAL_LAYER3_NONFINITE")

    state_dir = args.state_dir.resolve()
    state_records = dict(prior_state["tensors"])
    new_tensors = {
        f"hidden_after_layer_{layer_id}": output,
        f"mla_{layer_id}_keys": mla_state.keys,
        f"mla_{layer_id}_values": mla_state.values,
        f"mla_{layer_id}_shared_keys": mla_state.shared_keys,
    }
    if layer_id % config.attn_res_block_size == 0:
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
        "format": "k3x-official-mla-moe-layer-execution-v1",
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
        "state_manifest_sha256": state_manifest["record_sha256"],
        "downloaded_payload_bytes": downloaded_bytes,
        "requested_payload_bytes": topology["layers"][layer_id][
            "single_token_source_bytes"
        ],
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
