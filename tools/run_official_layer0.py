# 공식 Kimi K3의 실제 임베딩 행과 dense 0번 레이어를 범위 스트리밍으로 실행합니다.
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from pathlib import Path

import torch

from k3x_converter.format import K3XError
from k3x_converter.official_layer import _state_digest, _tensor_digest
from k3x_converter.official_moe import (
    materialize_official_range_object,
    prepare_official_moe_hidden,
)
from k3x_converter.official_source import (
    discover_official_snapshot,
    inspect_official_shard_header,
    load_official_config,
    load_official_index,
)
from k3x_converter.official_transport import UrllibTransport
from k3x_converter.official_two_layer import _dense_ffn
from k3x_ref.official_kda import (
    OfficialKdaConfig,
    OfficialKdaWeights,
    official_kda,
    zero_official_kda_state,
)
from k3x_ref.ops import rms_norm
from tools.official_k3x_source import (
    k3x_set_identity,
    logical_torch_dtype,
    open_official_fragment,
)


_EMBEDDING = "language_model.model.embed_tokens.weight"
_LAYER_PREFIX = "language_model.model.layers.0."


def _write_json_atomic(path: Path, value: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    with partial.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)


def _write_bytes_atomic(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)
    return hashlib.sha256(payload).hexdigest()


def _tensor_payload(tensor: torch.Tensor) -> bytes:
    return tensor.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()


def _load_topology(path: Path) -> dict[str, object]:
    record = json.loads(path.read_text(encoding="utf-8"))
    digest = record.pop("record_sha256", None)
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    if digest != hashlib.sha256(encoded).hexdigest():
        raise K3XError("OFFICIAL_TOPOLOGY_DIGEST")
    record["record_sha256"] = digest
    if record.get("format") != "k3x-official-topology-v2":
        raise K3XError("OFFICIAL_TOPOLOGY_FORMAT")
    return record


def _validate_contract(
    contract: list[dict[str, object]], index, headers
) -> None:
    for item in contract:
        name = item["name"]
        shard = item["shard"]
        tensor = headers[shard].tensors.get(name)
        if (
            index.weight_map.get(name) != shard
            or tensor is None
            or tensor.dtype != item["dtype"]
            or list(tensor.shape) != item["shape"]
            or tensor.offset != item["offset"]
            or tensor.length != item["length"]
        ):
            raise K3XError("OFFICIAL_LAYER0_CONTRACT", str(name))


def _load_tensor(path: Path, dtype: str, shape: list[int], device: torch.device):
    torch_dtype = {"BF16": torch.bfloat16, "F32": torch.float32}.get(dtype)
    if torch_dtype is None:
        raise K3XError("OFFICIAL_LAYER0_DTYPE", dtype)
    return torch.from_file(
        str(path), shared=False, size=math.prod(shape), dtype=torch_dtype
    ).reshape(shape).to(device)


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--object-dir", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--token-id", type=int, default=1)
    parser.add_argument("--k3x-set", type=Path)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    if args.token_id < 0 or args.token_id >= 163_840:
        raise K3XError("OFFICIAL_TOKEN_ID")
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
        or topology["snapshot_sha256"] != snapshot.canonical_sha256
        or topology["index_sha256"] != index.sha256
        or topology["config_sha256"] != config.sha256
    ):
        raise K3XError("OFFICIAL_TOPOLOGY_SOURCE_DRIFT")

    contracts = topology["execution_contracts"]
    globals_contract = contracts["global"]
    layer_contract = contracts["dense_layer_0"]
    shard_paths = {item["shard"] for item in (*globals_contract, *layer_contract)}
    headers = {
        shard: inspect_official_shard_header(snapshot, shard, transport)
        for shard in sorted(shard_paths)
    }
    _validate_contract(globals_contract, index, headers)
    _validate_contract(layer_contract, index, headers)

    download_start = time.perf_counter()
    objects = {}
    requests = 0
    downloaded_bytes = 0
    reused_objects = 0
    stores = (
        {
            shard: open_official_fragment(args.k3x_set, shard)
            for shard in sorted(shard_paths)
        }
        if args.k3x_set is not None
        else {}
    )
    set_identity = (
        k3x_set_identity(args.k3x_set) if args.k3x_set is not None else None
    )
    if not stores:
        for position, item in enumerate(layer_contract, 1):
            result = materialize_official_range_object(
                snapshot,
                item["shard"],
                item["offset"],
                item["length"],
                UrllibTransport(),
                object_dir,
            )
            objects[item["name"]] = result
            requests += result.requests
            downloaded_bytes += result.response_bytes
            reused_objects += int(result.reused)
            print(
                f"layer0_objects={position}/{len(layer_contract)} "
                f"downloaded_bytes={downloaded_bytes}",
                flush=True,
            )
    else:
        reused_objects += len(layer_contract)

    embedding_item = next(
        item for item in globals_contract if item["name"] == _EMBEDDING
    )
    embedding_row_bytes = 7_168 * 2
    embedding = None
    if not stores:
        embedding = materialize_official_range_object(
            snapshot,
            embedding_item["shard"],
            embedding_item["offset"] + args.token_id * embedding_row_bytes,
            embedding_row_bytes,
            UrllibTransport(),
            object_dir,
        )
        requests += embedding.requests
        downloaded_bytes += embedding.response_bytes
        reused_objects += int(embedding.reused)
    else:
        reused_objects += 1
    download_seconds = time.perf_counter() - download_start

    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    weights = (
        {
            item["name"][len(_LAYER_PREFIX) :]: stores[item["shard"]].load(
                item["name"].removeprefix("language_model."),
                device=device,
                dtype=logical_torch_dtype(item["dtype"]),
            )
            for item in layer_contract
        }
        if stores
        else {
            item["name"][len(_LAYER_PREFIX) :]: _load_tensor(
                objects[item["name"]].path,
                item["dtype"],
                item["shape"],
                device,
            )
            for item in layer_contract
        }
    )
    hidden = (
        stores[embedding_item["shard"]]
        .load_rows("model.embed_tokens.weight", args.token_id, 1, device=device)
        .reshape(7_168)
        if stores
        else _load_tensor(embedding.path, "BF16", [7_168], device)
    )
    kda_config = OfficialKdaConfig(7_168, 96, 128, 4, 1.0e-5, -5.0)
    kda_weights = OfficialKdaWeights(
        q_proj=weights["self_attn.q_proj.weight"],
        k_proj=weights["self_attn.k_proj.weight"],
        v_proj=weights["self_attn.v_proj.weight"],
        q_conv=weights["self_attn.q_conv1d.weight"].reshape(12_288, 4),
        k_conv=weights["self_attn.k_conv1d.weight"].reshape(12_288, 4),
        v_conv=weights["self_attn.v_conv1d.weight"].reshape(12_288, 4),
        f_a_proj=weights["self_attn.f_a_proj.weight"],
        f_b_proj=weights["self_attn.f_b_proj.weight"],
        a_log=weights["self_attn.A_log"],
        dt_bias=weights["self_attn.dt_bias"],
        b_proj=weights["self_attn.b_proj.weight"],
        g_proj=weights["self_attn.g_proj.weight"],
        o_norm=weights["self_attn.o_norm.weight"],
        o_proj=weights["self_attn.o_proj.weight"],
    )

    torch.cuda.synchronize(device)
    compute_start = time.perf_counter()
    normalized = rms_norm(
        hidden, weights["input_layernorm.weight"], config.rms_norm_eps
    ).to(torch.bfloat16)
    zero = zero_official_kda_state(kda_config, 1, device)
    kda = official_kda(normalized.reshape(1, 1, 7_168), kda_weights, zero, kda_config)
    prefix_sum = kda.output.reshape(7_168)
    ffn_hidden = prepare_official_moe_hidden(
        prefix_sum,
        hidden,
        weights["mlp_res_norm.weight"],
        weights["mlp_res_proj.weight"].reshape(-1),
        weights["post_attention_layernorm.weight"],
        rms_norm_eps=config.rms_norm_eps,
    )
    ffn = _dense_ffn(
        ffn_hidden,
        weights["mlp.gate_proj.weight"],
        weights["mlp.up_proj.weight"],
        weights["mlp.down_proj.weight"],
        config.activation_situ_beta,
        config.activation_situ_linear_beta,
    )
    output = (prefix_sum.float() + ffn.float()).to(torch.bfloat16)
    torch.cuda.synchronize(device)
    compute_seconds = time.perf_counter() - compute_start
    if not torch.isfinite(output).all():
        raise K3XError("OFFICIAL_LAYER0_NONFINITE")

    state_dir = args.state_dir.resolve()
    state_tensors = {
        "block_source_0": hidden,
        "hidden_after_layer_0": output,
        "kda_0_conv_q": kda.state.conv_q,
        "kda_0_conv_k": kda.state.conv_k,
        "kda_0_conv_v": kda.state.conv_v,
        "kda_0_recurrent_v_first": kda.state.recurrent_v_first,
    }
    state_records = {}
    for name, tensor in state_tensors.items():
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
        "token_id": args.token_id,
        "completed_layer": 0,
        "tensors": state_records,
    }
    if set_identity is not None:
        state_manifest["k3x_set_manifest_sha256"] = set_identity
    state_encoded = json.dumps(
        state_manifest, sort_keys=True, separators=(",", ":")
    ).encode()
    state_manifest["record_sha256"] = hashlib.sha256(state_encoded).hexdigest()
    state_manifest_path = state_dir / "state.json"
    _write_json_atomic(state_manifest_path, state_manifest)

    result = {
        "format": "k3x-official-layer0-execution-v1",
        "repository": snapshot.repository,
        "resolved_revision": snapshot.resolved_revision,
        "snapshot_sha256": snapshot.canonical_sha256,
        "index_sha256": index.sha256,
        "config_sha256": config.sha256,
        "topology_record_sha256": topology["record_sha256"],
        "token_id": args.token_id,
        "completed_layers": [0],
        "layer0_output_sha256": _tensor_digest(
            output, b"k3x-official-layer0-output-bf16\0"
        ),
        "layer0_kda_state_sha256": _state_digest(kda.state),
        "state_manifest_sha256": state_manifest["record_sha256"],
        "downloaded_payload_bytes": downloaded_bytes,
        "requested_payload_bytes": sum(item["length"] for item in layer_contract)
        + embedding_row_bytes,
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


def main() -> int:
    return run(_parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
