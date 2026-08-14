# 공식 Kimi K3의 output AttnRes, final norm, LM-head scan으로 greedy token을 생성합니다.
from __future__ import annotations

import argparse
import hashlib
import json
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import torch

from k3x_converter.format import K3XError
from k3x_converter.official_moe import materialize_official_range_object
from k3x_converter.official_source import (
    discover_official_snapshot,
    inspect_official_shard_header,
    load_official_config,
    load_official_index,
)
from k3x_converter.official_transport import UrllibTransport
from k3x_ref.ops import rms_norm
from tools.run_official_layer0 import (
    _load_tensor,
    _load_topology,
    _tensor_payload,
    _write_bytes_atomic,
    _write_json_atomic,
)
from tools.run_official_layer1 import _load_state, _residual_input
from tools.official_k3x_source import (
    k3x_set_identity,
    logical_torch_dtype,
    open_official_fragment,
    require_k3x_state_identity,
)


_HEAD = "language_model.lm_head.weight"
_GLOBAL_ROLES = {
    "language_model.model.norm.weight": "final_norm",
    "language_model.model.output_attn_res_norm.weight": "output_res_norm",
    "language_model.model.output_attn_res_proj.weight": "output_res_proj",
}
_ROWS_PER_CHUNK = 4_096


def _fetch(snapshot, item, start: int, length: int, object_dir: Path):
    return materialize_official_range_object(
        snapshot,
        item["shard"],
        start,
        length,
        UrllibTransport(),
        object_dir,
    )


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--object-dir", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k3x-set", type=Path)
    return parser.parse_args(argv)


def run(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        raise K3XError("CUDA_UNAVAILABLE")

    context = getattr(args, "runtime_context", None)
    topology = context.topology if context is not None else _load_topology(
        args.topology.resolve()
    )
    object_dir = context.object_dir if context is not None else args.object_dir.resolve()
    transport = (
        context.transport
        if context is not None
        else UrllibTransport(cache_directory=object_dir / "official-metadata-cache")
    )
    snapshot = (
        context.snapshot if context is not None else discover_official_snapshot(transport)
    )
    index = context.index if context is not None else load_official_index(
        snapshot, transport
    )
    config = context.config if context is not None else load_official_config(
        snapshot, transport
    )
    if (
        topology["resolved_revision"] != snapshot.resolved_revision
        or topology["index_sha256"] != index.sha256
        or topology["config_sha256"] != config.sha256
    ):
        raise K3XError("OFFICIAL_TOPOLOGY_SOURCE_DRIFT")

    global_contract = {
        item["name"]: item
        for item in topology["execution_contracts"]["global"]
    }
    if set(_GLOBAL_ROLES) - set(global_contract) or _HEAD not in global_contract:
        raise K3XError("OFFICIAL_HEAD_CONTRACT")
    head = global_contract[_HEAD]
    shard = head["shard"]
    header = (
        context.header(shard)
        if context is not None
        else inspect_official_shard_header(snapshot, shard, transport)
    )
    for name in (*_GLOBAL_ROLES, _HEAD):
        item = global_contract[name]
        tensor = header.tensors.get(name)
        if (
            index.weight_map.get(name) != item["shard"]
            or tensor is None
            or tensor.dtype != item["dtype"]
            or list(tensor.shape) != item["shape"]
            or tensor.offset != item["offset"]
            or tensor.length != item["length"]
        ):
            raise K3XError("OFFICIAL_HEAD_CONTRACT", name)
    if head["dtype"] != "BF16" or head["shape"] != [163_840, 7_168]:
        raise K3XError("OFFICIAL_HEAD_SHAPE")

    download_start = time.perf_counter()
    requests = downloaded_bytes = reused_objects = 0
    small_objects = {}
    stores = (
        {
            shard: (
                context.store(shard)
                if context is not None
                else open_official_fragment(args.k3x_set, shard)
            )
            for shard in {item["shard"] for item in global_contract.values()}
        }
        if args.k3x_set is not None
        else {}
    )
    set_identity = (
        (
            context.set_identity
            if context is not None
            else k3x_set_identity(args.k3x_set)
        )
        if args.k3x_set is not None
        else None
    )
    if not stores:
        for name in _GLOBAL_ROLES:
            item = global_contract[name]
            result = _fetch(
                snapshot, item, item["offset"], item["length"], object_dir
            )
            small_objects[name] = result
            requests += result.requests
            downloaded_bytes += result.response_bytes
            reused_objects += int(result.reused)
    else:
        reused_objects += len(_GLOBAL_ROLES)

    device = torch.device("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    prior_state, hidden, block_sources = _load_state(
        args.state_dir.resolve() / "state.json", device, 93
    )
    if set_identity is not None:
        require_k3x_state_identity(prior_state, set_identity)
    global_weights = (
        {
            role: stores[global_contract[name]["shard"]].load(
                name.removeprefix("language_model."),
                device=device,
                dtype=logical_torch_dtype(global_contract[name]["dtype"]),
            )
            for name, role in _GLOBAL_ROLES.items()
        }
        if stores
        else {
            role: _load_tensor(
                small_objects[name].path,
                global_contract[name]["dtype"],
                global_contract[name]["shape"],
                device,
            )
            for name, role in _GLOBAL_ROLES.items()
        }
    )
    mixed = _residual_input(
        hidden,
        block_sources,
        global_weights["output_res_norm"],
        global_weights["output_res_proj"],
        config.rms_norm_eps,
    )
    normalized = rms_norm(
        mixed, global_weights["final_norm"], config.rms_norm_eps
    ).to(torch.bfloat16)

    row_bytes = 7_168 * 2
    chunks = []
    for first_row in range(0, 163_840, _ROWS_PER_CHUNK):
        rows = min(_ROWS_PER_CHUNK, 163_840 - first_row)
        chunks.append(
            (
                first_row,
                rows,
                head["offset"] + first_row * row_bytes,
                rows * row_bytes,
            )
        )

    best_token = -1
    best_logit = float("-inf")
    torch.cuda.synchronize(device)
    compute_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = (
            {
                executor.submit(
                    stores[head["shard"]].load_rows,
                    "lm_head.weight",
                    first_row,
                    rows,
                    device="cpu",
                ): (first_row, rows)
                for first_row, rows, _start, _length in chunks
            }
            if stores
            else {
                executor.submit(
                    _fetch, snapshot, head, start, length, object_dir
                ): (first_row, rows)
                for first_row, rows, start, length in chunks
            }
        )
        for position, future in enumerate(as_completed(futures), 1):
            first_row, rows = futures[future]
            if stores:
                weight = future.result().to(device)
                reused_objects += 1
            else:
                result = future.result()
                requests += result.requests
                downloaded_bytes += result.response_bytes
                reused_objects += int(result.reused)
                weight = _load_tensor(
                    result.path, "BF16", [rows, 7_168], device
                )
            logits = weight.float() @ normalized.float()
            local_index = int(torch.argmax(logits).item())
            local_logit = float(logits[local_index].item())
            token = first_row + local_index
            if local_logit > best_logit or (
                local_logit == best_logit and token < best_token
            ):
                best_logit = local_logit
                best_token = token
            del weight, logits
            print(
                f"lm_head_chunks={position}/{len(chunks)} "
                f"downloaded_bytes={downloaded_bytes}",
                flush=True,
            )
    torch.cuda.synchronize(device)
    total_seconds = time.perf_counter() - download_start
    compute_and_wait_seconds = time.perf_counter() - compute_start
    if best_token < 0 or not torch.isfinite(torch.tensor(best_logit)):
        raise K3XError("OFFICIAL_HEAD_NONFINITE")

    state_dir = args.state_dir.resolve()
    normalized_path = state_dir / "final_normalized_hidden.bin"
    normalized_payload = _tensor_payload(normalized)
    normalized_sha256 = _write_bytes_atomic(normalized_path, normalized_payload)
    result = {
        "format": "k3x-official-first-token-v1",
        "repository": snapshot.repository,
        "resolved_revision": snapshot.resolved_revision,
        "snapshot_sha256": snapshot.canonical_sha256,
        "index_sha256": index.sha256,
        "config_sha256": config.sha256,
        "topology_record_sha256": topology["record_sha256"],
        "input_token_id": prior_state["token_id"],
        "generated_token_id": best_token,
        "generated_logit_fp32": best_logit,
        "completed_layers": list(range(93)),
        "final_normalized_hidden_sha256": normalized_sha256,
        "downloaded_payload_bytes": downloaded_bytes,
        "requested_payload_bytes": head["length"]
        + sum(global_contract[name]["length"] for name in _GLOBAL_ROLES),
        "range_requests": requests,
        "reused_objects": reused_objects,
        "wall_seconds": total_seconds,
        "head_compute_and_wait_seconds": compute_and_wait_seconds,
        "cuda_device": torch.cuda.get_device_name(device),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "token_generated": True,
        "throughput_measured": False,
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
