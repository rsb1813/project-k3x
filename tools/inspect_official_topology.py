# 공식 Kimi K3 전체 체크포인트의 메타데이터 전용 토폴로지 증거를 생성합니다.
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from k3x_converter.official_source import (
    discover_official_snapshot,
    inspect_official_shard_header,
    load_official_config,
    load_official_index,
)
from k3x_converter.official_topology import build_official_topology
from k3x_converter.official_transport import UrllibTransport


_EXPERT = re.compile(
    r"language_model\.model\.layers\.(\d+)\.block_sparse_moe\.experts\.(\d+)\."
)
_DTYPE_BYTES = {"BF16": 2, "F32": 4, "U8": 1}


def _write_json_atomic(path: Path, record: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    payload = json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n"
    with partial.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    transport = UrllibTransport()
    snapshot = discover_official_snapshot(transport)
    index = load_official_index(snapshot, transport)
    config = load_official_config(snapshot, transport)
    headers = {}
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(
                inspect_official_shard_header,
                snapshot,
                shard_path,
                UrllibTransport(),
            ): shard_path
            for shard_path in index.shard_paths
        }
        for position, future in enumerate(as_completed(futures), 1):
            shard_path = futures[future]
            headers[shard_path] = future.result()
            if position % 8 == 0 or position == len(index.shard_paths):
                print(
                    f"inspected_headers={position}/{len(index.shard_paths)}",
                    file=sys.stderr,
                    flush=True,
                )

    topology = build_official_topology(index, config, headers)
    topology_record = topology.to_record()
    layer_active_bytes = []
    contracts: dict[str, list[dict[str, object]]] = {
        "global": [],
        "dense_layer_0": [],
        "kda_moe_layer_1": [],
        "mla_moe_layer_3": [],
    }

    def tensor_record(name: str) -> dict[str, object]:
        shard_path = index.weight_map[name]
        tensor = headers[shard_path].tensors[name]
        return {
            "name": name,
            "dtype": tensor.dtype,
            "shape": list(tensor.shape),
            "shard": shard_path,
            "offset": tensor.offset,
            "length": tensor.length,
        }

    global_names = topology.global_text_tensors
    contracts["global"] = [tensor_record(name) for name in global_names]
    for layer in range(config.num_hidden_layers):
        prefix = f"language_model.model.layers.{layer}."
        names = sorted(name for name in index.weight_map if name.startswith(prefix))
        nonexpert_bytes = 0
        experts: dict[int, int] = {}
        for name in names:
            tensor = headers[index.weight_map[name]].tensors[name]
            match = _EXPERT.match(name)
            if match is None:
                nonexpert_bytes += tensor.length
            else:
                expert_id = int(match.group(2))
                experts[expert_id] = experts.get(expert_id, 0) + tensor.length
        if layer == 0:
            active_bytes = nonexpert_bytes
        else:
            if set(experts) != set(range(config.num_experts)):
                raise RuntimeError(f"layer {layer} expert set mismatch")
            expert_bytes = set(experts.values())
            if len(expert_bytes) != 1:
                raise RuntimeError(f"layer {layer} expert byte mismatch")
            active_bytes = nonexpert_bytes + config.top_k * expert_bytes.pop()
        layer_active_bytes.append(active_bytes)
        topology_record["layers"][layer]["single_token_source_bytes"] = active_bytes
        if layer in (0, 1, 3):
            contract_key = {
                0: "dense_layer_0",
                1: "kda_moe_layer_1",
                3: "mla_moe_layer_3",
            }[layer]
            contracts[contract_key] = [
                tensor_record(name)
                for name in names
                if _EXPERT.match(name) is None or ".experts.0." in name
            ]

    embedding_name = "language_model.model.embed_tokens.weight"
    embedding = headers[index.weight_map[embedding_name]].tensors[embedding_name]
    item_bytes = _DTYPE_BYTES.get(embedding.dtype)
    if item_bytes is None or embedding.shape != (163_840, 7_168):
        raise RuntimeError("embedding contract mismatch")
    embedding_row_bytes = item_bytes * embedding.shape[1]
    global_active_bytes = sum(
        headers[index.weight_map[name]].tensors[name].length
        for name in global_names
        if name != embedding_name
    ) + embedding_row_bytes
    record = {
        "format": "k3x-official-topology-v2",
        "repository": snapshot.repository,
        "requested_revision": snapshot.requested_revision,
        "resolved_revision": snapshot.resolved_revision,
        "snapshot_sha256": snapshot.canonical_sha256,
        "index_sha256": index.sha256,
        "config_sha256": config.sha256,
        "checkpoint_tensor_bytes": index.total_size,
        "checkpoint_tensor_count": index.tensor_count,
        "shard_count": len(index.shard_paths),
        "header_bytes": sum(8 + header.header_length for header in headers.values()),
        "tensor_payload_bytes": 0,
        "single_token_source_bytes": global_active_bytes + sum(layer_active_bytes),
        "embedding_row_bytes": embedding_row_bytes,
        "full_embedding_avoided_bytes": embedding.length - embedding_row_bytes,
        "execution_contracts": contracts,
        **topology_record,
    }
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    record["record_sha256"] = hashlib.sha256(encoded).hexdigest()
    _write_json_atomic(args.output.resolve(), record)
    print(json.dumps(record, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
