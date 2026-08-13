# 공식 Kimi K3 전체 체크포인트의 메타데이터 전용 토폴로지 증거를 생성합니다.
from __future__ import annotations

import argparse
import hashlib
import json
import os
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
    record = {
        "format": "k3x-official-topology-v1",
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
        **topology.to_record(),
    }
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    record["record_sha256"] = hashlib.sha256(encoded).hexdigest()
    _write_json_atomic(args.output.resolve(), record)
    print(json.dumps(record, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
