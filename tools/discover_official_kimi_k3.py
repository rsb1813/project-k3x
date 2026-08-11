# 공식 Kimi K3 expert range를 dry-run하거나 bounded K3X로 변환합니다.
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

from k3x_converter.format import (
    K3XError,
    OPTIONAL_OFFICIAL_MOE_FIXTURE,
    OPTIONAL_STORAGE_FIXTURE,
)
from k3x_converter.official_layer import (
    OFFICIAL_KDA_SOURCE_BLOB_ID,
    materialize_official_kda_layer,
    plan_official_kda_layer,
)
from k3x_converter.official_moe import (
    materialize_official_moe_slice,
    plan_official_moe_slice,
)
from k3x_converter.official_source import (
    Transport,
    discover_official_snapshot,
    inspect_official_shard_header,
    load_official_config,
    load_official_index,
    materialize_official_expert_slice,
    plan_official_expert,
)
from k3x_converter.official_transport import UrllibTransport
from tools.verify_official_discovery import (
    CSV_FIELDS,
    canonical_record_sha256,
    summary_csv_row,
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


def _write_csv_atomic(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".partial")
    with partial.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)


def _inside(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _validate_output_dir(path: Path) -> Path:
    resolved = path.resolve()
    repository = Path(__file__).resolve().parents[1]
    if _inside(resolved, repository):
        relative = resolved.relative_to(repository)
        if not relative.parts or relative.parts[0] != "artifacts":
            raise K3XError("OFFICIAL_OUTPUT_LOCATION")
    return resolved


def _transport_numbers(transport: Transport) -> tuple[int, int]:
    stats = getattr(transport, "stats", None)
    if stats is None:
        calls = getattr(transport, "calls", ())
        return len(calls), 0
    return stats.requests, stats.maximum_response_bytes


def _build_record(
    *,
    mode: str,
    snapshot,
    index,
    config,
    header,
    plan,
    transport,
    wall_seconds: float,
    materialization=None,
) -> dict[str, object]:
    shard = snapshot.files[plan.shard_path]
    requests, maximum_response = _transport_numbers(transport)
    materialized = materialization is not None
    artifacts: dict[str, object] = {
        "payload_sha256": None,
        "microshard_sha256": None,
        "k3x_root_sha256": None,
        "tensor_sha256": {},
    }
    if materialized:
        artifacts = {
            "payload_sha256": materialization.payload_sha256,
            "microshard_sha256": materialization.microshard_sha256,
            "k3x_root_sha256": materialization.k3x_root_sha256,
            "tensor_sha256": dict(materialization.tensor_sha256),
        }
    record: dict[str, object] = {
        "format": "k3x-official-discovery-v1",
        "benchmark_id": "B-0027" if materialized else None,
        "mode": mode,
        "repository": snapshot.repository,
        "requested_revision": snapshot.requested_revision,
        "resolved_revision": snapshot.resolved_revision,
        "observed_at": snapshot.observed_at,
        "snapshot_sha256": snapshot.canonical_sha256,
        "file_count": snapshot.file_count,
        "repository_bytes": snapshot.repository_bytes,
        "index": {
            "path": "model.safetensors.index.json",
            "bytes": snapshot.files["model.safetensors.index.json"].size,
            "sha256": index.sha256,
            "tensor_count": index.tensor_count,
            "shard_count": len(index.shard_paths),
            "total_tensor_bytes": index.total_size,
        },
        "config": {
            "path": "config.json",
            "bytes": snapshot.files["config.json"].size,
            "sha256": config.sha256,
            "git_blob_id": config.git_blob_id,
        },
        "expert": {
            "layer_id": plan.layer_id,
            "expert_id": plan.expert_id,
            "shard_path": plan.shard_path,
            "shard_bytes": shard.size,
            "shard_lfs_sha256": shard.lfs_sha256,
            "payload_start": plan.payload_start,
            "payload_end": plan.payload_end,
            "payload_bytes": plan.payload_bytes,
            "tensor_count": len(plan.tensors),
        },
        "traffic": {
            "http_requests": requests,
            "metadata_bytes": (
                snapshot.api_bytes
                + snapshot.files["model.safetensors.index.json"].size
                + snapshot.files["config.json"].size
            ),
            "header_bytes": 8 + header.header_length,
            "tensor_payload_bytes": plan.payload_bytes if materialized else 0,
            "maximum_response_bytes": maximum_response,
        },
        "reader_valid": materialized,
        "optional_features": materialization and 1 or 0,
        "provenance": "transport-pinned-range",
        "full_shard_verified": False,
        "artifacts": artifacts,
        "wall_seconds": wall_seconds,
    }
    record["record_sha256"] = canonical_record_sha256(record)
    return record


def _build_moe_dry_run_record(
    *, snapshot, index, config, header, plan, transport, wall_seconds: float
) -> dict[str, object]:
    requests, maximum_response = _transport_numbers(transport)
    return {
        "format": "k3x-official-moe-discovery-v1",
        "scope": "moe-ffn",
        "mode": "dry-run",
        "repository": snapshot.repository,
        "requested_revision": snapshot.requested_revision,
        "resolved_revision": snapshot.resolved_revision,
        "observed_at": snapshot.observed_at,
        "snapshot_sha256": snapshot.canonical_sha256,
        "index_sha256": index.sha256,
        "config_sha256": config.sha256,
        "shard_path": plan.shard_path,
        "always_active_tensor_count": len(plan.always_active),
        "always_active_bytes": plan.always_active_bytes,
        "expert_payload_bytes": plan.expert_payload_bytes,
        "maximum_two_case_bytes": plan.maximum_two_case_bytes,
        "selected_experts": list(plan.selected_experts),
        "traffic": {
            "http_requests": requests,
            "metadata_bytes": (
                snapshot.api_bytes
                + snapshot.files["model.safetensors.index.json"].size
                + snapshot.files["config.json"].size
            ),
            "header_bytes": 8 + header.header_length,
            "tensor_payload_bytes": 0,
            "maximum_response_bytes": maximum_response,
        },
        "provenance": "transport-pinned-ranges",
        "full_shard_verified": False,
        "wall_seconds": wall_seconds,
    }


def _build_moe_materialization_record(
    *,
    snapshot,
    index,
    config,
    header,
    plan,
    transport,
    materialization,
    wall_seconds: float,
) -> dict[str, object]:
    record = _build_moe_dry_run_record(
        snapshot=snapshot,
        index=index,
        config=config,
        header=header,
        plan=plan,
        transport=transport,
        wall_seconds=wall_seconds,
    )
    record.update(
        format="k3x-official-moe-materialization-v1",
        mode="materialize",
        selected_experts=list(materialization.selected_experts),
        reader_valid=True,
        optional_features=(
            OPTIONAL_STORAGE_FIXTURE | OPTIONAL_OFFICIAL_MOE_FIXTURE
        ),
        artifacts={
            "microshard_sha256": materialization.microshard_sha256,
            "tensor_sha256": dict(materialization.tensor_sha256),
            "k3x_root_sha256": materialization.k3x_root_sha256,
        },
    )
    record["traffic"].update(
        tensor_payload_bytes=materialization.downloaded_payload_bytes,
        source_object_bytes=materialization.requested_payload_bytes,
        materialization_requests=materialization.requests,
        reused_objects=materialization.reused_objects,
        maximum_response_bytes=max(
            record["traffic"]["maximum_response_bytes"],
            materialization.maximum_response_bytes,
        ),
    )
    return record


def _build_layer_dry_run_record(
    *, snapshot, index, config, header, plan, transport, wall_seconds: float
) -> dict[str, object]:
    requests, maximum_response = _transport_numbers(transport)
    return {
        "format": "k3x-official-kda-layer-discovery-v1",
        "scope": "kda-layer",
        "mode": "dry-run",
        "repository": snapshot.repository,
        "requested_revision": snapshot.requested_revision,
        "resolved_revision": snapshot.resolved_revision,
        "observed_at": snapshot.observed_at,
        "snapshot_sha256": snapshot.canonical_sha256,
        "index_sha256": index.sha256,
        "config_sha256": config.sha256,
        "source_blob_id": plan.source_blob_id,
        "shard_path": plan.shard_path,
        "kda_tensor_count": len(plan.kda_tensors),
        "kda_payload_bytes": plan.kda_payload_bytes,
        "base_payload_bytes": plan.base_payload_bytes,
        "maximum_two_token_bytes": plan.maximum_two_token_bytes,
        "selected_experts": [],
        "traffic": {
            "http_requests": requests,
            "metadata_bytes": (
                snapshot.api_bytes
                + snapshot.files["model.safetensors.index.json"].size
                + snapshot.files["config.json"].size
            ),
            "header_bytes": 8 + header.header_length,
            "tensor_payload_bytes": 0,
            "maximum_response_bytes": maximum_response,
        },
        "provenance": "transport-pinned-ranges",
        "full_shard_verified": False,
        "wall_seconds": wall_seconds,
    }


def _build_layer_materialization_record(
    *, snapshot, index, config, header, plan, transport, materialization,
    wall_seconds: float,
) -> dict[str, object]:
    record = _build_layer_dry_run_record(
        snapshot=snapshot,
        index=index,
        config=config,
        header=header,
        plan=plan,
        transport=transport,
        wall_seconds=wall_seconds,
    )
    record.update(
        format="k3x-official-kda-layer-materialization-v1",
        mode="materialize",
        selected_experts=list(materialization.selected_experts),
        reader_valid=True,
        optional_features=(
            OPTIONAL_STORAGE_FIXTURE | OPTIONAL_OFFICIAL_MOE_FIXTURE
        ),
        artifacts={
            "microshard_sha256": materialization.microshard_sha256,
            "tensor_sha256": dict(materialization.tensor_sha256),
            "k3x_root_sha256": materialization.k3x_root_sha256,
        },
    )
    record["traffic"].update(
        tensor_payload_bytes=materialization.downloaded_payload_bytes,
        source_object_bytes=materialization.requested_payload_bytes,
        materialization_requests=materialization.requests,
        reused_objects=materialization.reused_objects,
        maximum_response_bytes=max(
            record["traffic"]["maximum_response_bytes"],
            materialization.maximum_response_bytes,
        ),
    )
    return record


def main(
    argv: list[str] | None = None,
    *,
    transport: Transport | None = None,
) -> int:
    parser = argparse.ArgumentParser(allow_abbrev=False)
    parser.add_argument(
        "--scope", choices=("expert", "moe-ffn", "kda-layer"), default="expert"
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--dry-run", action="store_true")
    mode.add_argument("--materialize-expert", action="store_true")
    mode.add_argument("--materialize", action="store_true")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--summary-json", type=Path)
    parser.add_argument("--summary-csv", type=Path)
    parser.add_argument("--chunk-bytes", type=int, default=257 * 1024)
    args = parser.parse_args(argv)
    if args.materialize_expert and args.output_dir is None:
        parser.error("--materialize-expert requires --output-dir")
    if args.materialize and args.output_dir is None:
        parser.error("--materialize requires --output-dir")
    if args.materialize and args.scope not in {"moe-ffn", "kda-layer"}:
        parser.error("--materialize requires --scope moe-ffn or kda-layer")
    if args.scope in {"moe-ffn", "kda-layer"} and args.materialize_expert:
        parser.error("--materialize-expert requires --scope expert")
    if args.summary_csv is not None and not args.materialize_expert:
        parser.error("--summary-csv requires --materialize-expert")
    if transport is None:
        if os.environ.get("K3X_TEST_OFFICIAL_DISCOVERY") != "1":
            raise K3XError("OFFICIAL_LIVE_OPT_IN_REQUIRED")
        transport = UrllibTransport()

    started = time.perf_counter()
    snapshot = discover_official_snapshot(transport)
    index = load_official_index(snapshot, transport)
    config = load_official_config(snapshot, transport)
    if args.scope == "kda-layer":
        selected_name = "language_model.model.layers.1.self_attn.q_proj.weight"
    elif args.scope == "moe-ffn":
        selected_name = "language_model.model.layers.1.block_sparse_moe.gate.weight"
    else:
        selected_name = (
            "language_model.model.layers.1.block_sparse_moe.experts.0."
            "w1.weight_packed"
        )
    shard_path = index.weight_map.get(selected_name)
    if shard_path is None:
        raise K3XError("INVALID_OFFICIAL_EXPERT")
    header = inspect_official_shard_header(snapshot, shard_path, transport)
    if args.scope == "kda-layer":
        source_file = snapshot.files.get("modeling_kimi_linear.py")
        if (
            source_file is None
            or source_file.size != 51_506
            or source_file.blob_id != OFFICIAL_KDA_SOURCE_BLOB_ID
            or source_file.lfs_sha256 is not None
        ):
            raise K3XError("INVALID_OFFICIAL_LAYER_SOURCE")
        layer_plan = plan_official_kda_layer(
            index,
            header,
            config,
            source_blob_id=source_file.blob_id,
            layer_id=1,
        )
        if args.materialize:
            output_dir = _validate_output_dir(args.output_dir)
            materialization = materialize_official_kda_layer(
                snapshot,
                index,
                config,
                header,
                layer_plan,
                transport,
                output_dir,
                chunk_bytes=args.chunk_bytes,
            )
            record = _build_layer_materialization_record(
                snapshot=snapshot,
                index=index,
                config=config,
                header=header,
                plan=layer_plan,
                transport=transport,
                materialization=materialization,
                wall_seconds=time.perf_counter() - started,
            )
        else:
            record = _build_layer_dry_run_record(
                snapshot=snapshot,
                index=index,
                config=config,
                header=header,
                plan=layer_plan,
                transport=transport,
                wall_seconds=time.perf_counter() - started,
            )
        if args.summary_json is not None:
            _write_json_atomic(args.summary_json, record)
        json.dump(record, sys.stdout, sort_keys=True, separators=(",", ":"))
        sys.stdout.write("\n")
        return 0
    if args.scope == "moe-ffn":
        moe_plan = plan_official_moe_slice(index, header, config, layer_id=1)
        if args.materialize:
            output_dir = _validate_output_dir(args.output_dir)
            materialization = materialize_official_moe_slice(
                snapshot,
                index,
                config,
                header,
                moe_plan,
                transport,
                output_dir,
                chunk_bytes=args.chunk_bytes,
            )
            record = _build_moe_materialization_record(
                snapshot=snapshot,
                index=index,
                config=config,
                header=header,
                plan=moe_plan,
                transport=transport,
                materialization=materialization,
                wall_seconds=time.perf_counter() - started,
            )
        else:
            record = _build_moe_dry_run_record(
                snapshot=snapshot,
                index=index,
                config=config,
                header=header,
                plan=moe_plan,
                transport=transport,
                wall_seconds=time.perf_counter() - started,
            )
        if args.summary_json is not None:
            _write_json_atomic(args.summary_json, record)
        json.dump(record, sys.stdout, sort_keys=True, separators=(",", ":"))
        sys.stdout.write("\n")
        return 0
    plan = plan_official_expert(index, header, layer_id=1, expert_id=0)
    materialization = None
    selected_mode = "dry-run"
    if args.materialize_expert:
        selected_mode = "materialize-expert"
        output_dir = _validate_output_dir(args.output_dir)
        materialization = materialize_official_expert_slice(
            snapshot,
            config,
            plan,
            transport,
            output_dir,
            chunk_bytes=args.chunk_bytes,
        )
    record = _build_record(
        mode=selected_mode,
        snapshot=snapshot,
        index=index,
        config=config,
        header=header,
        plan=plan,
        transport=transport,
        wall_seconds=time.perf_counter() - started,
        materialization=materialization,
    )
    if args.summary_csv is not None:
        _write_csv_atomic(args.summary_csv, summary_csv_row(record))
        record["summary_csv_sha256"] = hashlib.sha256(
            args.summary_csv.read_bytes()
        ).hexdigest()
    if args.summary_json is not None:
        _write_json_atomic(args.summary_json, record)
    json.dump(record, sys.stdout, sort_keys=True, separators=(",", ":"))
    sys.stdout.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
