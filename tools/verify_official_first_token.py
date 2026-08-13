# 공식 Kimi K3 첫 토큰의 레이어 연속성, digest, state, traffic을 검증합니다.
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

from k3x_converter.format import K3XError
from tools.run_official_layer0 import _write_json_atomic


def _canonical_digest(record: dict[str, object]) -> str:
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def _load_record(path: Path, *, embedded_digest: bool = True) -> dict[str, object]:
    record = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise K3XError("OFFICIAL_EVIDENCE_SCHEMA", str(path))
    if embedded_digest:
        digest = record.pop("record_sha256", None)
        actual = _canonical_digest(record)
        if digest != actual:
            raise K3XError("OFFICIAL_EVIDENCE_DIGEST", str(path))
        record["record_sha256"] = digest
    return record


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--state", type=Path, required=True)
    parser.add_argument("--head", type=Path, required=True)
    parser.add_argument("--progress-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    topology = _load_record(args.topology.resolve())
    source = (
        topology["resolved_revision"],
        topology["index_sha256"],
        topology["config_sha256"],
        topology["record_sha256"],
    )
    fixed = {
        0: Path("results/b0038-official-layer0/cold-summary.json"),
        1: Path("results/b0039-official-layer1/summary.json"),
        2: Path("results/b0040-official-layer2/summary.json"),
        3: Path("results/b0041-official-layer3/summary.json"),
        4: Path("results/b0042-official-layer4/summary.json"),
    }
    records = []
    requested_bytes = downloaded_bytes = compute_seconds = 0
    peak_allocated = peak_reserved = 0
    for layer_id in range(93):
        path = fixed.get(
            layer_id,
            args.progress_dir.resolve() / f"layer-{layer_id:02d}.json",
        )
        record = _load_record(path)
        observed_layer = record.get("layer_id", record["completed_layers"][-1])
        if (
            observed_layer != layer_id
            or record["completed_layers"] != list(range(layer_id + 1))
            or record.get("token_generated") is not False
            or (
                record["resolved_revision"],
                record["index_sha256"],
                record["config_sha256"],
                record["topology_record_sha256"],
            )
            != source
        ):
            raise K3XError("OFFICIAL_LAYER_CHAIN", str(layer_id))
        requested_bytes += record["requested_payload_bytes"]
        downloaded_bytes += record["downloaded_payload_bytes"]
        compute_seconds += record["compute_seconds"]
        peak_allocated = max(peak_allocated, record["peak_cuda_allocated_bytes"])
        peak_reserved = max(peak_reserved, record["peak_cuda_reserved_bytes"])
        records.append(record["record_sha256"])

    state_path = args.state.resolve()
    state = _load_record(state_path)
    if (
        state.get("format") != "k3x-official-prefix-state-v1"
        or state.get("completed_layer") != 92
        or (
            state.get("resolved_revision"),
            state.get("index_sha256"),
            state.get("config_sha256"),
            state.get("topology_record_sha256"),
        )
        != source
    ):
        raise K3XError("OFFICIAL_FINAL_STATE")
    state_bytes = 0
    for name, item in state["tensors"].items():
        path = state_path.parent / item["path"]
        if (
            path.stat().st_size != item["bytes"]
            or hashlib.sha256(path.read_bytes()).hexdigest() != item["sha256"]
        ):
            raise K3XError("OFFICIAL_FINAL_STATE_TENSOR", name)
        state_bytes += item["bytes"]

    head = _load_record(args.head.resolve())
    if (
        head.get("format") != "k3x-official-first-token-v1"
        or head.get("token_generated") is not True
        or head.get("throughput_measured") is not False
        or head.get("completed_layers") != list(range(93))
        or head.get("input_token_id") != state.get("token_id")
        or not isinstance(head.get("generated_token_id"), int)
        or not 0 <= head["generated_token_id"] < 163_840
        or not isinstance(head.get("generated_logit_fp32"), (int, float))
        or not math.isfinite(head["generated_logit_fp32"])
        or (
            head["resolved_revision"],
            head["index_sha256"],
            head["config_sha256"],
            head["topology_record_sha256"],
        )
        != source
    ):
        raise K3XError("OFFICIAL_HEAD_CHAIN")
    normalized_path = state_path.parent / "final_normalized_hidden.bin"
    if (
        normalized_path.stat().st_size != 7_168 * 2
        or hashlib.sha256(normalized_path.read_bytes()).hexdigest()
        != head.get("final_normalized_hidden_sha256")
    ):
        raise K3XError("OFFICIAL_HEAD_HIDDEN")
    requested_bytes += head["requested_payload_bytes"]
    downloaded_bytes += head["downloaded_payload_bytes"]
    peak_allocated = max(peak_allocated, head["peak_cuda_allocated_bytes"])
    peak_reserved = max(peak_reserved, head["peak_cuda_reserved_bytes"])
    if requested_bytes != topology["single_token_source_bytes"]:
        raise K3XError("OFFICIAL_SOURCE_BYTE_CLOSURE")

    recovery_path = Path("results/b0041-official-layer3/failed-trunk-download.json")
    recovery = _load_record(recovery_path, embedded_digest=False)
    recovered_bytes = recovery.get("downloaded_payload_bytes")
    if (
        recovery.get("format") != "k3x-official-recovered-download-v1"
        or recovery.get("layer_id") != 3
        or recovery.get("published") is not False
        or recovered_bytes != 844_335_616
    ):
        raise K3XError("OFFICIAL_RECOVERED_DOWNLOAD")

    summary = {
        "format": "k3x-official-first-token-verification-v1",
        "repository": topology["repository"],
        "resolved_revision": source[0],
        "snapshot_sha256": topology["snapshot_sha256"],
        "index_sha256": source[1],
        "config_sha256": source[2],
        "topology_record_sha256": source[3],
        "verified_layer_count": len(records),
        "layer_record_sha256": records,
        "final_state_record_sha256": state["record_sha256"],
        "final_state_tensor_count": len(state["tensors"]),
        "final_state_bytes": state_bytes,
        "head_record_sha256": head["record_sha256"],
        "input_token_id": head["input_token_id"],
        "generated_token_id": head["generated_token_id"],
        "generated_logit_fp32": head["generated_logit_fp32"],
        "requested_source_bytes": requested_bytes,
        "successful_run_downloaded_bytes": downloaded_bytes,
        "recovered_failed_download_bytes": recovered_bytes,
        "observed_total_downloaded_bytes": downloaded_bytes + recovered_bytes,
        "layer_compute_seconds": compute_seconds,
        "peak_cuda_allocated_bytes": peak_allocated,
        "peak_cuda_reserved_bytes": peak_reserved,
        "token_generated": True,
        "throughput_measured": False,
    }
    summary["record_sha256"] = _canonical_digest(summary)
    _write_json_atomic(args.output.resolve(), summary)
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
