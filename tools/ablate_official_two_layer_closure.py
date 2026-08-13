# 공식 2레이어 closure의 B-0034 증거를 원자적으로 생성하고 검증합니다.
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
from pathlib import Path
from typing import Mapping

if __package__:
    from tools import ablate_official_layer as base
else:
    import ablate_official_layer as base


CASES = (
    ("host-round-trip", "host-round-trip"),
    ("device-closure", "device-closure"),
)

_FORMAT = "k3x-official-two-layer-closure-v1"
_BENCHMARK = "B-0034"
_SCOPE = "official-two-layer-device-closure"
_REVISION = "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"
_SNAPSHOT = "deaa6394b80afe12976ce8efbbf2463f6808c291d83b029e6b0cfb98de90a4e5"
_INDEX = "a1c5210650ce71d2d3ae9ec5a101ac4afd3cf4b10091be589853437eb967febd"
_CONFIG = "9710e121a58d03ac92c8d6da287a19541994319afbbe6d6202af001ffd379213"
_SOURCE_BLOB = "b8c41e8bfce768d74d8da3a37e693f5ee43876a0"
_SHARDS = (
    "model-00002-of-000096.safetensors",
    "model-00003-of-000096.safetensors",
)
_STEP_ORDER = (("a", 1), ("a", 2), ("b", 1), ("b", 2))
_TOP_K = 16
_HIDDEN_BYTES = 2 * 7_168 * 4
_STATE_BYTES = 2 * 6_512_640
_KDA_OUTPUT_BYTES = 4 * 7_168 * 4
_ROUTER_BYTES = 4 * 896 * 4
_KDA_WEIGHT_BYTES = 887_800_832
_MOE_COMMON_BYTES = 367_008_768
_EXPERT_BYTES = 17_547_264
_ROUTE_PREPARATION_BYTES = 12_888_064
_DEVICE_FRONT_BYTES = 3 * 7_168 * 2
_MAXIMUM_ERROR = 2.0e-3
_OFFICIAL_RESIDENT_CAPACITY = 4_294_967_296
_FORBIDDEN = {
    "decode_tok_s",
    "prefill_tok_s",
    "ttft",
    "gpu_utilization",
    "gpu_memory_bandwidth",
    "nvme_gb_per_token",
    "nvme_read_gb_per_token",
    "physical_nvme_bytes",
    "physical_h2d_bytes",
    "quality",
    "quality_score",
    "quality_benchmark_results",
}
_RUNNER_FIELDS = (
    "schema",
    "mode",
    "warmup",
    "iterations",
    "wall_nanoseconds",
    "maximum_absolute_error",
    "weight_h2d_bytes",
    "activation_h2d_bytes",
    "device_to_host_bytes",
    "state_h2d_bytes",
    "state_d2h_bytes",
    "kda_output_d2h_bytes",
    "router_logit_d2h_bytes",
    "inter_layer_hidden_h2d_bytes",
    "inter_layer_hidden_d2h_bytes",
    "final_hidden_d2h_bytes",
    "layer_front_calls",
    "layer_tail_calls",
    "state_seeds",
    "state_continuations",
    "state_publications",
    "state_invalidations",
    "prepared_seeds",
    "prepared_consumes",
    "prepared_discards",
    "prepared_invalidations",
    "resident_weight_bytes",
    "peak_device_bytes",
    "k3x_root_sha256",
    "route_expert_ids",
    "route_contribution_sha256",
    "final_output_sha256",
    "final_state_sha256",
)
_RAW_FIELDS = _RUNNER_FIELDS
_CSV_FIELDS = ("name", "raw_json_sha256", *_RAW_FIELDS)
_SUMMARY_FIELDS = {
    "format",
    "benchmark",
    "scope",
    "evidence",
    "warmups",
    "iterations",
    "resident_bytes",
    "artifact_sha256",
    "manifest_sha256",
    "oracle_sha256",
    "runner_sha256",
    "aggregate_sha256",
    "artifact_bytes",
    "oracle_bytes",
    "manifest_identity",
    "records",
    "summary_csv_sha256",
}


def _hex(value: object, size: int = 64) -> bool:
    return (
        isinstance(value, str)
        and len(value) == size
        and all(character in "0123456789abcdef" for character in value)
    )


def _finite(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _manifest_value(source: Path | Mapping[str, object]) -> dict[str, object]:
    if isinstance(source, Path):
        return base._parse_json(source.read_bytes(), "two-layer manifest")
    if not isinstance(source, Mapping):
        raise RuntimeError("manifest identity diverged")
    return dict(source)


def manifest_identity(source: Path | Mapping[str, object]) -> dict[str, object]:
    manifest = _manifest_value(source)
    fixed = {
        "format": "k3x-official-two-layer-v1",
        "repository": "moonshotai/Kimi-K3",
        "resolved_revision": _REVISION,
        "snapshot_sha256": _SNAPSHOT,
        "index_sha256": _INDEX,
        "config_sha256": _CONFIG,
        "source_blob_id": _SOURCE_BLOB,
        "shard_paths": list(_SHARDS),
        "layer_ids": [1, 2],
        "step_order": [f"{position}:{layer}" for position, layer in _STEP_ORDER],
    }
    if any(manifest.get(field) != value for field, value in fixed.items()):
        raise RuntimeError("manifest official identity diverged")
    steps = manifest.get("steps")
    selected = manifest.get("selected_experts")
    final_states = manifest.get("final_state_sha256")
    oracle = manifest.get("oracle")
    artifact = manifest.get("artifact")
    if (
        not isinstance(steps, list)
        or len(steps) != 4
        or not isinstance(selected, list)
        or len(selected) != 2
        or not isinstance(final_states, list)
        or len(final_states) != 2
        or not isinstance(oracle, dict)
        or not isinstance(artifact, dict)
    ):
        raise RuntimeError("manifest graph identity diverged")
    identity_steps: list[dict[str, object]] = []
    for index, ((position, layer), step) in enumerate(
        zip(_STEP_ORDER, steps, strict=True)
    ):
        if not isinstance(step, dict):
            raise RuntimeError("manifest step identity diverged")
        route = step.get("expert_ids")
        contributions = step.get("contributions")
        if (
            step.get("position") != position
            or step.get("layer_id") != layer
            or not isinstance(route, list)
            or len(route) != _TOP_K
            or len(set(route)) != _TOP_K
            or any(type(value) is not int or not 0 <= value < 896 for value in route)
            or not isinstance(contributions, list)
            or len(contributions) != _TOP_K
            or any(not _finite(value) or value <= 0 for value in contributions)
            or abs(sum(contributions) - 1.0) > 1.0e-5
        ):
            raise RuntimeError("manifest route identity diverged")
        digest_fields = (
            "hidden_input_sha256",
            "block_sha256",
            "consumes_state_sha256",
            "state_sha256",
            "kda_output_sha256",
            "contribution_sha256",
            "output_sha256",
        )
        if any(not _hex(step.get(field)) for field in digest_fields):
            raise RuntimeError("manifest step digest identity diverged")
        identity_steps.append(
            {
                "position": position,
                "layer_id": layer,
                "expert_ids": route,
                "contributions": contributions,
                **{field: step[field] for field in digest_fields},
            }
        )
    if (
        identity_steps[1]["hidden_input_sha256"]
        != identity_steps[0]["output_sha256"]
        or identity_steps[3]["hidden_input_sha256"]
        != identity_steps[2]["output_sha256"]
        or identity_steps[2]["consumes_state_sha256"]
        != identity_steps[0]["state_sha256"]
        or identity_steps[3]["consumes_state_sha256"]
        != identity_steps[1]["state_sha256"]
    ):
        raise RuntimeError("manifest state or hidden chain diverged")
    expected_selected = [
        list(dict.fromkeys(steps[layer - 1]["expert_ids"] + steps[layer + 1]["expert_ids"]))
        for layer in (1, 2)
    ]
    if selected != expected_selected:
        raise RuntimeError("manifest selected expert identity diverged")
    if final_states != [steps[2]["state_sha256"], steps[3]["state_sha256"]]:
        raise RuntimeError("manifest final state identity diverged")
    if (
        oracle.get("format") != "k3x-official-two-layer-oracle-v1"
        or oracle.get("filename") != "official-two-layer-oracle-v1.bin"
        or oracle.get("bytes") != 13_053_992
        or not _hex(oracle.get("sha256"))
    ):
        raise RuntimeError("manifest oracle identity diverged")
    tensors = artifact.get("tensor_sha256")
    if (
        artifact.get("filename") != "official-two-layer.k3x"
        or not _hex(artifact.get("k3x_root_sha256"))
        or not _hex(artifact.get("source_sha256"))
        or not isinstance(tensors, dict)
        or any(not isinstance(name, str) or not _hex(digest) for name, digest in tensors.items())
    ):
        raise RuntimeError("manifest artifact identity diverged")
    return {
        **fixed,
        "steps": identity_steps,
        "selected_experts": selected,
        "final_state_sha256": final_states,
        "oracle": {
            "format": oracle["format"],
            "filename": oracle["filename"],
            "sha256": oracle["sha256"],
            "bytes": oracle["bytes"],
        },
        "artifact": {
            "filename": artifact["filename"],
            "k3x_root_sha256": artifact["k3x_root_sha256"],
            "source_sha256": artifact["source_sha256"],
            "tensor_sha256": tensors,
        },
    }


def _expected_resident(identity: Mapping[str, object], *, host: bool) -> int:
    selected = identity["selected_experts"]
    assert isinstance(selected, list)
    layer_weights = sum(
        _KDA_WEIGHT_BYTES + _MOE_COMMON_BYTES + len(layer) * _EXPERT_BYTES
        for layer in selected
    )
    route_preparation = len(selected) * _ROUTE_PREPARATION_BYTES
    device_front = 0 if host else len(selected) * _DEVICE_FRONT_BYTES
    return layer_weights + route_preparation + device_front


def _validate_record(
    record: Mapping[str, object],
    *,
    name: str,
    mode: str,
    warmups: int,
    iterations: int,
    resident_bytes: int,
    identity: Mapping[str, object],
) -> None:
    forbidden = _FORBIDDEN.intersection(record)
    if forbidden:
        raise RuntimeError(f"{name} contains forbidden metric {min(forbidden)}")
    if set(record) != set(_RAW_FIELDS):
        raise RuntimeError(f"{name} schema diverged")
    expected_identity = {
        "schema": "k3x-official-two-layer-bench-v1",
        "mode": mode,
        "warmup": warmups,
        "iterations": iterations,
        "k3x_root_sha256": identity["artifact"]["k3x_root_sha256"],
    }
    if any(record.get(field) != value for field, value in expected_identity.items()):
        label = "mode identity" if record.get("mode") != mode else "graph identity"
        raise RuntimeError(f"{name} {label} diverged")
    walls = record.get("wall_nanoseconds")
    if (
        not isinstance(walls, list)
        or len(walls) != iterations
        or any(type(value) is not int or value <= 0 for value in walls)
    ):
        raise RuntimeError(f"{name} wall sample identity diverged")
    error = record.get("maximum_absolute_error")
    if not _finite(error) or not 0 <= error <= _MAXIMUM_ERROR:
        raise RuntimeError(f"{name} numerical divergence")
    routes = record.get("route_expert_ids")
    contribution_digests = record.get("route_contribution_sha256")
    output_digests = record.get("final_output_sha256")
    state_digests = record.get("final_state_sha256")
    steps = identity["steps"]
    if (
        not isinstance(routes, list)
        or len(routes) != 4
        or any(
            not isinstance(route, list)
            or sorted(route) != sorted(steps[index]["expert_ids"])
            for index, route in enumerate(routes)
        )
        or not isinstance(contribution_digests, list)
        or len(contribution_digests) != 4
        or any(not _hex(value) for value in contribution_digests)
        or not isinstance(output_digests, list)
        or len(output_digests) != 2
        or any(not _hex(value) for value in output_digests)
        or not isinstance(state_digests, list)
        or len(state_digests) != 2
        or any(not _hex(value) for value in state_digests)
    ):
        raise RuntimeError(f"{name} measured identity diverged")
    if record.get("weight_h2d_bytes") != 0:
        raise RuntimeError(f"{name} warm weight transfer diverged")
    host = mode == "host-round-trip"
    expected_transfer = {
        "state_h2d_bytes": _STATE_BYTES,
        "state_d2h_bytes": _STATE_BYTES,
        "kda_output_d2h_bytes": _KDA_OUTPUT_BYTES if host else 0,
        "router_logit_d2h_bytes": _ROUTER_BYTES,
        "inter_layer_hidden_h2d_bytes": _HIDDEN_BYTES if host else 0,
        "inter_layer_hidden_d2h_bytes": _HIDDEN_BYTES if host else 0,
        "final_hidden_d2h_bytes": _HIDDEN_BYTES,
    }
    labels = {
        "state_h2d_bytes": "state transfer",
        "state_d2h_bytes": "state transfer",
        "kda_output_d2h_bytes": "KDA output transfer",
        "router_logit_d2h_bytes": "router transfer",
        "inter_layer_hidden_h2d_bytes": "inter-layer transfer",
        "inter_layer_hidden_d2h_bytes": "inter-layer transfer",
        "final_hidden_d2h_bytes": "final hidden transfer",
    }
    for field, value in expected_transfer.items():
        if record.get(field) != value:
            raise RuntimeError(f"{name} {labels[field]} diverged")
    expected_d2h = sum(
        expected_transfer[field]
        for field in (
            "state_d2h_bytes",
            "kda_output_d2h_bytes",
            "router_logit_d2h_bytes",
            "inter_layer_hidden_d2h_bytes",
            "final_hidden_d2h_bytes",
        )
    )
    if record.get("device_to_host_bytes") != expected_d2h:
        raise RuntimeError(f"{name} aggregate transfer formula diverged")
    activation = record.get("activation_h2d_bytes")
    if type(activation) is not int or activation < _STATE_BYTES:
        raise RuntimeError(f"{name} activation transfer diverged")
    runs = warmups + iterations
    lifetime = {
        "layer_front_calls": 0 if host else 4,
        "layer_tail_calls": 0 if host else 4,
        "state_seeds": 2 * runs,
        "state_continuations": 2 * runs,
        "state_publications": 2 * runs,
        "state_invalidations": 0,
        "prepared_seeds": 4 * runs,
        "prepared_consumes": 4 * runs,
        "prepared_discards": 0,
        "prepared_invalidations": 0,
    }
    if any(record.get(field) != value for field, value in lifetime.items()):
        raise RuntimeError(f"{name} lifetime counter diverged")
    expected_resident = _expected_resident(identity, host=host)
    if (
        expected_resident > resident_bytes
        or record.get("resident_weight_bytes") != expected_resident
    ):
        raise RuntimeError(f"{name} resident weight diverged")
    peak = record.get("peak_device_bytes")
    if type(peak) is not int or peak < expected_resident:
        raise RuntimeError(f"{name} peak device residency diverged")


def _run_case(
    artifact: Path,
    manifest: Path,
    oracle: Path,
    runner: Path,
    *,
    mode: str,
    resident_bytes: int,
    warmups: int,
    iterations: int,
) -> dict[str, object]:
    command = [
        str(runner),
        "--artifact",
        str(artifact),
        "--manifest",
        str(manifest),
        "--oracle",
        str(oracle),
        "--mode",
        mode,
        "--resident-bytes",
        str(resident_bytes),
        "--warmup",
        str(warmups),
        "--iterations",
        str(iterations),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "official two-layer benchmark failed")
    return base._parse_json(result.stdout, f"{mode} output")


def _write_csv(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=_CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            {field: base._scalar(record[field]) for field in _CSV_FIELDS}
            for record in records
        )
        stream.flush()
        os.fsync(stream.fileno())


def _cross_row_parity(records: list[dict[str, object]]) -> None:
    for field in (
        "route_expert_ids",
        "weight_h2d_bytes",
    ):
        if records[1][field] != records[0][field]:
            raise RuntimeError(f"cross-row {field} parity diverged")


def run_ablation(
    artifact: Path,
    manifest: Path,
    oracle: Path,
    runner: Path,
    *,
    output_dir: Path,
    warmups: int,
    iterations: int,
    resident_bytes: int = _OFFICIAL_RESIDENT_CAPACITY,
) -> dict[str, object]:
    if type(warmups) is not int or warmups < 0:
        raise ValueError("warmups must be non-negative")
    if type(iterations) is not int or iterations <= 0:
        raise ValueError("iterations must be positive")
    if type(resident_bytes) is not int or resident_bytes <= 0:
        raise ValueError("resident bytes must be positive")
    artifact, manifest, oracle, runner = (
        Path(value).resolve() for value in (artifact, manifest, oracle, runner)
    )
    for path in (artifact, manifest, oracle, runner):
        if not path.is_file():
            raise FileNotFoundError(path)
    identity = manifest_identity(manifest)
    if (
        oracle.stat().st_size != identity["oracle"]["bytes"]
        or base._sha256(oracle) != identity["oracle"]["sha256"]
    ):
        raise RuntimeError("oracle file identity diverged")
    output_dir = Path(output_dir).resolve()
    partial = output_dir.with_name(f".{output_dir.name}.partial")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    try:
        records: list[dict[str, object]] = []
        for name, mode in CASES:
            runner_record = _run_case(
                artifact,
                manifest,
                oracle,
                runner,
                mode=mode,
                resident_bytes=resident_bytes,
                warmups=warmups,
                iterations=iterations,
            )
            forbidden = _FORBIDDEN.intersection(runner_record)
            if forbidden:
                raise RuntimeError(f"{name} contains forbidden metric {min(forbidden)}")
            if set(runner_record) != set(_RUNNER_FIELDS):
                raise RuntimeError(f"{name} schema diverged")
            raw = runner_record
            _validate_record(
                raw,
                name=name,
                mode=mode,
                warmups=warmups,
                iterations=iterations,
                resident_bytes=resident_bytes,
                identity=identity,
            )
            raw_path = partial / f"{name}.json"
            base._write_file(raw_path, base._canonical(raw))
            records.append(
                {
                    "name": name,
                    "raw_json_sha256": base._sha256(raw_path),
                    **raw,
                }
            )
        _cross_row_parity(records)
        aggregate = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
        summary: dict[str, object] = {
            "format": _FORMAT,
            "benchmark": _BENCHMARK,
            "scope": _SCOPE,
            "evidence": "measured",
            "warmups": warmups,
            "iterations": iterations,
            "resident_bytes": resident_bytes,
            "artifact_sha256": base._sha256(artifact),
            "manifest_sha256": base._sha256(manifest),
            "oracle_sha256": base._sha256(oracle),
            "runner_sha256": base._sha256(runner),
            "aggregate_sha256": hashlib.sha256(aggregate).hexdigest(),
            "artifact_bytes": artifact.stat().st_size,
            "oracle_bytes": oracle.stat().st_size,
            "manifest_identity": identity,
            "records": records,
        }
        csv_path = partial / "summary.csv"
        _write_csv(csv_path, records)
        summary["summary_csv_sha256"] = base._sha256(csv_path)
        base._write_file(partial / "summary.json", base._summary_bytes(summary))
        base._fsync_directory(partial)
        os.replace(partial, output_dir)
        base._fsync_directory(output_dir.parent)
        return summary
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise


def verify_summary(
    summary_json: Path,
    summary_csv: Path,
    *,
    artifact: Path | None = None,
    manifest: Path | None = None,
    oracle: Path | None = None,
    runner: Path | None = None,
    strict_official: bool = True,
) -> dict[str, object]:
    summary_json, summary_csv = Path(summary_json), Path(summary_csv)
    summary = base._parse_json(summary_json.read_bytes(), "summary JSON")
    if summary_json.read_bytes() != base._summary_bytes(summary) or set(summary) != _SUMMARY_FIELDS:
        raise RuntimeError("summary schema or encoding diverged")
    if (
        summary.get("format"),
        summary.get("benchmark"),
        summary.get("scope"),
        summary.get("evidence"),
    ) != (_FORMAT, _BENCHMARK, _SCOPE, "measured"):
        raise RuntimeError("summary identity diverged")
    warmups = summary.get("warmups")
    iterations = summary.get("iterations")
    resident_bytes = summary.get("resident_bytes")
    if (
        not base._integer(warmups)
        or not base._integer(iterations, positive=True)
        or not base._integer(resident_bytes, positive=True)
    ):
        raise RuntimeError("summary transaction identity diverged")
    paths = (artifact, manifest, oracle, runner)
    if strict_official and any(path is None for path in paths):
        raise RuntimeError("strict verification requires all transaction inputs")
    if strict_official and (
        warmups != 3
        or iterations != 20
        or resident_bytes != _OFFICIAL_RESIDENT_CAPACITY
    ):
        raise RuntimeError("official transaction gate diverged")
    for field, path in {
        "artifact_sha256": artifact,
        "manifest_sha256": manifest,
        "oracle_sha256": oracle,
        "runner_sha256": runner,
    }.items():
        if path is not None and summary.get(field) != base._sha256(Path(path)):
            raise RuntimeError(f"{field} diverged")
    identity = summary.get("manifest_identity")
    if not isinstance(identity, dict):
        raise RuntimeError("summary manifest identity diverged")
    checked_identity = manifest_identity(identity)
    if identity != checked_identity:
        raise RuntimeError("summary manifest identity diverged")
    if manifest is not None and manifest_identity(Path(manifest)) != identity:
        raise RuntimeError("summary manifest identity diverged")
    artifact_bytes = summary.get("artifact_bytes")
    oracle_bytes = summary.get("oracle_bytes")
    digest_fields = (
        "artifact_sha256",
        "manifest_sha256",
        "oracle_sha256",
        "runner_sha256",
        "aggregate_sha256",
        "summary_csv_sha256",
    )
    if any(not _hex(summary.get(field)) for field in digest_fields):
        raise RuntimeError("summary digest identity diverged")
    if (
        summary.get("oracle_sha256") != identity["oracle"]["sha256"]
        or not base._integer(artifact_bytes, positive=True)
        or oracle_bytes != identity["oracle"]["bytes"]
    ):
        raise RuntimeError("summary input size diverged")
    if artifact is not None and Path(artifact).stat().st_size != artifact_bytes:
        raise RuntimeError("summary artifact bytes diverged")
    if oracle is not None and Path(oracle).stat().st_size != oracle_bytes:
        raise RuntimeError("summary oracle bytes diverged")
    records = summary.get("records")
    if not isinstance(records, list) or len(records) != len(CASES):
        raise RuntimeError("summary record count diverged")
    for record, (name, mode) in zip(records, CASES, strict=True):
        if (
            not isinstance(record, dict)
            or record.get("name") != name
            or set(record) != set(_CSV_FIELDS)
        ):
            raise RuntimeError("summary case order or schema diverged")
        raw = {field: record[field] for field in _RAW_FIELDS}
        _validate_record(
            raw,
            name=name,
            mode=mode,
            warmups=warmups,
            iterations=iterations,
            resident_bytes=resident_bytes,
            identity=identity,
        )
        raw_path = summary_json.parent / f"{name}.json"
        if record["raw_json_sha256"] != base._sha256(raw_path):
            raise RuntimeError(f"{name} raw JSON digest diverged")
        payload = base._parse_json(raw_path.read_bytes(), f"{name} raw JSON")
        if raw_path.read_bytes() != base._canonical(payload) or payload != raw:
            raise RuntimeError(f"{name} raw JSON payload diverged")
    _cross_row_parity(records)
    aggregate = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    if summary.get("aggregate_sha256") != hashlib.sha256(aggregate).hexdigest():
        raise RuntimeError("aggregate digest diverged")
    csv_bytes = summary_csv.read_bytes()
    if b"\r\n" in csv_bytes or summary.get("summary_csv_sha256") != hashlib.sha256(csv_bytes).hexdigest():
        raise RuntimeError("summary CSV digest or newline diverged")
    with summary_csv.open(newline="", encoding="utf-8") as stream:
        reader_value = csv.DictReader(stream)
        rows, fields = list(reader_value), tuple(reader_value.fieldnames or ())
    expected = [
        {field: base._scalar(record[field]) for field in _CSV_FIELDS}
        for record in records
    ]
    if fields != _CSV_FIELDS or rows != expected:
        raise RuntimeError("summary CSV parity diverged")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--oracle", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--resident-bytes", type=int, default=_OFFICIAL_RESIDENT_CAPACITY
    )
    parser.add_argument("--verify-existing", action="store_true")
    arguments = parser.parse_args(argv)
    if not arguments.verify_existing:
        run_ablation(
            arguments.artifact,
            arguments.manifest,
            arguments.oracle,
            arguments.runner,
            output_dir=arguments.output_dir,
            warmups=arguments.warmups,
            iterations=arguments.iterations,
            resident_bytes=arguments.resident_bytes,
        )
    verify_summary(
        arguments.output_dir / "summary.json",
        arguments.output_dir / "summary.csv",
        artifact=arguments.artifact,
        manifest=arguments.manifest,
        oracle=arguments.oracle,
        runner=arguments.runner,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
