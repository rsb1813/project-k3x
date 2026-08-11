# B-0029 공식 MoE FFN CUDA 측정과 digest 기반 증거 검증을 수행합니다.
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Mapping


CASES = (
    ("a-transient", "a", "transient"),
    ("a-resident", "a", "resident"),
    ("alternating-resident", "alternating", "resident"),
)

_FORMAT = "k3x-official-moe-cuda-v1"
_BENCHMARK = "B-0029"
_SCOPE = "official-moe-ffn-cuda"
_REVISION = "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"
_HIDDEN = 7_168
_TOP_K = 16
_EXPERT_BYTES = 17_547_264
_COMMON_BYTES = (
    3_584 * 7_168 * 2
    + 3_584 * 2
    + 7_168 * 3_584 * 2
    + 6_144 * 7_168 * 2
    + 6_144 * 7_168 * 2
    + 7_168 * 6_144 * 2
)
_ACTIVATION_BYTES = 2 * _HIDDEN * 4 + _TOP_K * 4 + 3 * _TOP_K * 16
_D2H_BYTES = _HIDDEN * 4
_MAXIMUM_ERROR = 2.0e-2

_RAW_FIELDS = (
    "artifact_kind", "repository", "resolved_revision", "case", "weight_mode",
    "token_semantics", "routing_semantics", "full_moe_ffn",
    "full_transformer_layer", "quality_measured", "k3x_root_sha256", "warmup",
    "iterations", "input_elements", "output_elements", "selected_union", "route_a",
    "route_b", "route_a_contributions", "route_b_contributions", "source_bytes",
    "k3x_bytes", "cpu_oracle_nanoseconds", "attention_residual_nanoseconds",
    "router_nanoseconds", "cold_latency_nanoseconds", "cold_kernel_nanoseconds",
    "cold_weight_h2d_bytes", "cold_bf16_weight_h2d_bytes",
    "cold_mxfp4_weight_h2d_bytes", "latency_nanoseconds_p05",
    "latency_nanoseconds_median", "latency_nanoseconds_p95", "kernel_nanoseconds",
    "orchestration_nanoseconds", "weight_h2d_bytes", "bf16_weight_h2d_bytes",
    "mxfp4_weight_h2d_bytes", "activation_h2d_bytes", "device_to_host_bytes",
    "resident_weight_bytes", "peak_resident_weight_bytes", "weight_cache_hits",
    "weight_cache_misses", "weight_cache_bypasses", "device_allocation_count",
    "stream_synchronization_count", "peak_vram_bytes", "maximum_absolute_error",
    "all_finite",
)
_CSV_FIELDS = ("name", "raw_json_sha256", *_RAW_FIELDS)
_SUMMARY_FIELDS = {
    "format", "benchmark", "scope", "evidence", "warmup", "iterations",
    "artifact_sha256", "manifest_sha256", "runner_sha256", "aggregate_sha256",
    "artifact_bytes", "manifest_identity", "records", "summary_csv_sha256",
}
_FORBIDDEN = {
    "decode_tok_s", "prefill_tok_s", "ttft", "gpu_utilization",
    "gpu_memory_bandwidth", "nvme_gb_per_token", "nvme_read_gb_per_token",
    "physical_nvme_bytes", "quality", "quality_score", "quality_benchmark_results",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_json(payload: str | bytes, label: str) -> dict[str, object]:
    def pairs(items: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in items:
            if key in result:
                raise RuntimeError(f"{label} contains duplicate keys")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload,
            object_pairs_hook=pairs,
            parse_constant=lambda value: (_ for _ in ()).throw(
                RuntimeError(f"{label} contains non-finite {value}")
            ),
        )
    except RuntimeError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is not JSON") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} is not an object")
    return value


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _summary_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode()


def _integer(value: object, *, positive: bool = False) -> bool:
    return type(value) is int and value >= (1 if positive else 0)


def _finite(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _manifest_identity(manifest: Mapping[str, object]) -> dict[str, object]:
    try:
        artifact = manifest["artifact"]
        routes = manifest["routes"]
        selected = manifest["selected_experts"]
        assert isinstance(artifact, dict) and isinstance(routes, list)
        assert isinstance(selected, list) and len(routes) == 2
        root = artifact["k3x_root_sha256"]
        route_a, route_b = routes
        assert isinstance(route_a, dict) and isinstance(route_b, dict)
        result = {
            "root": root,
            "selected": selected,
            "route_a": route_a["expert_ids"],
            "route_b": route_b["expert_ids"],
            "contributions_a": route_a["contributions"],
            "contributions_b": route_b["contributions"],
        }
    except (AssertionError, KeyError, TypeError) as error:
        raise RuntimeError("manifest identity diverged") from error
    if (
        manifest.get("repository") != "moonshotai/Kimi-K3"
        or manifest.get("resolved_revision") != _REVISION
        or not isinstance(root, str)
        or len(root) != 64
        or any(character not in "0123456789abcdef" for character in root)
        or any(
            not isinstance(result[field], list)
            for field in (
                "selected", "route_a", "route_b", "contributions_a",
                "contributions_b",
            )
        )
    ):
        raise RuntimeError("manifest identity diverged")
    selected = result["selected"]
    route_a = result["route_a"]
    route_b = result["route_b"]
    contribution_a = result["contributions_a"]
    contribution_b = result["contributions_b"]
    assert isinstance(selected, list) and isinstance(route_a, list)
    assert isinstance(route_b, list) and isinstance(contribution_a, list)
    assert isinstance(contribution_b, list)
    if any(type(value) is not int or not 0 <= value < 896 for value in route_a + route_b):
        raise RuntimeError("manifest route identity diverged")
    if (
        len(route_a) != _TOP_K
        or len(route_b) != _TOP_K
        or len(set(route_a)) != _TOP_K
        or len(set(route_b)) != _TOP_K
        or len(contribution_a) != _TOP_K
        or len(contribution_b) != _TOP_K
        or any(
            not _finite(value) or value <= 0
            for value in contribution_a + contribution_b
        )
        or abs(sum(contribution_a) - 1.0) > 1.0e-5
        or abs(sum(contribution_b) - 1.0) > 1.0e-5
    ):
        raise RuntimeError("manifest route identity diverged")
    expected_union = list(dict.fromkeys(route_a + route_b))
    if selected != expected_union:
        raise RuntimeError("manifest selected union diverged")
    return result


def _identity_manifest(manifest: Mapping[str, object]) -> dict[str, object]:
    identity = _manifest_identity(manifest)
    return {
        "repository": "moonshotai/Kimi-K3",
        "resolved_revision": _REVISION,
        "artifact": {"k3x_root_sha256": identity["root"]},
        "selected_experts": identity["selected"],
        "routes": [
            {
                "expert_ids": identity["route_a"],
                "contributions": identity["contributions_a"],
            },
            {
                "expert_ids": identity["route_b"],
                "contributions": identity["contributions_b"],
            },
        ],
    }


def _validate_record(
    record: Mapping[str, object], *, name: str, case: str, mode: str,
    warmup: int, iterations: int, manifest: Mapping[str, object], artifact_bytes: int,
) -> None:
    forbidden = _FORBIDDEN.intersection(record)
    if forbidden:
        raise RuntimeError(f"{name} contains forbidden metric {min(forbidden)}")
    if set(record) != set(_RAW_FIELDS):
        raise RuntimeError(f"{name} schema diverged")
    identity = _manifest_identity(manifest)
    expected = {
        "artifact_kind": "official_kimi_k3_moe_ffn",
        "repository": "moonshotai/Kimi-K3", "resolved_revision": _REVISION,
        "case": case, "weight_mode": mode, "token_semantics": False,
        "routing_semantics": True, "full_moe_ffn": True,
        "full_transformer_layer": False, "quality_measured": False,
        "k3x_root_sha256": identity["root"], "warmup": warmup,
        "iterations": iterations, "input_elements": _HIDDEN,
        "output_elements": _HIDDEN, "selected_union": identity["selected"],
        "route_a": identity["route_a"], "route_b": identity["route_b"],
        "source_bytes": 379_900_416 + len(identity["selected"]) * _EXPERT_BYTES,
        "k3x_bytes": artifact_bytes, "all_finite": True,
    }
    for field, value in expected.items():
        if record.get(field) != value or type(record.get(field)) is not type(value):
            raise RuntimeError(f"{name} identity field {field} diverged")
    for field, expected_field in (
        ("route_a_contributions", "contributions_a"),
        ("route_b_contributions", "contributions_b"),
    ):
        observed = record.get(field)
        expected_values = identity[expected_field]
        if (
            not isinstance(observed, list)
            or not isinstance(expected_values, list)
            or len(observed) != _TOP_K
            or any(
                not _finite(value) or abs(value - expected_value) > 1.0e-6
                for value, expected_value in zip(observed, expected_values, strict=True)
            )
        ):
            raise RuntimeError(f"{name} route contribution diverged")
    for field in (
        "cpu_oracle_nanoseconds", "attention_residual_nanoseconds", "router_nanoseconds",
        "cold_latency_nanoseconds", "cold_kernel_nanoseconds",
        "latency_nanoseconds_p05", "latency_nanoseconds_median",
        "latency_nanoseconds_p95", "kernel_nanoseconds", "peak_vram_bytes",
    ):
        if not _integer(record.get(field), positive=True):
            raise RuntimeError(f"{name} positive field {field} diverged")
    for field in set(_RAW_FIELDS) - set(expected) - {
        "maximum_absolute_error",
    }:
        if field.endswith("_bytes") or field.endswith("_count") or field in {
            "weight_cache_hits", "weight_cache_misses", "weight_cache_bypasses",
            "orchestration_nanoseconds", "stream_synchronization_count",
        }:
            if not _integer(record.get(field)):
                raise RuntimeError(f"{name} counter field {field} diverged")
    p05, median, p95 = (record[f"latency_nanoseconds_{field}"] for field in ("p05", "median", "p95"))
    if not p05 <= median <= p95:
        raise RuntimeError(f"{name} latency distribution diverged")
    error = record["maximum_absolute_error"]
    if not _finite(error) or not 0 <= error <= _MAXIMUM_ERROR:
        raise RuntimeError(f"{name} numerical divergence")

    calls = 2 if case == "alternating" else 1
    total_calls = calls * iterations
    selected_count = len(identity["selected"]) if case == "alternating" else _TOP_K
    cold_bf16 = _COMMON_BYTES if mode == "resident" else calls * _COMMON_BYTES
    cold_mxfp4 = selected_count * _EXPERT_BYTES
    common = {
        "cold_bf16_weight_h2d_bytes": cold_bf16,
        "cold_mxfp4_weight_h2d_bytes": cold_mxfp4,
        "cold_weight_h2d_bytes": cold_bf16 + cold_mxfp4,
        "activation_h2d_bytes": total_calls * _ACTIVATION_BYTES,
        "device_to_host_bytes": total_calls * _D2H_BYTES,
        "stream_synchronization_count": total_calls,
        "weight_cache_misses": 0, "weight_cache_bypasses": 0,
    }
    if any(record.get(field) != value for field, value in common.items()):
        raise RuntimeError(f"{name} common traffic diverged")
    if mode == "transient":
        mode_expected = {
            "bf16_weight_h2d_bytes": total_calls * _COMMON_BYTES,
            "mxfp4_weight_h2d_bytes": total_calls * _TOP_K * _EXPERT_BYTES,
            "weight_h2d_bytes": total_calls * (_COMMON_BYTES + _TOP_K * _EXPERT_BYTES),
            "resident_weight_bytes": 0, "peak_resident_weight_bytes": 0,
            "weight_cache_hits": 0, "device_allocation_count": total_calls * 102,
        }
    else:
        resident = _COMMON_BYTES + selected_count * _EXPERT_BYTES
        mode_expected = {
            "bf16_weight_h2d_bytes": 0, "mxfp4_weight_h2d_bytes": 0,
            "weight_h2d_bytes": 0, "resident_weight_bytes": resident,
            "peak_resident_weight_bytes": resident,
            "weight_cache_hits": total_calls * 54, "device_allocation_count": 0,
        }
    if any(record.get(field) != value for field, value in mode_expected.items()):
        raise RuntimeError(f"{name} {mode} traffic diverged")


def _run_case(artifact: Path, manifest: Path, runner: Path, *, case: str,
              mode: str, warmup: int, iterations: int) -> dict[str, object]:
    command = [str(runner), "--model", str(artifact), "--manifest", str(manifest),
               "--case", case, "--weight-mode", mode, "--warmup", str(warmup),
               "--iterations", str(iterations)]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "official MoE benchmark failed")
    return _parse_json(result.stdout, f"{case}-{mode} output")


def _scalar(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, dict)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def _write_csv(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=_CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows({field: _scalar(record[field]) for field in _CSV_FIELDS}
                         for record in records)


def run_ablation(artifact: Path, manifest: Path, runner: Path, *, output_dir: Path,
                 warmup: int, iterations: int) -> dict[str, object]:
    if type(warmup) is not int or warmup < 0:
        raise ValueError("warmup must be non-negative")
    if type(iterations) is not int or iterations <= 0:
        raise ValueError("iterations must be positive")
    artifact, manifest, runner = (Path(value).resolve() for value in (artifact, manifest, runner))
    for path in (artifact, manifest, runner):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest_value = _parse_json(manifest.read_bytes(), "route manifest")
    _manifest_identity(manifest_value)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for name, case, mode in CASES:
        raw = _run_case(artifact, manifest, runner, case=case, mode=mode,
                        warmup=warmup, iterations=iterations)
        _validate_record(raw, name=name, case=case, mode=mode, warmup=warmup,
                         iterations=iterations, manifest=manifest_value,
                         artifact_bytes=artifact.stat().st_size)
        raw_path = output_dir / f"{name}.json"
        raw_path.write_bytes(_canonical(raw))
        records.append({"name": name, "raw_json_sha256": _sha256(raw_path), **raw})
    aggregate = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    summary: dict[str, object] = {
        "format": _FORMAT, "benchmark": _BENCHMARK, "scope": _SCOPE,
        "evidence": "measured", "warmup": warmup, "iterations": iterations,
        "artifact_sha256": _sha256(artifact), "manifest_sha256": _sha256(manifest),
        "runner_sha256": _sha256(runner),
        "aggregate_sha256": hashlib.sha256(aggregate).hexdigest(),
        "artifact_bytes": artifact.stat().st_size,
        "manifest_identity": _identity_manifest(manifest_value), "records": records,
    }
    csv_path = output_dir / "summary.csv"
    _write_csv(csv_path, records)
    summary["summary_csv_sha256"] = _sha256(csv_path)
    (output_dir / "summary.json").write_bytes(_summary_bytes(summary))
    return summary


def verify_summary(summary_json: Path, summary_csv: Path, *, artifact: Path | None = None,
                   manifest: Path | None = None, runner: Path | None = None,
                   strict_official: bool = True) -> dict[str, object]:
    summary_json, summary_csv = Path(summary_json), Path(summary_csv)
    summary = _parse_json(summary_json.read_bytes(), "summary JSON")
    if summary_json.read_bytes() != _summary_bytes(summary) or set(summary) != _SUMMARY_FIELDS:
        raise RuntimeError("summary schema or encoding diverged")
    if (summary.get("format"), summary.get("benchmark"), summary.get("scope"),
        summary.get("evidence")) != (_FORMAT, _BENCHMARK, _SCOPE, "measured"):
        raise RuntimeError("summary identity diverged")
    warmup, iterations = summary.get("warmup"), summary.get("iterations")
    if not _integer(warmup) or not _integer(iterations, positive=True):
        raise RuntimeError("summary iteration identity diverged")
    if strict_official and (artifact is None or manifest is None or runner is None):
        raise RuntimeError("strict verification requires artifact, manifest, and runner")
    paths = {"artifact_sha256": artifact, "manifest_sha256": manifest,
             "runner_sha256": runner}
    for field, path in paths.items():
        if path is not None and summary.get(field) != _sha256(Path(path)):
            raise RuntimeError(f"{field} diverged")
    if strict_official and (warmup != 3 or iterations != 20):
        raise RuntimeError("official iteration gate diverged")
    manifest_identity = summary.get("manifest_identity")
    if not isinstance(manifest_identity, dict):
        raise RuntimeError("summary manifest identity diverged")
    _manifest_identity(manifest_identity)
    if manifest is not None:
        actual_manifest = _parse_json(Path(manifest).read_bytes(), "route manifest")
        if _identity_manifest(actual_manifest) != manifest_identity:
            raise RuntimeError("summary manifest identity diverged")
    records = summary.get("records")
    if not isinstance(records, list) or len(records) != len(CASES):
        raise RuntimeError("summary record count diverged")
    artifact_bytes = summary.get("artifact_bytes")
    if not _integer(artifact_bytes, positive=True):
        raise RuntimeError("summary artifact bytes diverged")
    if artifact is not None and Path(artifact).stat().st_size != artifact_bytes:
        raise RuntimeError("summary artifact bytes diverged")
    for record, (name, case, mode) in zip(records, CASES, strict=True):
        if not isinstance(record, dict) or record.get("name") != name or set(record) != set(_CSV_FIELDS):
            raise RuntimeError("summary case order or schema diverged")
        raw = {field: record[field] for field in _RAW_FIELDS}
        _validate_record(raw, name=name, case=case, mode=mode, warmup=warmup,
                         iterations=iterations, manifest=manifest_identity,
                         artifact_bytes=artifact_bytes)
        raw_path = summary_json.parent / f"{name}.json"
        if record["raw_json_sha256"] != _sha256(raw_path):
            raise RuntimeError(f"{name} raw JSON digest diverged")
        payload = _parse_json(raw_path.read_bytes(), f"{name} raw JSON")
        if raw_path.read_bytes() != _canonical(payload) or payload != raw:
            raise RuntimeError(f"{name} raw JSON payload diverged")
    aggregate = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    if summary["aggregate_sha256"] != hashlib.sha256(aggregate).hexdigest():
        raise RuntimeError("aggregate digest diverged")
    csv_bytes = summary_csv.read_bytes()
    if b"\r\n" in csv_bytes or summary["summary_csv_sha256"] != hashlib.sha256(csv_bytes).hexdigest():
        raise RuntimeError("summary CSV digest or newline diverged")
    with summary_csv.open(newline="", encoding="utf-8") as stream:
        reader_value = csv.DictReader(stream)
        rows, fields = list(reader_value), tuple(reader_value.fieldnames or ())
    expected = [{field: _scalar(record[field]) for field in _CSV_FIELDS} for record in records]
    if fields != _CSV_FIELDS or rows != expected:
        raise RuntimeError("summary CSV parity diverged")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--verify-only", action="store_true")
    arguments = parser.parse_args(argv)
    if not arguments.verify_only:
        run_ablation(arguments.artifact, arguments.manifest, arguments.runner,
                     output_dir=arguments.output_dir, warmup=arguments.warmup,
                     iterations=arguments.iterations)
    verify_summary(arguments.output_dir / "summary.json",
                   arguments.output_dir / "summary.csv", artifact=arguments.artifact,
                   manifest=arguments.manifest, runner=arguments.runner)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
