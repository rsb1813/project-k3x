# B-0030 공식 complete-layer CUDA 측정과 digest 증거 검증을 수행합니다.
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


CASES = (
    ("a-transient", "a", "transient"),
    ("ab-incremental-resident", "ab-incremental", "resident"),
    ("ab-full-resident", "ab-full", "resident"),
)

_FORMAT = "k3x-official-layer-cuda-v1"
_BENCHMARK = "B-0030"
_SCOPE = "official-kda-complete-layer-cuda"
_REVISION = "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"
_TOP_K = 16
_STATE_BYTES = 6_512_640
_KDA_WEIGHT_BYTES = 887_800_832
_KDA_F32_BYTES = 640_000
_KDA_BF16_BYTES = _KDA_WEIGHT_BYTES - _KDA_F32_BYTES
_MOE_COMMON_BYTES = 367_008_768
_EXPERT_BYTES = 17_547_264
_SOURCE_BYTES = 1_829_256_704
_RESIDENT_BYTES = 1_816_322_048
_MAXIMUM_ERROR = 2.0e-2

_RAW_FIELDS = (
    "artifact_kind", "repository", "resolved_revision", "case", "weight_mode",
    "token_semantics", "routing_semantics", "full_transformer_layer",
    "quality_measured", "k3x_root_sha256", "warmups", "iterations",
    "selected_union", "route_a", "route_b", "route_a_contributions",
    "route_b_contributions", "output_sha256", "state_sha256", "source_bytes",
    "k3x_bytes", "cold_latency_nanoseconds", "cold_kernel_nanoseconds",
    "cold_weight_h2d_bytes", "cold_bf16_weight_h2d_bytes",
    "cold_f32_weight_h2d_bytes", "cold_mxfp4_weight_h2d_bytes",
    "latency_nanoseconds_p05",
    "latency_nanoseconds_median", "latency_nanoseconds_p95",
    "kernel_nanoseconds", "orchestration_nanoseconds", "weight_h2d_bytes",
    "bf16_weight_h2d_bytes", "f32_weight_h2d_bytes",
    "mxfp4_weight_h2d_bytes",
    "activation_h2d_bytes", "device_to_host_bytes", "official_kda_calls",
    "official_kda_kernel_launches", "official_kda_state_h2d_bytes",
    "official_kda_state_d2h_bytes", "official_kda_output_d2h_bytes",
    "resident_weight_bytes", "peak_resident_weight_bytes", "weight_cache_hits",
    "weight_cache_misses", "weight_cache_bypasses", "device_allocation_count",
    "stream_synchronization_count", "peak_vram_bytes",
    "process_peak_rss_bytes", "reader_read_calls", "reader_requested_bytes",
    "reader_completed_bytes", "reader_storage_submitted_bytes",
    "reader_storage_completed_bytes",
    "maximum_absolute_error", "all_finite",
)
_CSV_FIELDS = ("name", "raw_json_sha256", *_RAW_FIELDS)
_SUMMARY_FIELDS = {
    "format", "benchmark", "scope", "evidence", "warmups", "iterations",
    "artifact_sha256", "manifest_sha256", "runner_sha256", "aggregate_sha256",
    "artifact_bytes", "manifest_identity", "records", "summary_csv_sha256",
}
_FORBIDDEN = {
    "decode_tok_s", "prefill_tok_s", "ttft", "gpu_utilization",
    "gpu_memory_bandwidth", "nvme_gb_per_token", "nvme_read_gb_per_token",
    "physical_nvme_bytes", "physical_h2d_bytes", "quality", "quality_score",
    "quality_benchmark_results",
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


def _hex(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _manifest_identity(manifest: Mapping[str, object]) -> dict[str, object]:
    try:
        artifact = manifest["artifact"]
        steps = manifest["steps"]
        selected = manifest["selected_experts"]
        assert isinstance(artifact, dict) and isinstance(steps, list)
        assert isinstance(selected, list) and len(steps) == 2
        first, second = steps
        assert isinstance(first, dict) and isinstance(second, dict)
        result = {
            "root": artifact["k3x_root_sha256"],
            "selected": selected,
            "route_a": first["expert_ids"],
            "route_b": second["expert_ids"],
            "contributions_a": first["contributions"],
            "contributions_b": second["contributions"],
        }
    except (AssertionError, KeyError, TypeError) as error:
        raise RuntimeError("manifest identity diverged") from error
    if (
        manifest.get("repository") != "moonshotai/Kimi-K3"
        or manifest.get("resolved_revision") != _REVISION
        or not _hex(result["root"])
    ):
        raise RuntimeError("manifest identity diverged")
    for field in ("selected", "route_a", "route_b", "contributions_a",
                  "contributions_b"):
        if not isinstance(result[field], list):
            raise RuntimeError("manifest route identity diverged")
    route_a = result["route_a"]
    route_b = result["route_b"]
    selected = result["selected"]
    contribution_a = result["contributions_a"]
    contribution_b = result["contributions_b"]
    assert isinstance(route_a, list) and isinstance(route_b, list)
    assert isinstance(selected, list) and isinstance(contribution_a, list)
    assert isinstance(contribution_b, list)
    if (
        len(route_a) != _TOP_K
        or len(route_b) != _TOP_K
        or len(set(route_a)) != _TOP_K
        or len(set(route_b)) != _TOP_K
        or any(type(value) is not int or not 0 <= value < 896
               for value in route_a + route_b)
        or len(contribution_a) != _TOP_K
        or len(contribution_b) != _TOP_K
        or any(not _finite(value) or value <= 0
               for value in contribution_a + contribution_b)
        or abs(sum(contribution_a) - 1.0) > 1.0e-5
        or abs(sum(contribution_b) - 1.0) > 1.0e-5
        or selected != list(dict.fromkeys(route_a + route_b))
    ):
        raise RuntimeError("manifest route identity diverged")
    return result


def _identity_manifest(manifest: Mapping[str, object]) -> dict[str, object]:
    identity = _manifest_identity(manifest)
    return {
        "repository": "moonshotai/Kimi-K3",
        "resolved_revision": _REVISION,
        "artifact": {"k3x_root_sha256": identity["root"]},
        "selected_experts": identity["selected"],
        "steps": [
            {"expert_ids": identity["route_a"],
             "contributions": identity["contributions_a"]},
            {"expert_ids": identity["route_b"],
             "contributions": identity["contributions_b"]},
        ],
    }


def _traffic(case: str, mode: str, iterations: int,
             selected_count: int) -> dict[str, int]:
    tokens = 1 if case == "a" else 2
    kda_calls = 2 if case == "ab-incremental" else 1
    state = _STATE_BYTES * kda_calls * iterations
    kda_output = 28_672 * tokens * iterations
    activation = (
        _STATE_BYTES * kda_calls + 28_672 * tokens + 58_176 * tokens
    ) * iterations
    measured_weight = 0 if mode == "resident" else (
        _KDA_WEIGHT_BYTES + _MOE_COMMON_BYTES + _TOP_K * _EXPERT_BYTES
    ) * iterations
    cold_experts = _TOP_K if case == "a" else selected_count
    cold_weight = (
        _KDA_WEIGHT_BYTES + _MOE_COMMON_BYTES + cold_experts * _EXPERT_BYTES
    )
    measured_bf16 = 0 if mode == "resident" else (
        _KDA_BF16_BYTES + _MOE_COMMON_BYTES
    ) * iterations
    measured_f32 = 0 if mode == "resident" else _KDA_F32_BYTES * iterations
    measured_mxfp4 = 0 if mode == "resident" else (
        _TOP_K * _EXPERT_BYTES * iterations
    )
    return {
        "cold_weight_h2d_bytes": cold_weight,
        "cold_bf16_weight_h2d_bytes": _KDA_BF16_BYTES + _MOE_COMMON_BYTES,
        "cold_f32_weight_h2d_bytes": _KDA_F32_BYTES,
        "cold_mxfp4_weight_h2d_bytes": cold_experts * _EXPERT_BYTES,
        "weight_h2d_bytes": measured_weight,
        "bf16_weight_h2d_bytes": measured_bf16,
        "f32_weight_h2d_bytes": measured_f32,
        "mxfp4_weight_h2d_bytes": measured_mxfp4,
        "activation_h2d_bytes": activation,
        "device_to_host_bytes": state + 2 * kda_output,
        "official_kda_calls": kda_calls * iterations,
        "official_kda_kernel_launches": (
            24 if case == "ab-full" else 16 * kda_calls
        ) * iterations,
        "official_kda_state_h2d_bytes": state,
        "official_kda_state_d2h_bytes": state,
        "official_kda_output_d2h_bytes": kda_output,
        "stream_synchronization_count": (kda_calls + tokens) * iterations,
    }


def _validate_record(record: Mapping[str, object], *, name: str, case: str,
                     mode: str, warmups: int, iterations: int,
                     manifest: Mapping[str, object], artifact_bytes: int) -> None:
    forbidden = _FORBIDDEN.intersection(record)
    if forbidden:
        raise RuntimeError(f"{name} contains forbidden metric {min(forbidden)}")
    if set(record) != set(_RAW_FIELDS):
        raise RuntimeError(f"{name} schema diverged")
    identity = _manifest_identity(manifest)
    expected = {
        "artifact_kind": "official_kimi_k3_kda_layer",
        "repository": "moonshotai/Kimi-K3",
        "resolved_revision": _REVISION,
        "case": case,
        "weight_mode": mode,
        "token_semantics": False,
        "routing_semantics": True,
        "full_transformer_layer": True,
        "quality_measured": False,
        "k3x_root_sha256": identity["root"],
        "warmups": warmups,
        "iterations": iterations,
        "selected_union": identity["selected"],
        "route_a": identity["route_a"],
        "route_b": identity["route_b"],
        "source_bytes": _SOURCE_BYTES,
        "k3x_bytes": artifact_bytes,
        "all_finite": True,
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
            or len(observed) != len(expected_values)
            or any(
            not _finite(value) or abs(value - expected_value) > 2.0e-6
            for value, expected_value in zip(observed, expected_values, strict=True)
            )
        ):
            raise RuntimeError(f"{name} route contribution diverged")
    if not _hex(record.get("output_sha256")) or not _hex(record.get("state_sha256")):
        raise RuntimeError(f"{name} output/state identity diverged")
    for field in (
        "cold_latency_nanoseconds", "cold_kernel_nanoseconds",
        "latency_nanoseconds_p05", "latency_nanoseconds_median",
        "latency_nanoseconds_p95", "kernel_nanoseconds", "peak_vram_bytes",
        "process_peak_rss_bytes",
    ):
        if not _integer(record.get(field), positive=True):
            raise RuntimeError(f"{name} positive field {field} diverged")
    for field in set(_RAW_FIELDS) - set(expected) - {
        "maximum_absolute_error", "output_sha256", "state_sha256",
        "route_a_contributions", "route_b_contributions",
    }:
        if field.endswith("_bytes") or field.endswith("_count") or field in {
            "official_kda_calls", "official_kda_kernel_launches",
            "weight_cache_hits", "weight_cache_misses", "weight_cache_bypasses",
            "reader_read_calls",
            "orchestration_nanoseconds", "stream_synchronization_count",
        }:
            if not _integer(record.get(field)):
                raise RuntimeError(f"{name} counter field {field} diverged")
    p05 = record["latency_nanoseconds_p05"]
    median = record["latency_nanoseconds_median"]
    p95 = record["latency_nanoseconds_p95"]
    if not p05 <= median <= p95:
        raise RuntimeError(f"{name} latency distribution diverged")
    error = record["maximum_absolute_error"]
    if not _finite(error) or not 0 <= error <= _MAXIMUM_ERROR:
        raise RuntimeError(f"{name} numerical divergence")
    traffic = _traffic(case, mode, iterations, len(identity["selected"]))
    if any(record.get(field) != value for field, value in traffic.items()):
        raise RuntimeError(f"{name} traffic diverged")
    resident = _RESIDENT_BYTES if mode == "resident" else 0
    expected_resident = {
        "resident_weight_bytes": resident,
        "peak_resident_weight_bytes": resident,
        "weight_cache_misses": 0,
        "weight_cache_bypasses": 0,
    }
    if any(record.get(field) != value for field, value in expected_resident.items()):
        raise RuntimeError(f"{name} resident traffic diverged")
    if mode == "resident":
        hits_per_iteration = 136 if case == "ab-incremental" else 122
        if record.get("weight_cache_hits") != hits_per_iteration * iterations or record.get(
            "device_allocation_count"
        ) != 0:
            raise RuntimeError(f"{name} resident traffic diverged")
    elif record.get("weight_cache_hits") != 0:
        raise RuntimeError(f"{name} transient traffic diverged")
    if (
        record.get("reader_requested_bytes") != record.get("reader_completed_bytes")
        or record.get("reader_storage_submitted_bytes") !=
            record.get("reader_storage_completed_bytes")
        or not _integer(record.get("reader_read_calls"), positive=True)
        or not _integer(record.get("reader_requested_bytes"), positive=True)
        or not _integer(record.get("reader_storage_submitted_bytes"), positive=True)
    ):
        raise RuntimeError(f"{name} Reader traffic diverged")


def _run_case(artifact: Path, manifest: Path, runner: Path, *, case: str,
              mode: str, warmups: int, iterations: int) -> dict[str, object]:
    command = [
        str(runner), "--artifact", str(artifact), "--manifest", str(manifest),
        "--case", case, "--weight-mode", mode, "--warmups", str(warmups),
        "--iterations", str(iterations),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stderr.strip() or "official layer benchmark failed")
    return _parse_json(result.stdout, f"{case}-{mode} output")


def _scalar(value: object) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (list, dict)):
        return json.dumps(value, separators=(",", ":"))
    return str(value)


def _write_file(path: Path, payload: bytes) -> None:
    with path.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())


def _write_csv(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=_CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            {field: _scalar(record[field]) for field in _CSV_FIELDS}
            for record in records
        )
        stream.flush()
        os.fsync(stream.fileno())


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def run_ablation(artifact: Path, manifest: Path, runner: Path, *, output_dir: Path,
                 warmups: int, iterations: int) -> dict[str, object]:
    if type(warmups) is not int or warmups < 0:
        raise ValueError("warmups must be non-negative")
    if type(iterations) is not int or iterations <= 0:
        raise ValueError("iterations must be positive")
    artifact, manifest, runner = (
        Path(value).resolve() for value in (artifact, manifest, runner)
    )
    for path in (artifact, manifest, runner):
        if not path.is_file():
            raise FileNotFoundError(path)
    manifest_value = _parse_json(manifest.read_bytes(), "route manifest")
    _manifest_identity(manifest_value)
    output_dir = Path(output_dir).resolve()
    partial = output_dir.with_name(f".{output_dir.name}.partial")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    try:
        records: list[dict[str, object]] = []
        for name, case, mode in CASES:
            raw = _run_case(
                artifact, manifest, runner, case=case, mode=mode,
                warmups=warmups, iterations=iterations,
            )
            _validate_record(
                raw, name=name, case=case, mode=mode, warmups=warmups,
                iterations=iterations, manifest=manifest_value,
                artifact_bytes=artifact.stat().st_size,
            )
            raw_path = partial / f"{name}.json"
            _write_file(raw_path, _canonical(raw))
            records.append({"name": name, "raw_json_sha256": _sha256(raw_path), **raw})
        if (
            records[1]["output_sha256"] != records[2]["output_sha256"]
            or records[1]["state_sha256"] != records[2]["state_sha256"]
        ):
            raise RuntimeError("full/incremental parity diverged")
        aggregate = json.dumps(
            records, sort_keys=True, separators=(",", ":")
        ).encode()
        summary: dict[str, object] = {
            "format": _FORMAT,
            "benchmark": _BENCHMARK,
            "scope": _SCOPE,
            "evidence": "measured",
            "warmups": warmups,
            "iterations": iterations,
            "artifact_sha256": _sha256(artifact),
            "manifest_sha256": _sha256(manifest),
            "runner_sha256": _sha256(runner),
            "aggregate_sha256": hashlib.sha256(aggregate).hexdigest(),
            "artifact_bytes": artifact.stat().st_size,
            "manifest_identity": _identity_manifest(manifest_value),
            "records": records,
        }
        csv_path = partial / "summary.csv"
        _write_csv(csv_path, records)
        summary["summary_csv_sha256"] = _sha256(csv_path)
        _write_file(partial / "summary.json", _summary_bytes(summary))
        _fsync_directory(partial)
        os.replace(partial, output_dir)
        _fsync_directory(output_dir.parent)
        return summary
    except BaseException:
        shutil.rmtree(partial, ignore_errors=True)
        raise


def verify_summary(summary_json: Path, summary_csv: Path, *,
                   artifact: Path | None = None, manifest: Path | None = None,
                   runner: Path | None = None,
                   strict_official: bool = True) -> dict[str, object]:
    summary_json, summary_csv = Path(summary_json), Path(summary_csv)
    summary = _parse_json(summary_json.read_bytes(), "summary JSON")
    if summary_json.read_bytes() != _summary_bytes(summary) or set(summary) != _SUMMARY_FIELDS:
        raise RuntimeError("summary schema or encoding diverged")
    if (
        summary.get("format"), summary.get("benchmark"), summary.get("scope"),
        summary.get("evidence"),
    ) != (_FORMAT, _BENCHMARK, _SCOPE, "measured"):
        raise RuntimeError("summary identity diverged")
    warmups, iterations = summary.get("warmups"), summary.get("iterations")
    if not _integer(warmups) or not _integer(iterations, positive=True):
        raise RuntimeError("summary iteration identity diverged")
    if strict_official and (artifact is None or manifest is None or runner is None):
        raise RuntimeError("strict verification requires artifact, manifest, and runner")
    if strict_official and (warmups != 3 or iterations != 20):
        raise RuntimeError("official iteration gate diverged")
    for field, path in {
        "artifact_sha256": artifact,
        "manifest_sha256": manifest,
        "runner_sha256": runner,
    }.items():
        if path is not None and summary.get(field) != _sha256(Path(path)):
            raise RuntimeError(f"{field} diverged")
    manifest_identity = summary.get("manifest_identity")
    if not isinstance(manifest_identity, dict):
        raise RuntimeError("summary manifest identity diverged")
    _manifest_identity(manifest_identity)
    if manifest is not None:
        actual = _parse_json(Path(manifest).read_bytes(), "route manifest")
        if _identity_manifest(actual) != manifest_identity:
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
        if not isinstance(record, dict) or record.get("name") != name or set(record) != set(
            _CSV_FIELDS
        ):
            raise RuntimeError("summary case order or schema diverged")
        raw = {field: record[field] for field in _RAW_FIELDS}
        _validate_record(
            raw, name=name, case=case, mode=mode, warmups=warmups,
            iterations=iterations, manifest=manifest_identity,
            artifact_bytes=artifact_bytes,
        )
        raw_path = summary_json.parent / f"{name}.json"
        if record["raw_json_sha256"] != _sha256(raw_path):
            raise RuntimeError(f"{name} raw JSON digest diverged")
        payload = _parse_json(raw_path.read_bytes(), f"{name} raw JSON")
        if raw_path.read_bytes() != _canonical(payload) or payload != raw:
            raise RuntimeError(f"{name} raw JSON payload diverged")
    if (
        records[1]["output_sha256"] != records[2]["output_sha256"]
        or records[1]["state_sha256"] != records[2]["state_sha256"]
    ):
        raise RuntimeError("full/incremental parity diverged")
    aggregate = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    if summary.get("aggregate_sha256") != hashlib.sha256(aggregate).hexdigest():
        raise RuntimeError("aggregate digest diverged")
    csv_bytes = summary_csv.read_bytes()
    if b"\r\n" in csv_bytes or summary.get("summary_csv_sha256") != hashlib.sha256(
        csv_bytes
    ).hexdigest():
        raise RuntimeError("summary CSV digest or newline diverged")
    with summary_csv.open(newline="", encoding="utf-8") as stream:
        reader_value = csv.DictReader(stream)
        rows, fields = list(reader_value), tuple(reader_value.fieldnames or ())
    expected = [
        {field: _scalar(record[field]) for field in _CSV_FIELDS}
        for record in records
    ]
    if fields != _CSV_FIELDS or rows != expected:
        raise RuntimeError("summary CSV parity diverged")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--verify-existing", action="store_true")
    arguments = parser.parse_args(argv)
    if not arguments.verify_existing:
        run_ablation(
            arguments.artifact, arguments.manifest, arguments.runner,
            output_dir=arguments.output_dir, warmups=arguments.warmups,
            iterations=arguments.iterations,
        )
    verify_summary(
        arguments.output_dir / "summary.json",
        arguments.output_dir / "summary.csv",
        artifact=arguments.artifact, manifest=arguments.manifest,
        runner=arguments.runner,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
