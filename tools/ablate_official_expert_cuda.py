# B-0028 공식 Kimi K3 expert CUDA 측정과 digest 기반 증거 검증을 수행합니다.
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path
from typing import Mapping


CASES = (("transient", "transient"), ("resident", "resident"))

_FORMAT = "k3x-official-expert-cuda-v1"
_BENCHMARK = "B-0028"
_SCOPE = "official-single-expert-cuda"
_ARTIFACT_SHA256 = (
    "e08293cd854ed11913bd8f1bc3a51d1eb577202fd5fd9b5b7e3c96ef1bccecc7"
)
_B0027_SUMMARY_SHA256 = (
    "57ebd9d85ed3ae55a4e2ab01f023bc451faf02cd7b6e69f478d11e3ea73e982a"
)
_REVISION = "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"
_K3X_ROOT_SHA256 = (
    "d585d283325e13e1316a0194c2d6274dd89ef75a28b96b02f02733290b7658be"
)
_ORDERED_SHA256 = (
    "4e23bd960dfb5e8b10def10e12a94bac1119500f72918698986bd332d56d33ff"
)
_PAYLOAD_BYTES = 17_547_264
_ACTIVATION_BYTES = 14_336
_MAXIMUM_ERROR = 1.0e-6

_RAW_FIELDS = (
    "artifact_kind",
    "repository",
    "resolved_revision",
    "token_semantics",
    "routing_semantics",
    "full_moe_layer",
    "layer_id",
    "expert_id",
    "weight_mode",
    "k3x_root_sha256",
    "ordered_sha256",
    "expert_payload_bytes",
    "input_elements",
    "output_elements",
    "warmup",
    "iterations",
    "cpu_oracle_nanoseconds",
    "cold_latency_nanoseconds",
    "cold_kernel_nanoseconds",
    "cold_weight_h2d_bytes",
    "cold_activation_h2d_bytes",
    "cold_device_to_host_bytes",
    "latency_nanoseconds_median",
    "latency_nanoseconds_p05",
    "latency_nanoseconds_p95",
    "kernel_nanoseconds",
    "weight_h2d_bytes",
    "activation_h2d_bytes",
    "device_to_host_bytes",
    "device_allocation_count",
    "stream_synchronization_count",
    "weight_cache_hits",
    "weight_cache_misses",
    "weight_cache_bypasses",
    "resident_weight_bytes",
    "peak_resident_weight_bytes",
    "peak_vram_bytes",
    "maximum_absolute_error",
    "all_finite",
)
_CSV_FIELDS = ("name", "raw_json_sha256", *_RAW_FIELDS)
_SUMMARY_FIELDS = {
    "format",
    "benchmark",
    "scope",
    "evidence",
    "warmup",
    "iterations",
    "artifact_sha256",
    "source_b0027_summary_sha256",
    "runner_sha256",
    "aggregate_sha256",
    "records",
    "summary_csv_sha256",
}
_FORBIDDEN_FIELDS = {
    "decode_tok_s",
    "prefill_tok_s",
    "ttft",
    "gpu_utilization",
    "gpu_memory_bandwidth",
    "nvme_gb_per_token",
    "nvme_read_gb_per_token",
    "physical_nvme_bytes",
    "quality",
    "quality_benchmark_results",
}


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_bytes(value: object) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n"
    ).encode("utf-8")


def _summary_bytes(value: object) -> bytes:
    return (json.dumps(value, indent=2, sort_keys=True) + "\n").encode("utf-8")


def _write_bytes(path: Path, payload: bytes) -> None:
    path.write_bytes(payload)


def _parse_json(payload: str | bytes, label: str) -> dict[str, object]:
    def reject_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise RuntimeError(f"{label} contains duplicate keys")
            result[key] = value
        return result

    try:
        value = json.loads(
            payload,
            object_pairs_hook=reject_pairs,
            parse_constant=lambda constant: (_ for _ in ()).throw(
                RuntimeError(f"{label} contains non-finite {constant}")
            ),
        )
    except RuntimeError:
        raise
    except (UnicodeError, json.JSONDecodeError) as error:
        raise RuntimeError(f"{label} is not canonical JSON") from error
    if not isinstance(value, dict):
        raise RuntimeError(f"{label} is not a JSON object")
    return value


def _positive_integer(value: object) -> bool:
    return type(value) is int and value > 0


def _nonnegative_integer(value: object) -> bool:
    return type(value) is int and value >= 0


def _finite_number(value: object) -> bool:
    return type(value) in (int, float) and math.isfinite(value)


def _validate_record(
    record: Mapping[str, object],
    *,
    name: str,
    mode: str,
    warmup: int,
    iterations: int,
) -> None:
    forbidden = _FORBIDDEN_FIELDS.intersection(record)
    if forbidden:
        raise RuntimeError(f"{name} contains forbidden metric {min(forbidden)}")
    if set(record) != set(_RAW_FIELDS):
        raise RuntimeError(f"{name} schema diverged")

    identity = {
        "artifact_kind": "official_kimi_k3_expert",
        "repository": "moonshotai/Kimi-K3",
        "resolved_revision": _REVISION,
        "token_semantics": False,
        "routing_semantics": False,
        "full_moe_layer": False,
        "layer_id": 1,
        "expert_id": 0,
        "weight_mode": mode,
        "k3x_root_sha256": _K3X_ROOT_SHA256,
        "ordered_sha256": _ORDERED_SHA256,
        "expert_payload_bytes": _PAYLOAD_BYTES,
        "input_elements": 3_584,
        "output_elements": 3_584,
        "warmup": warmup,
        "iterations": iterations,
        "all_finite": True,
    }
    for field, expected in identity.items():
        if record.get(field) != expected or type(record.get(field)) is not type(expected):
            raise RuntimeError(f"{name} identity field {field} diverged")

    for field in (
        "cpu_oracle_nanoseconds",
        "cold_latency_nanoseconds",
        "cold_kernel_nanoseconds",
        "latency_nanoseconds_median",
        "latency_nanoseconds_p05",
        "latency_nanoseconds_p95",
        "kernel_nanoseconds",
        "peak_vram_bytes",
    ):
        if not _positive_integer(record.get(field)):
            raise RuntimeError(f"{name} timing or memory field {field} diverged")
    for field in (
        "cold_weight_h2d_bytes",
        "cold_activation_h2d_bytes",
        "cold_device_to_host_bytes",
        "weight_h2d_bytes",
        "activation_h2d_bytes",
        "device_to_host_bytes",
        "device_allocation_count",
        "stream_synchronization_count",
        "weight_cache_hits",
        "weight_cache_misses",
        "weight_cache_bypasses",
        "resident_weight_bytes",
        "peak_resident_weight_bytes",
    ):
        if not _nonnegative_integer(record.get(field)):
            raise RuntimeError(f"{name} counter field {field} diverged")

    p05 = record["latency_nanoseconds_p05"]
    median = record["latency_nanoseconds_median"]
    p95 = record["latency_nanoseconds_p95"]
    assert isinstance(p05, int) and isinstance(median, int) and isinstance(p95, int)
    if not p05 <= median <= p95:
        raise RuntimeError(f"{name} latency distribution diverged")
    error = record["maximum_absolute_error"]
    if not _finite_number(error) or not 0 <= error <= _MAXIMUM_ERROR:
        raise RuntimeError(f"{name} numerical divergence")

    common_traffic = {
        "cold_weight_h2d_bytes": _PAYLOAD_BYTES,
        "cold_activation_h2d_bytes": _ACTIVATION_BYTES,
        "cold_device_to_host_bytes": _ACTIVATION_BYTES,
        "activation_h2d_bytes": _ACTIVATION_BYTES * iterations,
        "device_to_host_bytes": _ACTIVATION_BYTES * iterations,
        "device_allocation_count": 0,
        "stream_synchronization_count": iterations,
        "weight_cache_misses": 0,
        "weight_cache_bypasses": 0,
    }
    if any(record.get(field) != expected for field, expected in common_traffic.items()):
        raise RuntimeError(f"{name} common traffic diverged")

    if mode == "transient":
        expected = {
            "weight_h2d_bytes": _PAYLOAD_BYTES * iterations,
            "weight_cache_hits": 0,
            "resident_weight_bytes": 0,
            "peak_resident_weight_bytes": 0,
        }
        message = "transient traffic diverged"
    elif mode == "resident":
        expected = {
            "weight_h2d_bytes": 0,
            "weight_cache_hits": 3 * iterations,
            "resident_weight_bytes": _PAYLOAD_BYTES,
            "peak_resident_weight_bytes": _PAYLOAD_BYTES,
        }
        message = "resident traffic diverged"
    else:
        raise RuntimeError(f"{name} mode diverged")
    if any(record.get(field) != value for field, value in expected.items()):
        raise RuntimeError(f"{name} {message}")


def _run_case(
    artifact: Path,
    runner: Path,
    *,
    mode: str,
    warmup: int,
    iterations: int,
) -> dict[str, object]:
    command = [
        str(runner),
        "--model",
        str(artifact),
        "--weight-mode",
        mode,
        "--warmup",
        str(warmup),
        "--iterations",
        str(iterations),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "official CUDA benchmark failed")
    return _parse_json(result.stdout, f"{mode} runner output")


def _scalar(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def _csv_row(record: Mapping[str, object]) -> dict[str, str]:
    return {field: _scalar(record[field]) for field in _CSV_FIELDS}


def _write_csv(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=_CSV_FIELDS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(_csv_row(record) for record in records)


def run_ablation(
    artifact: Path,
    runner: Path,
    *,
    output_dir: Path,
    warmup: int,
    iterations: int,
) -> dict[str, object]:
    if type(warmup) is not int or warmup < 0:
        raise ValueError("warmup must be non-negative")
    if type(iterations) is not int or iterations <= 0:
        raise ValueError("iterations must be positive")
    artifact = Path(artifact).resolve()
    runner = Path(runner).resolve()
    output_dir = Path(output_dir)
    if not artifact.is_file():
        raise FileNotFoundError(artifact)
    if not runner.is_file():
        raise FileNotFoundError(runner)
    output_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict[str, object]] = []
    for name, mode in CASES:
        raw = _run_case(
            artifact,
            runner,
            mode=mode,
            warmup=warmup,
            iterations=iterations,
        )
        _validate_record(
            raw,
            name=name,
            mode=mode,
            warmup=warmup,
            iterations=iterations,
        )
        raw_path = output_dir / f"{name}.json"
        _write_bytes(raw_path, _canonical_bytes(raw))
        records.append(
            {
                "name": name,
                "raw_json_sha256": _sha256(raw_path),
                **raw,
            }
        )

    aggregate = json.dumps(
        records, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    summary: dict[str, object] = {
        "format": _FORMAT,
        "benchmark": _BENCHMARK,
        "scope": _SCOPE,
        "evidence": "measured",
        "warmup": warmup,
        "iterations": iterations,
        "artifact_sha256": _sha256(artifact),
        "source_b0027_summary_sha256": _B0027_SUMMARY_SHA256,
        "runner_sha256": _sha256(runner),
        "aggregate_sha256": hashlib.sha256(aggregate).hexdigest(),
        "records": records,
    }
    summary_csv = output_dir / "summary.csv"
    _write_csv(summary_csv, records)
    summary["summary_csv_sha256"] = _sha256(summary_csv)
    _write_bytes(output_dir / "summary.json", _summary_bytes(summary))
    return summary


def verify_summary(
    summary_json: Path,
    summary_csv: Path,
    *,
    artifact: Path | None = None,
    runner: Path | None = None,
    strict_official: bool = True,
) -> dict[str, object]:
    summary_json = Path(summary_json)
    summary_csv = Path(summary_csv)
    summary = _parse_json(summary_json.read_bytes(), "summary JSON")
    if summary_json.read_bytes() != _summary_bytes(summary):
        raise RuntimeError("summary JSON encoding diverged")
    if set(summary) != _SUMMARY_FIELDS:
        raise RuntimeError("summary schema diverged")
    if (
        summary.get("format") != _FORMAT
        or summary.get("benchmark") != _BENCHMARK
        or summary.get("scope") != _SCOPE
        or summary.get("evidence") != "measured"
        or summary.get("source_b0027_summary_sha256") != _B0027_SUMMARY_SHA256
    ):
        raise RuntimeError("summary identity diverged")
    warmup = summary.get("warmup")
    iterations = summary.get("iterations")
    if not _nonnegative_integer(warmup) or not _positive_integer(iterations):
        raise RuntimeError("summary iteration identity diverged")
    assert isinstance(warmup, int) and isinstance(iterations, int)

    if strict_official and (artifact is None or runner is None):
        raise RuntimeError("strict verification requires artifact and runner")
    if artifact is not None and summary.get("artifact_sha256") != _sha256(
        Path(artifact)
    ):
        raise RuntimeError("artifact digest diverged")
    if runner is not None and summary.get("runner_sha256") != _sha256(Path(runner)):
        raise RuntimeError("runner digest diverged")
    if strict_official and (
        summary.get("artifact_sha256") != _ARTIFACT_SHA256
        or warmup != 3
        or iterations != 20
    ):
        raise RuntimeError("official artifact identity or iteration gate diverged")

    records = summary.get("records")
    if not isinstance(records, list) or len(records) != len(CASES):
        raise RuntimeError("summary record count diverged")
    for record, (name, mode) in zip(records, CASES, strict=True):
        if not isinstance(record, dict) or record.get("name") != name:
            raise RuntimeError("summary case order diverged")
        if set(record) != set(_CSV_FIELDS):
            raise RuntimeError(f"{name} summary record schema diverged")
        raw = {
            field: record[field]
            for field in _RAW_FIELDS
        }
        _validate_record(
            raw,
            name=name,
            mode=mode,
            warmup=warmup,
            iterations=iterations,
        )
        raw_path = summary_json.parent / f"{name}.json"
        if record.get("raw_json_sha256") != _sha256(raw_path):
            raise RuntimeError(f"{name} raw JSON digest diverged")
        raw_payload = _parse_json(raw_path.read_bytes(), f"{name} raw JSON")
        if raw_path.read_bytes() != _canonical_bytes(raw_payload):
            raise RuntimeError(f"{name} raw JSON encoding diverged")
        if raw_payload != raw:
            raise RuntimeError(f"{name} raw JSON payload diverged")

    aggregate = json.dumps(
        records, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    if summary.get("aggregate_sha256") != hashlib.sha256(aggregate).hexdigest():
        raise RuntimeError("aggregate digest diverged")
    csv_bytes = summary_csv.read_bytes()
    if b"\r\n" in csv_bytes:
        raise RuntimeError("summary CSV is not LF-only")
    if summary.get("summary_csv_sha256") != hashlib.sha256(csv_bytes).hexdigest():
        raise RuntimeError("summary CSV digest diverged")
    with summary_csv.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        fieldnames = tuple(reader.fieldnames or ())
    if fieldnames != _CSV_FIELDS or len(rows) != len(records):
        raise RuntimeError("summary CSV schema diverged")
    expected_rows = [_csv_row(record) for record in records]
    if rows != expected_rows:
        raise RuntimeError("summary CSV parity diverged")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--verify-only", action="store_true")
    arguments = parser.parse_args(argv)
    summary_json = arguments.output_dir / "summary.json"
    summary_csv = arguments.output_dir / "summary.csv"
    if arguments.verify_only:
        verify_summary(
            summary_json,
            summary_csv,
            artifact=arguments.artifact,
            runner=arguments.runner,
        )
        return 0
    run_ablation(
        arguments.artifact,
        arguments.runner,
        output_dir=arguments.output_dir,
        warmup=arguments.warmup,
        iterations=arguments.iterations,
    )
    verify_summary(
        summary_json,
        summary_csv,
        artifact=arguments.artifact,
        runner=arguments.runner,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
