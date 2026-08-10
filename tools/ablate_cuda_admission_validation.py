# CUDA immutable-weight admission validation의 18행 ablation 증거를 생성합니다.
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path


CASES = tuple(
    (
        f"{kind}-{experts}-{validation}-profiler-{profiler_name}",
        "ffn-block" if kind == "split" else "moe-layer",
        experts,
        validation,
        profiler_name == "on",
    )
    for kind, validation in (
        ("split", "per-call"),
        ("layer", "per-call"),
        ("layer", "admission"),
    )
    for experts in (1, 4, 16)
    for profiler_name in ("off", "on")
)

_IMMUTABLE_BYTES = 469_776_384


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_csv(path: Path, payload: dict) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=list(payload), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerow(payload)


def _run_case(
    artifact: Path,
    runner: Path,
    *,
    boundary: str,
    experts: int,
    validation: str,
    profiler: bool,
    warmup: int,
    iterations: int,
) -> dict:
    result = subprocess.run(
        [
            str(runner), "--model", str(artifact),
            "--boundary", boundary, "--experts", str(experts),
            "--validation", validation,
            "--profiler", "on" if profiler else "off",
            "--warmup", str(warmup), "--iterations", str(iterations),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "CUDA benchmark failed")
    payload = json.loads(result.stdout)
    if not isinstance(payload, dict):
        raise RuntimeError("CUDA benchmark did not emit an object")
    return payload


def _validate_record(
    record: dict,
    *,
    name: str,
    boundary: str,
    experts: int,
    validation: str,
    profiler: bool,
    warmup: int,
    iterations: int,
) -> None:
    expected = {
        "artifact_kind": "released_dimension_moe_layer",
        "routing_semantics": False,
        "boundary": boundary,
        "experts": experts,
        "hidden_width": 7168,
        "latent_width": 3584,
        "expert_intermediate_width": 3072,
        "expert_payload_bytes": 17_547_264,
        "resident_capacity_bytes": 1 << 30,
        "warmup": warmup,
        "iterations": iterations,
        "validation": validation,
        "profiler": profiler,
    }
    for field, value in expected.items():
        if record.get(field) != value:
            raise RuntimeError(f"{name} identity field {field} diverged")
    if record.get("maximum_absolute_error", float("inf")) > 1.0e-5:
        raise RuntimeError(f"{name} numerical divergence")
    if record.get("latency_nanoseconds_median", 0) <= 0:
        raise RuntimeError(f"{name} latency is not positive")
    if profiler != (record.get("kernel_nanoseconds") is not None):
        raise RuntimeError(f"{name} profiler attribution diverged")
    for field in (
        "weight_h2d_bytes", "weight_cache_bypasses",
        "resident_grid_fallbacks", "resident_moe_layer_fallbacks",
    ):
        if record.get(field) != 0:
            raise RuntimeError(f"{name} {field} is nonzero")

    layer = boundary == "moe-layer"
    cold_scans = 6 if layer else 0
    cold_bytes = _IMMUTABLE_BYTES if layer else 0
    if (
        record.get("cold_immutable_validation_scans") != cold_scans
        or record.get("cold_immutable_validation_bytes") != cold_bytes
    ):
        raise RuntimeError(f"{name} cold validation accounting diverged")
    if not layer:
        expected_scans = expected_hits = expected_bytes = 0
    elif validation == "per-call":
        expected_scans = iterations * 6
        expected_hits = 0
        expected_bytes = iterations * _IMMUTABLE_BYTES
    else:
        expected_scans = 0
        expected_hits = iterations * 6
        expected_bytes = 0
    if (
        record.get("immutable_validation_scans") != expected_scans
        or record.get("immutable_validation_hits") != expected_hits
        or record.get("immutable_validation_bytes") != expected_bytes
    ):
        raise RuntimeError(f"{name} warm validation accounting diverged")


def run_ablation(
    artifact: Path,
    runner: Path,
    *,
    output_dir: Path,
    warmup: int,
    iterations: int,
) -> dict:
    if warmup < 0 or iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations positive")
    artifact = Path(artifact).resolve()
    runner = Path(runner).resolve()
    if not artifact.is_file():
        raise FileNotFoundError(artifact)
    if not runner.is_file():
        raise FileNotFoundError(runner)
    output_dir.mkdir(parents=True, exist_ok=True)
    records = []
    for name, boundary, experts, validation, profiler in CASES:
        raw = _run_case(
            artifact, runner, boundary=boundary, experts=experts,
            validation=validation, profiler=profiler,
            warmup=warmup, iterations=iterations,
        )
        _validate_record(
            raw, name=name, boundary=boundary, experts=experts,
            validation=validation, profiler=profiler,
            warmup=warmup, iterations=iterations,
        )
        json_path = output_dir / f"{name}.json"
        csv_path = output_dir / f"{name}.csv"
        _write_json(json_path, raw)
        _write_csv(csv_path, raw)
        records.append({
            **raw,
            "name": name,
            "raw_json_sha256": _sha256(json_path),
            "raw_csv_sha256": _sha256(csv_path),
        })
    aggregate = json.dumps(
        records, sort_keys=True, separators=(",", ":")
    ).encode()
    summary = {
        "scope": "released-dimension-validation-attribution",
        "evidence": "measured",
        "benchmark": "B-0024",
        "warmup": warmup,
        "iterations": iterations,
        "artifact_sha256": _sha256(artifact),
        "runner_sha256": _sha256(runner),
        "aggregate_sha256": hashlib.sha256(aggregate).hexdigest(),
        "records": records,
    }
    fieldnames = list(dict.fromkeys(
        field for record in records for field in record
    ))
    with (output_dir / "summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(records)
    _write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("artifact", type=Path)
    parser.add_argument("runner", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--iterations", type=int, default=10)
    args = parser.parse_args()
    run_ablation(
        args.artifact, args.runner, output_dir=args.output_dir,
        warmup=args.warmup, iterations=args.iterations,
    )


if __name__ == "__main__":
    main()
