# 공식 MoE device-routing의 B-0033 증거를 원자적으로 생성하고 검증합니다.
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path
from typing import Mapping

if __package__:
    from tools import ablate_official_kda_device_state as state
    from tools import ablate_official_layer as base
else:
    import ablate_official_kda_device_state as state
    import ablate_official_layer as base


CASES = (
    ("ab-incremental-device-state-host-route", "host"),
    ("ab-incremental-device-state-device-route", "device"),
)

_FORMAT = "k3x-official-moe-device-routing-v1"
_BENCHMARK = "B-0033"
_SCOPE = "official-moe-device-routing"
_ROUTE_WEIGHT_BYTES = 12_888_064
_ROUTER_LOGIT_BYTES = 896 * 4
_PREPARED_SLOT_BYTES = 2 * 7_168 * 4
_EXTRA_FIELDS = (
    "route_preparation",
    "cold_official_moe_route_prepare_calls",
    "cold_official_moe_route_prepare_kernel_launches",
    "cold_official_moe_router_logit_d2h_bytes",
    "cold_official_moe_prepared_seeds",
    "cold_official_moe_prepared_consumes",
    "cold_official_moe_prepared_discards",
    "cold_official_moe_prepared_invalidations",
    "official_moe_route_prepare_calls",
    "official_moe_route_prepare_kernel_launches",
    "official_moe_router_logit_d2h_bytes",
    "official_moe_prepared_seeds",
    "official_moe_prepared_consumes",
    "official_moe_prepared_discards",
    "official_moe_prepared_invalidations",
    "official_moe_prepared_slot_bytes",
)
_RAW_FIELDS = (*state._RAW_FIELDS, *_EXTRA_FIELDS)
_CSV_FIELDS = ("name", "raw_json_sha256", *_RAW_FIELDS)
_SUMMARY_FIELDS = {
    "format",
    "benchmark",
    "scope",
    "evidence",
    "warmups",
    "iterations",
    "artifact_sha256",
    "manifest_sha256",
    "runner_sha256",
    "aggregate_sha256",
    "artifact_bytes",
    "manifest_identity",
    "records",
    "summary_csv_sha256",
}


def _route_formula(route_preparation: str, iterations: int) -> dict[str, int]:
    device = route_preparation == "device"
    calls = 2 if device else 0
    return {
        "cold_official_moe_route_prepare_calls": calls,
        "cold_official_moe_route_prepare_kernel_launches": calls * 2,
        "cold_official_moe_router_logit_d2h_bytes": _ROUTER_LOGIT_BYTES * calls,
        "cold_official_moe_prepared_seeds": calls,
        "cold_official_moe_prepared_consumes": calls,
        "cold_official_moe_prepared_discards": 0,
        "cold_official_moe_prepared_invalidations": 0,
        "official_moe_route_prepare_calls": calls * iterations,
        "official_moe_route_prepare_kernel_launches": calls * 2 * iterations,
        "official_moe_router_logit_d2h_bytes": (
            _ROUTER_LOGIT_BYTES * calls * iterations
        ),
        "official_moe_prepared_seeds": calls * iterations,
        "official_moe_prepared_consumes": calls * iterations,
        "official_moe_prepared_discards": 0,
        "official_moe_prepared_invalidations": 0,
        "official_moe_prepared_slot_bytes": _PREPARED_SLOT_BYTES if device else 0,
    }


def _normalize_state_record(
    record: Mapping[str, object], route_preparation: str, iterations: int
) -> dict[str, object]:
    legacy = {field: record[field] for field in state._RAW_FIELDS}
    if route_preparation == "host":
        return legacy
    deltas = {
        "cold_weight_h2d_bytes": _ROUTE_WEIGHT_BYTES,
        "cold_bf16_weight_h2d_bytes": _ROUTE_WEIGHT_BYTES,
        "resident_weight_bytes": _ROUTE_WEIGHT_BYTES,
        "peak_resident_weight_bytes": _ROUTE_WEIGHT_BYTES,
        "device_to_host_bytes": _ROUTER_LOGIT_BYTES * 2 * iterations,
        "stream_synchronization_count": 2 * iterations,
        "weight_cache_hits": 8 * iterations,
        "cold_immutable_validation_scans": 4,
        "cold_immutable_validation_hits": 4,
        "cold_immutable_validation_bytes": _ROUTE_WEIGHT_BYTES,
        "immutable_validation_hits": 8 * iterations,
    }
    for field, delta in deltas.items():
        value = legacy[field]
        if type(value) is not int or value < delta:
            raise RuntimeError("route formula diverged")
        legacy[field] = value - delta
    return legacy


def _validate_record(
    record: Mapping[str, object],
    *,
    name: str,
    route_preparation: str,
    warmups: int,
    iterations: int,
    manifest: Mapping[str, object],
    artifact_bytes: int,
) -> None:
    if set(record) != set(_RAW_FIELDS):
        raise RuntimeError(f"{name} schema diverged")
    if record.get("route_preparation") != route_preparation:
        raise RuntimeError(f"{name} route-preparation identity diverged")
    expected = _route_formula(route_preparation, iterations)
    if any(record.get(field) != value for field, value in expected.items()):
        raise RuntimeError(f"{name} route formula diverged")
    legacy = _normalize_state_record(record, route_preparation, iterations)
    try:
        state._validate_record(
            legacy,
            name=name,
            case="ab-incremental",
            state_transfer="device",
            warmups=warmups,
            iterations=iterations,
            manifest=manifest,
            artifact_bytes=artifact_bytes,
        )
    except RuntimeError as error:
        if route_preparation == "device":
            raise RuntimeError(f"{name} route formula diverged") from error
        raise


def _run_case(
    artifact: Path,
    manifest: Path,
    runner: Path,
    *,
    route_preparation: str,
    warmups: int,
    iterations: int,
) -> dict[str, object]:
    command = [
        str(runner),
        "--artifact",
        str(artifact),
        "--manifest",
        str(manifest),
        "--case",
        "ab-incremental",
        "--weight-mode",
        "resident",
        "--validation",
        "admission",
        "--state-transfer",
        "device",
        "--route-preparation",
        route_preparation,
        "--warmups",
        str(warmups),
        "--iterations",
        str(iterations),
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(
            result.stderr.strip() or "official MoE device-routing benchmark failed"
        )
    return base._parse_json(result.stdout, f"{route_preparation}-route output")


def _write_csv(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=_CSV_FIELDS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(
            {field: base._scalar(record[field]) for field in _CSV_FIELDS}
            for record in records
        )
        stream.flush()
        os.fsync(stream.fileno())


def _require_cross_row_parity(records: list[dict[str, object]]) -> None:
    for field in (
        "output_sha256",
        "state_sha256",
        "selected_union",
        "route_a",
        "route_b",
        "route_a_contributions",
        "route_b_contributions",
        "weight_h2d_bytes",
    ):
        if records[1][field] != records[0][field]:
            raise RuntimeError(f"cross-row {field} parity diverged")


def run_ablation(
    artifact: Path,
    manifest: Path,
    runner: Path,
    *,
    output_dir: Path,
    warmups: int,
    iterations: int,
) -> dict[str, object]:
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
    manifest_value = base._parse_json(manifest.read_bytes(), "route manifest")
    base._manifest_identity(manifest_value)
    output_dir = Path(output_dir).resolve()
    partial = output_dir.with_name(f".{output_dir.name}.partial")
    if output_dir.exists():
        raise FileExistsError(output_dir)
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    try:
        records: list[dict[str, object]] = []
        for name, route_preparation in CASES:
            raw = _run_case(
                artifact,
                manifest,
                runner,
                route_preparation=route_preparation,
                warmups=warmups,
                iterations=iterations,
            )
            _validate_record(
                raw,
                name=name,
                route_preparation=route_preparation,
                warmups=warmups,
                iterations=iterations,
                manifest=manifest_value,
                artifact_bytes=artifact.stat().st_size,
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
        _require_cross_row_parity(records)
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
            "artifact_sha256": base._sha256(artifact),
            "manifest_sha256": base._sha256(manifest),
            "runner_sha256": base._sha256(runner),
            "aggregate_sha256": hashlib.sha256(aggregate).hexdigest(),
            "artifact_bytes": artifact.stat().st_size,
            "manifest_identity": base._identity_manifest(manifest_value),
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
    runner: Path | None = None,
    strict_official: bool = True,
) -> dict[str, object]:
    summary_json, summary_csv = Path(summary_json), Path(summary_csv)
    summary = base._parse_json(summary_json.read_bytes(), "summary JSON")
    if (
        summary_json.read_bytes() != base._summary_bytes(summary)
        or set(summary) != _SUMMARY_FIELDS
    ):
        raise RuntimeError("summary schema or encoding diverged")
    if (
        summary.get("format"),
        summary.get("benchmark"),
        summary.get("scope"),
        summary.get("evidence"),
    ) != (_FORMAT, _BENCHMARK, _SCOPE, "measured"):
        raise RuntimeError("summary identity diverged")
    warmups, iterations = summary.get("warmups"), summary.get("iterations")
    if not base._integer(warmups) or not base._integer(iterations, positive=True):
        raise RuntimeError("summary iteration identity diverged")
    if strict_official and (artifact is None or manifest is None or runner is None):
        raise RuntimeError(
            "strict verification requires artifact, manifest, and runner"
        )
    if strict_official and (warmups != 3 or iterations != 20):
        raise RuntimeError("official iteration gate diverged")
    for field, path in {
        "artifact_sha256": artifact,
        "manifest_sha256": manifest,
        "runner_sha256": runner,
    }.items():
        if path is not None and summary.get(field) != base._sha256(Path(path)):
            raise RuntimeError(f"{field} diverged")
    manifest_identity = summary.get("manifest_identity")
    if not isinstance(manifest_identity, dict):
        raise RuntimeError("summary manifest identity diverged")
    base._manifest_identity(manifest_identity)
    if manifest is not None:
        actual_manifest = base._parse_json(
            Path(manifest).read_bytes(), "route manifest"
        )
        if base._identity_manifest(actual_manifest) != manifest_identity:
            raise RuntimeError("summary manifest identity diverged")
    records = summary.get("records")
    if not isinstance(records, list) or len(records) != len(CASES):
        raise RuntimeError("summary record count diverged")
    artifact_bytes = summary.get("artifact_bytes")
    if not base._integer(artifact_bytes, positive=True):
        raise RuntimeError("summary artifact bytes diverged")
    if artifact is not None and Path(artifact).stat().st_size != artifact_bytes:
        raise RuntimeError("summary artifact bytes diverged")
    for record, (name, route_preparation) in zip(records, CASES, strict=True):
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
            route_preparation=route_preparation,
            warmups=warmups,
            iterations=iterations,
            manifest=manifest_identity,
            artifact_bytes=artifact_bytes,
        )
        raw_path = summary_json.parent / f"{name}.json"
        if record["raw_json_sha256"] != base._sha256(raw_path):
            raise RuntimeError(f"{name} raw JSON digest diverged")
        payload = base._parse_json(raw_path.read_bytes(), f"{name} raw JSON")
        if raw_path.read_bytes() != base._canonical(payload) or payload != raw:
            raise RuntimeError(f"{name} raw JSON payload diverged")
    _require_cross_row_parity(records)
    aggregate = json.dumps(
        records, sort_keys=True, separators=(",", ":")
    ).encode()
    if summary.get("aggregate_sha256") != hashlib.sha256(aggregate).hexdigest():
        raise RuntimeError("aggregate digest diverged")
    csv_bytes = summary_csv.read_bytes()
    if b"\r\n" in csv_bytes or summary.get(
        "summary_csv_sha256"
    ) != hashlib.sha256(csv_bytes).hexdigest():
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
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmups", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--verify-existing", action="store_true")
    arguments = parser.parse_args(argv)
    if not arguments.verify_existing:
        run_ablation(
            arguments.artifact,
            arguments.manifest,
            arguments.runner,
            output_dir=arguments.output_dir,
            warmups=arguments.warmups,
            iterations=arguments.iterations,
        )
    verify_summary(
        arguments.output_dir / "summary.json",
        arguments.output_dir / "summary.csv",
        artifact=arguments.artifact,
        manifest=arguments.manifest,
        runner=arguments.runner,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
