# 공식 두 레이어 폐쇄 구간의 B-0035 계측 증거를 원자적으로 생성하고 검증합니다.
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
    from tools import ablate_official_layer as common
    from tools import ablate_official_two_layer_closure as closure
else:
    import ablate_official_layer as common
    import ablate_official_two_layer_closure as closure


CASES = closure.CASES
_FORMAT = "k3x-official-two-layer-attribution-v1"
_BENCHMARK = "B-0035"
_SCOPE = "official-two-layer-closure-attribution"
_ATTRIBUTION_FIELDS = (
    "total_wall_nanoseconds",
    "front_wall_nanoseconds",
    "front_device_nanoseconds",
    "route_wall_nanoseconds",
    "tail_wall_nanoseconds",
    "tail_device_nanoseconds",
    "unattributed_wall_nanoseconds",
)
_RUNNER_FIELDS = (*closure._RUNNER_FIELDS, *_ATTRIBUTION_FIELDS)
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
    forbidden = closure._FORBIDDEN.intersection(record)
    if forbidden:
        raise RuntimeError(f"{name} contains forbidden metric {min(forbidden)}")
    if set(record) != set(_RAW_FIELDS):
        raise RuntimeError(f"{name} schema diverged")
    baseline = {field: record[field] for field in closure._RAW_FIELDS}
    baseline["schema"] = "k3x-official-two-layer-bench-v1"
    closure._validate_record(
        baseline,
        name=name,
        mode=mode,
        warmups=warmups,
        iterations=iterations,
        resident_bytes=resident_bytes,
        identity=identity,
    )
    if record.get("schema") != _FORMAT:
        raise RuntimeError(f"{name} attribution schema diverged")
    values = {field: record.get(field) for field in _ATTRIBUTION_FIELDS}
    if any(type(value) is not int or value < 0 for value in values.values()):
        raise RuntimeError(f"{name} attribution timing diverged")
    total = values["total_wall_nanoseconds"]
    closed = (
        values["front_wall_nanoseconds"]
        + values["route_wall_nanoseconds"]
        + values["tail_wall_nanoseconds"]
        + values["unattributed_wall_nanoseconds"]
    )
    if total != closed or total <= 0 or total > sum(record["wall_nanoseconds"]):
        raise RuntimeError(f"{name} attribution formula diverged")
    if mode == "host-round-trip":
        if any(values[field] != 0 for field in (
            "front_wall_nanoseconds",
            "front_device_nanoseconds",
            "route_wall_nanoseconds",
            "tail_wall_nanoseconds",
            "tail_device_nanoseconds",
        )) or values["unattributed_wall_nanoseconds"] != total:
            raise RuntimeError(f"{name} attribution formula diverged")
    elif (
        values["front_wall_nanoseconds"] <= 0
        or values["front_device_nanoseconds"] <= 0
        or values["route_wall_nanoseconds"] <= 0
        or values["tail_wall_nanoseconds"] <= 0
        or values["tail_device_nanoseconds"] <= 0
        or values["front_device_nanoseconds"] > values["front_wall_nanoseconds"]
        or values["tail_device_nanoseconds"] > values["tail_wall_nanoseconds"]
    ):
        raise RuntimeError(f"{name} attribution timing diverged")


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
        "--artifact", str(artifact),
        "--manifest", str(manifest),
        "--oracle", str(oracle),
        "--mode", mode,
        "--resident-bytes", str(resident_bytes),
        "--warmup", str(warmups),
        "--iterations", str(iterations),
        "--attribution", "true",
    ]
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(
            result.stderr.strip() or "official two-layer attribution failed"
        )
    return common._parse_json(result.stdout, f"{mode} attribution output")


def _write_csv(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=_CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(
            {field: common._scalar(record[field]) for field in _CSV_FIELDS}
            for record in records
        )
        stream.flush()
        os.fsync(stream.fileno())


def run_ablation(
    artifact: Path,
    manifest: Path,
    oracle: Path,
    runner: Path,
    *,
    output_dir: Path,
    warmups: int,
    iterations: int,
    resident_bytes: int = closure._OFFICIAL_RESIDENT_CAPACITY,
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
    identity = closure.manifest_identity(manifest)
    if (
        oracle.stat().st_size != identity["oracle"]["bytes"]
        or common._sha256(oracle) != identity["oracle"]["sha256"]
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
            raw = _run_case(
                artifact, manifest, oracle, runner,
                mode=mode,
                resident_bytes=resident_bytes,
                warmups=warmups,
                iterations=iterations,
            )
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
            common._write_file(raw_path, common._canonical(raw))
            records.append(
                {"name": name, "raw_json_sha256": common._sha256(raw_path), **raw}
            )
        closure._cross_row_parity(records)
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
            "resident_bytes": resident_bytes,
            "artifact_sha256": common._sha256(artifact),
            "manifest_sha256": common._sha256(manifest),
            "oracle_sha256": common._sha256(oracle),
            "runner_sha256": common._sha256(runner),
            "aggregate_sha256": hashlib.sha256(aggregate).hexdigest(),
            "artifact_bytes": artifact.stat().st_size,
            "oracle_bytes": oracle.stat().st_size,
            "manifest_identity": identity,
            "records": records,
        }
        csv_path = partial / "summary.csv"
        _write_csv(csv_path, records)
        summary["summary_csv_sha256"] = common._sha256(csv_path)
        common._write_file(partial / "summary.json", common._summary_bytes(summary))
        common._fsync_directory(partial)
        os.replace(partial, output_dir)
        common._fsync_directory(output_dir.parent)
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
    summary = common._parse_json(summary_json.read_bytes(), "summary JSON")
    if (
        summary_json.read_bytes() != common._summary_bytes(summary)
        or set(summary) != _SUMMARY_FIELDS
        or (summary.get("format"), summary.get("benchmark"),
            summary.get("scope"), summary.get("evidence"))
        != (_FORMAT, _BENCHMARK, _SCOPE, "measured")
    ):
        raise RuntimeError("summary schema or identity diverged")
    warmups = summary.get("warmups")
    iterations = summary.get("iterations")
    resident_bytes = summary.get("resident_bytes")
    if (
        not common._integer(warmups)
        or not common._integer(iterations, positive=True)
        or not common._integer(resident_bytes, positive=True)
    ):
        raise RuntimeError("summary transaction identity diverged")
    paths = (artifact, manifest, oracle, runner)
    if strict_official and any(path is None for path in paths):
        raise RuntimeError("strict verification requires all transaction inputs")
    if strict_official and (
        warmups != 3
        or iterations != 20
        or resident_bytes != closure._OFFICIAL_RESIDENT_CAPACITY
    ):
        raise RuntimeError("official transaction gate diverged")
    for field, path in {
        "artifact_sha256": artifact,
        "manifest_sha256": manifest,
        "oracle_sha256": oracle,
        "runner_sha256": runner,
    }.items():
        if path is not None and summary.get(field) != common._sha256(Path(path)):
            raise RuntimeError(f"{field} diverged")
    identity = summary.get("manifest_identity")
    if not isinstance(identity, dict) or closure.manifest_identity(identity) != identity:
        raise RuntimeError("summary manifest identity diverged")
    if manifest is not None and closure.manifest_identity(Path(manifest)) != identity:
        raise RuntimeError("summary manifest identity diverged")
    if any(not closure._hex(summary.get(field)) for field in (
        "artifact_sha256", "manifest_sha256", "oracle_sha256", "runner_sha256",
        "aggregate_sha256", "summary_csv_sha256",
    )):
        raise RuntimeError("summary digest identity diverged")
    if (
        summary.get("oracle_sha256") != identity["oracle"]["sha256"]
        or not common._integer(summary.get("artifact_bytes"), positive=True)
        or summary.get("oracle_bytes") != identity["oracle"]["bytes"]
    ):
        raise RuntimeError("summary input identity diverged")
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
        if record["raw_json_sha256"] != common._sha256(raw_path):
            raise RuntimeError(f"{name} raw JSON digest diverged")
        payload = common._parse_json(raw_path.read_bytes(), f"{name} raw JSON")
        if raw_path.read_bytes() != common._canonical(payload) or payload != raw:
            raise RuntimeError(f"{name} raw JSON payload diverged")
    closure._cross_row_parity(records)
    aggregate = json.dumps(records, sort_keys=True, separators=(",", ":")).encode()
    if summary["aggregate_sha256"] != hashlib.sha256(aggregate).hexdigest():
        raise RuntimeError("aggregate digest diverged")
    csv_bytes = summary_csv.read_bytes()
    if (
        b"\r\n" in csv_bytes
        or summary["summary_csv_sha256"] != hashlib.sha256(csv_bytes).hexdigest()
    ):
        raise RuntimeError("summary CSV digest or newline diverged")
    with summary_csv.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows, fields = list(reader), tuple(reader.fieldnames or ())
    expected = [
        {field: common._scalar(record[field]) for field in _CSV_FIELDS}
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
        "--resident-bytes", type=int, default=closure._OFFICIAL_RESIDENT_CAPACITY
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
