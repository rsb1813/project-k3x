# B-0026 변환 무결성 감사 증거를 생성하고 검증합니다.
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import tempfile
import time
from pathlib import Path

from k3x_converter.format import SUPERBLOCK_BYTES
from k3x_converter.reader import K3XReader
from k3x_converter.resume import read_resume_manifest
from k3x_converter.writer import convert
from k3x_ref.fixtures import write_source_checkpoint

_SCHEMA = "k3x-converter-integrity-audit-v1"
_SCENARIOS = ("fresh", "resume-clean", "resume-orphan")
_RECORD_KEYS = (
    "scenario",
    "wall_nanoseconds",
    "maximum_source_read_bytes",
    "output_bytes",
    "reused_extent_count",
    "artifact_valid",
    "root_sha256",
    "artifact_sha256",
    "initial_stop_after_extents",
    "orphan_suffix_bytes",
    "committed_prefix_bytes",
    "rss_measurement",
    "peak_rss_bytes",
)
_SUMMARY_KEYS = {
    "schema",
    "benchmark_id",
    "evidence",
    "environment_label",
    "chunk_bytes",
    "stop_after_extents",
    "orphan_suffix_bytes",
    "runner_sha256",
    "aggregate_sha256",
    "summary_csv_sha256",
    "records",
}
_FORBIDDEN_FIELD_PARTS = (
    "decode",
    "prefill",
    "tok",
    "ttft",
    "top_k",
    "top-k",
    "specul",
    "gpu",
    "vram",
    "nvme",
    "h2d",
    "quality",
)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _aggregate_sha256(records: list[dict[str, object]]) -> str:
    encoded = json.dumps(records, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _orphan_bytes(length: int) -> bytes:
    pattern = bytes(range(1, 256))
    return (pattern * ((length + len(pattern) - 1) // len(pattern)))[:length]


def _record(
    scenario: str,
    output: Path,
    *,
    wall_nanoseconds: int,
    maximum_source_read_bytes: int,
    reused_extent_count: int,
    initial_stop_after_extents: int,
    orphan_suffix_bytes: int,
    committed_prefix_bytes: int,
) -> dict[str, object]:
    reader = K3XReader.open(output)
    return {
        "scenario": scenario,
        "wall_nanoseconds": wall_nanoseconds,
        "maximum_source_read_bytes": maximum_source_read_bytes,
        "output_bytes": output.stat().st_size,
        "reused_extent_count": reused_extent_count,
        "artifact_valid": True,
        "root_sha256": reader.superblock.root_sha256.hex(),
        "artifact_sha256": _sha256_path(output),
        "initial_stop_after_extents": initial_stop_after_extents,
        "orphan_suffix_bytes": orphan_suffix_bytes,
        "committed_prefix_bytes": committed_prefix_bytes,
        "rss_measurement": "not-measured",
        "peak_rss_bytes": None,
    }


def _resume_record(
    source: Path,
    output: Path,
    *,
    chunk_bytes: int,
    stop_after_extents: int,
    orphan_suffix_bytes: int,
    scenario: str,
) -> dict[str, object]:
    started = time.perf_counter_ns()
    interrupted = convert(
        source,
        output,
        chunk_bytes=chunk_bytes,
        stop_after_extents=stop_after_extents,
    )
    ledger = read_resume_manifest(output.with_suffix(".k3x.resume.json"))
    committed_prefix = (
        ledger.completed[-1].offset + ledger.completed[-1].length
        if ledger.completed
        else SUPERBLOCK_BYTES
    )
    if orphan_suffix_bytes:
        with output.with_suffix(".k3x.partial").open("ab") as stream:
            stream.write(_orphan_bytes(orphan_suffix_bytes))
    resumed = convert(source, output, chunk_bytes=chunk_bytes)
    return _record(
        scenario,
        output,
        wall_nanoseconds=time.perf_counter_ns() - started,
        maximum_source_read_bytes=max(
            interrupted.maximum_source_read_bytes,
            resumed.maximum_source_read_bytes,
        ),
        reused_extent_count=len(resumed.reused_extent_ids),
        initial_stop_after_extents=stop_after_extents,
        orphan_suffix_bytes=orphan_suffix_bytes,
        committed_prefix_bytes=committed_prefix,
    )


def _write_csv(path: Path, records: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=_RECORD_KEYS, lineterminator="\n")
        writer.writeheader()
        for record in records:
            writer.writerow({
                key: json.dumps(record[key], sort_keys=True, separators=(",", ":"))
                for key in _RECORD_KEYS
            })


def _write_json(path: Path, summary: dict[str, object]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(summary, stream, indent=2, sort_keys=True)
        stream.write("\n")


def run_converter_integrity_audit(
    output_dir: Path,
    *,
    environment_label: str,
    chunk_bytes: int = 257,
    stop_after_extents: int = 2,
    orphan_suffix_bytes: int = 8192,
) -> dict[str, object]:
    if (
        not environment_label
        or chunk_bytes <= 0
        or stop_after_extents <= 0
        or orphan_suffix_bytes <= 0
    ):
        raise ValueError("invalid audit configuration")
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="k3x-converter-integrity-") as temporary:
        work = Path(temporary)
        source = work / "source"
        write_source_checkpoint(source)

        fresh_output = work / "fresh.k3x"
        started = time.perf_counter_ns()
        fresh = convert(source, fresh_output, chunk_bytes=chunk_bytes)
        records = [
            _record(
                "fresh",
                fresh_output,
                wall_nanoseconds=time.perf_counter_ns() - started,
                maximum_source_read_bytes=fresh.maximum_source_read_bytes,
                reused_extent_count=len(fresh.reused_extent_ids),
                initial_stop_after_extents=0,
                orphan_suffix_bytes=0,
                committed_prefix_bytes=0,
            ),
            _resume_record(
                source,
                work / "resume-clean.k3x",
                chunk_bytes=chunk_bytes,
                stop_after_extents=stop_after_extents,
                orphan_suffix_bytes=0,
                scenario="resume-clean",
            ),
            _resume_record(
                source,
                work / "resume-orphan.k3x",
                chunk_bytes=chunk_bytes,
                stop_after_extents=stop_after_extents,
                orphan_suffix_bytes=orphan_suffix_bytes,
                scenario="resume-orphan",
            ),
        ]
    if len({record["output_bytes"] for record in records}) != 1:
        raise ValueError("finalized output sizes differ")
    if any(record["maximum_source_read_bytes"] > chunk_bytes for record in records):
        raise ValueError("source read exceeded chunk size")
    summary_csv = output_dir / "summary.csv"
    _write_csv(summary_csv, records)
    summary = {
        "schema": _SCHEMA,
        "benchmark_id": "B-0026",
        "evidence": "measured",
        "environment_label": environment_label,
        "chunk_bytes": chunk_bytes,
        "stop_after_extents": stop_after_extents,
        "orphan_suffix_bytes": orphan_suffix_bytes,
        "runner_sha256": _sha256_path(Path(__file__)),
        "aggregate_sha256": _aggregate_sha256(records),
        "summary_csv_sha256": _sha256_path(summary_csv),
        "records": records,
    }
    _write_json(output_dir / "summary.json", summary)
    return summary


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _is_hex_digest(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )


def _reject_forbidden_fields(value: object) -> None:
    if isinstance(value, dict):
        for key, nested in value.items():
            lowered = key.lower()
            if any(part in lowered for part in _FORBIDDEN_FIELD_PARTS):
                raise ValueError("forbidden inference field")
            _reject_forbidden_fields(nested)
    elif isinstance(value, list):
        for nested in value:
            _reject_forbidden_fields(nested)


def _read_csv_records(path: Path) -> list[dict[str, object]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream)
        _require(tuple(reader.fieldnames or ()) == _RECORD_KEYS, "invalid CSV columns")
        return [
            {key: json.loads(value) for key, value in row.items()}
            for row in reader
        ]


def verify_evidence(
    evidence_dir: Path,
    *,
    runner: Path | None = None,
) -> dict[str, object]:
    evidence_dir = Path(evidence_dir)
    json_path = evidence_dir / "summary.json"
    csv_path = evidence_dir / "summary.csv"
    try:
        json_bytes = json_path.read_bytes()
        csv_bytes = csv_path.read_bytes()
        _require(b"\r" not in json_bytes, "summary JSON is not LF-only")
        _require(json_bytes.endswith(b"\n"), "summary JSON is not LF-terminated")
        _require(b"\r" not in csv_bytes, "CSV is not LF-only")
        _require(csv_bytes.endswith(b"\n"), "CSV is not LF-terminated")
        summary = json.loads(json_bytes)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("invalid audit evidence") from error
    _require(isinstance(summary, dict) and set(summary) == _SUMMARY_KEYS, "invalid summary keys")
    _reject_forbidden_fields(summary)
    _require(summary["schema"] == _SCHEMA, "invalid schema")
    _require(summary["benchmark_id"] == "B-0026", "invalid benchmark id")
    _require(summary["evidence"] == "measured", "invalid evidence kind")
    _require(isinstance(summary["environment_label"], str), "invalid environment label")
    for key in ("chunk_bytes", "stop_after_extents", "orphan_suffix_bytes"):
        _require(isinstance(summary[key], int) and not isinstance(summary[key], bool) and summary[key] > 0, "invalid audit setting")
    _require(_is_hex_digest(summary["runner_sha256"]), "invalid runner digest")
    _require(_is_hex_digest(summary["aggregate_sha256"]), "invalid aggregate digest")
    _require(_is_hex_digest(summary["summary_csv_sha256"]), "invalid CSV digest")
    records = summary["records"]
    _require(isinstance(records, list) and len(records) == len(_SCENARIOS), "invalid records")
    _require([record.get("scenario") if isinstance(record, dict) else None for record in records] == list(_SCENARIOS), "invalid scenario order")
    for index, record in enumerate(records):
        _require(isinstance(record, dict) and set(record) == set(_RECORD_KEYS), "invalid record keys")
        for key in ("wall_nanoseconds", "maximum_source_read_bytes", "output_bytes", "reused_extent_count", "initial_stop_after_extents", "orphan_suffix_bytes", "committed_prefix_bytes"):
            _require(isinstance(record[key], int) and not isinstance(record[key], bool) and record[key] >= 0, "invalid record integer")
        _require(record["maximum_source_read_bytes"] <= summary["chunk_bytes"], "unbounded source read")
        _require(record["output_bytes"] > 0, "empty output")
        _require(record["artifact_valid"] is True, "invalid artifact")
        _require(_is_hex_digest(record["root_sha256"]), "invalid root digest")
        _require(_is_hex_digest(record["artifact_sha256"]), "invalid artifact digest")
        _require(record["rss_measurement"] == "not-measured" and record["peak_rss_bytes"] is None, "invalid RSS record")
        if index == 0:
            _require(record["reused_extent_count"] == 0 and record["initial_stop_after_extents"] == 0 and record["orphan_suffix_bytes"] == 0 and record["committed_prefix_bytes"] == 0, "invalid fresh record")
        else:
            _require(record["reused_extent_count"] > 0 and record["initial_stop_after_extents"] == summary["stop_after_extents"] and record["committed_prefix_bytes"] > 0, "invalid resume record")
            _require(record["orphan_suffix_bytes"] == (0 if index == 1 else summary["orphan_suffix_bytes"]), "invalid orphan record")
    _require(len({record["output_bytes"] for record in records}) == 1, "output sizes differ")
    _require(summary["aggregate_sha256"] == _aggregate_sha256(records), "aggregate digest mismatch")
    _require(summary["summary_csv_sha256"] == _sha256_path(csv_path), "CSV digest mismatch")
    _require(_read_csv_records(csv_path) == records, "JSON and CSV records differ")
    if runner is not None:
        _require(summary["runner_sha256"] == _sha256_path(Path(runner)), "runner digest mismatch")
    return summary


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence_dir", type=Path)
    parser.add_argument("--environment-label")
    parser.add_argument("--chunk-bytes", type=int, default=257)
    parser.add_argument("--stop-after-extents", type=int, default=2)
    parser.add_argument("--orphan-suffix-bytes", type=int, default=8192)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args(argv)
    if args.verify_only:
        summary = verify_evidence(args.evidence_dir, runner=Path(__file__))
    else:
        if args.environment_label is None:
            parser.error("--environment-label is required unless --verify-only is used")
        summary = run_converter_integrity_audit(
            args.evidence_dir,
            environment_label=args.environment_label,
            chunk_bytes=args.chunk_bytes,
            stop_after_extents=args.stop_after_extents,
            orphan_suffix_bytes=args.orphan_suffix_bytes,
        )
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
