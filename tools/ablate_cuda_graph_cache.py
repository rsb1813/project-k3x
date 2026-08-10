# CUDA Graph 캐시의 15행 추적 ablation과 증거 검증을 수행합니다.
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
from pathlib import Path


_TRACES = {
    "stable-1": (0,),
    "alternating-2": (0, 1),
    "rotating-5": (0, 1, 2, 3, 4),
}
_MODES = (
    ("disabled", "disabled", 0),
    ("update-1", "update", 1),
    ("cache-1", "cache", 1),
    ("cache-2", "cache", 2),
    ("cache-4", "cache", 4),
)


def case_matrix() -> tuple[tuple[str, str, str, int], ...]:
    return tuple(
        (f"{trace}-{name}", trace, graph, entries)
        for trace in _TRACES
        for name, graph, entries in _MODES
    )


def _counter_snapshot(
    trace: str, graph: str, entries: int, calls: int
) -> dict[str, int]:
    counters = {
        "cuda_graph_cache_hits": 0,
        "cuda_graph_cache_misses": 0,
        "cuda_graph_cache_evictions": 0,
        "cuda_graph_instantiations": 0,
        "cuda_graph_update_attempts": 0,
        "cuda_graph_update_successes": 0,
        "cuda_graph_update_failures": 0,
        "cuda_graph_launches": 0,
        "cuda_graph_invalidations": 0,
        "cuda_graph_host_nanoseconds": 0,
        "cuda_graph_resident_entries": 0,
        "cuda_graph_peak_entries": 0,
    }
    if graph == "disabled":
        return counters
    if graph == "update":
        if calls:
            counters["cuda_graph_instantiations"] = 1
            counters["cuda_graph_update_attempts"] = calls - 1
            counters["cuda_graph_update_successes"] = calls - 1
            counters["cuda_graph_launches"] = calls
            counters["cuda_graph_resident_entries"] = 1
            counters["cuda_graph_peak_entries"] = 1
        return counters
    if graph != "cache" or entries <= 0:
        raise ValueError("invalid graph configuration")

    resident: list[int] = []
    sequence = _TRACES[trace]
    for call in range(calls):
        key = sequence[call % len(sequence)]
        if key in resident:
            counters["cuda_graph_cache_hits"] += 1
            resident.remove(key)
            resident.append(key)
        else:
            counters["cuda_graph_cache_misses"] += 1
            counters["cuda_graph_instantiations"] += 1
            if len(resident) == entries:
                resident.pop(0)
                counters["cuda_graph_cache_evictions"] += 1
            resident.append(key)
            counters["cuda_graph_peak_entries"] = max(
                counters["cuda_graph_peak_entries"], len(resident)
            )
        counters["cuda_graph_launches"] += 1
    counters["cuda_graph_resident_entries"] = len(resident)
    return counters


def expected_graph_counters(
    trace: str,
    graph: str,
    entries: int,
    *,
    warmup: int,
    iterations: int,
) -> dict[str, int]:
    if trace not in _TRACES:
        raise ValueError(f"unknown trace: {trace}")
    before = _counter_snapshot(trace, graph, entries, 1 + warmup)
    after = _counter_snapshot(trace, graph, entries, 1 + warmup + iterations)
    absolute = {"cuda_graph_resident_entries", "cuda_graph_peak_entries"}
    return {
        field: after[field] if field in absolute else after[field] - before[field]
        for field in after
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _write_csv(path: Path, records: list[dict]) -> None:
    fieldnames = list(dict.fromkeys(
        field for record in records for field in record
    ))
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=fieldnames, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(records)


def _run_case(
    artifact: Path,
    runner: Path,
    *,
    trace: str,
    graph: str,
    entries: int,
    warmup: int,
    iterations: int,
) -> dict:
    result = subprocess.run(
        [
            str(runner), "--model", str(artifact),
            "--boundary", "moe-layer", "--experts", "4",
            "--validation", "admission", "--profiler", "off",
            "--graph", graph, "--graph-entries", str(entries),
            "--trace", trace,
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
    trace: str,
    graph: str,
    entries: int,
    warmup: int,
    iterations: int,
) -> None:
    expected_identity = {
        "artifact_kind": "released_dimension_moe_layer",
        "routing_semantics": False,
        "boundary": "moe-layer",
        "trace": trace,
        "trace_period": len(_TRACES[trace]),
        "experts": 4,
        "hidden_width": 7168,
        "latent_width": 3584,
        "expert_intermediate_width": 3072,
        "expert_payload_bytes": 17_547_264,
        "resident_capacity_bytes": 1 << 30,
        "warmup": warmup,
        "iterations": iterations,
        "validation": "admission",
        "cuda_graph": graph,
        "cuda_graph_entries": entries,
        "profiler": False,
        "kernel_nanoseconds": None,
    }
    for field, value in expected_identity.items():
        if record.get(field) != value:
            raise RuntimeError(f"{name} identity field {field} diverged")
    if record.get("maximum_absolute_error") != 0.0:
        raise RuntimeError(f"{name} numerical divergence")
    p10 = record.get("latency_nanoseconds_p10", 0)
    median = record.get("latency_nanoseconds_median", 0)
    p90 = record.get("latency_nanoseconds_p90", 0)
    if not (0 < p10 <= median <= p90):
        raise RuntimeError(f"{name} latency distribution diverged")
    for field in (
        "weight_h2d_bytes",
        "weight_cache_bypasses",
        "resident_grid_fallbacks",
        "resident_moe_layer_fallbacks",
    ):
        if record.get(field) != 0:
            raise RuntimeError(f"{name} {field} is nonzero")
    if record.get("cold_weight_h2d_bytes", 0) <= 0:
        raise RuntimeError(f"{name} cold weight load is missing")
    if any(
        marker in key.lower()
        for key in record
        for marker in ("token", "prefill", "ttft")
    ):
        raise RuntimeError(f"{name} contains token-like telemetry")

    expected_physical = {
        "activation_h2d_bytes": 28_880 * iterations,
        "device_to_host_bytes": 28_672 * iterations,
        "stream_synchronization_count": iterations,
        "immutable_validation_scans": 0,
        "immutable_validation_hits": 6 * iterations,
        "immutable_validation_bytes": 0,
        "resident_grid_calls": iterations,
        "resident_grid_kernel_launches": 4 * iterations,
        "resident_moe_layer_calls": iterations,
        "resident_moe_layer_experts": 4 * iterations,
        "resident_moe_layer_kernel_launches": 13 * iterations,
        "resident_moe_layer_contribution_h2d_bytes": 16 * iterations,
    }
    if any(record.get(field) != value for field, value in expected_physical.items()):
        raise RuntimeError(f"{name} physical accounting diverged")

    expected_graph = expected_graph_counters(
        trace, graph, entries, warmup=warmup, iterations=iterations
    )
    ignored = {"cuda_graph_host_nanoseconds"}
    if any(
        record.get(field) != value
        for field, value in expected_graph.items()
        if field not in ignored
    ) or record.get("cuda_graph_host_nanoseconds", -1) < 0:
        raise RuntimeError(f"{name} graph accounting diverged")


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
    for name, trace, graph, entries in case_matrix():
        raw = _run_case(
            artifact, runner, trace=trace, graph=graph, entries=entries,
            warmup=warmup, iterations=iterations,
        )
        _validate_record(
            raw, name=name, trace=trace, graph=graph, entries=entries,
            warmup=warmup, iterations=iterations,
        )
        raw_json = output_dir / f"{name}.json"
        raw_csv = output_dir / f"{name}.csv"
        _write_json(raw_json, raw)
        _write_csv(raw_csv, [raw])
        records.append({
            **raw,
            "name": name,
            "raw_json_sha256": _sha256(raw_json),
            "raw_csv_sha256": _sha256(raw_csv),
        })

    aggregate = json.dumps(
        records, sort_keys=True, separators=(",", ":")
    ).encode()
    summary = {
        "scope": "released-dimension-bounded-cuda-graph-cache",
        "evidence": "measured",
        "benchmark": "B-0025",
        "warmup": warmup,
        "iterations": iterations,
        "artifact_sha256": _sha256(artifact),
        "runner_sha256": _sha256(runner),
        "aggregate_sha256": hashlib.sha256(aggregate).hexdigest(),
        "records": records,
    }
    summary_csv = output_dir / "summary.csv"
    _write_csv(summary_csv, records)
    summary["summary_csv_sha256"] = _sha256(summary_csv)
    _write_json(output_dir / "summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    args = parser.parse_args()
    run_ablation(
        args.artifact, args.runner, output_dir=args.output_dir,
        warmup=args.warmup, iterations=args.iterations,
    )


if __name__ == "__main__":
    main()
