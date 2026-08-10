# CUDA Graph 캐시 ablation의 행렬, 수식, 증거 무결성을 검증합니다.
import csv
import hashlib
import json
from pathlib import Path

import pytest

from tools.ablate_cuda_graph_cache import (
    case_matrix,
    expected_graph_counters,
    run_ablation,
    verify_evidence,
)


def _record(
    trace: str,
    graph: str,
    entries: int,
    warmup: int,
    iterations: int,
) -> dict:
    counters = expected_graph_counters(
        trace, graph, entries, warmup=warmup, iterations=iterations
    )
    return {
        "artifact_kind": "released_dimension_moe_layer",
        "routing_semantics": False,
        "boundary": "moe-layer",
        "trace": trace,
        "trace_period": {"stable-1": 1, "alternating-2": 2, "rotating-5": 5}[trace],
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
        "maximum_absolute_error": 0.0,
        "latency_nanoseconds_median": 100,
        "latency_nanoseconds_p10": 90,
        "latency_nanoseconds_p90": 110,
        "kernel_nanoseconds": None,
        "activation_h2d_bytes": 28_880 * iterations,
        "device_to_host_bytes": 28_672 * iterations,
        "weight_h2d_bytes": 0,
        "stream_synchronization_count": iterations,
        "cold_weight_h2d_bytes": 539_965_440,
        "cold_immutable_validation_scans": 6,
        "cold_immutable_validation_bytes": 469_776_384,
        "immutable_validation_scans": 0,
        "immutable_validation_hits": 6 * iterations,
        "immutable_validation_bytes": 0,
        "immutable_validation_nanoseconds": 0,
        **counters,
        "resident_weight_bytes": 750_532_608,
        "peak_resident_weight_bytes": 750_532_608,
        "oracle_peak_vram_bytes": 750_518_272,
        "peak_vram_bytes": 751_547_200,
        "weight_cache_bypasses": 0,
        "resident_grid_calls": iterations,
        "resident_grid_kernel_launches": 4 * iterations,
        "resident_grid_fallbacks": 0,
        "resident_moe_layer_calls": iterations,
        "resident_moe_layer_experts": 4 * iterations,
        "resident_moe_layer_kernel_launches": 13 * iterations,
        "resident_moe_layer_fallbacks": 0,
        "resident_moe_layer_contribution_h2d_bytes": 16 * iterations,
    }


def test_case_matrix_is_canonical_trace_major_order() -> None:
    assert case_matrix() == tuple(
        (f"{trace}-{name}", trace, graph, entries)
        for trace in ("stable-1", "alternating-2", "rotating-5")
        for name, graph, entries in (
            ("disabled", "disabled", 0),
            ("update-1", "update", 1),
            ("cache-1", "cache", 1),
            ("cache-2", "cache", 2),
            ("cache-4", "cache", 4),
        )
    )


def test_graph_counter_formula_models_warm_state_and_lru() -> None:
    assert expected_graph_counters(
        "stable-1", "cache", 1, warmup=3, iterations=20
    )["cuda_graph_cache_hits"] == 20
    alternating = expected_graph_counters(
        "alternating-2", "cache", 1, warmup=3, iterations=20
    )
    assert alternating["cuda_graph_cache_misses"] == 20
    assert alternating["cuda_graph_cache_evictions"] == 20
    assert expected_graph_counters(
        "alternating-2", "cache", 2, warmup=3, iterations=20
    )["cuda_graph_cache_hits"] == 20
    rotating = expected_graph_counters(
        "rotating-5", "cache", 4, warmup=3, iterations=20
    )
    assert rotating["cuda_graph_cache_misses"] == 20
    assert rotating["cuda_graph_cache_evictions"] == 20
    update = expected_graph_counters(
        "rotating-5", "update", 1, warmup=3, iterations=20
    )
    assert update["cuda_graph_update_attempts"] == 20
    assert update["cuda_graph_update_successes"] == 20


def test_run_ablation_writes_digest_backed_lf_evidence(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    artifact = tmp_path / "released.k3x"
    runner = tmp_path / "runner"
    artifact.write_bytes(b"artifact")
    runner.write_bytes(b"runner")
    calls = []

    def fake_run(*args: object, **kwargs: object) -> dict:
        calls.append(kwargs)
        return _record(
            str(kwargs["trace"]), str(kwargs["graph"]), int(kwargs["entries"]),
            int(kwargs["warmup"]), int(kwargs["iterations"]),
        )

    monkeypatch.setattr("tools.ablate_cuda_graph_cache._run_case", fake_run)
    output = tmp_path / "b0025"
    summary = run_ablation(
        artifact, runner, output_dir=output, warmup=3, iterations=20
    )

    assert len(summary["records"]) == 15
    assert [(call["trace"], call["graph"], call["entries"]) for call in calls] == [
        row[1:] for row in case_matrix()
    ]
    assert summary["artifact_sha256"] == hashlib.sha256(b"artifact").hexdigest()
    assert summary["runner_sha256"] == hashlib.sha256(b"runner").hexdigest()
    aggregate = json.dumps(
        summary["records"], sort_keys=True, separators=(",", ":")
    ).encode()
    assert summary["aggregate_sha256"] == hashlib.sha256(aggregate).hexdigest()
    assert summary["summary_csv_sha256"] == hashlib.sha256(
        (output / "summary.csv").read_bytes()
    ).hexdigest()
    assert b"\r\n" not in (output / "summary.csv").read_bytes()
    for record in summary["records"]:
        for suffix, digest_field in (
            ("json", "raw_json_sha256"), ("csv", "raw_csv_sha256")
        ):
            raw = output / f"{record['name']}.{suffix}"
            assert record[digest_field] == hashlib.sha256(raw.read_bytes()).hexdigest()
        with (output / f"{record['name']}.csv").open(
            newline="", encoding="utf-8"
        ) as stream:
            assert len(list(csv.DictReader(stream))) == 1
    assert verify_evidence(output, artifact=artifact, runner=runner) == summary

    raw = output / "stable-1-disabled.json"
    raw.write_bytes(raw.read_bytes() + b" ")
    with pytest.raises(RuntimeError, match="raw JSON digest diverged"):
        verify_evidence(output, artifact=artifact, runner=runner)


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("maximum_absolute_error", 1.0e-4, "numerical divergence"),
        ("cuda_graph_cache_hits", 999, "graph accounting diverged"),
        ("cuda_graph_cache_misses", 999, "graph accounting diverged"),
        ("cuda_graph_cache_evictions", 999, "graph accounting diverged"),
        ("weight_h2d_bytes", 1, "weight_h2d_bytes is nonzero"),
        ("stream_synchronization_count", 0, "physical accounting diverged"),
        ("resident_moe_layer_kernel_launches", 0, "physical accounting diverged"),
        ("activation_h2d_bytes", 0, "physical accounting diverged"),
        ("resident_moe_layer_fallbacks", 1, "resident_moe_layer_fallbacks is nonzero"),
    ],
)
def test_run_ablation_rejects_mutated_evidence(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
    value: object,
    message: str,
) -> None:
    artifact = tmp_path / "released.k3x"
    runner = tmp_path / "runner"
    artifact.write_bytes(b"artifact")
    runner.write_bytes(b"runner")

    def fake_run(*args: object, **kwargs: object) -> dict:
        record = _record(
            str(kwargs["trace"]), str(kwargs["graph"]), int(kwargs["entries"]),
            int(kwargs["warmup"]), int(kwargs["iterations"]),
        )
        record[field] = value
        return record

    monkeypatch.setattr("tools.ablate_cuda_graph_cache._run_case", fake_run)
    with pytest.raises(RuntimeError, match=message):
        run_ablation(
            artifact, runner, output_dir=tmp_path / "bad",
            warmup=3, iterations=20,
        )
