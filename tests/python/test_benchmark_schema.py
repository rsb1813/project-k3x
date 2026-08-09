# Synthetic benchmark 결과 schema와 measured/projected 구분을 검증합니다.
import csv
import json
import os
from pathlib import Path

import pytest

from conftest import cpp_binary
from k3x_converter.writer import convert
from tools.ablate_cuda_residency import cuda_residency_matrix, run_ablation
from tools.benchmark_synthetic import BenchmarkRecord, benchmark_once, write_results


def _record() -> BenchmarkRecord:
    return BenchmarkRecord(
        scope="synthetic-milestone-zero",
        evidence="measured",
        platform="test-machine",
        iterations=3,
        prompt_tokens=4,
        generated_tokens=6,
        prefill_tokens_per_second=100.0,
        decode_tokens_per_second=50.0,
        ttft_ms=12.5,
        peak_rss_bytes=123456,
        file_read_bytes_per_token=789.0,
        backend="cpu",
        device="CPU",
        dense_precision="fp32",
        cuda_allocation="per-operation",
        cuda_weights="transient",
        cuda_batching="scalar",
        cuda_boundary="operation",
        cuda_resident_bytes=0,
        kernel_nanoseconds=0,
        host_to_device_bytes=0,
        weight_h2d_bytes=0,
        activation_h2d_bytes=0,
        device_to_host_bytes=0,
        peak_vram_bytes=None,
        device_allocation_count=0,
        device_free_count=0,
        stream_synchronization_count=0,
        weight_cache_hits=0,
        weight_cache_misses=0,
        weight_cache_bypasses=0,
        resident_weight_bytes=0,
        peak_resident_weight_bytes=0,
        scratch_bytes=0,
        peak_scratch_bytes=0,
        grouped_projection_calls=0,
        grouped_projection_members=0,
        ffn_block_calls=0,
        ffn_block_experts=0,
        max_absolute_error=None,
        max_relative_error=None,
        kda_state_bytes=1024,
        mla_kv_bytes=2048,
        per_layer_nanoseconds=(1, 2, 3, 4),
    )


def test_benchmark_json_and_csv_preserve_schema(tmp_path: Path) -> None:
    json_path, csv_path = tmp_path / "result.json", tmp_path / "result.csv"
    write_results(_record(), json_path, csv_path)
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload["scope"] == "synthetic-milestone-zero"
    assert payload["evidence"] == "measured"
    assert isinstance(payload["peak_rss_bytes"], int)
    assert payload["backend"] == "cpu"
    assert payload["device"] == "CPU"
    assert payload["dense_precision"] == "fp32"
    assert payload["cuda_allocation"] == "per-operation"
    assert payload["cuda_weights"] == "transient"
    assert payload["cuda_batching"] == "scalar"
    assert payload["cuda_boundary"] == "operation"
    assert payload["cuda_resident_bytes"] == 0
    assert payload["device_allocation_count"] == 0
    assert payload["weight_h2d_bytes"] == 0
    assert payload["activation_h2d_bytes"] == 0
    assert payload["peak_vram_bytes"] is None
    with csv_path.open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert row["decode_tokens_per_second"] == "50.0"
    assert row["host_to_device_bytes"] == "0"
    assert row["cuda_allocation"] == "per-operation"
    assert row["weight_cache_bypasses"] == "0"
    assert row["grouped_projection_members"] == "0"
    assert row["ffn_block_calls"] == "0"
    assert row["ffn_block_experts"] == "0"
    assert row["peak_vram_bytes"] == ""
    assert row["per_layer_nanoseconds"] == "1;2;3;4"


def test_schema_accepts_explicit_milestone_one_scope() -> None:
    record = BenchmarkRecord(
        **{**_record().__dict__, "scope": "synthetic-milestone-one"}
    )
    assert record.scope == "synthetic-milestone-one"


def test_cuda_residency_matrix_changes_one_axis_at_a_time() -> None:
    matrix = cuda_residency_matrix()
    assert tuple(item["name"] for item in matrix) == (
        "reference",
        "reuse",
        "residency",
        "grouped",
    )
    assert tuple(
        (
            item["cuda_allocation"],
            item["cuda_weights"],
            item["cuda_batching"],
        )
        for item in matrix
    ) == (
        ("per-operation", "transient", "scalar"),
        ("reused", "transient", "scalar"),
        ("reused", "resident", "scalar"),
        ("reused", "resident", "grouped"),
    )


def test_cuda_residency_ablation_runs_sequential_stages_and_writes_summary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    observed: list[tuple[str, str, str, int]] = []

    def fake_benchmark_once(*args: object, **kwargs: object) -> BenchmarkRecord:
        observed.append(
            (
                str(kwargs["cuda_allocation"]),
                str(kwargs["cuda_weights"]),
                str(kwargs["cuda_batching"]),
                int(kwargs["cuda_resident_bytes"]),
            )
        )
        return BenchmarkRecord(
            **{
                **_record().__dict__,
                "backend": str(kwargs["backend"]),
                "cuda_allocation": str(kwargs["cuda_allocation"]),
                "cuda_weights": str(kwargs["cuda_weights"]),
                "cuda_batching": str(kwargs["cuda_batching"]),
                "cuda_resident_bytes": int(kwargs["cuda_resident_bytes"]),
                "device_allocation_count": len(observed),
            }
        )

    monkeypatch.setattr(
        "tools.ablate_cuda_residency.benchmark_once", fake_benchmark_once
    )
    output_dir = tmp_path / "ablation"
    summary = run_ablation(
        tmp_path / "model.k3x",
        tmp_path / "k3x_run",
        backend="cuda-custom",
        dense_precision="fp32",
        cuda_resident_bytes=4096,
        warmup=0,
        iterations=1,
        output_dir=output_dir,
    )
    assert observed == [
        ("per-operation", "transient", "scalar", 0),
        ("reused", "transient", "scalar", 0),
        ("reused", "resident", "scalar", 4096),
        ("reused", "resident", "grouped", 4096),
    ]
    assert [item["name"] for item in summary["records"]] == [
        "reference",
        "reuse",
        "residency",
        "grouped",
    ]
    assert summary["deltas"][0]["device_allocation_count"] == 1
    assert (output_dir / "summary.json").is_file()
    for name in ("reference", "reuse", "residency", "grouped"):
        assert (output_dir / f"{name}.json").is_file()
        assert (output_dir / f"{name}.csv").is_file()


def test_cuda_residency_ablation_requires_positive_resident_capacity(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="positive"):
        run_ablation(
            tmp_path / "model.k3x",
            tmp_path / "k3x_run",
            backend="cuda-custom",
            dense_precision="fp32",
            cuda_resident_bytes=0,
            warmup=0,
            iterations=1,
            output_dir=tmp_path / "ablation",
        )


def test_schema_rejects_projected_values_as_measured() -> None:
    with pytest.raises(ValueError, match="synthetic-milestone"):
        BenchmarkRecord(**{**_record().__dict__, "scope": "projected-full-model"})


def test_benchmark_once_collects_cpu_backend_profile(
    synthetic_source: Path, tmp_path: Path
) -> None:
    artifact = tmp_path / "synthetic.k3x"
    convert(synthetic_source, artifact, chunk_bytes=257)
    record = benchmark_once(
        artifact,
        cpp_binary("k3x_run"),
        warmup=0,
        iterations=1,
        backend="cpu",
        dense_precision="fp32",
    )
    assert record.scope == "synthetic-milestone-one"
    assert record.backend == "cpu"
    assert record.device == "CPU"
    assert record.dense_precision == "fp32"
    assert record.cuda_allocation == "per-operation"
    assert record.cuda_weights == "transient"
    assert record.cuda_batching == "scalar"
    assert record.cuda_boundary == "operation"
    assert record.cuda_resident_bytes == 0
    assert record.kernel_nanoseconds == 0
    assert record.host_to_device_bytes == 0
    assert record.weight_h2d_bytes == 0
    assert record.activation_h2d_bytes == 0
    assert record.device_to_host_bytes == 0
    assert record.peak_vram_bytes == 0
    assert record.device_allocation_count == 0
    assert record.device_free_count == 0
    assert record.stream_synchronization_count == 0
    assert record.weight_cache_hits == 0
    assert record.weight_cache_misses == 0
    assert record.weight_cache_bypasses == 0
    assert record.resident_weight_bytes == 0
    assert record.peak_resident_weight_bytes == 0
    assert record.scratch_bytes == 0
    assert record.peak_scratch_bytes == 0
    assert record.grouped_projection_calls == 0
    assert record.grouped_projection_members == 0
    assert record.ffn_block_calls == 0
    assert record.ffn_block_experts == 0
    assert record.max_absolute_error == 0.0
    assert record.max_relative_error == 0.0


@pytest.mark.parametrize(
    ("backend", "dense_precision", "tolerance"),
    [
        ("cuda-dense", "fp32", 1e-4),
        ("cuda-custom", "fp32", 1e-4),
        ("cuda-custom", "bf16", 2e-2),
    ],
)
def test_benchmark_once_measures_cuda_error_against_cpu(
    synthetic_source: Path,
    tmp_path: Path,
    backend: str,
    dense_precision: str,
    tolerance: float,
) -> None:
    if Path(os.environ.get("K3X_BUILD_DIR", "build")).name != "build-cuda":
        pytest.skip("CUDA benchmark contract is exercised only against build-cuda")
    artifact = tmp_path / "synthetic.k3x"
    convert(synthetic_source, artifact, chunk_bytes=257)
    record = benchmark_once(
        artifact,
        cpp_binary("k3x_run"),
        warmup=0,
        iterations=1,
        backend=backend,
        dense_precision=dense_precision,
    )
    assert record.backend == backend
    assert record.device != "CPU"
    assert record.kernel_nanoseconds > 0
    assert record.host_to_device_bytes > 0
    assert record.weight_h2d_bytes > 0
    assert record.activation_h2d_bytes > 0
    assert (
        record.weight_h2d_bytes + record.activation_h2d_bytes
        == record.host_to_device_bytes
    )
    assert record.device_to_host_bytes > 0
    assert record.peak_vram_bytes > 0
    assert record.device_allocation_count > 0
    assert record.device_allocation_count == record.device_free_count
    assert record.stream_synchronization_count > 0
    assert record.scratch_bytes == 0
    assert record.max_absolute_error is not None
    assert record.max_absolute_error <= tolerance
    assert record.max_relative_error is not None
    assert record.max_relative_error >= 0.0


def test_benchmark_once_reports_reused_cuda_scratch(
    synthetic_source: Path, tmp_path: Path
) -> None:
    if Path(os.environ.get("K3X_BUILD_DIR", "build")).name != "build-cuda":
        pytest.skip("CUDA benchmark contract is exercised only against build-cuda")
    artifact = tmp_path / "synthetic.k3x"
    convert(synthetic_source, artifact, chunk_bytes=257)
    record = benchmark_once(
        artifact,
        cpp_binary("k3x_run"),
        warmup=0,
        iterations=1,
        backend="cuda-dense",
        dense_precision="fp32",
        cuda_allocation="reused",
    )
    assert record.cuda_allocation == "reused"
    assert record.device_allocation_count > record.device_free_count
    assert record.scratch_bytes > 0
    assert record.peak_scratch_bytes >= record.scratch_bytes
    assert record.max_absolute_error is not None
    assert record.max_absolute_error <= 1.0e-4
