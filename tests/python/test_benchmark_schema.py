# Synthetic benchmark 결과 schema와 measured/projected 구분을 검증합니다.
import csv
import json
import os
from pathlib import Path

import pytest

from conftest import cpp_binary
from k3x_converter.writer import convert
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
        kernel_nanoseconds=0,
        host_to_device_bytes=0,
        device_to_host_bytes=0,
        peak_vram_bytes=None,
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
    assert payload["peak_vram_bytes"] is None
    with csv_path.open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert row["decode_tokens_per_second"] == "50.0"
    assert row["host_to_device_bytes"] == "0"
    assert row["peak_vram_bytes"] == ""
    assert row["per_layer_nanoseconds"] == "1;2;3;4"


def test_schema_accepts_explicit_milestone_one_scope() -> None:
    record = BenchmarkRecord(
        **{**_record().__dict__, "scope": "synthetic-milestone-one"}
    )
    assert record.scope == "synthetic-milestone-one"


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
    assert record.kernel_nanoseconds == 0
    assert record.host_to_device_bytes == 0
    assert record.device_to_host_bytes == 0
    assert record.peak_vram_bytes == 0
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
    assert record.device_to_host_bytes > 0
    assert record.peak_vram_bytes > 0
    assert record.max_absolute_error is not None
    assert record.max_absolute_error <= tolerance
    assert record.max_relative_error is not None
    assert record.max_relative_error >= 0.0
