# Synthetic benchmark 결과 schema와 measured/projected 구분을 검증합니다.
import csv
import json
from pathlib import Path

import pytest

from tools.benchmark_synthetic import BenchmarkRecord, write_results


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
    with csv_path.open(newline="", encoding="utf-8") as stream:
        row = next(csv.DictReader(stream))
    assert row["decode_tokens_per_second"] == "50.0"
    assert row["per_layer_nanoseconds"] == "1;2;3;4"


def test_schema_rejects_projected_values_as_measured() -> None:
    with pytest.raises(ValueError, match="synthetic-milestone-zero"):
        BenchmarkRecord(**{**_record().__dict__, "scope": "projected-full-model"})
