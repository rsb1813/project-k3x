# L1 expert cache ablation의 축 격리와 exact acceptance 계약을 검증합니다.
import os
from pathlib import Path

import pytest

from conftest import cpp_binary
from k3x_converter.writer import convert
from tools.ablate_l1_expert_cache import l1_cache_matrix, run_l1_cache_ablation


def test_l1_cache_matrix_crosses_only_cache_and_transfer() -> None:
    assert l1_cache_matrix(65536, 1024) == (
        {
            "name": "disabled-synchronous",
            "l1_expert_cache": "disabled",
            "l1_expert_cache_bytes": 0,
            "cuda_transfer": "synchronous",
            "cuda_pinned_bytes": 0,
        },
        {
            "name": "static-synchronous",
            "l1_expert_cache": "static",
            "l1_expert_cache_bytes": 65536,
            "cuda_transfer": "synchronous",
            "cuda_pinned_bytes": 0,
        },
        {
            "name": "disabled-prefetch",
            "l1_expert_cache": "disabled",
            "l1_expert_cache_bytes": 0,
            "cuda_transfer": "prefetch",
            "cuda_pinned_bytes": 1024,
        },
        {
            "name": "static-prefetch",
            "l1_expert_cache": "static",
            "l1_expert_cache_bytes": 65536,
            "cuda_transfer": "prefetch",
            "cuda_pinned_bytes": 1024,
        },
    )


def test_l1_cache_ablation_runs_exact_cuda_matrix(
    synthetic_source: Path, tmp_path: Path
) -> None:
    if Path(os.environ.get("K3X_BUILD_DIR", "build")).name != "build-cuda":
        pytest.skip("L1 CUDA ablation is exercised only against build-cuda")
    artifact = tmp_path / "synthetic.k3x"
    convert(synthetic_source, artifact, chunk_bytes=257)
    summary = run_l1_cache_ablation(
        artifact,
        cpp_binary("k3x_run"),
        dense_precision="fp32",
        l1_expert_cache_bytes=65536,
        cuda_pinned_bytes=1024 * 1024,
        warmup=0,
        iterations=1,
        output_dir=tmp_path / "ablation",
    )
    assert len(summary["records"]) == 4
    assert len(summary["deltas"]) == 2
    for record in summary["records"]:
        assert record["parity_status"] == "exact"
        assert (tmp_path / "ablation" / f"{record['name']}.json").is_file()
        assert (tmp_path / "ablation" / f"{record['name']}.csv").is_file()

