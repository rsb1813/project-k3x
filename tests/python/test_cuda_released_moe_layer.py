# released-dimension resident MoE-layer CUDA 벤치마크의 CLI 경계를 검증합니다.
import os
import subprocess
from pathlib import Path

import pytest

from conftest import cpp_binary
from k3x_converter.writer import convert
from k3x_ref.storage_fixture import write_bounded_expert_source


def _require_cuda_build() -> Path:
    if Path(os.environ.get("K3X_BUILD_DIR", "build")).name != "build-cuda":
        pytest.skip("released MoE-layer benchmark requires build-cuda")
    return cpp_binary("k3x_cuda_moe_layer_bench")


def _released_artifact(tmp_path: Path) -> Path:
    source = tmp_path / "source"
    write_bounded_expert_source(source, chunk_bytes=257 * 1024)
    artifact = tmp_path / "bounded.k3x"
    convert(source, artifact, chunk_bytes=193 * 1024)
    return artifact


@pytest.mark.parametrize("boundary", ["ffn-block", "moe-layer"])
def test_released_moe_layer_bench_executes(
    boundary: str, tmp_path: Path
) -> None:
    runner = _require_cuda_build()
    artifact = _released_artifact(tmp_path)
    result = subprocess.run(
        [
            str(runner),
            "--model",
            str(artifact),
            "--boundary",
            boundary,
            "--experts",
            "1",
            "--warmup",
            "0",
            "--iterations",
            "1",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (("--boundary", "bogus"), "unknown boundary: bogus"),
        (("--experts", "0"), "experts must be one of 1, 4, or 16"),
        (("--experts", "2"), "experts must be one of 1, 4, or 16"),
        (("--experts", "17"), "experts must be one of 1, 4, or 16"),
        ((), "model path is required"),
        (("--iterations", "0"), "iterations must be positive"),
    ],
)
def test_released_moe_layer_bench_rejects_invalid_arguments(
    arguments: tuple[str, ...], message: str
) -> None:
    runner = _require_cuda_build()
    result = subprocess.run(
        [str(runner), *arguments], capture_output=True, text=True
    )
    assert result.returncode == 2
    assert result.stderr.strip() == message
