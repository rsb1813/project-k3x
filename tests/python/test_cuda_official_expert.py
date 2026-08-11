# 공식 Kimi K3 expert 전용 CUDA harness의 CLI, identity, 수치, traffic 계약을 검증합니다.
from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

from conftest import cpp_binary
from k3x_converter.writer import convert
from k3x_ref.storage_fixture import write_bounded_expert_source


def _runner() -> Path:
    if Path(os.environ.get("K3X_BUILD_DIR", "build")).name != "build-cuda":
        pytest.skip("official expert benchmark requires build-cuda")
    return cpp_binary("k3x_cuda_official_expert_bench")


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        ((), "model path is required"),
        (("--weight-mode", "other"), "unknown weight mode: other"),
        (("--iterations", "0"), "iterations must be positive"),
    ],
)
def test_official_expert_bench_rejects_invalid_arguments(
    arguments: tuple[str, ...], message: str
) -> None:
    result = subprocess.run(
        [str(_runner()), *arguments], capture_output=True, text=True
    )
    assert result.returncode == 2
    assert result.stderr.strip() == message


def test_official_expert_bench_rejects_synthetic_fixture_before_cuda(
    tmp_path: Path,
) -> None:
    source = tmp_path / "source"
    write_bounded_expert_source(source, chunk_bytes=257 * 1024)
    artifact = tmp_path / "synthetic.k3x"
    convert(source, artifact, chunk_bytes=193 * 1024)

    result = subprocess.run(
        [
            str(_runner()),
            "--model",
            str(artifact),
            "--weight-mode",
            "transient",
            "--warmup",
            "0",
            "--iterations",
            "1",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 4
    assert result.stderr.strip() == (
        "INVALID_MXFP4: official Kimi K3 expert identity mismatch"
    )


@pytest.mark.parametrize("weight_mode", ["transient", "resident"])
def test_official_expert_bench_executes_exact_real_artifact(
    weight_mode: str,
) -> None:
    artifact_value = os.environ.get("K3X_TEST_OFFICIAL_EXPERT")
    if artifact_value is None:
        pytest.skip("set K3X_TEST_OFFICIAL_EXPERT for the ignored real artifact")
    artifact = Path(artifact_value)
    result = subprocess.run(
        [
            str(_runner()),
            "--model",
            str(artifact),
            "--weight-mode",
            weight_mode,
            "--warmup",
            "0",
            "--iterations",
            "1",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    record = json.loads(result.stdout)
    assert record["artifact_kind"] == "official_kimi_k3_expert"
    assert record["repository"] == "moonshotai/Kimi-K3"
    assert record["resolved_revision"] == (
        "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"
    )
    assert record["token_semantics"] is False
    assert record["routing_semantics"] is False
    assert record["full_moe_layer"] is False
    assert record["layer_id"] == 1
    assert record["expert_id"] == 0
    assert record["weight_mode"] == weight_mode
    assert record["k3x_root_sha256"] == (
        "d585d283325e13e1316a0194c2d6274dd89ef75a28b96b02f02733290b7658be"
    )
    assert record["ordered_sha256"] == (
        "4e23bd960dfb5e8b10def10e12a94bac1119500f72918698986bd332d56d33ff"
    )
    assert record["expert_payload_bytes"] == 17_547_264
    assert record["input_elements"] == 3_584
    assert record["output_elements"] == 3_584
    assert record["warmup"] == 0
    assert record["iterations"] == 1
    assert record["cpu_oracle_nanoseconds"] > 0
    assert record["cold_latency_nanoseconds"] > 0
    assert record["cold_kernel_nanoseconds"] > 0
    assert record["latency_nanoseconds_median"] > 0
    assert record["latency_nanoseconds_p05"] > 0
    assert record["latency_nanoseconds_p95"] > 0
    assert record["kernel_nanoseconds"] > 0
    assert record["activation_h2d_bytes"] == 14_336
    assert record["device_to_host_bytes"] == 14_336
    assert record["maximum_absolute_error"] <= 1.0e-6
    assert record["all_finite"] is True
    assert record["weight_cache_bypasses"] == 0
    assert record["peak_vram_bytes"] > 0
    if weight_mode == "transient":
        assert record["cold_weight_h2d_bytes"] == 17_547_264
        assert record["weight_h2d_bytes"] == 17_547_264
        assert record["resident_weight_bytes"] == 0
    else:
        assert record["cold_weight_h2d_bytes"] == 17_547_264
        assert record["weight_h2d_bytes"] == 0
        assert record["resident_weight_bytes"] == 17_547_264
        assert record["peak_resident_weight_bytes"] == 17_547_264
