# 독립 C++ runtime의 greedy token이 PyTorch golden과 일치하는지 검증합니다.
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np
import torch

from k3x_converter.reader import K3XReader
from k3x_converter.writer import convert
from k3x_ref.fixtures import build_synthetic_model


import pytest

from conftest import cpp_binary


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--backend", "warp"], "unknown backend: warp"),
        (["--dense-precision", "fp8"], "unknown dense precision: fp8"),
    ],
)
def test_cpp_runner_rejects_unknown_backend_values(
    arguments: list[str], message: str
) -> None:
    result = subprocess.run(
        [str(cpp_binary("k3x_run")), *arguments],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert result.stderr.strip() == message


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (["--cuda-allocation", "pool"], "unknown CUDA allocation mode: pool"),
        (["--cuda-weights", "lru"], "unknown CUDA weight mode: lru"),
        (["--cuda-batching", "graph"], "unknown CUDA batching mode: graph"),
        (
            ["--cuda-resident-bytes", "-1"],
            "invalid CUDA resident byte capacity: -1",
        ),
    ],
)
def test_cpp_runner_rejects_invalid_cuda_execution_options(
    arguments: list[str], message: str
) -> None:
    result = subprocess.run(
        [str(cpp_binary("k3x_run")), *arguments],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert result.stderr.strip() == message


@pytest.mark.parametrize(
    "arguments",
    [
        ["--backend", "cpu", "--cuda-allocation", "reused"],
        ["--backend", "cpu", "--cuda-weights", "resident", "--cuda-resident-bytes", "1"],
        ["--backend", "cpu", "--cuda-batching", "grouped"],
        ["--backend", "cpu", "--cuda-resident-bytes", "1"],
    ],
)
def test_cpp_runner_rejects_cuda_execution_options_for_cpu(
    arguments: list[str],
) -> None:
    result = subprocess.run(
        [str(cpp_binary("k3x_run")), *arguments],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert result.stderr.strip() == "CUDA execution options require a CUDA backend"


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (
            ["--backend", "cuda-dense", "--cuda-weights", "resident"],
            "resident CUDA weights require a positive resident byte capacity",
        ),
        (
            ["--backend", "cuda-dense", "--cuda-resident-bytes", "1"],
            "transient CUDA weights require a zero resident byte capacity",
        ),
    ],
)
def test_cpp_runner_rejects_invalid_cuda_weight_capacity_combinations(
    arguments: list[str], message: str
) -> None:
    result = subprocess.run(
        [str(cpp_binary("k3x_run")), *arguments],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert result.stderr.strip() == message


def test_cpu_build_reports_explicit_cuda_request_as_unavailable() -> None:
    if Path(os.environ.get("K3X_BUILD_DIR", "build")).name != "build-cpu":
        pytest.skip("CPU-build contract is exercised only against build-cpu")
    result = subprocess.run(
        [str(cpp_binary("k3x_run")), "--backend", "cuda-custom"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 4
    assert result.stderr.startswith("BACKEND_UNAVAILABLE")


def test_cpp_runner_rejects_bf16_for_cpu_backend() -> None:
    result = subprocess.run(
        [
            str(cpp_binary("k3x_run")),
            "--backend",
            "cpu",
            "--dense-precision",
            "bf16",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert result.stderr.strip() == "bf16 dense precision requires a CUDA backend"


@pytest.mark.parametrize("mode", ["incremental", "full"])
def test_cpp_generation_matches_python_golden(
    synthetic_source: Path, tmp_path: Path, mode: str
) -> None:
    runner = cpp_binary("k3x_run")
    assert runner.exists(), "build k3x_run before running cross-language parity"
    artifact = tmp_path / "synthetic.k3x"
    output = tmp_path / "result.json"
    convert(synthetic_source, artifact, chunk_bytes=257)
    subprocess.run(
        [str(runner), "--model", str(artifact), "--prompt-ids", "1,7,3,9",
         "--generate", "6", "--mode", mode, "--json", str(output)],
        check=True,
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    expected = build_synthetic_model().generate_greedy(
        [1, 7, 3, 9], 6, mode == "incremental"
    )
    assert result["token_ids"] == expected
    assert result["read_bytes"] > 0
    assert result["decode_nanoseconds"] > 0
    assert result["backend"] == "cpu"
    assert result["device"] == "CPU"
    assert result["dense_precision"] == "fp32"
    assert result["cuda_allocation"] == "per-operation"
    assert result["cuda_weights"] == "transient"
    assert result["cuda_batching"] == "scalar"
    assert result["cuda_resident_bytes"] == 0
    assert result["device_allocation_count"] == 0
    assert result["weight_cache_hits"] == 0


def test_cpp_prefill_layers_logits_and_state_match_python(
    synthetic_source: Path, tmp_path: Path
) -> None:
    runner = cpp_binary("k3x_run")
    artifact = tmp_path / "synthetic.k3x"
    output = tmp_path / "result.json"
    convert(synthetic_source, artifact, chunk_bytes=257)
    subprocess.run(
        [str(runner), "--model", str(artifact), "--prompt-ids", "1,7,3,9",
         "--generate", "1", "--mode", "incremental", "--diagnostics", "true",
         "--json", str(output)],
        check=True,
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    model = build_synthetic_model()
    expected_logits, expected_state, expected_layers = model.prefill_with_trace(
        torch.tensor([[1, 7, 3, 9]], dtype=torch.long)
    )
    np.testing.assert_allclose(
        result["prefill_logits"], expected_logits.numpy().reshape(-1), atol=1e-6, rtol=1e-6
    )
    for actual, expected in zip(
        result["prefill_layer_outputs"], expected_layers, strict=True
    ):
        np.testing.assert_allclose(
            actual, expected.numpy().reshape(-1), atol=1e-6, rtol=1e-6
        )
    state_values: list[np.ndarray] = []
    for state in expected_state.attention:
        tensors = (
            (state.conv_q, state.conv_k, state.conv_v, state.recurrent)
            if hasattr(state, "recurrent")
            else (state.keys, state.values, state.shared_keys)
        )
        state_values.extend(tensor.numpy().reshape(-1) for tensor in tensors)
    np.testing.assert_allclose(
        result["prefill_state"], np.concatenate(state_values), atol=1e-6, rtol=1e-6
    )


@pytest.mark.parametrize(
    (
        "backend",
        "dense_precision",
        "cuda_allocation",
        "cuda_weights",
        "cuda_batching",
        "cuda_resident_bytes",
        "tolerance",
    ),
    [
        (backend, "fp32", allocation, weights, batching,
         8 * 1024 * 1024 if weights == "resident" else 0,
         1e-5 if backend == "cuda-dense" else 1e-4)
        for backend in ("cuda-dense", "cuda-custom")
        for allocation in ("per-operation", "reused")
        for weights in ("transient", "resident")
        for batching in ("scalar", "grouped")
    ]
    + [
        ("cuda-dense", "bf16", "reused", "resident", "grouped",
         8 * 1024 * 1024, 2e-2),
        ("cuda-custom", "bf16", "reused", "resident", "grouped",
         8 * 1024 * 1024, 2e-2),
    ],
)
def test_cuda_backends_match_synthetic_graph_and_tokens(
    synthetic_source: Path,
    tmp_path: Path,
    backend: str,
    dense_precision: str,
    cuda_allocation: str,
    cuda_weights: str,
    cuda_batching: str,
    cuda_resident_bytes: int,
    tolerance: float,
) -> None:
    if Path(os.environ.get("K3X_BUILD_DIR", "build")).name != "build-cuda":
        pytest.skip("CUDA parity is exercised only against build-cuda")
    runner = cpp_binary("k3x_run")
    artifact = tmp_path / "synthetic.k3x"
    output = tmp_path / (
        f"{backend}-{dense_precision}-{cuda_allocation}-"
        f"{cuda_weights}-{cuda_batching}.json"
    )
    convert(synthetic_source, artifact, chunk_bytes=257)
    subprocess.run(
        [
            str(runner),
            "--model",
            str(artifact),
            "--prompt-ids",
            "1,7,3,9",
            "--generate",
            "6",
            "--mode",
            "incremental",
            "--diagnostics",
            "true",
            "--backend",
            backend,
            "--dense-precision",
            dense_precision,
            "--cuda-allocation",
            cuda_allocation,
            "--cuda-weights",
            cuda_weights,
            "--cuda-batching",
            cuda_batching,
            "--cuda-resident-bytes",
            str(cuda_resident_bytes),
            "--json",
            str(output),
        ],
        check=True,
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    model = build_synthetic_model()
    expected_tokens = model.generate_greedy([1, 7, 3, 9], 6, True)
    expected_logits, expected_state, expected_layers = model.prefill_with_trace(
        torch.tensor([[1, 7, 3, 9]], dtype=torch.long)
    )
    assert result["token_ids"] == expected_tokens == [43, 32, 28, 49, 9, 28]
    assert result["backend"] == backend
    assert result["device"] != "CPU"
    assert result["dense_precision"] == dense_precision
    assert result["cuda_allocation"] == cuda_allocation
    assert result["cuda_weights"] == cuda_weights
    assert result["cuda_batching"] == cuda_batching
    assert result["cuda_resident_bytes"] == cuda_resident_bytes
    assert result["kernel_nanoseconds"] > 0
    assert result["host_to_device_bytes"] > 0
    assert result["device_to_host_bytes"] > 0
    assert result["peak_vram_bytes"] > 0
    assert result["failed_operations"] == 0
    if cuda_batching == "grouped":
        assert result["grouped_projection_calls"] > 0
        assert result["grouped_projection_members"] > result["grouped_projection_calls"]
    if cuda_weights == "resident":
        assert result["weight_cache_misses"] > 0
        assert result["resident_weight_bytes"] <= cuda_resident_bytes
    np.testing.assert_allclose(
        result["prefill_logits"],
        expected_logits.numpy().reshape(-1),
        atol=tolerance,
        rtol=tolerance,
    )
    for actual, expected in zip(
        result["prefill_layer_outputs"], expected_layers, strict=True
    ):
        np.testing.assert_allclose(
            actual,
            expected.numpy().reshape(-1),
            atol=tolerance,
            rtol=tolerance,
        )
    state_values: list[np.ndarray] = []
    for state in expected_state.attention:
        tensors = (
            (state.conv_q, state.conv_k, state.conv_v, state.recurrent)
            if hasattr(state, "recurrent")
            else (state.keys, state.values, state.shared_keys)
        )
        state_values.extend(tensor.numpy().reshape(-1) for tensor in tensors)
    np.testing.assert_allclose(
        result["prefill_state"],
        np.concatenate(state_values),
        atol=tolerance,
        rtol=tolerance,
    )


def test_cpp_runner_rejects_corrupt_model_before_generation(
    synthetic_source: Path, tmp_path: Path
) -> None:
    runner = cpp_binary("k3x_run")
    valid = tmp_path / "valid.k3x"
    corrupt = tmp_path / "corrupt.k3x"
    convert(synthetic_source, valid, chunk_bytes=257)
    shutil.copyfile(valid, corrupt)
    first = K3XReader.open(valid).tensor_records[0]
    with corrupt.open("r+b") as stream:
        stream.seek(first.data_offset)
        value = stream.read(1)
        stream.seek(first.data_offset)
        stream.write(bytes([value[0] ^ 1]))
    result = subprocess.run(
        [str(runner), "--model", str(corrupt), "--prompt-ids", "1,7,3,9",
         "--generate", "1", "--mode", "incremental", "--json", str(tmp_path / "x.json")],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 3
    assert result.stderr.strip() == "DATA_CRC_MISMATCH"


def test_cpp_first_generated_token_is_not_counted_as_decode(
    synthetic_source: Path, tmp_path: Path
) -> None:
    runner = cpp_binary("k3x_run")
    artifact = tmp_path / "synthetic.k3x"
    output = tmp_path / "result.json"
    convert(synthetic_source, artifact, chunk_bytes=257)
    subprocess.run(
        [str(runner), "--model", str(artifact), "--prompt-ids", "1,7,3,9",
         "--generate", "1", "--mode", "incremental", "--json", str(output)],
        check=True,
    )
    result = json.loads(output.read_text(encoding="utf-8"))
    assert result["token_ids"] == [43]
    assert result["decode_nanoseconds"] == 0
