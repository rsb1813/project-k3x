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


def cpu_only_build() -> bool:
    return cpp_binary("test_backend_unavailable").is_file()


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
        (["--cuda-boundary", "layer"], "unknown CUDA boundary mode: layer"),
        (["--cuda-transfer", "queue"], "unknown CUDA transfer mode: queue"),
        (
            ["--cuda-resident-bytes", "-1"],
            "invalid CUDA resident byte capacity: -1",
        ),
        (
            ["--cuda-pinned-bytes", "-1"],
            "invalid CUDA pinned byte capacity: -1",
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
    ("arguments", "message"),
    [
        (["--l1-expert-cache", "clock"], "unknown L1 expert cache mode: clock"),
        (
            ["--l1-expert-cache-bytes", "-1"],
            "invalid L1 expert cache byte capacity: -1",
        ),
        (
            ["--l1-expert-cache", "static"],
            "static L1 expert cache requires a positive byte capacity",
        ),
        (
            ["--l1-expert-cache-bytes", "1"],
            "disabled L1 expert cache requires a zero byte capacity",
        ),
    ],
)
def test_cpp_runner_rejects_invalid_l1_expert_cache_options(
    arguments: list[str], message: str
) -> None:
    result = subprocess.run(
        [str(cpp_binary("k3x_run")), *arguments],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert result.stderr.strip() == message


def test_cpp_runner_accepts_static_l1_expert_cache_for_cpu() -> None:
    result = subprocess.run(
        [
            str(cpp_binary("k3x_run")),
            "--l1-expert-cache",
            "static",
            "--l1-expert-cache-bytes",
            "65536",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 3


@pytest.mark.parametrize(
    "arguments",
    [
        ["--backend", "cpu", "--cuda-allocation", "reused"],
        ["--backend", "cpu", "--cuda-weights", "resident", "--cuda-resident-bytes", "1"],
        ["--backend", "cpu", "--cuda-batching", "grouped"],
        ["--backend", "cpu", "--cuda-resident-bytes", "1"],
        ["--backend", "cpu", "--cuda-transfer", "prefetch"],
        ["--backend", "cpu", "--cuda-pinned-bytes", "1"],
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


@pytest.mark.parametrize("backend", ["cpu", "cuda-dense"])
def test_cpp_runner_rejects_ffn_block_boundary_without_custom_cuda(
    backend: str,
) -> None:
    result = subprocess.run(
        [
            str(cpp_binary("k3x_run")),
            "--backend",
            backend,
            "--cuda-boundary",
            "ffn-block",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert result.stderr.strip() == "ffn-block boundary requires cuda-custom"


@pytest.mark.parametrize(
    ("arguments", "message"),
    [
        (
            ["--backend", "cuda-custom", "--cuda-pinned-bytes", "1"],
            "synchronous CUDA transfer requires a zero pinned byte capacity",
        ),
        (
            ["--backend", "cuda-custom", "--cuda-transfer", "prefetch"],
            "prefetch CUDA transfer requires a positive pinned byte capacity",
        ),
        (
            [
                "--backend", "cuda-dense", "--cuda-transfer", "prefetch",
                "--cuda-pinned-bytes", "1",
            ],
            "prefetch CUDA transfer requires cuda-custom",
        ),
        (
            [
                "--backend", "cuda-custom", "--cuda-transfer", "prefetch",
                "--cuda-pinned-bytes", "1",
            ],
            "prefetch CUDA transfer requires ffn-block boundary",
        ),
        (
            [
                "--backend", "cuda-custom", "--cuda-boundary", "ffn-block",
                "--cuda-transfer", "prefetch", "--cuda-pinned-bytes", "1",
            ],
            "prefetch CUDA transfer requires reused allocation",
        ),
        (
            [
                "--backend", "cuda-custom", "--cuda-boundary", "ffn-block",
                "--cuda-allocation", "reused", "--cuda-weights", "resident",
                "--cuda-resident-bytes", "1", "--cuda-transfer", "prefetch",
                "--cuda-pinned-bytes", "1",
            ],
            "prefetch CUDA transfer requires transient weights",
        ),
    ],
)
def test_cpp_runner_rejects_invalid_cuda_transfer_combinations(
    arguments: list[str], message: str
) -> None:
    result = subprocess.run(
        [str(cpp_binary("k3x_run")), *arguments],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert result.stderr.strip() == message


def test_cpp_runner_accepts_exact_cuda_prefetch_capability_combination() -> None:
    result = subprocess.run(
        [
            str(cpp_binary("k3x_run")),
            "--backend", "cuda-custom",
            "--cuda-boundary", "ffn-block",
            "--cuda-allocation", "reused",
            "--cuda-weights", "transient",
            "--cuda-transfer", "prefetch",
            "--cuda-pinned-bytes", "1048576",
        ],
        capture_output=True,
        text=True,
    )
    if cpu_only_build():
        assert result.returncode == 4
        assert result.stderr.startswith("BACKEND_UNAVAILABLE")
    else:
        assert result.returncode == 3


def test_cpu_build_reports_explicit_cuda_request_as_unavailable() -> None:
    if not cpu_only_build():
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


def test_static_l1_cache_preserves_cpu_graph_and_avoids_reader_calls(
    synthetic_source: Path, tmp_path: Path
) -> None:
    runner = cpp_binary("k3x_run")
    artifact = tmp_path / "synthetic.k3x"
    convert(synthetic_source, artifact, chunk_bytes=257)
    results: dict[str, dict] = {}
    cases = (
        ("disabled", "disabled", 0),
        ("static", "static", 65536),
        ("static-tiny", "static", 1),
    )
    for name, cache_mode, capacity in cases:
        output = tmp_path / f"cpu-{name}.json"
        subprocess.run(
            [
                str(runner),
                "--model", str(artifact),
                "--prompt-ids", "1,7,3,9",
                "--generate", "6",
                "--mode", "incremental",
                "--diagnostics", "true",
                "--l1-expert-cache", cache_mode,
                "--l1-expert-cache-bytes", str(capacity),
                "--json", str(output),
            ],
            check=True,
        )
        results[name] = json.loads(output.read_text(encoding="utf-8"))

    disabled = results["disabled"]
    static = results["static"]
    tiny = results["static-tiny"]
    assert static["token_ids"] == disabled["token_ids"] == [43, 32, 28, 49, 9, 28]
    assert static["prefill_routed_experts"] == disabled["prefill_routed_experts"]
    np.testing.assert_allclose(static["prefill_logits"], disabled["prefill_logits"])
    assert disabled["l1_expert_cache_hits"] == 0
    assert disabled["l1_expert_cache_misses"] == 0
    assert disabled["l1_expert_cache_resident_bytes"] == 0
    assert static["l1_expert_cache_hits"] > 0
    assert static["l1_expert_cache_misses"] > 0
    assert static["l1_expert_cache_bypasses"] == 0
    assert 0 < static["l1_expert_cache_resident_bytes"] <= 65536
    assert static["read_calls"] < disabled["read_calls"]
    assert static["read_bytes"] < disabled["read_bytes"]
    assert static["reader_read_calls"] == static["read_calls"]
    assert static["reader_completed_bytes"] == static["read_bytes"]
    assert static["reader_requested_bytes"] >= static["reader_completed_bytes"]
    assert tiny["token_ids"] == disabled["token_ids"]
    assert tiny["prefill_routed_experts"] == disabled["prefill_routed_experts"]
    assert tiny["l1_expert_cache_hits"] == 0
    assert tiny["l1_expert_cache_misses"] > 0
    assert tiny["l1_expert_cache_bypasses"] == tiny["l1_expert_cache_misses"]
    assert tiny["l1_expert_cache_resident_bytes"] == 0
    assert tiny["read_calls"] == disabled["read_calls"]
    assert tiny["read_bytes"] == disabled["read_bytes"]


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


@pytest.mark.parametrize(
    ("dense_precision", "cuda_batching", "tolerance"),
    [
        ("fp32", "scalar", 1e-4),
        ("fp32", "grouped", 1e-4),
        ("bf16", "scalar", 2e-2),
        ("bf16", "grouped", 2e-2),
    ],
)
def test_cuda_ffn_block_matches_operation_graph_and_routing(
    synthetic_source: Path,
    tmp_path: Path,
    dense_precision: str,
    cuda_batching: str,
    tolerance: float,
) -> None:
    if Path(os.environ.get("K3X_BUILD_DIR", "build")).name != "build-cuda":
        pytest.skip("CUDA FFN parity is exercised only against build-cuda")
    runner = cpp_binary("k3x_run")
    artifact = tmp_path / "synthetic.k3x"
    convert(synthetic_source, artifact, chunk_bytes=257)

    results: dict[str, dict] = {}
    cases = (
        ("operation", "operation", "resident", 8 * 1024 * 1024, "synchronous", 0),
        ("synchronous", "ffn-block", "transient", 0, "synchronous", 0),
        ("prefetch", "ffn-block", "transient", 0, "prefetch", 1024 * 1024),
    )
    for name, boundary, weights, resident_bytes, transfer, pinned_bytes in cases:
        output = tmp_path / f"{name}-{dense_precision}-{cuda_batching}.json"
        subprocess.run(
            [
                str(runner),
                "--model", str(artifact),
                "--prompt-ids", "1,7,3,9",
                "--generate", "6",
                "--mode", "incremental",
                "--diagnostics", "true",
                "--backend", "cuda-custom",
                "--dense-precision", dense_precision,
                "--cuda-allocation", "reused",
                "--cuda-weights", weights,
                "--cuda-batching", cuda_batching,
                "--cuda-boundary", boundary,
                "--cuda-resident-bytes", str(resident_bytes),
                "--cuda-transfer", transfer,
                "--cuda-pinned-bytes", str(pinned_bytes),
                "--json", str(output),
            ],
            check=True,
        )
        results[name] = json.loads(output.read_text(encoding="utf-8"))

    reference = results["operation"]
    synchronous = results["synchronous"]
    prefetch = results["prefetch"]
    for name, candidate in (("synchronous", synchronous), ("prefetch", prefetch)):
        assert candidate["cuda_boundary"] == "ffn-block"
        assert candidate["cuda_transfer"] == name
        assert candidate["token_ids"] == reference["token_ids"] == [43, 32, 28, 49, 9, 28]
        assert candidate["prefill_routed_experts"] == reference["prefill_routed_experts"]
        assert candidate["ffn_block_calls"] > 0
        assert candidate["ffn_block_experts"] > 0
        np.testing.assert_allclose(
            candidate["prefill_logits"], reference["prefill_logits"],
            atol=tolerance, rtol=tolerance,
        )
        for actual, expected in zip(
            candidate["prefill_layer_outputs"],
            reference["prefill_layer_outputs"],
            strict=True,
        ):
            np.testing.assert_allclose(
                actual, expected, atol=tolerance, rtol=tolerance
            )
        np.testing.assert_allclose(
            candidate["prefill_state"], reference["prefill_state"],
            atol=tolerance, rtol=tolerance,
        )
    assert synchronous["cuda_pinned_bytes"] == 0
    assert synchronous["pinned_host_bytes"] == 0
    assert synchronous["async_prefetch_calls"] == 0
    assert synchronous["async_prefetch_bytes"] == 0
    assert synchronous["transfer_stream_wait_count"] == 0
    assert prefetch["cuda_pinned_bytes"] == 1024 * 1024
    assert prefetch["pinned_host_bytes"] == 1024 * 1024
    assert prefetch["async_prefetch_calls"] > 0
    assert prefetch["async_prefetch_bytes"] > 0
    assert prefetch["transfer_stream_wait_count"] == prefetch["async_prefetch_calls"]
    assert prefetch["stream_synchronization_count"] <= synchronous[
        "stream_synchronization_count"
    ]


@pytest.mark.parametrize(
    ("dense_precision", "tolerance"), [("fp32", 1e-4), ("bf16", 2e-2)]
)
def test_static_l1_cache_preserves_cuda_ffn_block_and_exact_bypass(
    synthetic_source: Path,
    tmp_path: Path,
    dense_precision: str,
    tolerance: float,
) -> None:
    if Path(os.environ.get("K3X_BUILD_DIR", "build")).name != "build-cuda":
        pytest.skip("CUDA L1 parity is exercised only against build-cuda")
    runner = cpp_binary("k3x_run")
    artifact = tmp_path / "synthetic.k3x"
    convert(synthetic_source, artifact, chunk_bytes=257)
    results: dict[str, dict] = {}
    for name, cache_mode, capacity in (
        ("disabled", "disabled", 0),
        ("static", "static", 65536),
        ("static-tiny", "static", 1),
    ):
        output = tmp_path / f"cuda-{dense_precision}-{name}.json"
        subprocess.run(
            [
                str(runner),
                "--model", str(artifact),
                "--prompt-ids", "1,7,3,9",
                "--generate", "6",
                "--mode", "incremental",
                "--diagnostics", "true",
                "--backend", "cuda-custom",
                "--dense-precision", dense_precision,
                "--cuda-allocation", "reused",
                "--cuda-weights", "transient",
                "--cuda-batching", "scalar",
                "--cuda-boundary", "ffn-block",
                "--cuda-transfer", "synchronous",
                "--l1-expert-cache", cache_mode,
                "--l1-expert-cache-bytes", str(capacity),
                "--json", str(output),
            ],
            check=True,
        )
        results[name] = json.loads(output.read_text(encoding="utf-8"))

    disabled = results["disabled"]
    static = results["static"]
    tiny = results["static-tiny"]
    for candidate in (static, tiny):
        assert candidate["token_ids"] == disabled["token_ids"]
        assert candidate["prefill_routed_experts"] == disabled[
            "prefill_routed_experts"
        ]
        np.testing.assert_allclose(
            candidate["prefill_logits"], disabled["prefill_logits"],
            atol=tolerance, rtol=tolerance,
        )
        assert candidate["host_to_device_bytes"] == disabled[
            "host_to_device_bytes"
        ]
        assert candidate["ffn_block_calls"] == disabled["ffn_block_calls"]
        assert candidate["ffn_block_experts"] == disabled["ffn_block_experts"]
    assert static["l1_expert_cache_hits"] > 0
    assert static["l1_expert_cache_misses"] > 0
    assert static["l1_expert_cache_bypasses"] == 0
    assert static["read_calls"] < disabled["read_calls"]
    assert static["read_bytes"] < disabled["read_bytes"]
    assert tiny["l1_expert_cache_hits"] == 0
    assert tiny["l1_expert_cache_bypasses"] == tiny["l1_expert_cache_misses"]
    assert tiny["read_calls"] == disabled["read_calls"]
    assert tiny["read_bytes"] == disabled["read_bytes"]


@pytest.mark.parametrize(
    ("dense_precision", "cuda_batching", "tolerance"),
    [
        ("fp32", "scalar", 1e-4),
        ("fp32", "grouped", 1e-4),
        ("bf16", "scalar", 2e-2),
        ("bf16", "grouped", 2e-2),
    ],
)
def test_static_l1_cache_feeds_exact_cuda_prefetch(
    synthetic_source: Path,
    tmp_path: Path,
    dense_precision: str,
    cuda_batching: str,
    tolerance: float,
) -> None:
    if Path(os.environ.get("K3X_BUILD_DIR", "build")).name != "build-cuda":
        pytest.skip("CUDA L1 prefetch parity is exercised only against build-cuda")
    runner = cpp_binary("k3x_run")
    artifact = tmp_path / "synthetic.k3x"
    convert(synthetic_source, artifact, chunk_bytes=257)
    results: dict[str, dict] = {}
    for name, cache_mode, capacity in (
        ("disabled", "disabled", 0),
        ("static", "static", 65536),
        ("static-tiny", "static", 1),
    ):
        output = tmp_path / f"prefetch-{dense_precision}-{cuda_batching}-{name}.json"
        subprocess.run(
            [
                str(runner),
                "--model", str(artifact),
                "--prompt-ids", "1,7,3,9",
                "--generate", "6",
                "--mode", "incremental",
                "--diagnostics", "true",
                "--backend", "cuda-custom",
                "--dense-precision", dense_precision,
                "--cuda-allocation", "reused",
                "--cuda-weights", "transient",
                "--cuda-batching", cuda_batching,
                "--cuda-boundary", "ffn-block",
                "--cuda-transfer", "prefetch",
                "--cuda-pinned-bytes", str(1024 * 1024),
                "--l1-expert-cache", cache_mode,
                "--l1-expert-cache-bytes", str(capacity),
                "--json", str(output),
            ],
            check=True,
        )
        results[name] = json.loads(output.read_text(encoding="utf-8"))

    disabled = results["disabled"]
    static = results["static"]
    tiny = results["static-tiny"]
    for candidate in (static, tiny):
        assert candidate["token_ids"] == disabled["token_ids"]
        assert candidate["prefill_routed_experts"] == disabled[
            "prefill_routed_experts"
        ]
        np.testing.assert_allclose(
            candidate["prefill_logits"], disabled["prefill_logits"],
            atol=tolerance, rtol=tolerance,
        )
        for field in (
            "host_to_device_bytes",
            "weight_h2d_bytes",
            "activation_h2d_bytes",
            "async_prefetch_calls",
            "async_prefetch_bytes",
            "transfer_stream_wait_count",
            "ffn_block_calls",
            "ffn_block_experts",
        ):
            assert candidate[field] == disabled[field]
        assert candidate["async_prefetch_calls"] > 0
        assert candidate["transfer_stream_wait_count"] == candidate[
            "async_prefetch_calls"
        ]
    assert static["l1_expert_cache_hits"] > 0
    assert static["l1_expert_cache_misses"] > 0
    assert static["l1_expert_cache_bypasses"] == 0
    assert static["read_calls"] < disabled["read_calls"]
    assert static["read_bytes"] < disabled["read_bytes"]
    assert tiny["l1_expert_cache_hits"] == 0
    assert tiny["l1_expert_cache_bypasses"] == tiny["l1_expert_cache_misses"]
    assert tiny["read_calls"] == disabled["read_calls"]
    assert tiny["read_bytes"] == disabled["read_bytes"]


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
