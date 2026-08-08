# 독립 C++ runtime의 greedy token이 PyTorch golden과 일치하는지 검증합니다.
import json
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
