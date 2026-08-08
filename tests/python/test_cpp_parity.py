# 독립 C++ runtime의 greedy token이 PyTorch golden과 일치하는지 검증합니다.
import json
import os
import subprocess
from pathlib import Path

from k3x_converter.writer import convert
from k3x_ref.fixtures import build_synthetic_model


import pytest


@pytest.mark.parametrize("mode", ["incremental", "full"])
def test_cpp_generation_matches_python_golden(
    synthetic_source: Path, tmp_path: Path, mode: str
) -> None:
    suffix = ".exe" if os.name == "nt" else ""
    runner = Path(f"build/k3x_run{suffix}").resolve()
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
