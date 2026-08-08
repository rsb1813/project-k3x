# C++ reader가 Python writer artifact의 정확한 corruption code를 반환하는지 검증합니다.
import dataclasses
import os
import shutil
import subprocess
from pathlib import Path

from k3x_converter.format import SUPERBLOCK_BYTES, Superblock
from k3x_converter.reader import K3XReader
from k3x_converter.writer import convert


def _run_reader(path: Path) -> subprocess.CompletedProcess[str]:
    runner = Path("build/test_reader.exe").resolve()
    assert runner.exists(), "build test_reader before running cross-language reader tests"
    return subprocess.run([str(runner), str(path)], capture_output=True, text=True)


def test_cpp_reader_accepts_python_artifact_and_rejects_corruption(
    synthetic_source: Path, tmp_path: Path
) -> None:
    valid = tmp_path / "valid.k3x"
    convert(synthetic_source, valid, chunk_bytes=257)
    assert _run_reader(valid).returncode == 0

    corrupted = tmp_path / "corrupted.k3x"
    shutil.copyfile(valid, corrupted)
    first = K3XReader.open(valid).tensor_records[0]
    with corrupted.open("r+b") as stream:
        stream.seek(first.data_offset)
        value = stream.read(1)
        stream.seek(first.data_offset)
        stream.write(bytes([value[0] ^ 1]))
    result = _run_reader(corrupted)
    assert result.returncode == 1
    assert result.stderr.strip() == "DATA_CRC_MISMATCH"

    unsupported = tmp_path / "unsupported.k3x"
    shutil.copyfile(valid, unsupported)
    with unsupported.open("r+b") as stream:
        block = Superblock.decode(stream.read(SUPERBLOCK_BYTES))
        stream.seek(0)
        stream.write(dataclasses.replace(block, required_features=1).encode())
    result = _run_reader(unsupported)
    assert result.returncode == 1
    assert result.stderr.strip() == "UNSUPPORTED_REQUIRED_FEATURE"

    truncated = tmp_path / "truncated.k3x"
    shutil.copyfile(valid, truncated)
    os.truncate(truncated, truncated.stat().st_size - 1)
    result = _run_reader(truncated)
    assert result.returncode == 1
    assert result.stderr.strip() == "TRUNCATED_FILE"
