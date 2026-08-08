# C++ reader가 Python writer artifact의 정확한 corruption code를 반환하는지 검증합니다.
import dataclasses
import hashlib
import os
import shutil
import subprocess
from pathlib import Path

from k3x_converter.format import SUPERBLOCK_BYTES, Superblock, root_sha256
from k3x_converter.reader import K3XReader
from k3x_converter.writer import convert


def _run_reader(path: Path) -> subprocess.CompletedProcess[str]:
    suffix = ".exe" if os.name == "nt" else ""
    runner = Path(f"build/test_reader{suffix}").resolve()
    assert runner.exists(), "build test_reader before running cross-language reader tests"
    return subprocess.run([str(runner), str(path)], capture_output=True, text=True)


def _refinalize_metadata(path: Path, mutation) -> None:
    with path.open("r+b") as stream:
        block = Superblock.decode(stream.read(SUPERBLOCK_BYTES))
        mutation(stream, block)
        directories = []
        for offset, length in (
            (block.tensor_directory_offset, block.tensor_directory_length),
            (block.layer_directory_offset, block.layer_directory_length),
            (block.expert_directory_offset, block.expert_directory_length),
            (block.model_config_offset, block.model_config_length),
        ):
            stream.seek(offset)
            directories.append(stream.read(length))
        block = dataclasses.replace(
            block,
            directory_sha256=hashlib.sha256(b"".join(directories)).digest(),
            root_sha256=bytes(32),
        )
        stream.seek(0)
        stream.write(block.encode())
        digest = root_sha256(stream, block.file_length)
        stream.seek(0)
        stream.write(dataclasses.replace(block, root_sha256=digest).encode())


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


def test_cpp_reader_rejects_invalid_layer_and_expert_directories(
    synthetic_source: Path, tmp_path: Path
) -> None:
    valid = tmp_path / "valid.k3x"
    convert(synthetic_source, valid, chunk_bytes=257)

    invalid_layer = tmp_path / "invalid-layer.k3x"
    shutil.copyfile(valid, invalid_layer)
    _refinalize_metadata(
        invalid_layer,
        lambda stream, block: (
            stream.seek(block.layer_directory_offset), stream.write(b"BAD!")
        ),
    )
    result = _run_reader(invalid_layer)
    assert result.returncode == 1
    assert result.stderr.strip() == "INVALID_DIRECTORY"

    invalid_expert = tmp_path / "invalid-expert.k3x"
    shutil.copyfile(valid, invalid_expert)

    def corrupt_expert_reserved(stream, block: Superblock) -> None:
        stream.seek(block.expert_directory_offset + 16 + 48)
        stream.write(b"\x01")

    _refinalize_metadata(invalid_expert, corrupt_expert_reserved)
    result = _run_reader(invalid_expert)
    assert result.returncode == 1
    assert result.stderr.strip() == "INVALID_DIRECTORY"
