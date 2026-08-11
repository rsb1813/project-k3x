# C++ reader가 Python writer artifact의 정확한 corruption code를 반환하는지 검증합니다.
import dataclasses
import hashlib
import json
import os
import shutil
import struct
import subprocess
from pathlib import Path

import pytest

from conftest import cpp_binary
from k3x_converter.format import (
    SUPERBLOCK_BYTES,
    Superblock,
    root_sha256,
)
from k3x_converter.reader import K3XReader
from k3x_converter.writer import convert
from k3x_ref.storage_fixture import write_bounded_expert_source


def _write_bf16_source(source: Path, config_source: Path) -> None:
    source.mkdir()
    name = "model.layers.0.bf16_probe.weight"
    payload = bytes.fromhex("0000803f004040408040a040")
    header = json.dumps(
        {name: {"dtype": "BF16", "shape": [2, 3], "data_offsets": [0, 12]}},
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    shard_name = "bf16.safetensors"
    (source / shard_name).write_bytes(struct.pack("<Q", len(header)) + header + payload)
    config = json.loads(
        (config_source / "source-manifest.json").read_text(encoding="utf-8")
    )["config"]
    (source / "source-manifest.json").write_text(
        json.dumps(
            {
                "format": "k3-official-moe-slice-v1",
                "artifact_kind": "official_moe_fixture",
                "config": config,
                "packed_shapes": {},
                "weight_map": {name: shard_name},
            },
            sort_keys=True,
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )


def _run_reader(path: Path) -> subprocess.CompletedProcess[str]:
    runner = cpp_binary("test_reader")
    assert runner.exists(), "build test_reader before running cross-language reader tests"
    return subprocess.run([str(runner), str(path)], capture_output=True, text=True)


def test_cpp_reader_keeps_one_linux_data_plane_descriptor(
    synthetic_source: Path, tmp_path: Path
) -> None:
    artifact = tmp_path / "persistent.k3x"
    convert(synthetic_source, artifact, chunk_bytes=257)
    result = subprocess.run(
        [str(cpp_binary("test_reader")), str(artifact), "persistent"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    assert artifact.is_file()


@pytest.mark.parametrize(
    "mode",
    [
        None,
        pytest.param(
            "io-uring",
            marks=pytest.mark.skipif(
                os.environ.get("K3X_TEST_IO_URING") != "1",
                reason="requires the optional liburing build",
            ),
        ),
    ],
)
def test_cpp_reader_serializes_shared_data_plane_and_counters(
    synthetic_source: Path, tmp_path: Path, mode: str | None
) -> None:
    artifact = tmp_path / "concurrent.k3x"
    convert(synthetic_source, artifact, chunk_bytes=257)
    arguments = [str(cpp_binary("test_reader_concurrency")), str(artifact)]
    if mode is not None:
        arguments.append(mode)
    result = subprocess.run(arguments, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_cpp_reader_exposes_storage_fixture_optional_identity(
    tmp_path: Path,
) -> None:
    source = tmp_path / "bounded-source"
    write_bounded_expert_source(source, chunk_bytes=257 * 1024)
    artifact = tmp_path / "bounded.k3x"
    convert(source, artifact, chunk_bytes=193 * 1024)

    result = subprocess.run(
        [str(cpp_binary("test_reader")), str(artifact), "storage-fixture"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_cpp_reader_accepts_exact_bf16_tensor(
    synthetic_source: Path, tmp_path: Path
) -> None:
    source = tmp_path / "bf16-source"
    _write_bf16_source(source, synthetic_source)
    artifact = tmp_path / "bf16.k3x"
    convert(source, artifact, chunk_bytes=5)

    result = subprocess.run(
        [str(cpp_binary("test_reader")), str(artifact), "bf16"],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr


def test_cpp_reader_rejects_bf16_feature_mismatch_and_quantization(
    synthetic_source: Path, tmp_path: Path
) -> None:
    source = tmp_path / "bf16-source"
    _write_bf16_source(source, synthetic_source)
    valid = tmp_path / "bf16.k3x"
    convert(source, valid, chunk_bytes=5)

    missing_feature = tmp_path / "missing-feature.k3x"
    shutil.copyfile(valid, missing_feature)
    with missing_feature.open("r+b") as stream:
        block = Superblock.decode(stream.read(SUPERBLOCK_BYTES))
        stream.seek(0)
        stream.write(dataclasses.replace(block, required_features=0).encode())
    result = _run_reader(missing_feature)
    assert result.returncode == 1
    assert result.stderr.strip() == "INVALID_DIRECTORY"

    invalid_quantization = tmp_path / "invalid-quantization.k3x"
    shutil.copyfile(valid, invalid_quantization)

    def set_mxfp4_quantization(stream, block: Superblock) -> None:
        stream.seek(block.tensor_directory_offset + 16 + 14)
        stream.write(struct.pack("<H", 1))

    _refinalize_metadata(invalid_quantization, set_mxfp4_quantization)
    result = _run_reader(invalid_quantization)
    assert result.returncode == 1
    assert result.stderr.strip() == "INVALID_DIRECTORY"


@pytest.mark.skipif(
    os.environ.get("K3X_TEST_IO_URING") != "1",
    reason="requires the optional liburing build",
)
def test_cpp_reader_uses_real_io_uring_for_exact_tensor_reads(
    synthetic_source: Path, tmp_path: Path
) -> None:
    artifact = tmp_path / "io-uring.k3x"
    convert(synthetic_source, artifact, chunk_bytes=257)
    result = subprocess.run(
        [str(cpp_binary("test_reader")), str(artifact), "io-uring"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.skipif(
    os.environ.get("K3X_TEST_IO_URING") == "1",
    reason="exercises a build without liburing support",
)
def test_cpp_reader_reports_unbuilt_io_uring_mode(
    synthetic_source: Path, tmp_path: Path
) -> None:
    artifact = tmp_path / "unbuilt-io-uring.k3x"
    convert(synthetic_source, artifact, chunk_bytes=257)
    result = subprocess.run(
        [str(cpp_binary("test_reader")), str(artifact), "io-uring"],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 12
    assert result.stderr.strip() == (
        "STORAGE_UNAVAILABLE: requested L2 mode is not built"
    )


@pytest.mark.skipif(
    os.environ.get("K3X_TEST_DIRECT") != "1",
    reason="requires a Linux filesystem with STATX_DIOALIGN",
)
@pytest.mark.parametrize("mode", ["direct", "io-uring-direct"])
def test_cpp_reader_uses_direct_bounce_buffers_for_small_extents(
    synthetic_source: Path, tmp_path: Path, mode: str
) -> None:
    if mode == "io-uring-direct" and os.environ.get("K3X_TEST_IO_URING") != "1":
        pytest.skip("requires the optional liburing build")
    artifact = tmp_path / f"{mode}.k3x"
    convert(synthetic_source, artifact, chunk_bytes=257)
    result = subprocess.run(
        [str(cpp_binary("test_reader")), str(artifact), mode],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


def test_cpp_binary_uses_configured_build_directory(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("K3X_BUILD_DIR", str(tmp_path / "native-build"))
    assert cpp_binary("test_reader").parent == tmp_path / "native-build"


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
        stream.write(dataclasses.replace(block, required_features=1 << 63).encode())
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
