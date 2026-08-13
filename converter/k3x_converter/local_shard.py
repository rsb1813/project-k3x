# 공식 Kimi K3 shard 하나를 bounded K3X 제조 단위로 변환합니다.
from __future__ import annotations

import errno
import hashlib
import json
import math
import os
import re
import shutil
import struct
import tempfile
from dataclasses import dataclass
from pathlib import Path

import torch

from k3x_ref.quant8 import quantize_groupwise_8bit

from .format import K3XError
from .reader import K3XReader
from .safetensors_reader import SourceTensor, inspect_shard, iter_tensor_chunks
from .writer import convert


_EXPERT = re.compile(
    r"^language_model\.model\.layers\.(\d+)\.block_sparse_moe\."
    r"experts\.(\d+)\.(w1|w2|w3)\.(weight_packed|weight_scale)$"
)
_ROLES = {"w1": "gate", "w2": "down", "w3": "up"}
_EXPERT_SHAPES = {
    "gate": (3072, 3584),
    "down": (3584, 3072),
    "up": (3072, 3584),
}


@dataclass(frozen=True)
class LocalShardReport:
    source_path: Path
    source_sha256: str
    output_path: Path
    output_sha256: str
    output_bytes: int
    tensor_count: int
    quant8_tensor_count: int
    native_expert_tensor_count: int


@dataclass(frozen=True)
class _OutputTensor:
    name: str
    dtype: str
    shape: tuple[int, ...]
    length: int
    source: SourceTensor
    kind: str


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_name(name: str) -> str:
    match = _EXPERT.match(name)
    if match is not None:
        layer, expert, matrix, suffix = match.groups()
        role = _ROLES[matrix]
        return (
            f"model.layers.{layer}.feed_forward.experts.{expert}."
            f"{role}.{suffix}"
        )
    if name.startswith("language_model."):
        return name.removeprefix("language_model.")
    return name


def _preserve(name: str, tensor: SourceTensor) -> bool:
    return (
        tensor.dtype != "BF16"
        or len(tensor.shape) <= 1
        or "norm" in name
        or ".block_sparse_moe.gate." in name
        or name
        in {
            "language_model.lm_head.weight",
            "language_model.model.embed_tokens.weight",
        }
    )


def _plan(
    tensors: dict[str, SourceTensor],
) -> tuple[list[_OutputTensor], dict[str, list[int]], dict[str, list[int]], int, int]:
    outputs: list[_OutputTensor] = []
    quant8_shapes: dict[str, list[int]] = {}
    packed_shapes: dict[str, list[int]] = {}
    quant8_count = 0
    native_expert_count = 0
    for name in sorted(tensors):
        tensor = tensors[name]
        canonical = _canonical_name(name)
        expert = _EXPERT.match(name)
        if expert is not None:
            role = _ROLES[expert.group(3)]
            base = canonical.rsplit(".", 1)[0]
            packed_shapes[base] = list(_EXPERT_SHAPES[role])
            outputs.append(
                _OutputTensor(
                    canonical, tensor.dtype, tensor.shape, tensor.length, tensor, "copy"
                )
            )
            native_expert_count += 1
            continue
        if _preserve(name, tensor):
            if tensor.dtype not in {"BF16", "F32"}:
                raise K3XError("UNSUPPORTED_LOCAL_SHARD_DTYPE", name)
            outputs.append(
                _OutputTensor(
                    canonical, tensor.dtype, tensor.shape, tensor.length, tensor, "copy"
                )
            )
            continue
        values = math.prod(tensor.shape)
        if tensor.length != values * 2:
            raise K3XError("INVALID_LOCAL_SHARD_BF16", name)
        groups = (values + 127) // 128
        quant8_shapes[canonical] = list(tensor.shape)
        outputs.extend(
            (
                _OutputTensor(
                    canonical + ".q8_codes",
                    "U8",
                    (groups * 128,),
                    groups * 128,
                    tensor,
                    "q8_codes",
                ),
                _OutputTensor(
                    canonical + ".q8_scale",
                    "U8",
                    (groups * 2,),
                    groups * 2,
                    tensor,
                    "q8_scale",
                ),
            )
        )
        quant8_count += 1
    return outputs, quant8_shapes, packed_shapes, quant8_count, native_expert_count


def _write_microshard(
    path: Path,
    outputs: list[_OutputTensor],
    *,
    chunk_bytes: int,
) -> None:
    if chunk_bytes < 256:
        raise K3XError("INVALID_LOCAL_SHARD_CHUNK")
    chunk_bytes -= chunk_bytes % 256
    offsets: dict[str, tuple[int, int]] = {}
    cursor = 0
    header: dict[str, object] = {}
    for output in outputs:
        offsets[output.name] = (cursor, cursor + output.length)
        header[output.name] = {
            "dtype": output.dtype,
            "shape": list(output.shape),
            "data_offsets": [cursor, cursor + output.length],
        }
        cursor += output.length
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    data_start = 8 + len(encoded)
    partial = path.with_name(path.name + ".partial")
    with partial.open("w+b") as stream:
        stream.write(struct.pack("<Q", len(encoded)))
        stream.write(encoded)
        stream.truncate(data_start + cursor)
        handled_q8: set[str] = set()
        for output in outputs:
            if output.kind == "copy":
                stream.seek(data_start + offsets[output.name][0])
                for chunk in iter_tensor_chunks(output.source, chunk_bytes):
                    stream.write(chunk)
                continue
            canonical = output.name.removesuffix(".q8_codes").removesuffix(".q8_scale")
            if canonical in handled_q8:
                continue
            handled_q8.add(canonical)
            code_name = canonical + ".q8_codes"
            scale_name = canonical + ".q8_scale"
            code_cursor = data_start + offsets[code_name][0]
            scale_cursor = data_start + offsets[scale_name][0]
            for chunk in iter_tensor_chunks(output.source, chunk_bytes):
                values = torch.frombuffer(
                    bytearray(chunk), dtype=torch.bfloat16
                ).clone()
                encoded_chunk = quantize_groupwise_8bit(values)
                stream.seek(code_cursor)
                stream.write(encoded_chunk.codes)
                stream.seek(scale_cursor)
                stream.write(encoded_chunk.scales_bf16)
                code_cursor += len(encoded_chunk.codes)
                scale_cursor += len(encoded_chunk.scales_bf16)
            if (
                code_cursor != data_start + offsets[code_name][1]
                or scale_cursor != data_start + offsets[scale_name][1]
            ):
                raise K3XError("LOCAL_SHARD_QUANT8_LENGTH", canonical)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)


def convert_local_official_shard(
    source_path: Path,
    output_directory: Path,
    *,
    config: dict[str, object],
    expected_sha256: str,
    chunk_bytes: int = 8 * 1024 * 1024,
    temporary_directory: Path | None = None,
) -> LocalShardReport:
    source_path = Path(source_path).resolve(strict=True)
    output_directory = Path(output_directory).resolve()
    output_directory.mkdir(parents=True, exist_ok=True)
    temporary_root = (
        Path(temporary_directory).resolve()
        if temporary_directory is not None
        else output_directory
    )
    temporary_root.mkdir(parents=True, exist_ok=True)
    output_path = output_directory / (source_path.stem + ".k3x")
    with tempfile.TemporaryDirectory(dir=temporary_root) as temporary:
        work = Path(temporary)
        source_link = work / "official.safetensors"
        try:
            os.link(source_path, source_link)
        except OSError as exc:
            if exc.errno != errno.EXDEV:
                raise K3XError("LOCAL_SOURCE_HARDLINK", source_path.name) from exc
            shutil.copyfile(source_path, source_link)
        source_sha256 = _sha256(source_link)
        if source_sha256 != expected_sha256:
            raise K3XError("LOCAL_SOURCE_SHA256", source_path.name)
        tensors = inspect_shard(source_link)
        outputs, quant8_shapes, packed_shapes, quant8_count, native_expert_count = _plan(
            tensors
        )
        copy_outputs = [output for output in outputs if output.kind == "copy"]
        quantized_outputs = [output for output in outputs if output.kind != "copy"]
        weight_map = {output.name: source_link.name for output in copy_outputs}
        source_names = {output.name: output.source.name for output in copy_outputs}
        if quantized_outputs:
            quantized_shard = work / "quantized.safetensors"
            _write_microshard(
                quantized_shard, quantized_outputs, chunk_bytes=chunk_bytes
            )
            for output in quantized_outputs:
                weight_map[output.name] = quantized_shard.name
                source_names[output.name] = output.name
        manifest = {
            "format": "k3-local-shard-v1",
            "config": config,
            "packed_shapes": packed_shapes,
            "quant8_shapes": quant8_shapes,
            "source_names": source_names,
            "weight_map": weight_map,
        }
        (work / "source-manifest.json").write_text(
            json.dumps(manifest, sort_keys=True, separators=(",", ":")) + "\n",
            encoding="utf-8",
        )
        report = convert(work, output_path, chunk_bytes=chunk_bytes)
        if not report.completed:
            raise K3XError("LOCAL_SHARD_CONVERSION_INCOMPLETE", source_path.name)
    reader = K3XReader.open(output_path)
    expected_records = len(outputs) - quant8_count - native_expert_count // 2
    if len(reader.tensor_records) != expected_records:
        raise K3XError("LOCAL_SHARD_TENSOR_COUNT", source_path.name)
    return LocalShardReport(
        source_path,
        source_sha256,
        output_path,
        _sha256(output_path),
        output_path.stat().st_size,
        len(reader.tensor_records),
        quant8_count,
        native_expert_count,
    )
