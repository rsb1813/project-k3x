# K3X fragment 집합에서 이름으로 dense 텐서를 읽고 역양자화하는 저장소입니다.
from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch

from k3x_ref.quant8 import Quant8Tensor, decode_groupwise_8bit

from .format import DType, K3XError, Quantization, TensorRecord, fnv1a64
from .reader import K3XReader


@dataclass(frozen=True)
class _LocatedTensor:
    reader: K3XReader
    record: TensorRecord


@dataclass(frozen=True)
class K3XTensorStore:
    tensors: dict[int, _LocatedTensor]

    @classmethod
    def open(
        cls,
        paths: Iterable[Path],
        *,
        verify_root: bool = True,
        verify_payload: bool = True,
    ) -> "K3XTensorStore":
        tensors: dict[int, _LocatedTensor] = {}
        for path in paths:
            reader = K3XReader.open(
                Path(path), verify_root=verify_root, verify_payload=verify_payload
            )
            for record in reader.tensor_records:
                if record.tensor_id in tensors:
                    raise K3XError("DUPLICATE_TENSOR_ID", f"{record.tensor_id:016x}")
                tensors[record.tensor_id] = _LocatedTensor(reader, record)
        return cls(tensors)

    def record(self, name: str) -> _LocatedTensor:
        tensor_id = fnv1a64(name)
        located = self.tensors.get(tensor_id)
        if located is None:
            raise K3XError("TENSOR_NOT_FOUND", name)
        return located

    def load(
        self,
        name: str,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype | None = None,
    ) -> torch.Tensor:
        located = self.record(name)
        record = located.record
        data, auxiliary = located.reader.read_tensor_extents(record)
        values = math.prod(record.dimensions)
        if record.quantization == Quantization.GROUPWISE_8BIT:
            tensor = decode_groupwise_8bit(
                Quant8Tensor(record.dimensions, values, 128, data, auxiliary)
            )
        elif record.quantization == Quantization.NONE and record.dtype == DType.BF16:
            tensor = torch.frombuffer(bytearray(data), dtype=torch.bfloat16).reshape(
                record.dimensions
            )
        elif record.quantization == Quantization.NONE and record.dtype == DType.FP32:
            tensor = torch.frombuffer(bytearray(data), dtype=torch.float32).reshape(
                record.dimensions
            )
        else:
            raise K3XError("TENSOR_REQUIRES_SPECIALIZED_DECODE", name)
        return tensor.to(device=device, dtype=dtype)

    def load_rows(
        self,
        name: str,
        first_row: int,
        rows: int,
        *,
        device: torch.device | str = "cpu",
    ) -> torch.Tensor:
        located = self.record(name)
        record = located.record
        if (
            record.quantization != Quantization.NONE
            or record.dtype != DType.BF16
            or len(record.dimensions) != 2
            or first_row < 0
            or rows <= 0
            or first_row + rows > record.dimensions[0]
        ):
            raise K3XError("INVALID_TENSOR_ROW_SLICE", name)
        columns = record.dimensions[1]
        length = rows * columns * 2
        with located.reader.path.open("rb") as stream:
            stream.seek(record.data_offset + first_row * columns * 2)
            data = stream.read(length)
        if len(data) != length:
            raise K3XError("TRUNCATED_FILE")
        return torch.frombuffer(bytearray(data), dtype=torch.bfloat16).reshape(
            rows, columns
        ).to(device)

    def mxfp4_matvec(
        self,
        name: str,
        value: torch.Tensor,
        *,
        device: torch.device | str | None = None,
    ) -> torch.Tensor:
        located = self.record(name)
        record = located.record
        if record.quantization != Quantization.MXFP4 or len(record.dimensions) != 2:
            raise K3XError("INVALID_MXFP4_TENSOR", name)
        rows, columns = record.dimensions
        if value.numel() != columns:
            raise K3XError("INVALID_MXFP4_INPUT", name)
        target = value.device if device is None else torch.device(device)
        packed, scale_bytes = located.reader.read_tensor_extents(record)
        scales = torch.frombuffer(bytearray(scale_bytes), dtype=torch.uint8).to(target)
        if bool((scales == 0xFF).any()):
            raise K3XError("INVALID_MXFP4", name)
        packed_tensor = torch.frombuffer(bytearray(packed), dtype=torch.uint8).to(target)
        lookup = torch.tensor(
            (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
             -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0),
            dtype=torch.float32,
            device=target,
        )
        nibbles = torch.stack(
            (packed_tensor.bitwise_and(0x0F), packed_tensor.bitwise_right_shift(4)),
            dim=1,
        ).reshape(-1)
        decoded = lookup[nibbles.long()]
        exponents = scales.to(torch.int32) - 127
        scale_values = torch.ldexp(
            torch.ones_like(exponents, dtype=torch.float32), exponents
        )
        weight = (decoded * scale_values.repeat_interleave(32)).reshape(rows, columns)
        return weight @ value.to(device=target, dtype=torch.float32).reshape(columns)
