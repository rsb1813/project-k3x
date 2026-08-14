# K3X fragment 집합에서 이름으로 dense 텐서를 읽고 역양자화하는 저장소입니다.
from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import torch

from k3x_ref.quant8 import Quant8Tensor, decode_groupwise_8bit

from .format import DType, K3XError, Quantization, TensorRecord, fnv1a64
from .reader import K3XReader


@dataclass(frozen=True)
class _LocatedTensor:
    reader: K3XReader
    record: TensorRecord


def _quant8_host_tensors(
    value: Quant8Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    if (
        value.group_size != 128
        or value.values <= 0
        or math.prod(value.shape) != value.values
    ):
        raise K3XError("INVALID_QUANT8_METADATA")
    groups = math.ceil(value.values / value.group_size)
    if (
        len(value.codes) != groups * value.group_size
        or len(value.scales_bf16) != groups * 2
    ):
        raise K3XError("INVALID_QUANT8_LENGTH")
    scales = torch.frombuffer(
        bytearray(value.scales_bf16), dtype=torch.bfloat16
    )
    if not torch.isfinite(scales).all() or torch.any(scales <= 0):
        raise K3XError("INVALID_QUANT8_SCALE")
    codes = torch.frombuffer(bytearray(value.codes), dtype=torch.int8)
    return codes, scales


def _decode_groupwise_8bit_cuda(
    value: Quant8Tensor,
    device: torch.device,
    dtype: torch.dtype | None,
) -> torch.Tensor:
    codes, scales = _quant8_host_tensors(value)
    groups = math.ceil(value.values / value.group_size)
    device_scales = scales.to(device)
    if dtype == torch.bfloat16:
        decoded = codes.to(device=device, dtype=torch.bfloat16).reshape(
            groups, value.group_size
        )
        decoded *= device_scales.unsqueeze(1)
    else:
        decoded = codes.to(device).float().reshape(groups, value.group_size)
        decoded *= device_scales.float().unsqueeze(1)
    return decoded.reshape(-1)[: value.values].reshape(value.shape)


@dataclass
class PackedQ8Cache:
    host_budget_bytes: int
    device_budget_bytes: int
    _host: dict[tuple[str, int], tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict, init=False
    )
    _device: dict[tuple[str, int, str], tuple[torch.Tensor, torch.Tensor]] = field(
        default_factory=dict, init=False
    )
    host_resident_bytes: int = field(default=0, init=False)
    device_resident_bytes: int = field(default=0, init=False)
    host_hits: int = field(default=0, init=False)
    device_hits: int = field(default=0, init=False)
    misses: int = field(default=0, init=False)
    host_admissions: int = field(default=0, init=False)
    device_admissions: int = field(default=0, init=False)
    rejected_bytes: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        for value in (self.host_budget_bytes, self.device_budget_bytes):
            if isinstance(value, bool) or not isinstance(value, int) or value < 0:
                raise K3XError("INVALID_Q8_CACHE_BUDGET")

    @staticmethod
    def _bytes(tensors: tuple[torch.Tensor, torch.Tensor]) -> int:
        return sum(tensor.numel() * tensor.element_size() for tensor in tensors)

    def acquire(
        self,
        key: tuple[str, int, str],
        loader: Callable[[], tuple[torch.Tensor, torch.Tensor]],
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cached_device = self._device.get(key)
        if cached_device is not None:
            self.device_hits += 1
            return cached_device

        host_key = key[:2]
        cached_host = self._host.get(host_key)
        if cached_host is not None:
            self.host_hits += 1
            return tuple(tensor.to(device) for tensor in cached_host)

        self.misses += 1
        host_tensors = loader()
        size = self._bytes(host_tensors)
        if self.device_resident_bytes + size <= self.device_budget_bytes:
            device_tensors = tuple(tensor.to(device) for tensor in host_tensors)
            self._device[key] = device_tensors
            self.device_resident_bytes += size
            self.device_admissions += 1
            return device_tensors
        if self.host_resident_bytes + size <= self.host_budget_bytes:
            self._host[host_key] = host_tensors
            self.host_resident_bytes += size
            self.host_admissions += 1
        else:
            self.rejected_bytes += size
        return tuple(tensor.to(device) for tensor in host_tensors)

    def snapshot(self) -> dict[str, int]:
        return {
            "host_budget_bytes": self.host_budget_bytes,
            "device_budget_bytes": self.device_budget_bytes,
            "host_resident_bytes": self.host_resident_bytes,
            "device_resident_bytes": self.device_resident_bytes,
            "host_hits": self.host_hits,
            "device_hits": self.device_hits,
            "misses": self.misses,
            "host_admissions": self.host_admissions,
            "device_admissions": self.device_admissions,
            "rejected_bytes": self.rejected_bytes,
        }


@dataclass(frozen=True)
class PackedQ8Matrix:
    located: _LocatedTensor
    name: str
    device: torch.device
    cache: PackedQ8Cache | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.located.record.dimensions

    @property
    def ndim(self) -> int:
        return 2

    @property
    def dtype(self) -> torch.dtype:
        return torch.bfloat16

    def matvec(self, value: torch.Tensor) -> torch.Tensor:
        from .q8_cuda import q8_matvec

        rows, columns = self.shape
        if value.numel() != columns:
            raise K3XError("INVALID_Q8_CUDA_INPUT", self.name)
        def load() -> tuple[torch.Tensor, torch.Tensor]:
            data, auxiliary = self.located.reader.read_tensor_extents(
                self.located.record
            )
            return _quant8_host_tensors(
                Quant8Tensor(self.shape, rows * columns, 128, data, auxiliary)
            )

        if self.cache is None:
            codes, scales = load()
            codes = codes.to(self.device)
            scales = scales.to(self.device)
        else:
            key = (
                str(self.located.reader.path.resolve()),
                self.located.record.tensor_id,
                str(self.device),
            )
            codes, scales = self.cache.acquire(key, load, self.device)
        return q8_matvec(
            value.to(self.device),
            codes,
            scales,
            rows,
            columns,
        )


@dataclass(frozen=True)
class K3XTensorStore:
    tensors: dict[int, _LocatedTensor]
    packed_q8_cache: PackedQ8Cache | None = None

    @classmethod
    def open(
        cls,
        paths: Iterable[Path],
        *,
        verify_root: bool = True,
        verify_payload: bool = True,
        packed_q8_cache: PackedQ8Cache | None = None,
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
        return cls(tensors, packed_q8_cache)

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
        target = torch.device(device)
        if record.quantization == Quantization.GROUPWISE_8BIT:
            encoded = Quant8Tensor(record.dimensions, values, 128, data, auxiliary)
            if target.type == "cuda":
                tensor = _decode_groupwise_8bit_cuda(encoded, target, dtype)
            else:
                tensor = decode_groupwise_8bit(encoded)
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
        return tensor.to(device=target, dtype=dtype)

    def packed_q8_matrix(
        self,
        name: str,
        *,
        device: torch.device | str,
    ) -> PackedQ8Matrix:
        located = self.record(name)
        record = located.record
        if (
            record.quantization != Quantization.GROUPWISE_8BIT
            or len(record.dimensions) != 2
            or record.dimensions[1] % 128
        ):
            raise K3XError("INVALID_Q8_CUDA_MATRIX", name)
        target = torch.device(device)
        if target.type != "cuda":
            raise K3XError("INVALID_Q8_CUDA_DEVICE", name)
        if target.index is None:
            target = torch.device("cuda", torch.cuda.current_device())
        return PackedQ8Matrix(located, name, target, self.packed_q8_cache)

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
