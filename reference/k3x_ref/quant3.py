# 저민감도 routed expert를 위한 결정적 group-wise 3-bit 참조 codec입니다.
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from k3x_converter.format import K3XError


@dataclass(frozen=True)
class Quant3Tensor:
    shape: tuple[int, ...]
    values: int
    group_size: int
    packed: bytes
    scales_bf16: bytes


def _scale_payload(scales: torch.Tensor) -> bytes:
    return scales.contiguous().view(torch.uint8).numpy().tobytes()


def quantize_groupwise_3bit(
    tensor: torch.Tensor, *, group_size: int = 32
) -> Quant3Tensor:
    if group_size != 32 or tensor.numel() <= 0 or not torch.isfinite(tensor).all():
        raise K3XError("INVALID_QUANT3_INPUT")
    source = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().flatten()
    values = source.numel()
    groups = math.ceil(values / group_size)
    padded = torch.zeros(groups * group_size, dtype=torch.float32)
    padded[:values] = source
    grouped = padded.reshape(groups, group_size)
    scales_float = (grouped.abs().amax(dim=1, keepdim=True) / 3.0).clamp_min(1.0e-30)
    for _ in range(3):
        fitted_codes = torch.round(grouped / scales_float).clamp(-3, 3)
        scales_float = (
            (grouped * fitted_codes).sum(dim=1, keepdim=True)
            / fitted_codes.square().sum(dim=1, keepdim=True).clamp_min(1.0)
        ).clamp_min(1.0e-30)
    scales = scales_float.flatten().to(torch.bfloat16)
    scales = torch.where(scales == 0, torch.ones_like(scales), scales)
    codes = torch.round(grouped / scales.float().unsqueeze(1)).clamp(-3, 3)
    codes = (codes.to(torch.int64) + 3).reshape(-1, 8)
    shifts = torch.arange(0, 24, 3, dtype=torch.int64)
    words = torch.sum(codes.bitwise_left_shift(shifts), dim=1)
    payload_tensor = torch.stack(
        (
            words.bitwise_and(0xFF),
            words.bitwise_right_shift(8).bitwise_and(0xFF),
            words.bitwise_right_shift(16).bitwise_and(0xFF),
        ),
        dim=1,
    ).to(torch.uint8)
    return Quant3Tensor(
        tuple(tensor.shape),
        values,
        group_size,
        payload_tensor.numpy().tobytes(),
        _scale_payload(scales),
    )


def decode_groupwise_3bit(value: Quant3Tensor) -> torch.Tensor:
    if (
        value.group_size != 32
        or value.values <= 0
        or math.prod(value.shape) != value.values
    ):
        raise K3XError("INVALID_QUANT3_METADATA")
    groups = math.ceil(value.values / value.group_size)
    if len(value.packed) != groups * 12 or len(value.scales_bf16) != groups * 2:
        raise K3XError("INVALID_QUANT3_LENGTH")
    codes = torch.empty(groups * value.group_size, dtype=torch.int8)
    for block in range(groups * 4):
        offset = block * 3
        word = int.from_bytes(value.packed[offset : offset + 3], "little")
        first = block * 8
        for index in range(8):
            code = (word >> (index * 3)) & 0x7
            if code == 7:
                raise K3XError("QUANT3_RESERVED_CODE")
            codes[first + index] = code - 3
    scales = torch.frombuffer(
        bytearray(value.scales_bf16), dtype=torch.bfloat16
    ).clone()
    if not torch.isfinite(scales).all() or torch.any(scales <= 0):
        raise K3XError("INVALID_QUANT3_SCALE")
    decoded = codes.reshape(groups, value.group_size).float()
    decoded *= scales.float().unsqueeze(1)
    return decoded.reshape(-1)[: value.values].reshape(value.shape)
