# 비전문가 BF16 행렬을 위한 group-128 signed 8비트 참조 codec입니다.
from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from k3x_converter.format import K3XError


@dataclass(frozen=True)
class Quant8Tensor:
    shape: tuple[int, ...]
    values: int
    group_size: int
    codes: bytes
    scales_bf16: bytes


def quantize_groupwise_8bit(
    tensor: torch.Tensor, *, group_size: int = 128
) -> Quant8Tensor:
    if group_size != 128 or tensor.numel() <= 0 or not torch.isfinite(tensor).all():
        raise K3XError("INVALID_QUANT8_INPUT")
    source = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous().flatten()
    values = source.numel()
    groups = math.ceil(values / group_size)
    padded = torch.zeros(groups * group_size, dtype=torch.float32)
    padded[:values] = source
    grouped = padded.reshape(groups, group_size)
    scales = (grouped.abs().amax(dim=1) / 127.0).clamp_min(1.0e-30)
    codes = torch.round(grouped / scales.unsqueeze(1)).clamp(-127, 127)
    fitted = (
        (grouped * codes).sum(dim=1)
        / codes.square().sum(dim=1).clamp_min(1.0)
    ).clamp_min(1.0e-30)
    scales_bf16 = fitted.to(torch.bfloat16)
    scales_bf16 = torch.where(
        scales_bf16 == 0, torch.ones_like(scales_bf16), scales_bf16
    )
    codes = torch.round(grouped / scales_bf16.float().unsqueeze(1)).clamp(-127, 127)
    return Quant8Tensor(
        tuple(tensor.shape),
        values,
        group_size,
        codes.to(torch.int8).numpy().tobytes(),
        scales_bf16.contiguous().view(torch.uint8).numpy().tobytes(),
    )


def decode_groupwise_8bit(value: Quant8Tensor) -> torch.Tensor:
    if (
        value.group_size != 128
        or value.values <= 0
        or math.prod(value.shape) != value.values
    ):
        raise K3XError("INVALID_QUANT8_METADATA")
    groups = math.ceil(value.values / value.group_size)
    if len(value.codes) != groups * value.group_size or len(value.scales_bf16) != groups * 2:
        raise K3XError("INVALID_QUANT8_LENGTH")
    codes = torch.frombuffer(bytearray(value.codes), dtype=torch.int8).clone().float()
    scales = torch.frombuffer(
        bytearray(value.scales_bf16), dtype=torch.bfloat16
    ).clone()
    if not torch.isfinite(scales).all() or torch.any(scales <= 0):
        raise K3XError("INVALID_QUANT8_SCALE")
    decoded = codes.reshape(groups, value.group_size)
    decoded *= scales.float().unsqueeze(1)
    return decoded.reshape(-1)[: value.values].reshape(value.shape)
