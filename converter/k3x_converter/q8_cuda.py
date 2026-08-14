# group-128 Q8 직접 matvec CUDA 확장을 지연 빌드하고 호출합니다.
from __future__ import annotations

from functools import cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

from .format import K3XError


@cache
def _extension():
    root = Path(__file__).resolve().parents[2]
    source = root / "runtime" / "cuda" / "q8_torch_extension.cu"
    build_directory = root / "build" / "torch-extensions" / "q8-sm120"
    build_directory.mkdir(parents=True, exist_ok=True)
    return load(
        name="k3x_q8_cuda_sm120_v1",
        sources=[str(source)],
        build_directory=str(build_directory),
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_120,code=sm_120"],
        verbose=False,
    )


def q8_matvec(
    value: torch.Tensor,
    codes: torch.Tensor,
    scales: torch.Tensor,
    rows: int,
    columns: int,
) -> torch.Tensor:
    if (
        not value.is_cuda
        or not codes.is_cuda
        or not scales.is_cuda
        or value.device != codes.device
        or value.device != scales.device
        or codes.dtype != torch.int8
        or scales.dtype != torch.bfloat16
        or rows <= 0
        or columns <= 0
        or columns % 128
        or value.numel() != columns
        or codes.numel() != rows * columns
        or scales.numel() != rows * columns // 128
    ):
        raise K3XError("INVALID_Q8_CUDA_MATVEC")
    output = _extension().q8_matvec(
        value.to(dtype=torch.float32).contiguous(),
        codes.contiguous(),
        scales.contiguous(),
        rows,
        columns,
    )
    return output.to(torch.bfloat16)
