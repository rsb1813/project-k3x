# native MXFP4 직접 matvec CUDA 확장을 지연 빌드하고 검증해 호출합니다.
from __future__ import annotations

from functools import cache
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

from .format import K3XError


@cache
def _extension():
    root = Path(__file__).resolve().parents[2]
    source = root / "runtime" / "cuda" / "mxfp4_torch_extension.cu"
    build_directory = root / "build" / "torch-extensions" / "mxfp4-sm120"
    build_directory.mkdir(parents=True, exist_ok=True)
    return load(
        name="k3x_mxfp4_cuda_sm120_v1",
        sources=[str(source)],
        build_directory=str(build_directory),
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "-gencode=arch=compute_120,code=sm_120"],
        verbose=False,
    )


def mxfp4_matvec(
    value: torch.Tensor,
    packed: torch.Tensor,
    scales: torch.Tensor,
    rows: int,
    columns: int,
) -> torch.Tensor:
    if (
        not value.is_cuda
        or not packed.is_cuda
        or not scales.is_cuda
        or value.device != packed.device
        or value.device != scales.device
        or packed.dtype != torch.uint8
        or scales.dtype != torch.uint8
        or rows <= 0
        or columns <= 0
        or columns % 32
        or value.numel() != columns
        or packed.numel() != rows * columns // 2
        or scales.numel() != rows * columns // 32
    ):
        raise K3XError("INVALID_MXFP4_CUDA_MATVEC")
    output = _extension().mxfp4_matvec(
        value.to(dtype=torch.float32).contiguous(),
        packed.contiguous(),
        scales.contiguous(),
        rows,
        columns,
    )
    return output.to(torch.bfloat16)
