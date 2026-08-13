# 공식 MXFP4 행렬을 K3X group-32 3비트 표현으로 변환합니다.
from __future__ import annotations

from k3x_ref.mxfp4 import decode_mxfp4
from k3x_ref.quant3 import Quant3Tensor, quantize_groupwise_3bit


def quantize_mxfp4_payload(
    packed: bytes,
    scales: bytes,
    *,
    rows: int,
    cols: int,
) -> Quant3Tensor:
    source = decode_mxfp4(packed, scales, rows, cols)
    return quantize_groupwise_3bit(source)
