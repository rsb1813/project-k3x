# 공식 MXFP4 전문가를 3비트 제조 표현으로 변환하는 경계를 검증합니다.
import torch

from k3x_converter.official_quant3 import quantize_mxfp4_payload
from k3x_ref.quant3 import decode_groupwise_3bit
from tools.measure_official_quant3_expert import main


def test_quantize_mxfp4_payload_preserves_shape_and_budget() -> None:
    assert callable(main)
    packed = bytes([0x62] * 16)
    scales = bytes([127])

    result = quantize_mxfp4_payload(packed, scales, rows=1, cols=32)

    assert result.shape == (1, 32)
    assert len(result.packed) == 12
    assert len(result.scales_bf16) == 2
    decoded = decode_groupwise_3bit(result)
    assert torch.isfinite(decoded).all()
