# group-128 Q8 직접 CUDA matvec이 독립 복호화 기준과 일치하는지 검증합니다.
import pytest
import torch

from k3x_ref.quant8 import decode_groupwise_8bit, quantize_groupwise_8bit


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_direct_q8_cuda_matvec_matches_decoded_reference() -> None:
    from k3x_converter.q8_cuda import q8_matvec

    source = torch.linspace(-2.0, 2.0, 512, dtype=torch.float32).reshape(4, 128)
    encoded = quantize_groupwise_8bit(source)
    value = torch.linspace(-1.0, 1.0, 128, dtype=torch.bfloat16)
    expected = (decode_groupwise_8bit(encoded) @ value.float()).to(torch.bfloat16)
    codes = torch.frombuffer(bytearray(encoded.codes), dtype=torch.int8).to("cuda")
    scales = torch.frombuffer(
        bytearray(encoded.scales_bf16), dtype=torch.bfloat16
    ).to("cuda")

    actual = q8_matvec(value.to("cuda"), codes, scales, 4, 128).cpu()

    assert actual.dtype == torch.bfloat16
    assert torch.allclose(actual.float(), expected.float(), atol=0.125, rtol=0.01)
