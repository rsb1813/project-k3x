# native MXFP4 CUDA matvec이 독립 CPU 참조와 일치하는지 검증합니다.
import pytest
import torch

from k3x_ref.mxfp4 import mxfp4_matmul


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_native_mxfp4_cuda_matvec_matches_portable_reference() -> None:
    from k3x_converter.mxfp4_cuda import mxfp4_matvec

    rows = 4
    columns = 64
    packed_bytes = bytes((index * 29 + 7) & 0xFF for index in range(128))
    scale_bytes = bytes((126, 127, 128, 129, 127, 128, 126, 129))
    value = torch.linspace(-1.0, 1.0, columns, dtype=torch.bfloat16)
    expected = mxfp4_matmul(value, packed_bytes, scale_bytes, rows, columns)
    packed = torch.frombuffer(bytearray(packed_bytes), dtype=torch.uint8).to("cuda")
    scales = torch.frombuffer(bytearray(scale_bytes), dtype=torch.uint8).to("cuda")

    actual = mxfp4_matvec(value.to("cuda"), packed, scales, rows, columns).cpu()

    assert actual.dtype == torch.float32
    assert torch.allclose(actual, expected, atol=1e-5, rtol=1e-5)
