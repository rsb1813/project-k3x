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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
def test_native_mxfp4_expert_batch_matches_ordered_portable_mix() -> None:
    from k3x_converter.mxfp4_cuda import mxfp4_expert_batch
    from k3x_converter.official_two_layer import _situ

    latent_size = 64
    intermediate_size = 32
    value = torch.linspace(-0.75, 0.75, latent_size, dtype=torch.bfloat16)
    contributions = torch.tensor((0.625, 0.375), dtype=torch.float32)
    experts = []
    expected = torch.zeros(latent_size, dtype=torch.float32)
    for expert_index in range(2):
        tensors = []
        for role_index, (rows, columns) in enumerate(
            (
                (intermediate_size, latent_size),
                (intermediate_size, latent_size),
                (latent_size, intermediate_size),
            )
        ):
            packed_length = rows * columns // 2
            scale_length = rows * columns // 32
            packed_bytes = bytes(
                (index * 17 + expert_index * 31 + role_index * 11) & 0xFF
                for index in range(packed_length)
            )
            scale_bytes = bytes(
                126 + ((index + expert_index + role_index) % 4)
                for index in range(scale_length)
            )
            tensors.append(
                (
                    torch.frombuffer(
                        bytearray(packed_bytes), dtype=torch.uint8
                    ).to("cuda"),
                    torch.frombuffer(
                        bytearray(scale_bytes), dtype=torch.uint8
                    ).to("cuda"),
                )
            )
        gate = mxfp4_matmul(
            value,
            bytes(tensors[0][0].cpu().tolist()),
            bytes(tensors[0][1].cpu().tolist()),
            intermediate_size,
            latent_size,
        )
        up = mxfp4_matmul(
            value,
            bytes(tensors[1][0].cpu().tolist()),
            bytes(tensors[1][1].cpu().tolist()),
            intermediate_size,
            latent_size,
        )
        activated = _situ(gate, up, 4.0, 25.0)
        down = mxfp4_matmul(
            activated,
            bytes(tensors[2][0].cpu().tolist()),
            bytes(tensors[2][1].cpu().tolist()),
            latent_size,
            intermediate_size,
        )
        expected += contributions[expert_index] * down
        experts.append(tuple(tensors))

    actual = mxfp4_expert_batch(
        value.to("cuda"), experts, contributions.to("cuda")
    ).cpu()

    assert actual.dtype == torch.float32
    assert torch.allclose(actual, expected, atol=2e-5, rtol=1e-5)
