# 비전문가 BF16 행렬의 group-128 8비트 제조 표현을 검증합니다.
import json

import torch
from safetensors.torch import save_file

from k3x_converter.format import Quantization, REQUIRED_QUANT8_TENSORS, fnv1a64
from k3x_converter.reader import K3XReader
from k3x_converter.writer import convert
from k3x_ref.fixtures import write_source_checkpoint
from k3x_ref.quant8 import decode_groupwise_8bit, quantize_groupwise_8bit


def test_groupwise_8bit_round_trip_has_fixed_budget() -> None:
    assert Quantization.GROUPWISE_8BIT == 3
    assert REQUIRED_QUANT8_TENSORS == 1 << 2
    source = torch.linspace(-2.0, 2.0, 256, dtype=torch.float32).reshape(2, 128)

    encoded = quantize_groupwise_8bit(source)
    decoded = decode_groupwise_8bit(encoded)

    assert encoded.shape == (2, 128)
    assert encoded.group_size == 128
    assert len(encoded.codes) == 256
    assert len(encoded.scales_bf16) == 4
    assert torch.max(torch.abs(source - decoded)).item() < 0.02


def test_writer_round_trips_groupwise_8bit_extents(tmp_path) -> None:
    baseline = tmp_path / "baseline"
    write_source_checkpoint(baseline)
    config = json.loads((baseline / "source-manifest.json").read_text())["config"]
    source = torch.linspace(-1.0, 1.0, 128).reshape(1, 128)
    encoded = quantize_groupwise_8bit(source)
    fixture = tmp_path / "q8"
    fixture.mkdir()
    shard = fixture / "model.safetensors"
    save_file(
        {
            "model.test.weight.q8_codes": torch.frombuffer(
                bytearray(encoded.codes), dtype=torch.uint8
            ).clone(),
            "model.test.weight.q8_scale": torch.frombuffer(
                bytearray(encoded.scales_bf16), dtype=torch.uint8
            ).clone(),
        },
        shard,
    )
    manifest = {
        "format": "synthetic-k3-source-v1",
        "config": config,
        "packed_shapes": {},
        "quant8_shapes": {"model.test.weight": [1, 128]},
        "weight_map": {
            "model.test.weight.q8_codes": shard.name,
            "model.test.weight.q8_scale": shard.name,
        },
    }
    (fixture / "source-manifest.json").write_text(json.dumps(manifest))
    output = tmp_path / "q8.k3x"

    assert convert(fixture, output).completed
    reader = K3XReader.open(output)
    record = reader.tensor_records[0]
    assert record.tensor_id == fnv1a64("model.test.weight")
    assert record.quantization == Quantization.GROUPWISE_8BIT
    assert reader.superblock.required_features & REQUIRED_QUANT8_TENSORS
    assert reader.read_tensor_extents(record) == (encoded.codes, encoded.scales_bf16)
