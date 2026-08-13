# 공식 safetensors shard 하나를 독립 K3X 제조 단위로 변환하는 경계를 검증합니다.
import hashlib
import json

import torch
from safetensors.torch import save_file

import k3x_converter.local_shard as local_shard
from k3x_converter.format import Quantization, fnv1a64
from k3x_converter.local_shard import convert_local_official_shard
from k3x_converter.reader import K3XReader
from k3x_ref.fixtures import write_source_checkpoint
from k3x_ref.quant8 import Quant8Tensor, decode_groupwise_8bit


def test_local_shard_quantizes_matrix_and_preserves_sensitive_tensors(
    tmp_path, monkeypatch
) -> None:
    baseline = tmp_path / "baseline"
    write_source_checkpoint(baseline)
    config = json.loads((baseline / "source-manifest.json").read_text())["config"]
    source = tmp_path / "model-00001-of-000096.safetensors"
    matrix = torch.linspace(-1.0, 1.0, 256, dtype=torch.bfloat16).reshape(2, 128)
    payload = {
            "language_model.model.layers.0.input_layernorm.weight": torch.ones(
                128, dtype=torch.bfloat16
            ),
            "language_model.model.layers.0.mlp.gate_proj.weight": matrix,
            "language_model.model.layers.0.self_attn.A_log": torch.zeros(
                2, dtype=torch.float32
            ),
    }
    for expert_matrix in ("w1", "w2", "w3"):
        payload[
            f"language_model.model.layers.1.block_sparse_moe.experts.0.{expert_matrix}.weight_packed"
        ] = torch.zeros(1, dtype=torch.uint8)
        payload[
            f"language_model.model.layers.1.block_sparse_moe.experts.0.{expert_matrix}.weight_scale"
        ] = torch.ones(1, dtype=torch.uint8)
    save_file(payload, source)
    source_sha256 = hashlib.sha256(source.read_bytes()).hexdigest()
    written_kinds = []
    write_microshard = local_shard._write_microshard

    def capture(path, outputs, *, chunk_bytes):
        written_kinds.extend(output.kind for output in outputs)
        return write_microshard(path, outputs, chunk_bytes=chunk_bytes)

    monkeypatch.setattr(local_shard, "_write_microshard", capture)

    report = convert_local_official_shard(
        source,
        tmp_path / "output",
        config=config,
        expected_sha256=source_sha256,
        chunk_bytes=256,
        temporary_directory=tmp_path / "staging-work",
    )

    assert report.source_sha256 == source_sha256
    assert report.quant8_tensor_count == 1
    assert report.native_expert_tensor_count == 6
    assert "copy" not in written_kinds
    assert report.tensor_count == 6
    assert not any((tmp_path / "staging-work").iterdir())
    reader = K3XReader.open(report.output_path)
    by_id = {record.tensor_id: record for record in reader.tensor_records}
    quantized = by_id[
        fnv1a64("model.layers.0.mlp.gate_proj.weight")
    ]
    assert quantized.quantization == Quantization.GROUPWISE_8BIT
    codes, scales = reader.read_tensor_extents(quantized)
    decoded = decode_groupwise_8bit(
        Quant8Tensor((2, 128), 256, 128, codes, scales)
    )
    assert torch.max(torch.abs(matrix.float() - decoded)).item() < 0.02
    assert by_id[
        fnv1a64("model.layers.0.input_layernorm.weight")
    ].quantization == Quantization.NONE
