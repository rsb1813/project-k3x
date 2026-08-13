# 공식 safetensors shard 하나를 독립 K3X 제조 단위로 변환하는 경계를 검증합니다.
import errno
import hashlib
import json

import torch
from safetensors.torch import save_file

import k3x_converter.local_shard as local_shard
from k3x_converter.format import Quantization, fnv1a64
from k3x_converter.fragment_tensor_store import K3XTensorStore
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
    inspected_paths = []
    hashed_paths = []
    write_microshard = local_shard._write_microshard
    inspect_shard = local_shard.inspect_shard
    sha256 = local_shard._sha256
    staging_ready = tmp_path / "source.ram-ready"

    def capture(path, outputs, *, chunk_bytes):
        assert staging_ready.is_file()
        written_kinds.extend(output.kind for output in outputs)
        return write_microshard(path, outputs, chunk_bytes=chunk_bytes)

    monkeypatch.setattr(local_shard, "_write_microshard", capture)

    def capture_inspect(path):
        inspected_paths.append(path)
        return inspect_shard(path)

    monkeypatch.setattr(local_shard, "inspect_shard", capture_inspect)

    def capture_sha256(path):
        hashed_paths.append(path)
        return sha256(path)

    monkeypatch.setattr(local_shard, "_sha256", capture_sha256)

    def reject_cross_device_link(source_path, destination_path):
        raise OSError(errno.EXDEV, "cross-device link")

    monkeypatch.setattr(local_shard.os, "link", reject_cross_device_link)

    report = convert_local_official_shard(
        source,
        tmp_path / "output",
        config=config,
        expected_sha256=source_sha256,
        chunk_bytes=256,
        temporary_directory=tmp_path / "staging-work",
        staging_ready_path=staging_ready,
    )

    assert report.source_sha256 == source_sha256
    assert staging_ready.is_file()
    assert report.quant8_tensor_count == 1
    assert report.native_expert_tensor_count == 6
    assert [path.name for path in inspected_paths] == ["official.safetensors"]
    assert [path.name for path in hashed_paths] == ["official.safetensors"]
    assert report.output_sha256 == hashlib.sha256(
        report.output_path.read_bytes()
    ).hexdigest()
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
    store = K3XTensorStore.open([report.output_path])
    loaded_matrix = store.load("model.layers.0.mlp.gate_proj.weight")
    loaded_norm = store.load("model.layers.0.input_layernorm.weight")
    assert torch.max(torch.abs(matrix.float() - loaded_matrix)).item() < 0.02
    assert loaded_norm.dtype == torch.bfloat16
    assert torch.equal(loaded_norm, torch.ones(128, dtype=torch.bfloat16))
