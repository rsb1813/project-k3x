# 공식 Kimi K3 전체 텍스트 토폴로지의 메타데이터 계약을 검증합니다.
from types import MappingProxyType

from k3x_converter.official_source import (
    OfficialConfig,
    OfficialIndex,
    OfficialShardHeader,
)
from k3x_converter.official_topology import build_official_topology
from k3x_converter.safetensors_reader import TensorMetadata


def _config() -> OfficialConfig:
    kda_layers = tuple(
        index for index in range(1, 94) if index % 4 != 0 and index != 93
    )
    return OfficialConfig(
        "1" * 64,
        "2" * 40,
        7_168,
        896,
        16,
        3_584,
        3_072,
        2,
        4.0,
        25.0,
        True,
        1.0e-5,
        True,
        "sigmoid",
        1,
        1,
        1.0,
        93,
        kda_layers,
        96,
        128,
        4,
        -5.0,
        True,
        12,
    )


def test_build_official_topology_classifies_complete_text_graph() -> None:
    shard = "model-00001-of-000096.safetensors"
    names = [
        "language_model.model.embed_tokens.weight",
        "language_model.model.norm.weight",
        "language_model.model.output_attn_res_norm.weight",
        "language_model.model.output_attn_res_proj.weight",
        "language_model.lm_head.weight",
        "vision_tower.patch_embed.weight",
    ]
    names.extend(
        f"language_model.model.layers.{layer}.sentinel.weight"
        for layer in range(93)
    )
    tensors = {
        name: TensorMetadata(name, "BF16", (1,), 1_024 + 2 * offset, 2)
        for offset, name in enumerate(names)
    }
    index = OfficialIndex(
        2 * len(names),
        MappingProxyType({name: shard for name in names}),
        (shard,),
        len(names),
        "3" * 64,
    )
    header = OfficialShardHeader(
        shard,
        1_024 + 2 * len(names),
        1_016,
        1_024,
        MappingProxyType(tensors),
    )

    topology = build_official_topology(index, _config(), {shard: header})
    record = topology.to_record()

    assert record["layer_count"] == 93
    assert record["kda_layer_count"] == 69
    assert record["mla_layer_count"] == 24
    assert record["dense_layer_count"] == 1
    assert record["moe_layer_count"] == 92
    assert record["layers"][0] == {
        "layer_id": 0,
        "attention": "kda",
        "feed_forward": "dense",
        "tensor_count": 1,
        "tensor_bytes": 2,
        "shards": [shard],
    }
    assert record["layers"][92]["attention"] == "mla"
    assert record["kda_layers"][:3] == [0, 1, 2]
    assert record["mla_layers"][:2] == [3, 7]
    assert record["mla_layers"][-1] == 92
    assert record["global_text_tensors"] == sorted(names[:5])
    assert record["text_tensor_count"] == 98
    assert record["text_tensor_bytes"] == 196
    assert record["non_text_tensor_count"] == 1
    assert record["non_text_tensor_bytes"] == 2
