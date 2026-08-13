# 공식 Kimi K3 체크포인트의 전체 텍스트 토폴로지를 메타데이터만으로 분류합니다.
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping

from .format import K3XError
from .official_source import OfficialConfig, OfficialIndex, OfficialShardHeader


_LAYER = re.compile(r"language_model\.model\.layers\.(\d+)\.")


@dataclass(frozen=True)
class OfficialTopologyLayer:
    layer_id: int
    attention: str
    feed_forward: str
    tensor_count: int
    tensor_bytes: int
    shards: tuple[str, ...]

    def to_record(self) -> dict[str, object]:
        return {
            "layer_id": self.layer_id,
            "attention": self.attention,
            "feed_forward": self.feed_forward,
            "tensor_count": self.tensor_count,
            "tensor_bytes": self.tensor_bytes,
            "shards": list(self.shards),
        }


@dataclass(frozen=True)
class OfficialTopology:
    layer_count: int
    kda_layers: tuple[int, ...]
    mla_layers: tuple[int, ...]
    dense_layers: tuple[int, ...]
    moe_layers: tuple[int, ...]
    layers: tuple[OfficialTopologyLayer, ...]
    global_text_tensors: tuple[str, ...]
    text_tensor_count: int
    text_tensor_bytes: int
    non_text_tensor_count: int
    non_text_tensor_bytes: int

    def to_record(self) -> dict[str, object]:
        return {
            "layer_count": self.layer_count,
            "kda_layer_count": len(self.kda_layers),
            "mla_layer_count": len(self.mla_layers),
            "dense_layer_count": len(self.dense_layers),
            "moe_layer_count": len(self.moe_layers),
            "kda_layers": list(self.kda_layers),
            "mla_layers": list(self.mla_layers),
            "dense_layers": list(self.dense_layers),
            "moe_layers": list(self.moe_layers),
            "layers": [layer.to_record() for layer in self.layers],
            "global_text_tensors": list(self.global_text_tensors),
            "text_tensor_count": self.text_tensor_count,
            "text_tensor_bytes": self.text_tensor_bytes,
            "non_text_tensor_count": self.non_text_tensor_count,
            "non_text_tensor_bytes": self.non_text_tensor_bytes,
        }


def build_official_topology(
    index: OfficialIndex,
    config: OfficialConfig,
    headers: Mapping[str, OfficialShardHeader],
) -> OfficialTopology:
    if set(headers) != set(index.shard_paths):
        raise K3XError("OFFICIAL_TOPOLOGY_SHARD_SET")

    header_names: set[str] = set()
    tensor_bytes = 0
    for shard_path, header in headers.items():
        if header.shard_path != shard_path:
            raise K3XError("OFFICIAL_TOPOLOGY_SHARD_IDENTITY", shard_path)
        for name, tensor in header.tensors.items():
            if (
                name in header_names
                or tensor.name != name
                or index.weight_map.get(name) != shard_path
                or tensor.offset < header.data_start
                or tensor.offset + tensor.length > header.file_size
            ):
                raise K3XError("OFFICIAL_TOPOLOGY_TENSOR", name)
            header_names.add(name)
            tensor_bytes += tensor.length

    if header_names != set(index.weight_map) or tensor_bytes != index.total_size:
        raise K3XError("OFFICIAL_TOPOLOGY_INDEX_PARITY")

    if config.num_hidden_layers != 93:
        raise K3XError("OFFICIAL_TOPOLOGY_LAYER_COUNT")
    kda_layers = tuple(sorted(layer - 1 for layer in config.kda_layers))
    if (
        len(kda_layers) != 69
        or len(set(kda_layers)) != len(kda_layers)
        or any(layer < 0 or layer >= config.num_hidden_layers for layer in kda_layers)
    ):
        raise K3XError("OFFICIAL_TOPOLOGY_KDA_LAYOUT")
    kda_set = set(kda_layers)
    mla_layers = tuple(
        layer for layer in range(config.num_hidden_layers) if layer not in kda_set
    )
    if len(mla_layers) != 24 or mla_layers[-1] != 92:
        raise K3XError("OFFICIAL_TOPOLOGY_MLA_LAYOUT")

    layer_names: list[list[str]] = [[] for _ in range(config.num_hidden_layers)]
    global_text: list[str] = []
    text_bytes = 0
    text_count = 0
    for name in sorted(header_names):
        shard_path = index.weight_map[name]
        tensor = headers[shard_path].tensors[name]
        if not name.startswith("language_model."):
            continue
        text_count += 1
        text_bytes += tensor.length
        match = _LAYER.match(name)
        if match is None:
            global_text.append(name)
            continue
        layer = int(match.group(1))
        if layer < 0 or layer >= config.num_hidden_layers:
            raise K3XError("OFFICIAL_TOPOLOGY_LAYER_ID", name)
        layer_names[layer].append(name)

    if any(not names for names in layer_names):
        raise K3XError("OFFICIAL_TOPOLOGY_EMPTY_LAYER")

    layers = tuple(
        OfficialTopologyLayer(
            layer,
            "kda" if layer in kda_set else "mla",
            "dense" if layer == 0 else "moe",
            len(names),
            sum(
                headers[index.weight_map[name]].tensors[name].length for name in names
            ),
            tuple(sorted({index.weight_map[name] for name in names})),
        )
        for layer, names in enumerate(layer_names)
    )

    return OfficialTopology(
        config.num_hidden_layers,
        kda_layers,
        mla_layers,
        (0,),
        tuple(range(1, config.num_hidden_layers)),
        layers,
        tuple(global_text),
        text_count,
        text_bytes,
        len(header_names) - text_count,
        tensor_bytes - text_bytes,
    )
