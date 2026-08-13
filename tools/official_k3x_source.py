# 공식 Kimi K3 레이어 실행기가 K3X shard에서 텐서를 읽도록 연결합니다.
from __future__ import annotations

from pathlib import Path

import torch

from k3x_converter.fragment_tensor_store import K3XTensorStore
from k3x_converter.fragment_set import read_fragment_set_manifest
from k3x_converter.format import K3XError


def open_official_fragment(k3x_set: Path, source_shard: str) -> K3XTensorStore:
    suffix = ".safetensors"
    if not source_shard.endswith(suffix):
        raise K3XError("INVALID_OFFICIAL_SHARD", source_shard)
    filename = source_shard.removesuffix(suffix) + ".k3x"
    manifest = read_fragment_set_manifest(k3x_set)
    path = next((item for item in manifest.fragments if item.name == filename), None)
    if path is None:
        raise K3XError("K3X_FRAGMENT_NOT_FOUND", filename)
    return K3XTensorStore.open([path], verify_root=False, verify_payload=False)


def load_planned_tensors(store, planned, device: torch.device):
    return {
        item.role: store.load(item.canonical_name, device=device)
        for item in planned
    }


def expert_matvec(store, plan, role: str, value: torch.Tensor) -> torch.Tensor:
    packed = next(
        item
        for item in plan.tensors
        if item.role == role and item.canonical_name.endswith(".weight_packed")
    )
    return store.mxfp4_matvec(
        packed.canonical_name.removesuffix(".weight_packed"), value
    )
