# 공식 Kimi K3 레이어 실행기가 K3X shard에서 텐서를 읽도록 연결합니다.
from __future__ import annotations

from pathlib import Path

import torch

from k3x_converter.fragment_tensor_store import K3XTensorStore
from k3x_converter.fragment_set import read_fragment_set_manifest
from k3x_converter.format import K3XError, Quantization


def k3x_set_identity(k3x_set: Path) -> str:
    return read_fragment_set_manifest(k3x_set).record_sha256


def require_k3x_state_identity(record: dict[str, object], identity: str) -> None:
    if record.get("k3x_set_manifest_sha256") != identity:
        raise K3XError("K3X_STATE_SET_MISMATCH")


def logical_torch_dtype(name: str) -> torch.dtype:
    if name == "BF16":
        return torch.bfloat16
    if name == "F32":
        return torch.float32
    raise K3XError("UNSUPPORTED_SOURCE_DTYPE", name)


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


def load_official_tensor(
    store,
    name: str,
    dtype: torch.dtype,
    device: torch.device,
    *,
    direct_q8: bool,
):
    record = store.record(name).record
    if (
        direct_q8
        and (
            ".mlp." in name
            or ".block_sparse_moe." in name
            or name.startswith("model.layers.0.self_attn.")
        )
        and dtype == torch.bfloat16
        and record.quantization == Quantization.GROUPWISE_8BIT
        and len(record.dimensions) == 2
        and record.dimensions[1] >= 128
        and record.dimensions[1] % 128 == 0
    ):
        return store.packed_q8_matrix(name, device=device)
    return store.load(name, device=device, dtype=dtype)


def load_planned_tensors(
    store, planned, device: torch.device, *, direct_q8: bool = False
):
    return {
        item.role: load_official_tensor(
            store,
            item.canonical_name,
            logical_torch_dtype(item.dtype),
            device,
            direct_q8=direct_q8,
        )
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
