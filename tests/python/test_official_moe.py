# 공식 MoE FFN의 결정적 입력과 항상 활성 tensor 계획을 검증합니다.
from __future__ import annotations

import hashlib
import json
import struct
from pathlib import Path
from types import MappingProxyType

import pytest
import torch

from k3x_converter.official_moe import (
    OfficialMoeSourceTensor,
    assemble_official_moe_source,
    prepare_official_moe_hidden,
    materialize_official_range_object,
    official_moe_inputs,
    plan_official_moe_slice,
    route_official_hidden,
)
from k3x_converter.official_source import (
    OfficialConfig,
    OfficialFile,
    OfficialIndex,
    OfficialSnapshot,
    OfficialShardHeader,
)
from k3x_converter.safetensors_reader import TensorMetadata
from k3x_converter.format import K3XError
from k3x_converter.format import fnv1a64
from k3x_converter.official_transport import HttpResponse
from k3x_converter.reader import K3XReader
from k3x_converter.writer import convert


_SHARD = "model-00002-of-000096.safetensors"
_PREFIX = "language_model.model.layers.1"
_ALWAYS_ACTIVE_ORDER = (
    "mlp_res_norm.weight",
    "mlp_res_proj.weight",
    "post_attention_layernorm.weight",
    "block_sparse_moe.gate.weight",
    "block_sparse_moe.gate.e_score_correction_bias",
    "block_sparse_moe.routed_expert_down_proj.weight",
    "block_sparse_moe.routed_expert_norm.weight",
    "block_sparse_moe.routed_expert_up_proj.weight",
    "block_sparse_moe.shared_experts.gate_proj.weight",
    "block_sparse_moe.shared_experts.up_proj.weight",
    "block_sparse_moe.shared_experts.down_proj.weight",
)


def _config() -> OfficialConfig:
    return OfficialConfig(
        "5" * 64,
        "6" * 40,
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
    )


def _metadata() -> dict[str, TensorMetadata]:
    specifications = (
        ("block_sparse_moe.gate.e_score_correction_bias", "F32", (896,)),
        ("block_sparse_moe.gate.weight", "BF16", (896, 7_168)),
        ("block_sparse_moe.routed_expert_down_proj.weight", "BF16", (3_584, 7_168)),
        ("block_sparse_moe.routed_expert_norm.weight", "BF16", (3_584,)),
        ("block_sparse_moe.routed_expert_up_proj.weight", "BF16", (7_168, 3_584)),
        ("block_sparse_moe.shared_experts.gate_proj.weight", "BF16", (6_144, 7_168)),
        ("block_sparse_moe.shared_experts.up_proj.weight", "BF16", (6_144, 7_168)),
        ("block_sparse_moe.shared_experts.down_proj.weight", "BF16", (7_168, 6_144)),
        ("mlp_res_norm.weight", "BF16", (7_168,)),
        ("mlp_res_proj.weight", "BF16", (1, 7_168)),
        ("post_attention_layernorm.weight", "BF16", (7_168,)),
    )
    offset = 1_000_000
    result: dict[str, TensorMetadata] = {}
    for suffix, dtype, shape in specifications:
        name = f"{_PREFIX}.{suffix}"
        values = 1
        for dimension in shape:
            values *= dimension
        length = values * (4 if dtype == "F32" else 2)
        result[name] = TensorMetadata(name, dtype, shape, offset, length)
        offset += length
    return result


def test_official_moe_inputs_match_charter_formulas_and_little_endian_digests() -> None:
    cases = official_moe_inputs()

    assert tuple(case.name for case in cases) == ("a", "b")
    expected_formulas = (
        (17, 3, 257, 128, 29, 11, 251, 125),
        (31, 7, 263, 131, 43, 19, 269, 134),
    )
    for case, values in zip(cases, expected_formulas):
        pa, pb, pm, po, ba, bb, bm, bo = values
        expected_prefix = tuple(
            (((pa * index + pb) % pm) - po) / 1024.0
            for index in range(7_168)
        )
        expected_block = tuple(
            (((ba * index + bb) % bm) - bo) / 1024.0
            for index in range(7_168)
        )
        prefix_bytes = struct.pack("<7168f", *expected_prefix)
        block_bytes = struct.pack("<7168f", *expected_block)
        assert case.prefix_sum == expected_prefix
        assert case.block_residual == expected_block
        assert case.prefix_sha256 == hashlib.sha256(prefix_bytes).hexdigest()
        assert case.block_sha256 == hashlib.sha256(block_bytes).hexdigest()


def test_official_moe_plan_binds_exact_always_active_tensor_set_and_order() -> None:
    metadata = _metadata()
    weight_map = {name: _SHARD for name in metadata}
    index = OfficialIndex(
        sum(item.length for item in metadata.values()),
        MappingProxyType(weight_map),
        (_SHARD,),
        len(weight_map),
        "7" * 64,
    )
    header = OfficialShardHeader(
        _SHARD,
        16_990_911_504,
        818_696,
        818_704,
        MappingProxyType(metadata),
    )

    plan = plan_official_moe_slice(index, header, _config(), layer_id=1)

    assert tuple(item.official_name for item in plan.always_active) == tuple(
        f"{_PREFIX}.{suffix}" for suffix in _ALWAYS_ACTIVE_ORDER
    )
    assert tuple(item.dtype for item in plan.always_active) == (
        "BF16",
        "BF16",
        "BF16",
        "BF16",
        "F32",
        "BF16",
        "BF16",
        "BF16",
        "BF16",
        "BF16",
        "BF16",
    )
    assert plan.shard_path == _SHARD
    assert plan.always_active_bytes == 379_900_416
    assert plan.expert_payload_bytes == 17_547_264
    assert plan.maximum_two_case_bytes == 941_412_864
    assert plan.selected_experts == ()


def test_official_attention_residual_postnorm_and_router_match_literal_oracle() -> None:
    prefix = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.bfloat16)
    block = torch.tensor([-1.0, 0.0, 1.0, 2.0], dtype=torch.bfloat16)
    residual_norm = torch.tensor([1.0, 1.5, 0.5, 2.0], dtype=torch.bfloat16)
    residual_proj = torch.tensor([0.5, -0.25, 0.125, 0.0], dtype=torch.bfloat16)
    post_norm = torch.tensor([1.0, 0.5, 1.5, 2.0], dtype=torch.bfloat16)

    hidden = prepare_official_moe_hidden(
        prefix,
        block,
        residual_norm,
        residual_proj,
        post_norm,
        rms_norm_eps=1.0e-5,
    )

    assert hidden.dtype == torch.bfloat16
    assert hidden.view(torch.int16).tolist() == [15785, 16020, 16335, 16458]

    router_weight = torch.tensor(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [-1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 0.0, 0.0],
        ],
        dtype=torch.bfloat16,
    )
    correction = torch.tensor([0.0, 0.1, 0.2, 0.3, 0.4, 0.5])
    route = route_official_hidden(hidden, router_weight, correction, top_k=3)

    assert route.expert_ids == (3, 2, 5)
    assert route.contributions == pytest.approx(
        (0.4317025244, 0.3755553663, 0.1927421242), abs=1.0e-7
    )
    assert sum(route.contributions) == pytest.approx(1.0, abs=1.0e-7)


class _ObjectTransport:
    def __init__(self, body: bytes, *, fail_after: int | None = None) -> None:
        self.body = body
        self.fail_after = fail_after
        self.calls: list[tuple[int, int]] = []

    def get(
        self,
        url: str,
        *,
        headers: dict[str, str],
        max_bytes: int,
        timeout_seconds: float,
        expected_status: int = 200,
    ) -> HttpResponse:
        value = headers["Range"].removeprefix("bytes=")
        start_text, end_text = value.split("-", 1)
        start, end = int(start_text), int(end_text)
        assert end - start + 1 <= 4
        self.calls.append((start, end))
        if self.fail_after is not None and len(self.calls) > self.fail_after:
            raise K3XError("SIMULATED_RANGE_FAILURE")
        response = self.body[start : end + 1]
        return HttpResponse(
            206,
            url,
            {"content-range": f"bytes {start}-{end}/{len(self.body)}"},
            response,
        )


def _object_snapshot(length: int) -> OfficialSnapshot:
    shard = OfficialFile(_SHARD, length, "3" * 40, "4" * 64)
    return OfficialSnapshot(
        "moonshotai/Kimi-K3",
        "main",
        "9f62e4e9fffbd0a83ddd60e1c209d828994b3569",
        "2026-08-11T00:00:00Z",
        MappingProxyType({_SHARD: shard}),
        1,
        length,
        "5" * 64,
    )


def test_range_object_is_chunk_bounded_content_addressed_and_reused(
    tmp_path: Path,
) -> None:
    body = bytes(range(100))
    snapshot = _object_snapshot(len(body))
    transport = _ObjectTransport(body)

    first = materialize_official_range_object(
        snapshot,
        _SHARD,
        10,
        13,
        transport,
        tmp_path / "objects",
        chunk_bytes=4,
    )

    expected = body[10:23]
    assert first.path.read_bytes() == expected
    assert first.sha256 == hashlib.sha256(expected).hexdigest()
    assert first.reused is False
    assert first.requests == 4
    assert first.maximum_response_bytes == 4
    assert transport.calls == [(10, 13), (14, 17), (18, 21), (22, 22)]

    second_transport = _ObjectTransport(body)
    second = materialize_official_range_object(
        snapshot,
        _SHARD,
        10,
        13,
        second_transport,
        tmp_path / "objects",
        chunk_bytes=4,
    )
    assert second.path == first.path
    assert second.sha256 == first.sha256
    assert second.reused is True
    assert second.requests == 0
    assert second_transport.calls == []


def test_range_object_rejects_corrupt_partial_and_refetches_from_start(
    tmp_path: Path,
) -> None:
    body = bytes(range(100))
    snapshot = _object_snapshot(len(body))
    output = tmp_path / "objects"
    interrupted = _ObjectTransport(body, fail_after=2)

    with pytest.raises(K3XError, match="SIMULATED_RANGE_FAILURE"):
        materialize_official_range_object(
            snapshot,
            _SHARD,
            10,
            13,
            interrupted,
            output,
            chunk_bytes=4,
        )

    partial = next(output.glob("*.partial"))
    damaged = bytearray(partial.read_bytes())
    damaged[0] ^= 1
    partial.write_bytes(damaged)
    resumed = _ObjectTransport(body)

    result = materialize_official_range_object(
        snapshot,
        _SHARD,
        10,
        13,
        resumed,
        output,
        chunk_bytes=4,
    )

    assert result.path.read_bytes() == body[10:23]
    assert resumed.calls[0] == (10, 13)
    assert not tuple(output.glob("*.partial"))


def test_range_object_finalizes_complete_verified_partial_without_refetch(
    tmp_path: Path,
) -> None:
    body = bytes(range(100))
    snapshot = _object_snapshot(len(body))
    output = tmp_path / "objects"
    interrupted = _ObjectTransport(body, fail_after=2)
    with pytest.raises(K3XError, match="SIMULATED_RANGE_FAILURE"):
        materialize_official_range_object(
            snapshot,
            _SHARD,
            10,
            13,
            interrupted,
            output,
            chunk_bytes=4,
        )
    partial = next(output.glob("*.partial"))
    with partial.open("ab") as stream:
        stream.write(body[18:23])
    progress_path = next(output.glob("*.progress.json"))
    progress = json.loads(progress_path.read_text(encoding="utf-8"))
    progress["completed"] = 13
    progress["partial_sha256"] = hashlib.sha256(partial.read_bytes()).hexdigest()
    progress_path.write_text(
        json.dumps(progress, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    resumed = _ObjectTransport(body)

    result = materialize_official_range_object(
        snapshot,
        _SHARD,
        10,
        13,
        resumed,
        output,
        chunk_bytes=4,
    )

    assert result.path.read_bytes() == body[10:23]
    assert resumed.calls == []


def test_source_assembler_preserves_declared_physical_order_and_hashes(
    synthetic_source: Path, tmp_path: Path
) -> None:
    first_bytes = bytes.fromhex("803f0040")
    second_bytes = bytes.fromhex("40408040")
    first_object = tmp_path / "first.blob"
    second_object = tmp_path / "second.blob"
    first_object.write_bytes(first_bytes)
    second_object.write_bytes(second_bytes)
    config = json.loads(
        (synthetic_source / "source-manifest.json").read_text(encoding="utf-8")
    )["config"]
    first_name = "model.layers.0.first.weight"
    second_name = "model.layers.0.second.weight"

    assembled = assemble_official_moe_source(
        tmp_path / "assembled",
        (
            OfficialMoeSourceTensor(
                second_name, "BF16", (2,), second_object, 0, 4
            ),
            OfficialMoeSourceTensor(
                first_name, "BF16", (2,), first_object, 0, 4
            ),
        ),
        config,
        chunk_bytes=3,
    )

    manifest = json.loads(assembled.manifest_path.read_text(encoding="utf-8"))
    assert manifest["tensor_order"] == [second_name, first_name]
    assert manifest["source_sha256"] == hashlib.sha256(
        assembled.microshard_path.read_bytes()
    ).hexdigest()
    assert manifest["tensor_sha256"] == {
        first_name: hashlib.sha256(first_bytes).hexdigest(),
        second_name: hashlib.sha256(second_bytes).hexdigest(),
    }

    artifact = tmp_path / "assembled.k3x"
    convert(assembled.source_directory, artifact, chunk_bytes=3)
    reader = K3XReader.open(artifact)
    by_id = {record.tensor_id: record for record in reader.tensor_records}
    assert by_id[fnv1a64(second_name)].data_offset < by_id[
        fnv1a64(first_name)
    ].data_offset
