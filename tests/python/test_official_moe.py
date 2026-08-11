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
    AssembledOfficialMoeSource,
    OfficialMoeRoute,
    OfficialMoeRouteCase,
    OfficialMoeRoutes,
    OfficialMoeSourceTensor,
    MaterializedRangeObject,
    assemble_official_moe_source,
    build_official_moe_source_tensors,
    derive_official_moe_routes,
    prepare_official_moe_hidden,
    materialize_official_range_object,
    materialize_official_moe_slice,
    official_moe_inputs,
    plan_official_moe_slice,
    route_official_hidden,
)
from k3x_converter.official_source import (
    ExpertPlan,
    OfficialConfig,
    OfficialFile,
    OfficialIndex,
    OfficialSnapshot,
    OfficialShardHeader,
    PlannedTensor,
)
from k3x_converter.safetensors_reader import TensorMetadata
from k3x_converter.format import K3XError
from k3x_converter.format import fnv1a64
from k3x_converter.official_transport import HttpResponse
from k3x_converter.reader import K3XReader
from k3x_converter.writer import ConversionReport, convert


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
        93,
        tuple(index for index in range(1, 92) if index % 4 != 0),
        96,
        128,
        4,
        -5.0,
        True,
        12,
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


def _plan_inputs_for_layer(
    layer_id: int, shard: str
) -> tuple[OfficialIndex, OfficialShardHeader]:
    marker = f".layers.{layer_id}."
    metadata = {
        item.name.replace(".layers.1.", marker): TensorMetadata(
            item.name.replace(".layers.1.", marker),
            item.dtype,
            item.shape,
            item.offset,
            item.length,
        )
        for item in _metadata().values()
    }
    index = OfficialIndex(
        sum(item.length for item in metadata.values()),
        MappingProxyType({name: shard for name in metadata}),
        (shard,),
        len(metadata),
        "8" * 64,
    )
    header = OfficialShardHeader(
        shard,
        16_990_911_504,
        818_696,
        818_704,
        MappingProxyType(metadata),
    )
    return index, header


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


def test_official_moe_plan_accepts_bounded_layer_two_identity() -> None:
    shard = "model-00003-of-000096.safetensors"
    index, header = _plan_inputs_for_layer(2, shard)

    plan = plan_official_moe_slice(index, header, _config(), layer_id=2)

    assert plan.layer_id == 2
    assert plan.shard_path == shard
    assert tuple(item.official_name for item in plan.always_active) == tuple(
        f"language_model.model.layers.2.{suffix}"
        for suffix in _ALWAYS_ACTIVE_ORDER
    )
    assert tuple(item.canonical_name for item in plan.always_active) == tuple(
        f"model.layers.2.{suffix}" for suffix in _ALWAYS_ACTIVE_ORDER
    )
    assert plan.always_active_bytes == 379_900_416


def test_official_moe_plan_rejects_layer_two_cross_shard_binding() -> None:
    shard = "model-00003-of-000096.safetensors"
    index, header = _plan_inputs_for_layer(2, shard)
    weight_map = dict(index.weight_map)
    first_name = next(iter(weight_map))
    weight_map[first_name] = _SHARD
    mixed_index = OfficialIndex(
        index.total_size,
        MappingProxyType(weight_map),
        tuple(sorted({_SHARD, shard})),
        index.tensor_count,
        index.sha256,
    )

    with pytest.raises(K3XError, match="INVALID_OFFICIAL_MOE_TENSOR"):
        plan_official_moe_slice(mixed_index, header, _config(), layer_id=2)


@pytest.mark.parametrize("layer_id", [0, 3])
def test_official_moe_plan_rejects_layers_outside_bounded_pair(
    layer_id: int,
) -> None:
    index, header = _plan_inputs_for_layer(layer_id, _SHARD)

    with pytest.raises(K3XError, match="INVALID_OFFICIAL_MOE_CONFIG"):
        plan_official_moe_slice(index, header, _config(), layer_id=layer_id)


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
    assert first.response_bytes == 13
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
    assert second.response_bytes == 0
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
    assert result.response_bytes == 13
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
    assert result.response_bytes == 0
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


def test_route_derivation_decodes_exact_objects_and_builds_first_use_union(
    tmp_path: Path,
) -> None:
    metadata = _metadata()
    index = OfficialIndex(
        sum(item.length for item in metadata.values()),
        MappingProxyType({name: _SHARD for name in metadata}),
        (_SHARD,),
        len(metadata),
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
    prefix = f"{_PREFIX}."
    tensors = {
        prefix + "mlp_res_norm.weight": torch.linspace(
            0.5, 1.5, 7_168, dtype=torch.bfloat16
        ),
        prefix + "mlp_res_proj.weight": torch.linspace(
            -0.25, 0.25, 7_168, dtype=torch.bfloat16
        ).reshape(1, 7_168),
        prefix + "post_attention_layernorm.weight": torch.linspace(
            0.75, 1.25, 7_168, dtype=torch.bfloat16
        ),
        prefix + "block_sparse_moe.gate.e_score_correction_bias": (
            torch.arange(896, dtype=torch.float32).remainder(97) / 100.0
        ),
    }
    gate = torch.zeros((896, 7_168), dtype=torch.bfloat16)
    expert_ids = torch.arange(896)
    gate[expert_ids, (expert_ids * 37).remainder(7_168)] = (
        expert_ids.remainder(17).to(torch.float32) - 8.0
    ).to(torch.bfloat16) / 4.0
    tensors[prefix + "block_sparse_moe.gate.weight"] = gate
    objects = {}
    for name, tensor in tensors.items():
        path = tmp_path / f"{hashlib.sha256(name.encode()).hexdigest()}.blob"
        path.write_bytes(tensor.contiguous().view(torch.uint8).numpy().tobytes())
        objects[name] = MaterializedRangeObject(
            path,
            hashlib.sha256(path.read_bytes()).hexdigest(),
            path.stat().st_size,
            False,
            1,
            path.stat().st_size,
        )

    derived = derive_official_moe_routes(plan, objects)

    expected_routes = []
    for case_index, values in enumerate(
        (
            (17, 3, 257, 128, 29, 11, 251, 125),
            (31, 7, 263, 131, 43, 19, 269, 134),
        )
    ):
        pa, pb, pm, po, ba, bb, bm, bo = values
        prefix_sum = torch.tensor(
            [(((pa * i + pb) % pm) - po) / 1024.0 for i in range(7_168)],
            dtype=torch.bfloat16,
        )
        block = torch.tensor(
            [(((ba * i + bb) % bm) - bo) / 1024.0 for i in range(7_168)],
            dtype=torch.bfloat16,
        )
        stacked = torch.stack((block, prefix_sum)).float()
        normalized = stacked * torch.rsqrt(
            stacked.pow(2).mean(dim=-1, keepdim=True) + 1.0e-5
        )
        score_weight = tensors[prefix + "mlp_res_norm.weight"].float() * tensors[
            prefix + "mlp_res_proj.weight"
        ].flatten().float()
        probabilities = (normalized * score_weight).sum(dim=-1).softmax(dim=-1)
        hidden = (probabilities.unsqueeze(-1) * stacked).sum(dim=0).to(torch.bfloat16)
        hidden_float = hidden.float()
        hidden = (
            hidden_float
            * torch.rsqrt(hidden_float.pow(2).mean() + 1.0e-5)
            * tensors[prefix + "post_attention_layernorm.weight"].float()
        ).to(torch.bfloat16)
        scores = torch.sigmoid(gate.float() @ hidden.float())
        adjusted = scores + tensors[
            prefix + "block_sparse_moe.gate.e_score_correction_bias"
        ]
        selected = torch.topk(adjusted, 16, sorted=False).indices.tolist()
        canonical = tuple(sorted(selected, key=lambda e: (-float(adjusted[e]), e)))
        contributions = scores[list(canonical)]
        contributions = contributions / (contributions.sum() + 1.0e-20)
        expected_routes.append(
            ("a" if case_index == 0 else "b", canonical, tuple(float(v) for v in contributions))
        )

    assert tuple((case.name, case.route.expert_ids) for case in derived.cases) == tuple(
        (name, ids) for name, ids, _ in expected_routes
    )
    for case, (_, _, contributions) in zip(derived.cases, expected_routes):
        assert case.route.contributions == pytest.approx(contributions, abs=1.0e-7)
    expected_union = tuple(
        dict.fromkeys((*expected_routes[0][1], *expected_routes[1][1]))
    )
    assert derived.selected_experts == expected_union
    assert set(expected_routes[0][1]) != set(expected_routes[1][1])


def _small_expert_plan(expert_id: int, index_sha256: str) -> ExpertPlan:
    base = f"model.layers.1.feed_forward.experts.{expert_id}"
    specifications = (
        ("gate", "weight_packed", 100),
        ("gate", "weight_scale", 101),
        ("down", "weight_packed", 102),
        ("down", "weight_scale", 103),
        ("up", "weight_packed", 104),
        ("up", "weight_scale", 105),
    )
    tensors = tuple(
        PlannedTensor(
            f"official.{expert_id}.{role}.{kind}",
            f"{base}.{role}.{kind}",
            role,
            "U8",
            (1,),
            offset,
            1,
        )
        for role, kind, offset in specifications
    )
    return ExpertPlan(1, expert_id, _SHARD, 100, 106, 6, index_sha256, tensors)


def test_source_tensor_builder_uses_route_union_and_gate_up_down_first_use_order(
    tmp_path: Path,
) -> None:
    metadata = _metadata()
    index = OfficialIndex(
        sum(item.length for item in metadata.values()),
        MappingProxyType({name: _SHARD for name in metadata}),
        (_SHARD,),
        len(metadata),
        "7" * 64,
    )
    header = OfficialShardHeader(
        _SHARD, 16_990_911_504, 818_696, 818_704, MappingProxyType(metadata)
    )
    plan = plan_official_moe_slice(index, header, _config(), layer_id=1)
    routes = OfficialMoeRoutes(
        (
            OfficialMoeRouteCase("a", OfficialMoeRoute((9, 3), (0.6, 0.4))),
            OfficialMoeRouteCase("b", OfficialMoeRoute((3, 5), (0.7, 0.3))),
        ),
        (9, 3, 5),
    )
    always_objects = {
        item.official_name: MaterializedRangeObject(
            tmp_path / f"always-{position}.blob",
            "a" * 64,
            item.length,
            False,
            1,
            1,
        )
        for position, item in enumerate(plan.always_active)
    }
    expert_plans = {
        expert_id: _small_expert_plan(expert_id, index.sha256)
        for expert_id in routes.selected_experts
    }
    expert_objects = {
        expert_id: MaterializedRangeObject(
            tmp_path / f"expert-{expert_id}.blob", "b" * 64, 6, False, 1, 1
        )
        for expert_id in routes.selected_experts
    }

    tensors = build_official_moe_source_tensors(
        plan, routes, expert_plans, always_objects, expert_objects
    )

    expert_weights = [
        item
        for item in tensors
        if ".feed_forward.experts." in item.name
        and item.name.endswith("weight_packed")
    ]
    assert [item.name.rsplit(".", 2)[-2] for item in expert_weights] == [
        "gate", "up", "down", "gate", "up", "down", "gate", "up", "down"
    ]
    assert [
        item.name.split(".experts.", 1)[1].split(".", 1)[0]
        for item in expert_weights
    ] == ["9", "9", "9", "3", "3", "3", "5", "5", "5"]
    assert tensors[0].name.endswith("mlp_res_norm.weight")
    assert tensors[-1].name.endswith("shared_experts.down_proj.weight")


def test_materializer_publishes_routes_before_experts_and_returns_verified_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    metadata = _metadata()
    index = OfficialIndex(
        sum(item.length for item in metadata.values()),
        MappingProxyType({name: _SHARD for name in metadata}),
        (_SHARD,),
        len(metadata),
        "7" * 64,
    )
    header = OfficialShardHeader(
        _SHARD, 16_990_911_504, 818_696, 818_704, MappingProxyType(metadata)
    )
    plan = plan_official_moe_slice(index, header, _config(), layer_id=1)
    snapshot = OfficialSnapshot(
        "moonshotai/Kimi-K3",
        "main",
        "9" * 40,
        "2026-08-11T00:00:00Z",
        MappingProxyType(
            {
                _SHARD: OfficialFile(
                    _SHARD, header.file_size, "1" * 40, "2" * 64
                )
            }
        ),
        1,
        header.file_size,
        "3" * 64,
    )
    routes = OfficialMoeRoutes(
        (
            OfficialMoeRouteCase("a", OfficialMoeRoute((7,), (1.0,))),
            OfficialMoeRouteCase("b", OfficialMoeRoute((8,), (1.0,))),
        ),
        (7, 8),
    )
    events: list[str] = []
    object_counter = 0

    def fake_object(*args, **kwargs):
        nonlocal object_counter
        object_counter += 1
        path = tmp_path / f"object-{object_counter}.blob"
        path.write_bytes(b"x")
        return MaterializedRangeObject(
            path, f"{object_counter:064x}", args[3], False, 1, 1
        )

    def fake_routes(*args, **kwargs):
        events.append("routes")
        return routes

    def fake_expert(*args, expert_id: int, **kwargs):
        assert (tmp_path / "out" / "route-manifest.json").is_file()
        events.append(f"expert-{expert_id}")
        return _small_expert_plan(expert_id, index.sha256)

    assembled_dir = tmp_path / "assembled-source"
    assembled_dir.mkdir()
    manifest_path = assembled_dir / "source-manifest.json"
    microshard_path = assembled_dir / "model.safetensors"
    manifest_path.write_text("{}\n", encoding="utf-8")
    microshard_path.write_bytes(b"source")

    def fake_assemble(*args, **kwargs):
        events.append("assemble")
        return AssembledOfficialMoeSource(
            assembled_dir,
            manifest_path,
            microshard_path,
            "4" * 64,
            {"t": "5" * 64},
        )

    def fake_convert(source, output, **kwargs):
        events.append("convert")
        Path(output).write_bytes(b"k3x")
        return ConversionReport(True, (), 1, Path(output))

    class _Reader:
        class _Superblock:
            root_sha256 = bytes.fromhex("6" * 64)
            optional_features = 3

        superblock = _Superblock()

    import k3x_converter.official_moe as module

    monkeypatch.setattr(module, "materialize_official_range_object", fake_object)
    monkeypatch.setattr(module, "derive_official_moe_routes", fake_routes)
    monkeypatch.setattr(module, "plan_official_expert", fake_expert)
    monkeypatch.setattr(module, "assemble_official_moe_source", fake_assemble)
    monkeypatch.setattr(module, "convert", fake_convert)
    monkeypatch.setattr(module.K3XReader, "open", lambda path: _Reader())

    report = materialize_official_moe_slice(
        snapshot,
        index,
        _config(),
        header,
        plan,
        object(),
        tmp_path / "out",
        chunk_bytes=17,
    )

    assert events == ["routes", "expert-7", "expert-8", "assemble", "convert"]
    assert report.selected_experts == (7, 8)
    assert report.requested_payload_bytes == plan.always_active_bytes + 12
    assert report.k3x_root_sha256 == "6" * 64
    assert report.route_manifest_path.is_file()
    final_manifest = json.loads(report.route_manifest_path.read_text(encoding="utf-8"))
    assert final_manifest["artifact"] == {
        "filename": "official-moe-l1.k3x",
        "k3x_root_sha256": "6" * 64,
        "source_sha256": "4" * 64,
        "tensor_sha256": {"t": "5" * 64},
    }
