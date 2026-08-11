# 공식 Kimi K3 두 레이어 제조 계획과 의존 실행 순서를 검증합니다.
from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import struct
from pathlib import Path

import pytest
import torch

from k3x_ref.official_kda import OfficialKdaConfig, zero_official_kda_state

from k3x_converter.format import (
    K3XError,
    OPTIONAL_OFFICIAL_MOE_FIXTURE,
    OPTIONAL_STORAGE_FIXTURE,
    fnv1a64,
)
from k3x_converter.official_layer import OfficialLayerPlan
from k3x_converter.official_layer import OfficialLayerInput
from k3x_converter.official_moe import (
    OfficialMoePlan,
    OfficialMoeRoute,
    OfficialMoeSourceTensor,
)
from k3x_converter.official_two_layer import (
    OfficialLayerSourceBytes,
    OfficialMxfp4ExpertBytes,
    OfficialMxfp4MatrixBytes,
    OfficialSourceTensorBytes,
    OfficialTwoLayerState,
    OfficialTwoLayerStepExecution,
    derive_official_two_layer_trace,
    finish_official_source_step,
    make_official_source_byte_executor,
    manufacture_official_two_layer_fixture,
    official_two_layer_state,
    plan_official_two_layer,
    prepare_official_source_step,
)
from k3x_converter.reader import K3XReader


_SOURCE_BLOB = "b8c41e8bfce768d74d8da3a37e693f5ee43876a0"


def _text_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _float_digest(values: tuple[float, ...]) -> str:
    return hashlib.sha256(struct.pack(f"<{len(values)}f", *values)).hexdigest()


def _layer_plan(layer_id: int) -> OfficialLayerPlan:
    shard = f"model-{layer_id + 1:05d}-of-000096.safetensors"
    moe = OfficialMoePlan(
        layer_id,
        shard,
        "7" * 64,
        (),
        379_900_416,
        17_547_264,
        941_412_864,
    )
    return OfficialLayerPlan(
        layer_id,
        shard,
        "7" * 64,
        _SOURCE_BLOB,
        (),
        887_843_840,
        moe,
        1_267_744_256,
        1_829_256_704,
    )


def _source_tensor(value: torch.Tensor) -> OfficialSourceTensorBytes:
    contiguous = value.contiguous()
    return OfficialSourceTensorBytes(
        "BF16" if contiguous.dtype == torch.bfloat16 else "F32",
        tuple(contiguous.shape),
        contiguous.view(torch.uint8).numpy().tobytes(),
    )


def _tiny_source(layer_id: int) -> OfficialLayerSourceBytes:
    hidden = 2
    projection = 2
    latent = 32
    roles = {
        "self_res_norm": torch.ones(hidden, dtype=torch.bfloat16),
        "self_res_proj": torch.tensor([[0.25, -0.125]], dtype=torch.bfloat16),
        "input_norm": torch.ones(hidden, dtype=torch.bfloat16),
        "kda_q_proj": torch.tensor(
            [[0.25, 0.0], [0.0, 0.25]], dtype=torch.bfloat16
        ),
        "kda_k_proj": torch.tensor(
            [[0.125, 0.0], [0.0, 0.125]], dtype=torch.bfloat16
        ),
        "kda_v_proj": torch.tensor(
            [[0.5, 0.0], [0.0, 0.5]], dtype=torch.bfloat16
        ),
        "kda_q_conv": torch.full((projection, 2), 0.5, dtype=torch.float32),
        "kda_k_conv": torch.full((projection, 2), 0.5, dtype=torch.float32),
        "kda_v_conv": torch.full((projection, 2), 0.5, dtype=torch.float32),
        "kda_f_a": torch.tensor(
            [[0.125, -0.125], [0.25, 0.125]], dtype=torch.bfloat16
        ),
        "kda_f_b": torch.eye(projection, dtype=torch.bfloat16),
        "kda_a_log": torch.zeros(projection, dtype=torch.float32),
        "kda_dt_bias": torch.zeros(projection, dtype=torch.float32),
        "kda_beta": torch.tensor([[0.25, 0.25]], dtype=torch.bfloat16),
        "kda_output_gate": torch.eye(projection, dtype=torch.bfloat16),
        "kda_output_norm": torch.ones(projection, dtype=torch.float32),
        "kda_output_proj": torch.eye(hidden, dtype=torch.bfloat16),
        "mlp_res_norm": torch.ones(hidden, dtype=torch.bfloat16),
        "mlp_res_proj": torch.tensor([[0.125, -0.25]], dtype=torch.bfloat16),
        "post_attention_norm": torch.ones(hidden, dtype=torch.bfloat16),
        "router": torch.tensor(
            [[1.0, -0.5], [-0.5, 1.0]], dtype=torch.bfloat16
        ) if layer_id == 1 else torch.tensor(
            [[-0.5, 1.0], [1.0, -0.5]], dtype=torch.bfloat16
        ),
        "router_correction": torch.zeros(2, dtype=torch.float32),
        "routed_down": torch.full((latent, hidden), 0.015625, dtype=torch.bfloat16),
        "routed_norm": torch.ones(latent, dtype=torch.bfloat16),
        "routed_up": torch.full((hidden, latent), 0.0078125, dtype=torch.bfloat16),
        "shared_gate": torch.full((2, hidden), 0.03125, dtype=torch.bfloat16),
        "shared_up": torch.full((2, hidden), 0.015625, dtype=torch.bfloat16),
        "shared_down": torch.full((hidden, 2), 0.0625, dtype=torch.bfloat16),
    }
    matrix = OfficialMxfp4MatrixBytes(
        bytes([0x11]) * (latent * latent // 2),
        bytes([120]) * (latent * latent // 32),
        latent,
        latent,
        32,
    )
    experts = tuple(
        OfficialMxfp4ExpertBytes(expert_id, matrix, matrix, matrix)
        for expert_id in range(2)
    )
    return OfficialLayerSourceBytes(
        layer_id,
        OfficialKdaConfig(hidden, 1, projection, 2, 1.0e-5, -5.0),
        tuple((role, _source_tensor(value)) for role, value in roles.items()),
        experts,
        1,
        1.0e-5,
        4.0,
        25.0,
    )


def test_official_two_layer_plan_binds_exact_order_and_byte_bounds() -> None:
    first = _layer_plan(1)
    second = _layer_plan(2)

    plan = plan_official_two_layer(first, second)

    assert plan.layers == (first, second)
    assert plan.layer_ids == (1, 2)
    assert plan.base_payload_bytes == 2_535_488_512
    assert plan.maximum_two_position_bytes == 3_658_513_408
    assert plan.shard_paths == (
        "model-00002-of-000096.safetensors",
        "model-00003-of-000096.safetensors",
    )


@pytest.mark.parametrize("layer_ids", [(2, 1), (1, 1), (2, 2)])
def test_official_two_layer_plan_rejects_noncanonical_layer_order(
    layer_ids: tuple[int, int],
) -> None:
    with pytest.raises(K3XError, match="INVALID_OFFICIAL_TWO_LAYER_PLAN"):
        plan_official_two_layer(*(_layer_plan(layer_id) for layer_id in layer_ids))


@pytest.mark.parametrize("field", ["index_sha256", "source_blob_id"])
def test_official_two_layer_plan_rejects_cross_layer_source_drift(field: str) -> None:
    first = _layer_plan(1)
    second = replace(_layer_plan(2), **{field: "8" * 64})

    with pytest.raises(K3XError, match="INVALID_OFFICIAL_TWO_LAYER_PLAN"):
        plan_official_two_layer(first, second)


def test_official_two_layer_plan_rejects_matching_unpinned_source_blobs() -> None:
    first = replace(_layer_plan(1), source_blob_id="8" * 40)
    second = replace(_layer_plan(2), source_blob_id="8" * 40)

    with pytest.raises(K3XError, match="INVALID_OFFICIAL_TWO_LAYER_PLAN"):
        plan_official_two_layer(first, second)


def test_official_two_layer_trace_interleaves_positions_and_layer_states() -> None:
    plan = plan_official_two_layer(_layer_plan(1), _layer_plan(2))
    inputs = (
        OfficialLayerInput(
            "a",
            (1.0, 2.0),
            (10.0, 20.0),
            _float_digest((1.0, 2.0)),
            _float_digest((10.0, 20.0)),
        ),
        OfficialLayerInput(
            "b",
            (3.0, 4.0),
            (30.0, 40.0),
            _float_digest((3.0, 4.0)),
            _float_digest((30.0, 40.0)),
        ),
    )
    layer_1_initial = _text_digest("layer-1-initial")
    layer_2_initial = _text_digest("layer-2-initial")
    states = (
        OfficialTwoLayerState(1, layer_1_initial),
        OfficialTwoLayerState(2, layer_2_initial),
    )
    calls: list[tuple[str, int, tuple[float, ...], tuple[float, ...], str]] = []

    def execute(layer, item, state):
        calls.append(
            (
                item.name,
                layer.layer_id,
                item.hidden_input,
                item.block_source,
                state.sha256,
            )
        )
        output = tuple(value + layer.layer_id for value in item.hidden_input)
        return OfficialTwoLayerStepExecution(
            output,
            OfficialTwoLayerState(
                state.value + 10,
                _text_digest(f"layer-{layer.layer_id}-{item.name}"),
            ),
            _text_digest(f"kda-{layer.layer_id}-{item.name}"),
            OfficialMoeRoute((layer.layer_id, layer.layer_id + 10), (0.75, 0.25)),
        )

    trace = derive_official_two_layer_trace(plan, inputs, states, execute)

    assert tuple((step.position, step.layer_id) for step in trace.steps) == (
        ("a", 1),
        ("a", 2),
        ("b", 1),
        ("b", 2),
    )
    assert calls == [
        ("a", 1, (1.0, 2.0), (10.0, 20.0), layer_1_initial),
        ("a", 2, (2.0, 3.0), (10.0, 20.0), layer_2_initial),
        ("b", 1, (3.0, 4.0), (30.0, 40.0), _text_digest("layer-1-a")),
        ("b", 2, (4.0, 5.0), (30.0, 40.0), _text_digest("layer-2-a")),
    ]
    assert trace.steps[1].hidden_input_sha256 == trace.steps[0].output_sha256
    assert trace.steps[3].hidden_input_sha256 == trace.steps[2].output_sha256
    assert trace.steps[2].consumes_state_sha256 == trace.steps[0].state_sha256
    assert trace.steps[3].consumes_state_sha256 == trace.steps[1].state_sha256
    assert trace.selected_experts == ((1, 11), (2, 12))
    assert trace.final_state_sha256 == (
        _text_digest("layer-1-b"),
        _text_digest("layer-2-b"),
    )
    assert trace.outputs == ((4.0, 5.0), (6.0, 7.0))


def test_official_two_layer_source_bytes_execute_exact_interleaved_trace() -> None:
    config = OfficialKdaConfig(2, 1, 2, 2, 1.0e-5, -5.0)
    zero_1 = zero_official_kda_state(config, 1, torch.device("cpu"))
    zero_2 = zero_official_kda_state(config, 1, torch.device("cpu"))
    states = (official_two_layer_state(zero_1), official_two_layer_state(zero_2))
    inputs = (
        OfficialLayerInput(
            "a",
            (0.5, -0.25),
            (0.125, 0.375),
            _float_digest((0.5, -0.25)),
            _float_digest((0.125, 0.375)),
        ),
        OfficialLayerInput(
            "b",
            (-0.375, 0.625),
            (0.25, -0.125),
            _float_digest((-0.375, 0.625)),
            _float_digest((0.25, -0.125)),
        ),
    )
    execute = make_official_source_byte_executor((_tiny_source(1), _tiny_source(2)))

    trace = derive_official_two_layer_trace(
        plan_official_two_layer(_layer_plan(1), _layer_plan(2)),
        inputs,
        states,
        execute,
    )
    assert tuple((step.position, step.layer_id) for step in trace.steps) == (
        ("a", 1),
        ("a", 2),
        ("b", 1),
        ("b", 2),
    )
    assert trace.steps[1].hidden_input_sha256 == trace.steps[0].output_sha256
    assert trace.steps[3].hidden_input_sha256 == trace.steps[2].output_sha256
    assert trace.steps[2].consumes_state_sha256 == trace.steps[0].state_sha256
    assert trace.steps[3].consumes_state_sha256 == trace.steps[1].state_sha256
    for step in trace.steps:
        contribution_payload = struct.pack(
            f"<{len(step.route.expert_ids)}I", *step.route.expert_ids
        ) + struct.pack(
            f"<{len(step.route.contributions)}f", *step.route.contributions
        )
        assert step.contribution_sha256 == hashlib.sha256(
            contribution_payload
        ).hexdigest()
    assert trace.selected_experts == ((0, 1), (1,))
    assert trace.outputs == (
        (2.765625, -0.267578125),
        (1.296875, 1.6171875),
    )


def test_official_source_byte_executor_rejects_truncated_dense_payload() -> None:
    source = _tiny_source(1)
    role, tensor = source.tensors[0]
    damaged = replace(
        source,
        tensors=((role, replace(tensor, payload=tensor.payload[:-1])),)
        + source.tensors[1:],
    )
    execute = make_official_source_byte_executor((damaged, _tiny_source(2)))
    config = source.kda_config
    state = official_two_layer_state(
        zero_official_kda_state(config, 1, torch.device("cpu"))
    )
    item = OfficialLayerInput(
        "a",
        (0.5, -0.25),
        (0.125, 0.375),
        _float_digest((0.5, -0.25)),
        _float_digest((0.125, 0.375)),
    )

    with pytest.raises(K3XError, match="INVALID_OFFICIAL_TWO_LAYER_SOURCE"):
        execute(_layer_plan(1), item, state)


def test_official_source_prepare_routes_before_expert_bytes_arrive() -> None:
    source = replace(_tiny_source(1), experts=())
    state = official_two_layer_state(
        zero_official_kda_state(source.kda_config, 1, torch.device("cpu"))
    )
    item = OfficialLayerInput(
        "a",
        (0.5, -0.25),
        (0.125, 0.375),
        _float_digest((0.5, -0.25)),
        _float_digest((0.125, 0.375)),
    )

    prepared = prepare_official_source_step(_layer_plan(1), item, state, source)

    assert prepared.route.expert_ids == (0,)
    with pytest.raises(K3XError, match="INVALID_OFFICIAL_TWO_LAYER_SOURCE"):
        finish_official_source_step(prepared, source)


def _tiny_artifact_tensors(
    tmp_path: Path,
) -> tuple[tuple[OfficialMoeSourceTensor, ...], list[str]]:
    tensors: list[OfficialMoeSourceTensor] = []
    expected_order: list[str] = []
    for layer_id in (1, 2):
        dense_name = f"model.layers.{layer_id}.input_layernorm.weight"
        dense_path = tmp_path / f"layer-{layer_id}-dense.bin"
        dense_path.write_bytes(struct.pack("<2H", 0x3F80, 0x4000))
        tensors.append(
            OfficialMoeSourceTensor(
                dense_name, "BF16", (2,), dense_path, 0, 4
            )
        )
        expected_order.append(dense_name)
        for role in ("gate", "up", "down"):
            base = f"model.layers.{layer_id}.feed_forward.experts.0.{role}"
            packed_path = tmp_path / f"layer-{layer_id}-{role}-packed.bin"
            scale_path = tmp_path / f"layer-{layer_id}-{role}-scale.bin"
            packed_path.write_bytes(bytes([0x11]) * 512)
            scale_path.write_bytes(bytes([120]) * 32)
            tensors.extend(
                (
                    OfficialMoeSourceTensor(
                        f"{base}.weight_packed",
                        "U8",
                        (512,),
                        packed_path,
                        0,
                        512,
                        (32, 32),
                    ),
                    OfficialMoeSourceTensor(
                        f"{base}.weight_scale",
                        "U8",
                        (32,),
                        scale_path,
                        0,
                        32,
                    ),
                )
            )
            expected_order.append(base)
    return tuple(tensors), expected_order


def test_official_two_layer_fixture_round_trips_execution_order(
    synthetic_source: Path,
    tmp_path: Path,
) -> None:
    config = json.loads(
        (synthetic_source / "source-manifest.json").read_text(encoding="utf-8")
    )["config"]
    tensors, expected_order = _tiny_artifact_tensors(tmp_path)

    report = manufacture_official_two_layer_fixture(
        tmp_path / "manufactured",
        tensors,
        config,
        {
            "format": "k3x-official-two-layer-v1",
            "layer_ids": [1, 2],
            "step_order": ["a:1", "a:2", "b:1", "b:2"],
        },
        chunk_bytes=97,
    )
    reader = K3XReader.open(report.k3x_path)
    manifest = json.loads(report.manifest_path.read_text(encoding="utf-8"))
    by_id = {record.tensor_id: record for record in reader.tensor_records}

    assert report.completed is True
    assert manifest["tensor_order"] == expected_order
    assert manifest["official_two_layer"]["layer_ids"] == [1, 2]
    assert tuple(
        record.layer_index
        for record in reader.layer_records
        if record.tensor_count
    ) == (1, 2)
    assert tuple(
        (record.layer_index, record.expert_id) for record in reader.expert_records
    ) == ((1, 0), (2, 0))
    assert [by_id[fnv1a64(name)].data_offset for name in expected_order] == sorted(
        by_id[fnv1a64(name)].data_offset for name in expected_order
    )
    assert reader.superblock.optional_features == (
        OPTIONAL_STORAGE_FIXTURE | OPTIONAL_OFFICIAL_MOE_FIXTURE
    )


def test_official_two_layer_fixture_resumes_verified_extents(
    synthetic_source: Path,
    tmp_path: Path,
) -> None:
    config = json.loads(
        (synthetic_source / "source-manifest.json").read_text(encoding="utf-8")
    )["config"]
    tensors, _ = _tiny_artifact_tensors(tmp_path)
    metadata = {
        "format": "k3x-official-two-layer-v1",
        "layer_ids": [1, 2],
        "step_order": ["a:1", "a:2", "b:1", "b:2"],
    }
    output = tmp_path / "resume"

    interrupted = manufacture_official_two_layer_fixture(
        output,
        tensors,
        config,
        metadata,
        chunk_bytes=97,
        stop_after_extents=3,
    )
    partial_path = interrupted.k3x_path.with_suffix(".k3x.partial")
    resume_path = interrupted.k3x_path.with_suffix(".k3x.resume.json")

    assert interrupted.completed is False
    assert partial_path.is_file()
    assert resume_path.is_file()

    resumed = manufacture_official_two_layer_fixture(
        output,
        tensors,
        config,
        metadata,
        chunk_bytes=97,
    )

    assert resumed.completed is True
    assert not partial_path.exists()
    assert not resume_path.exists()
    K3XReader.open(resumed.k3x_path)


def test_official_two_layer_fixture_rejects_incomplete_expert_before_publication(
    synthetic_source: Path,
    tmp_path: Path,
) -> None:
    config = json.loads(
        (synthetic_source / "source-manifest.json").read_text(encoding="utf-8")
    )["config"]
    tensors, _ = _tiny_artifact_tensors(tmp_path)
    output = tmp_path / "incomplete"

    with pytest.raises(
        K3XError, match="INVALID_OFFICIAL_TWO_LAYER_MATERIALIZATION"
    ):
        manufacture_official_two_layer_fixture(
            output,
            tensors[:-1],
            config,
            {
                "format": "k3x-official-two-layer-v1",
                "layer_ids": [1, 2],
                "step_order": ["a:1", "a:2", "b:1", "b:2"],
            },
            chunk_bytes=97,
        )

    assert not output.exists()
