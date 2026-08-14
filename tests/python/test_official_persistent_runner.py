# 공식 K3X 그래프 단계의 상주형 호출 계약을 검증합니다.
from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

from tools import (
    run_official_head,
    run_official_layer0,
    run_official_layer1,
    run_official_layer3,
    run_official_remaining,
)
from tools import official_runtime_context


def test_official_stages_expose_callable_run_entrypoints() -> None:
    for module in (
        run_official_layer0,
        run_official_layer1,
        run_official_layer3,
        run_official_head,
    ):
        assert callable(module.run)


def test_in_process_dispatch_runs_stage_without_child_process(
    tmp_path, monkeypatch
) -> None:
    output = tmp_path / "layer-01.json"
    layer_args = argparse.Namespace(output=output, layer_id=1)

    def publish(args: argparse.Namespace) -> int:
        args.output.write_text(
            json.dumps({"layer_id": args.layer_id}), encoding="utf-8"
        )
        return 0

    monkeypatch.setitem(run_official_remaining._IN_PROCESS_RUNNERS, "kda", publish)
    result = run_official_remaining._execute_layer(
        "in-process",
        "kda",
        ["a-command-that-must-not-run"],
        layer_args,
    )

    assert result == 0
    assert json.loads(output.read_text(encoding="utf-8")) == {"layer_id": 1}


def test_token_driver_runs_stages_in_process_and_resumes_completed_prefix(
    tmp_path, monkeypatch
) -> None:
    from tools import run_official_token

    state_dir = tmp_path / "state"
    result_dir = tmp_path / "layers"
    output = tmp_path / "token.json"
    timing = tmp_path / "timing.json"
    calls: list[str] = []
    cache_record = {
        "device_hits": 3,
        "device_resident_bytes": 2_048,
        "misses": 1,
    }
    shared_context = SimpleNamespace(
        packed_q8_cache=SimpleNamespace(snapshot=lambda: cache_record),
        packed_mxfp4_cache=SimpleNamespace(snapshot=lambda: cache_record),
    )
    events: list[str] = []

    def clock() -> float:
        events.append("clock")
        return float(len(events))

    class ContextFactory:
        @classmethod
        def create(cls, **kwargs):
            assert kwargs["q8_host_cache_bytes"] == 1_024
            assert kwargs["q8_device_cache_bytes"] == 2_048
            assert kwargs["mxfp4_host_cache_bytes"] == 4_096
            assert kwargs["mxfp4_device_cache_bytes"] == 8_192
            events.append("context")
            return shared_context

    def layer0(args: argparse.Namespace) -> int:
        assert args.runtime_context is shared_context
        assert args.direct_q8 is True
        calls.append("layer0")
        args.state_dir.mkdir(parents=True, exist_ok=True)
        (args.state_dir / "state.json").write_text(
            json.dumps({"completed_layer": 0}), encoding="utf-8"
        )
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps({"completed_layers": [0]}), encoding="utf-8"
        )
        return 0

    def remaining(args: argparse.Namespace) -> int:
        assert args.runtime_context is shared_context
        assert args.direct_q8 is True
        calls.append("remaining")
        (args.state_dir / "state.json").write_text(
            json.dumps({"completed_layer": 92}), encoding="utf-8"
        )
        return 0

    def head(args: argparse.Namespace) -> int:
        assert args.runtime_context is shared_context
        calls.append("head")
        args.output.write_text(
            json.dumps({"token_generated": True, "generated_token_id": 9689}),
            encoding="utf-8",
        )
        return 0

    monkeypatch.setattr(
        run_official_token,
        "_IN_PROCESS_STAGES",
        {"layer0": layer0, "remaining": remaining, "head": head},
    )
    monkeypatch.setattr(run_official_token, "OfficialRuntimeContext", ContextFactory)
    monkeypatch.setattr(run_official_token.time, "perf_counter", clock)
    args = argparse.Namespace(
        topology=Path("topology.json"),
        object_dir=tmp_path / "objects",
        state_dir=state_dir,
        result_dir=result_dir,
        output=output,
        timing_output=timing,
        token_id=1,
        stop_layer=92,
        k3x_set=None,
        execution_mode="in-process",
        direct_q8=True,
        q8_host_cache_bytes=1_024,
        q8_device_cache_bytes=2_048,
        mxfp4_host_cache_bytes=4_096,
        mxfp4_device_cache_bytes=8_192,
    )

    assert run_official_token.run(args) == 0
    assert events[0] == "clock"
    assert calls == ["layer0", "remaining", "head"]
    timing_record = json.loads(timing.read_text(encoding="utf-8"))
    assert timing_record["resumed_from_layer"] == -1
    assert timing_record["q8_cache"] == cache_record
    assert timing_record["mxfp4_cache"] == cache_record

    calls.clear()
    assert run_official_token.run(args) == 0
    assert calls == ["remaining", "head"]
    assert json.loads(timing.read_text(encoding="utf-8"))["resumed_from_layer"] == 92


def test_official_runtime_context_caches_headers_and_stores(
    tmp_path, monkeypatch
) -> None:
    source_shard = "model-00001-of-000096.safetensors"
    fragment = tmp_path / "model-00001-of-000096.k3x"
    header = object()
    store = object()
    header_calls: list[str] = []
    store_calls: list[tuple[tuple[Path, ...], bool, bool, object, object]] = []

    def inspect(snapshot, shard, transport):
        header_calls.append(shard)
        return header

    def open_store(
        paths,
        *,
        verify_root,
        verify_payload,
        packed_q8_cache,
        packed_mxfp4_cache,
    ):
        store_calls.append(
            (
                tuple(paths),
                verify_root,
                verify_payload,
                packed_q8_cache,
                packed_mxfp4_cache,
            )
        )
        return store

    monkeypatch.setattr(
        official_runtime_context, "inspect_official_shard_header", inspect
    )
    monkeypatch.setattr(
        official_runtime_context.K3XTensorStore, "open", open_store
    )
    context = official_runtime_context.OfficialRuntimeContext(
        topology={},
        object_dir=tmp_path,
        transport=object(),
        snapshot=object(),
        index=object(),
        config=object(),
        set_manifest=SimpleNamespace(
            fragments=(fragment,), record_sha256="a" * 64
        ),
    )

    assert context.header(source_shard) is header
    assert context.header(source_shard) is header
    assert context.store(source_shard) is store
    assert context.store(source_shard) is store
    assert header_calls == [source_shard]
    assert store_calls == [
        (
            (fragment,),
            False,
            False,
            context.packed_q8_cache,
            context.packed_mxfp4_cache,
        )
    ]
