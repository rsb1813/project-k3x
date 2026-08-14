# 공식 K3X 그래프 단계의 상주형 호출 계약을 검증합니다.
from __future__ import annotations

import argparse
import json

from tools import (
    run_official_head,
    run_official_layer0,
    run_official_layer1,
    run_official_layer3,
    run_official_remaining,
)


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
