# 공식 K3X 첫 토큰 그래프를 하나의 재개 가능한 프로세스로 실행합니다.
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

from k3x_converter.format import K3XError
from tools.run_official_head import run as run_head
from tools.run_official_layer0 import _write_json_atomic, run as run_layer0
from tools.run_official_remaining import run as run_remaining
from tools.official_runtime_context import OfficialRuntimeContext


_IN_PROCESS_STAGES = {
    "layer0": run_layer0,
    "remaining": run_remaining,
    "head": run_head,
}


def _parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--object-dir", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--timing-output", type=Path, required=True)
    parser.add_argument("--token-id", type=int, default=1)
    parser.add_argument("--stop-layer", type=int, default=92)
    parser.add_argument("--k3x-set", type=Path)
    parser.add_argument("--direct-q8", action="store_true")
    parser.add_argument("--q8-host-cache-bytes", type=int, default=0)
    parser.add_argument("--q8-device-cache-bytes", type=int, default=0)
    parser.add_argument("--mxfp4-host-cache-bytes", type=int, default=0)
    parser.add_argument("--mxfp4-device-cache-bytes", type=int, default=0)
    parser.add_argument(
        "--execution-mode",
        choices=("subprocess", "in-process"),
        default="in-process",
    )
    return parser.parse_args(argv)


def _invoke(
    execution_mode: str,
    stage: str,
    stage_args: argparse.Namespace,
    command: list[str],
) -> None:
    if execution_mode == "subprocess":
        subprocess.run(command, check=True)
        return
    result = _IN_PROCESS_STAGES[stage](stage_args)
    if result != 0:
        raise K3XError("OFFICIAL_PERSISTENT_STAGE", stage)


def _published(path: Path) -> dict[str, object]:
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise K3XError("OFFICIAL_PERSISTENT_PUBLICATION", str(path)) from error
    if not isinstance(record, dict):
        raise K3XError("OFFICIAL_PERSISTENT_PUBLICATION", str(path))
    return record


def run(args: argparse.Namespace) -> int:
    if args.stop_layer != 92:
        raise K3XError("OFFICIAL_PERSISTENT_STOP_LAYER")

    topology = args.topology.resolve()
    object_dir = args.object_dir.resolve()
    state_dir = args.state_dir.resolve()
    result_dir = args.result_dir.resolve()
    output = args.output.resolve()
    timing_output = args.timing_output.resolve()
    k3x_set = args.k3x_set.resolve() if args.k3x_set is not None else None
    state_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    state_path = state_dir / "state.json"
    total_start = time.perf_counter()
    runtime_context = (
        OfficialRuntimeContext.create(
            topology_path=topology,
            object_dir=object_dir,
            k3x_set=k3x_set,
            q8_host_cache_bytes=getattr(args, "q8_host_cache_bytes", 0),
            q8_device_cache_bytes=getattr(args, "q8_device_cache_bytes", 0),
            mxfp4_host_cache_bytes=getattr(args, "mxfp4_host_cache_bytes", 0),
            mxfp4_device_cache_bytes=getattr(args, "mxfp4_device_cache_bytes", 0),
        )
        if args.execution_mode == "in-process"
        else None
    )

    resumed_from_layer = -1
    if state_path.exists():
        state = _published(state_path)
        completed = state.get("completed_layer")
        if not isinstance(completed, int) or not 0 <= completed <= 92:
            raise K3XError("OFFICIAL_PREFIX_STATE")
        resumed_from_layer = completed

    stage_seconds: dict[str, float] = {}
    layer0_output = result_dir / "layer-00.json"
    if resumed_from_layer < 0:
        layer0_args = argparse.Namespace(
            topology=topology,
            object_dir=object_dir,
            state_dir=state_dir,
            output=layer0_output,
            token_id=args.token_id,
            k3x_set=k3x_set,
            runtime_context=runtime_context,
            direct_q8=getattr(args, "direct_q8", False),
        )
        command = [
            sys.executable,
            "tools/run_official_layer0.py",
            "--topology",
            str(topology),
            "--object-dir",
            str(object_dir),
            "--state-dir",
            str(state_dir),
            "--output",
            str(layer0_output),
            "--token-id",
            str(args.token_id),
        ]
        if k3x_set is not None:
            command.extend(("--k3x-set", str(k3x_set)))
        if getattr(args, "direct_q8", False):
            command.append("--direct-q8")
        start = time.perf_counter()
        _invoke(args.execution_mode, "layer0", layer0_args, command)
        stage_seconds["layer0"] = time.perf_counter() - start
        if _published(layer0_output).get("completed_layers") != [0]:
            raise K3XError("OFFICIAL_LAYER_PUBLICATION")

    remaining_args = argparse.Namespace(
        topology=topology,
        object_dir=object_dir,
        state_dir=state_dir,
        result_dir=result_dir,
        stop_layer=92,
        k3x_set=k3x_set,
        execution_mode=args.execution_mode,
        runtime_context=runtime_context,
        direct_q8=getattr(args, "direct_q8", False),
    )
    command = [
        sys.executable,
        "tools/run_official_remaining.py",
        "--topology",
        str(topology),
        "--object-dir",
        str(object_dir),
        "--state-dir",
        str(state_dir),
        "--result-dir",
        str(result_dir),
        "--execution-mode",
        args.execution_mode,
    ]
    if k3x_set is not None:
        command.extend(("--k3x-set", str(k3x_set)))
    if getattr(args, "direct_q8", False):
        command.append("--direct-q8")
    start = time.perf_counter()
    _invoke(args.execution_mode, "remaining", remaining_args, command)
    stage_seconds["remaining"] = time.perf_counter() - start
    if _published(state_path).get("completed_layer") != 92:
        raise K3XError("OFFICIAL_PREFIX_STATE")

    head_args = argparse.Namespace(
        topology=topology,
        object_dir=object_dir,
        state_dir=state_dir,
        output=output,
        k3x_set=k3x_set,
        runtime_context=runtime_context,
    )
    command = [
        sys.executable,
        "tools/run_official_head.py",
        "--topology",
        str(topology),
        "--object-dir",
        str(object_dir),
        "--state-dir",
        str(state_dir),
        "--output",
        str(output),
    ]
    if k3x_set is not None:
        command.extend(("--k3x-set", str(k3x_set)))
    start = time.perf_counter()
    _invoke(args.execution_mode, "head", head_args, command)
    stage_seconds["head"] = time.perf_counter() - start
    head = _published(output)
    if head.get("token_generated") is not True or not isinstance(
        head.get("generated_token_id"), int
    ):
        raise K3XError("OFFICIAL_HEAD_PUBLICATION")

    timing = {
        "format": "k3x-official-persistent-token-timing-v1",
        "execution_mode": args.execution_mode,
        "input_token_id": args.token_id,
        "generated_token_id": head["generated_token_id"],
        "resumed_from_layer": resumed_from_layer,
        "stage_seconds": stage_seconds,
        "wall_seconds": time.perf_counter() - total_start,
        "throughput_measured": False,
        "direct_q8": getattr(args, "direct_q8", False),
        "q8_cache": (
            runtime_context.packed_q8_cache.snapshot()
            if runtime_context is not None
            else None
        ),
        "mxfp4_cache": (
            runtime_context.packed_mxfp4_cache.snapshot()
            if runtime_context is not None
            else None
        ),
    }
    encoded = json.dumps(timing, sort_keys=True, separators=(",", ":")).encode()
    timing["record_sha256"] = hashlib.sha256(encoded).hexdigest()
    _write_json_atomic(timing_output, timing)
    print(json.dumps(timing, sort_keys=True, separators=(",", ":")))
    return 0


def main() -> int:
    return run(_parse_args())


if __name__ == "__main__":
    raise SystemExit(main())
