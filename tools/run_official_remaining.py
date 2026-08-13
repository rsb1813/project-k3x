# 저장된 공식 Kimi K3 prefix state부터 92번 레이어까지 재개 실행합니다.
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from k3x_converter.format import K3XError


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--object-dir", type=Path, required=True)
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--stop-layer", type=int, default=92)
    parser.add_argument("--k3x-set", type=Path)
    args = parser.parse_args()

    topology_path = args.topology.resolve()
    topology = json.loads(topology_path.read_text(encoding="utf-8"))
    if topology.get("format") != "k3x-official-topology-v2":
        raise K3XError("OFFICIAL_TOPOLOGY_FORMAT")
    state_dir = args.state_dir.resolve()
    state_path = state_dir / "state.json"
    result_dir = args.result_dir.resolve()
    result_dir.mkdir(parents=True, exist_ok=True)

    while True:
        state = json.loads(state_path.read_text(encoding="utf-8"))
        completed = state.get("completed_layer")
        if not isinstance(completed, int):
            raise K3XError("OFFICIAL_PREFIX_STATE")
        layer_id = completed + 1
        if layer_id > args.stop_layer:
            return 0
        layer = topology["layers"][layer_id]
        runner = (
            "tools/run_official_layer1.py"
            if layer["attention"] == "kda"
            else "tools/run_official_layer3.py"
        )
        output = result_dir / f"layer-{layer_id:02d}.json"
        command = [
            sys.executable,
            runner,
            "--topology",
            str(topology_path),
            "--object-dir",
            str(args.object_dir.resolve()),
            "--state-dir",
            str(state_dir),
            "--output",
            str(output),
            "--layer-id",
            str(layer_id),
        ]
        if args.k3x_set is not None:
            command.extend(("--k3x-set", str(args.k3x_set.resolve())))
        print(
            f"starting_layer={layer_id} attention={layer['attention']}",
            flush=True,
        )
        subprocess.run(command, check=True)
        published = json.loads(output.read_text(encoding="utf-8"))
        if published.get("layer_id") != layer_id:
            raise K3XError("OFFICIAL_LAYER_PUBLICATION")
        print(
            f"completed_layer={layer_id} "
            f"output_sha256={published['layer_output_sha256']}",
            flush=True,
        )


if __name__ == "__main__":
    raise SystemExit(main())
