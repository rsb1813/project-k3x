# 공식 K3X 레이어 0을 같은 런타임에서 반복해 packed Q8 상주 효과를 측정합니다.
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import torch

from k3x_converter.format import K3XError
from tools.official_runtime_context import OfficialRuntimeContext
from tools.run_official_layer0 import run as run_layer0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--object-dir", type=Path, required=True)
    parser.add_argument("--k3x-set", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--q8-host-cache-bytes", type=int, default=0)
    parser.add_argument("--q8-device-cache-bytes", type=int, required=True)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not torch.cuda.is_available() or args.q8_device_cache_bytes <= 0:
        raise K3XError("Q8_RESIDENCY_BENCHMARK_UNAVAILABLE")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    context = OfficialRuntimeContext.create(
        topology_path=args.topology,
        object_dir=args.object_dir,
        k3x_set=args.k3x_set,
        q8_host_cache_bytes=args.q8_host_cache_bytes,
        q8_device_cache_bytes=args.q8_device_cache_bytes,
    )

    runs = []
    for label in ("cold", "warm"):
        state_dir = output_dir / f"{label}-state"
        result_path = output_dir / f"{label}-layer0.json"
        torch.cuda.synchronize()
        started = time.perf_counter()
        status = run_layer0(
            argparse.Namespace(
                topology=args.topology,
                object_dir=args.object_dir,
                state_dir=state_dir,
                output=result_path,
                token_id=1,
                k3x_set=args.k3x_set,
                runtime_context=context,
                direct_q8=True,
            )
        )
        torch.cuda.synchronize()
        wall_seconds = time.perf_counter() - started
        if status != 0:
            raise K3XError("Q8_RESIDENCY_LAYER0", label)
        result = json.loads(result_path.read_text(encoding="utf-8"))
        runs.append(
            {
                "label": label,
                "wall_seconds": wall_seconds,
                "download_seconds": result["download_seconds"],
                "compute_seconds": result["compute_seconds"],
                "layer0_output_sha256": result["layer0_output_sha256"],
                "layer0_kda_state_sha256": result["layer0_kda_state_sha256"],
                "peak_cuda_allocated_bytes": result[
                    "peak_cuda_allocated_bytes"
                ],
                "peak_cuda_reserved_bytes": result[
                    "peak_cuda_reserved_bytes"
                ],
                "cache": context.packed_q8_cache.snapshot(),
            }
        )

    if (
        runs[0]["layer0_output_sha256"] != runs[1]["layer0_output_sha256"]
        or runs[0]["layer0_kda_state_sha256"]
        != runs[1]["layer0_kda_state_sha256"]
    ):
        raise K3XError("Q8_RESIDENCY_OUTPUT_MISMATCH")
    summary = {
        "format": "k3x-official-q8-residency-benchmark-v1",
        "device": torch.cuda.get_device_name(),
        "q8_host_cache_bytes": args.q8_host_cache_bytes,
        "q8_device_cache_bytes": args.q8_device_cache_bytes,
        "runs": runs,
        "warm_speedup": runs[0]["wall_seconds"] / runs[1]["wall_seconds"],
        "output_and_kda_state_match": True,
        "physical_nvme_bytes_measured": False,
        "physical_h2d_bytes_measured": False,
        "token_throughput_measured": False,
    }
    encoded = json.dumps(summary, sort_keys=True, separators=(",", ":")).encode()
    summary["record_sha256"] = hashlib.sha256(encoded).hexdigest()
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
