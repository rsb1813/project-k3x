# 공식 Kimi K3 layer 1의 expert-major MXFP4 상주 성능과 scalar parity를 측정합니다.
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import tempfile
import time
from pathlib import Path

import torch

from k3x_converter.format import K3XError
from tools.official_runtime_context import OfficialRuntimeContext
from tools.run_official_layer0 import run as run_layer0
from tools.run_official_layer1 import run as run_layer1


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--object-dir", type=Path, required=True)
    parser.add_argument("--k3x-set", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--harness-commit", required=True)
    parser.add_argument("--q8-device-cache-bytes", type=int, required=True)
    parser.add_argument("--mxfp4-device-cache-bytes", type=int, required=True)
    parser.add_argument("--warm-runs", type=int, default=5)
    return parser.parse_args()


def _context(args, *, mxfp4_device_cache_bytes: int) -> OfficialRuntimeContext:
    return OfficialRuntimeContext.create(
        topology_path=args.topology,
        object_dir=args.object_dir,
        k3x_set=args.k3x_set,
        q8_device_cache_bytes=args.q8_device_cache_bytes,
        mxfp4_device_cache_bytes=mxfp4_device_cache_bytes,
    )


def _layer_args(
    args, state_dir: Path, output: Path, context, *, direct_q8: bool = True
):
    return argparse.Namespace(
        topology=args.topology,
        object_dir=args.object_dir,
        state_dir=state_dir,
        output=output,
        token_id=1,
        layer_id=1,
        k3x_set=args.k3x_set,
        runtime_context=context,
        direct_q8=direct_q8,
    )


def _hidden(state_dir: Path) -> torch.Tensor:
    state = json.loads((state_dir / "state.json").read_text(encoding="utf-8"))
    metadata = state["tensors"]["hidden_after_layer_1"]
    payload = (state_dir / metadata["path"]).read_bytes()
    return torch.frombuffer(bytearray(payload), dtype=torch.bfloat16).clone()


def main() -> int:
    args = _parse_args()
    if (
        not torch.cuda.is_available()
        or args.q8_device_cache_bytes <= 0
        or args.mxfp4_device_cache_bytes <= 0
        or args.warm_runs <= 0
    ):
        raise K3XError("MXFP4_LAYER_BATCH_BENCHMARK_UNAVAILABLE")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    device = torch.device("cuda", torch.cuda.current_device())
    with tempfile.TemporaryDirectory(prefix="k3x-layer1-batch-") as temporary:
        state_dir = Path(temporary)
        batch_context = _context(
            args, mxfp4_device_cache_bytes=args.mxfp4_device_cache_bytes
        )
        setup_output = output_dir / "setup-layer0.json"
        torch.cuda.synchronize(device)
        setup_started = time.perf_counter()
        if run_layer0(
            _layer_args(args, state_dir, setup_output, batch_context)
        ) != 0:
            raise K3XError("MXFP4_LAYER_BATCH_SETUP")
        torch.cuda.synchronize(device)
        setup_seconds = time.perf_counter() - setup_started
        layer0_state = (state_dir / "state.json").read_bytes()

        exact_context = _context(args, mxfp4_device_cache_bytes=0)
        exact_output = output_dir / "exact-q8-layer1.json"
        torch.cuda.synchronize(device)
        exact_started = time.perf_counter()
        if run_layer1(
            _layer_args(
                args,
                state_dir,
                exact_output,
                exact_context,
                direct_q8=False,
            )
        ) != 0:
            raise K3XError("MXFP4_LAYER_EXACT_REFERENCE")
        torch.cuda.synchronize(device)
        exact_seconds = time.perf_counter() - exact_started
        exact_hidden = _hidden(state_dir)
        exact_record = json.loads(exact_output.read_text(encoding="utf-8"))

        (state_dir / "state.json").write_bytes(layer0_state)
        scalar_context = _context(args, mxfp4_device_cache_bytes=0)
        scalar_output = output_dir / "scalar-layer1.json"
        torch.cuda.synchronize(device)
        scalar_started = time.perf_counter()
        if run_layer1(
            _layer_args(args, state_dir, scalar_output, scalar_context)
        ) != 0:
            raise K3XError("MXFP4_LAYER_SCALAR_REFERENCE")
        torch.cuda.synchronize(device)
        scalar_seconds = time.perf_counter() - scalar_started
        scalar_hidden = _hidden(state_dir)
        scalar_record = json.loads(scalar_output.read_text(encoding="utf-8"))

        runs = []
        for label in (
            "cold",
            *(f"warm-{index}" for index in range(args.warm_runs)),
        ):
            (state_dir / "state.json").write_bytes(layer0_state)
            result_path = output_dir / f"{label}-layer1.json"
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            if run_layer1(
                _layer_args(args, state_dir, result_path, batch_context)
            ) != 0:
                raise K3XError("MXFP4_LAYER_BATCH_RUN", label)
            torch.cuda.synchronize(device)
            wall_seconds = time.perf_counter() - started
            result = json.loads(result_path.read_text(encoding="utf-8"))
            hidden = _hidden(state_dir)
            difference = (hidden.float() - scalar_hidden.float()).abs()
            exact_difference = (hidden.float() - exact_hidden.float()).abs()
            runs.append(
                {
                    "label": label,
                    "wall_seconds": wall_seconds,
                    "download_seconds": result["download_seconds"],
                    "compute_seconds": result["compute_seconds"],
                    "layer_output_sha256": result["layer_output_sha256"],
                    "route_expert_ids": result["route_expert_ids"],
                    "maximum_absolute_error_vs_scalar": difference.max().item(),
                    "mean_absolute_error_vs_scalar": difference.mean().item(),
                    "cosine_similarity_vs_scalar": torch.nn.functional.cosine_similarity(
                        hidden.float(), scalar_hidden.float(), dim=0
                    ).item(),
                    "bf16_exact_ratio_vs_scalar": (
                        hidden == scalar_hidden
                    ).float().mean().item(),
                    "maximum_absolute_error_vs_exact_q8": exact_difference.max().item(),
                    "mean_absolute_error_vs_exact_q8": exact_difference.mean().item(),
                    "cosine_similarity_vs_exact_q8": torch.nn.functional.cosine_similarity(
                        hidden.float(), exact_hidden.float(), dim=0
                    ).item(),
                    "bf16_exact_ratio_vs_exact_q8": (
                        hidden == exact_hidden
                    ).float().mean().item(),
                    "route_overlap_vs_exact_q8": len(
                        set(result["route_expert_ids"])
                        & set(exact_record["route_expert_ids"])
                    ),
                    "q8_cache": batch_context.packed_q8_cache.snapshot(),
                    "mxfp4_cache": batch_context.packed_mxfp4_cache.snapshot(),
                }
            )

    if any(
        run["route_expert_ids"] != scalar_record["route_expert_ids"]
        or run["cosine_similarity_vs_scalar"] < 0.999999
        for run in runs
    ):
        raise K3XError("MXFP4_LAYER_BATCH_OUTPUT_MISMATCH")
    warm_median = statistics.median(
        float(run["wall_seconds"]) for run in runs[1:]
    )
    warm_compute_median = statistics.median(
        float(run["compute_seconds"]) for run in runs[1:]
    )
    summary = {
        "format": "k3x-official-layer1-expert-batch-benchmark-v1",
        "implementation_commit": args.implementation_commit,
        "harness_commit": args.harness_commit,
        "device": torch.cuda.get_device_name(device),
        "setup_layer0_seconds": setup_seconds,
        "exact_q8_reference_seconds": exact_seconds,
        "exact_q8_layer_output_sha256": exact_record["layer_output_sha256"],
        "exact_q8_route_expert_ids": exact_record["route_expert_ids"],
        "scalar_reference_seconds": scalar_seconds,
        "scalar_layer_output_sha256": scalar_record["layer_output_sha256"],
        "route_expert_ids": scalar_record["route_expert_ids"],
        "q8_device_cache_bytes": args.q8_device_cache_bytes,
        "mxfp4_device_cache_bytes": args.mxfp4_device_cache_bytes,
        "runs": runs,
        "warm_runs": args.warm_runs,
        "warm_median_wall_seconds": warm_median,
        "warm_median_compute_seconds": warm_compute_median,
        "scalar_to_warm_speedup": scalar_seconds / warm_median,
        "physical_nvme_bytes_measured": False,
        "physical_h2d_bytes_measured": False,
        "token_throughput_measured": False,
    }
    encoded = json.dumps(summary, sort_keys=True, separators=(",", ":")).encode()
    summary["record_sha256"] = hashlib.sha256(encoded).hexdigest()
    (output_dir / "summary.json").write_text(
        json.dumps(summary, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
