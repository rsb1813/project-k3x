# 공개 Kimi K3 expert의 native MXFP4 GPU 상주 효과와 정확도를 측정합니다.
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import time
from pathlib import Path

import torch

from k3x_converter.format import K3XError
from k3x_converter.official_source import plan_official_expert
from k3x_converter.official_two_layer import _situ
from tools.official_k3x_source import expert_matvec
from tools.official_runtime_context import OfficialRuntimeContext


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--object-dir", type=Path, required=True)
    parser.add_argument("--k3x-set", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--harness-commit", required=True)
    parser.add_argument("--layer", type=int, default=1)
    parser.add_argument("--expert", type=int, default=0)
    parser.add_argument("--mxfp4-host-cache-bytes", type=int, default=0)
    parser.add_argument("--mxfp4-device-cache-bytes", type=int, required=True)
    parser.add_argument("--warm-runs", type=int, default=5)
    return parser.parse_args()


def _forward(store, plan, value: torch.Tensor) -> torch.Tensor:
    gate = expert_matvec(store, plan, "gate", value)
    up = expert_matvec(store, plan, "up", value)
    activated = _situ(gate, up, 4.0, 25.0)
    return expert_matvec(store, plan, "down", activated)


def _timed_cuda_forward(store, plan, value: torch.Tensor):
    events = [torch.cuda.Event(enable_timing=True) for _ in range(5)]
    events[0].record()
    gate = expert_matvec(store, plan, "gate", value)
    events[1].record()
    up = expert_matvec(store, plan, "up", value)
    events[2].record()
    activated = _situ(gate, up, 4.0, 25.0)
    events[3].record()
    output = expert_matvec(store, plan, "down", activated)
    events[4].record()
    return output, events


def _digest(value: torch.Tensor) -> str:
    raw = value.detach().cpu().contiguous().view(torch.uint8).numpy().tobytes()
    return hashlib.sha256(raw).hexdigest()


def main() -> int:
    args = _parse_args()
    if (
        not torch.cuda.is_available()
        or args.layer < 1
        or args.expert < 0
        or args.warm_runs <= 0
        or args.mxfp4_device_cache_bytes <= 0
    ):
        raise K3XError("MXFP4_RESIDENCY_BENCHMARK_UNAVAILABLE")
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=False)
    context = OfficialRuntimeContext.create(
        topology_path=args.topology,
        object_dir=args.object_dir,
        k3x_set=args.k3x_set,
        mxfp4_host_cache_bytes=args.mxfp4_host_cache_bytes,
        mxfp4_device_cache_bytes=args.mxfp4_device_cache_bytes,
    )
    official_gate = (
        f"language_model.model.layers.{args.layer}.block_sparse_moe."
        f"experts.{args.expert}.w1.weight_packed"
    )
    source_shard = context.index.weight_map.get(official_gate)
    if source_shard is None:
        raise K3XError("INVALID_OFFICIAL_EXPERT", official_gate)
    plan = plan_official_expert(
        context.index,
        context.header(source_shard),
        layer_id=args.layer,
        expert_id=args.expert,
    )
    store = context.store(source_shard)
    cpu_value = torch.linspace(-1.0, 1.0, 3_584, dtype=torch.bfloat16)
    oracle_started = time.perf_counter()
    oracle = _forward(store, plan, cpu_value)
    oracle_seconds = time.perf_counter() - oracle_started

    device = torch.device("cuda", torch.cuda.current_device())
    device_value = cpu_value.to(device)
    runs: list[dict[str, object]] = []
    torch.cuda.reset_peak_memory_stats(device)
    for label in ("cold", *(f"warm-{index}" for index in range(args.warm_runs))):
        torch.cuda.synchronize(device)
        started = time.perf_counter()
        output, events = _timed_cuda_forward(store, plan, device_value)
        torch.cuda.synchronize(device)
        wall_seconds = time.perf_counter() - started
        cuda_stage_milliseconds = {
            name: events[index].elapsed_time(events[index + 1])
            for index, name in enumerate(("gate", "up", "situ", "down"))
        }
        output_cpu = output.cpu()
        difference = (output_cpu - oracle).abs()
        cosine = torch.nn.functional.cosine_similarity(
            output_cpu.double(), oracle.double(), dim=0
        ).item()
        bf16_exact_ratio = (
            output_cpu.to(torch.bfloat16) == oracle.to(torch.bfloat16)
        ).float().mean().item()
        runs.append(
            {
                "label": label,
                "wall_seconds": wall_seconds,
                "output_sha256": _digest(output_cpu),
                "max_abs_error": difference.max().item(),
                "mean_abs_error": difference.mean().item(),
                "cosine_similarity": cosine,
                "bf16_exact_ratio": bf16_exact_ratio,
                "cuda_stage_milliseconds": cuda_stage_milliseconds,
                "cache": context.packed_mxfp4_cache.snapshot(),
            }
        )

    if any(
        not torch.isfinite(torch.tensor(run["cosine_similarity"]))
        or run["cosine_similarity"] < 0.999999
        or run["bf16_exact_ratio"] < 0.999
        for run in runs
    ):
        raise K3XError("MXFP4_RESIDENCY_OUTPUT_MISMATCH")
    warm_median = statistics.median(
        float(run["wall_seconds"]) for run in runs[1:]
    )
    summary = {
        "format": "k3x-official-mxfp4-residency-benchmark-v2",
        "implementation_commit": args.implementation_commit,
        "harness_commit": args.harness_commit,
        "device": torch.cuda.get_device_name(device),
        "layer": args.layer,
        "expert": args.expert,
        "source_shard": source_shard,
        "expert_payload_bytes": plan.payload_bytes,
        "mxfp4_host_cache_bytes": args.mxfp4_host_cache_bytes,
        "mxfp4_device_cache_bytes": args.mxfp4_device_cache_bytes,
        "portable_oracle_seconds": oracle_seconds,
        "runs": runs,
        "warm_runs": args.warm_runs,
        "warm_median_wall_seconds": warm_median,
        "cold_to_warm_speedup": float(runs[0]["wall_seconds"]) / warm_median,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
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
