# 실제 K3X Q8 텐서의 materialized 경로와 direct-packed CUDA matvec을 비교합니다.
from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import time
from pathlib import Path

import torch

from k3x_converter.format import K3XError, Quantization
from k3x_converter.official_two_layer import _bf16_matvec
from k3x_converter.q8_cuda import q8_matvec
from tools.official_runtime_context import OfficialRuntimeContext


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--object-dir", type=Path, required=True)
    parser.add_argument("--k3x-set", type=Path, required=True)
    parser.add_argument("--source-shard", required=True)
    parser.add_argument("--tensor", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--kernel-repeats", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _timed(callable_) -> tuple[torch.Tensor, float, int]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    output = callable_()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return output, elapsed, torch.cuda.max_memory_allocated()


def main() -> int:
    args = _parse_args()
    if args.repeats <= 0 or args.kernel_repeats <= 0 or not torch.cuda.is_available():
        raise K3XError("Q8_CUDA_BENCHMARK_UNAVAILABLE")
    context = OfficialRuntimeContext.create(
        topology_path=args.topology,
        object_dir=args.object_dir,
        k3x_set=args.k3x_set,
    )
    store = context.store(args.source_shard)
    located = store.record(args.tensor)
    record = located.record
    if (
        record.quantization != Quantization.GROUPWISE_8BIT
        or len(record.dimensions) != 2
        or record.dimensions[1] % 128
    ):
        raise K3XError("Q8_CUDA_BENCHMARK_TENSOR")
    rows, columns = record.dimensions
    value = torch.linspace(
        -1.0, 1.0, columns, dtype=torch.bfloat16, device="cuda"
    )

    def materialized() -> torch.Tensor:
        weight = store.load(args.tensor, device="cuda", dtype=torch.bfloat16)
        return _bf16_matvec(value, weight)

    def direct() -> torch.Tensor:
        codes_bytes, scale_bytes = located.reader.read_tensor_extents(record)
        codes = torch.frombuffer(bytearray(codes_bytes), dtype=torch.int8).to("cuda")
        scales = torch.frombuffer(
            bytearray(scale_bytes), dtype=torch.bfloat16
        ).to("cuda")
        return q8_matvec(value, codes, scales, rows, columns)

    expected, _, _ = _timed(materialized)
    actual, _, _ = _timed(direct)
    difference = (actual.float() - expected.float()).abs()
    exact_ratio = torch.eq(actual, expected).float().mean().item()
    maximum_error = difference.max().item()
    mean_error = difference.mean().item()
    if not torch.isfinite(actual).all():
        raise K3XError("Q8_CUDA_BENCHMARK_NONFINITE")
    del actual, expected, difference

    resident_weight = store.load(args.tensor, device="cuda", dtype=torch.bfloat16)
    codes_bytes, scale_bytes = located.reader.read_tensor_extents(record)
    resident_codes = torch.frombuffer(
        bytearray(codes_bytes), dtype=torch.int8
    ).to("cuda")
    resident_scales = torch.frombuffer(
        bytearray(scale_bytes), dtype=torch.bfloat16
    ).to("cuda")

    def resident_materialized() -> torch.Tensor:
        return _bf16_matvec(value, resident_weight)

    def resident_direct() -> torch.Tensor:
        return q8_matvec(value, resident_codes, resident_scales, rows, columns)

    warm, _, _ = _timed(resident_materialized)
    del warm
    warm, _, _ = _timed(resident_direct)
    del warm
    resident_materialized_seconds = []
    resident_direct_seconds = []
    for _ in range(args.kernel_repeats):
        output, elapsed, _ = _timed(resident_materialized)
        resident_materialized_seconds.append(elapsed)
        del output
        output, elapsed, _ = _timed(resident_direct)
        resident_direct_seconds.append(elapsed)
        del output
    del resident_weight, resident_codes, resident_scales

    materialized_seconds = []
    direct_seconds = []
    materialized_peaks = []
    direct_peaks = []
    for _ in range(args.repeats):
        output, elapsed, peak = _timed(materialized)
        materialized_seconds.append(elapsed)
        materialized_peaks.append(peak)
        del output
        output, elapsed, peak = _timed(direct)
        direct_seconds.append(elapsed)
        direct_peaks.append(peak)
        del output

    materialized_median = statistics.median(materialized_seconds)
    direct_median = statistics.median(direct_seconds)
    result = {
        "format": "k3x-q8-direct-matvec-benchmark-v1",
        "device": torch.cuda.get_device_name(),
        "source_shard": args.source_shard,
        "tensor": args.tensor,
        "shape": list(record.dimensions),
        "codes_bytes": record.data_length,
        "scales_bytes": record.auxiliary_length,
        "materialized_bf16_bytes": math.prod(record.dimensions) * 2,
        "repeats": args.repeats,
        "kernel_repeats": args.kernel_repeats,
        "materialized_seconds": materialized_seconds,
        "direct_seconds": direct_seconds,
        "materialized_median_seconds": materialized_median,
        "direct_median_seconds": direct_median,
        "median_speedup": materialized_median / direct_median,
        "resident_materialized_median_seconds": statistics.median(
            resident_materialized_seconds
        ),
        "resident_direct_median_seconds": statistics.median(
            resident_direct_seconds
        ),
        "resident_median_speedup": statistics.median(
            resident_materialized_seconds
        )
        / statistics.median(resident_direct_seconds),
        "materialized_peak_cuda_allocated_bytes": max(materialized_peaks),
        "direct_peak_cuda_allocated_bytes": max(direct_peaks),
        "bf16_exact_ratio": exact_ratio,
        "maximum_absolute_error": maximum_error,
        "mean_absolute_error": mean_error,
        "physical_h2d_bytes_measured": False,
        "token_throughput_measured": False,
    }
    encoded = json.dumps(result, sort_keys=True, separators=(",", ":")).encode()
    result["record_sha256"] = hashlib.sha256(encoded).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
