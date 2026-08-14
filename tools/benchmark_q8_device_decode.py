# 실제 K3X Q8 텐서의 CPU 복호화 경로와 CUDA 복호화 경로를 비교합니다.
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
from k3x_ref.quant8 import Quant8Tensor, decode_groupwise_8bit
from tools.official_runtime_context import OfficialRuntimeContext


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--object-dir", type=Path, required=True)
    parser.add_argument("--k3x-set", type=Path, required=True)
    parser.add_argument("--source-shard", required=True)
    parser.add_argument("--tensor", required=True)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _timed(callable_) -> tuple[torch.Tensor, float, int]:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    start = time.perf_counter()
    tensor = callable_()
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return tensor, elapsed, torch.cuda.max_memory_allocated()


def main() -> int:
    args = _parse_args()
    if args.repeats <= 0 or not torch.cuda.is_available():
        raise K3XError("Q8_CUDA_BENCHMARK_UNAVAILABLE")
    context = OfficialRuntimeContext.create(
        topology_path=args.topology,
        object_dir=args.object_dir,
        k3x_set=args.k3x_set,
    )
    store = context.store(args.source_shard)
    located = store.record(args.tensor)
    record = located.record
    if record.quantization != Quantization.GROUPWISE_8BIT:
        raise K3XError("Q8_CUDA_BENCHMARK_TENSOR")
    values = math.prod(record.dimensions)

    def cpu_decode_then_copy() -> torch.Tensor:
        codes, scales = located.reader.read_tensor_extents(record)
        decoded = decode_groupwise_8bit(
            Quant8Tensor(record.dimensions, values, 128, codes, scales)
        )
        return decoded.to(device="cuda", dtype=torch.bfloat16)

    def device_decode() -> torch.Tensor:
        return store.load(args.tensor, device="cuda", dtype=torch.bfloat16)

    expected, _, _ = _timed(cpu_decode_then_copy)
    actual, _, _ = _timed(device_decode)
    equal = torch.equal(actual, expected)
    del actual, expected
    if not equal:
        raise K3XError("Q8_CUDA_BENCHMARK_MISMATCH")

    cpu_seconds = []
    device_seconds = []
    cpu_peaks = []
    device_peaks = []
    for _ in range(args.repeats):
        tensor, elapsed, peak = _timed(cpu_decode_then_copy)
        cpu_seconds.append(elapsed)
        cpu_peaks.append(peak)
        del tensor
        tensor, elapsed, peak = _timed(device_decode)
        device_seconds.append(elapsed)
        device_peaks.append(peak)
        del tensor

    cpu_median = statistics.median(cpu_seconds)
    device_median = statistics.median(device_seconds)
    result = {
        "format": "k3x-q8-device-decode-benchmark-v1",
        "device": torch.cuda.get_device_name(),
        "source_shard": args.source_shard,
        "tensor": args.tensor,
        "shape": list(record.dimensions),
        "codes_bytes": record.data_length,
        "scales_bytes": record.auxiliary_length,
        "logical_fp32_bytes": values * 4,
        "repeats": args.repeats,
        "cpu_decode_then_copy_seconds": cpu_seconds,
        "device_decode_seconds": device_seconds,
        "cpu_decode_then_copy_median_seconds": cpu_median,
        "device_decode_median_seconds": device_median,
        "median_speedup": cpu_median / device_median,
        "cpu_peak_cuda_allocated_bytes": max(cpu_peaks),
        "device_peak_cuda_allocated_bytes": max(device_peaks),
        "bf16_output_bit_exact": True,
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
