# K3X synthetic runtime의 latency, I/O, RSS와 state 크기를 재현 가능하게 측정합니다.
from __future__ import annotations

import argparse
import csv
import json
import platform
import statistics
import struct
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import psutil

from k3x_converter.reader import K3XReader


@dataclass(frozen=True)
class BenchmarkRecord:
    scope: str
    evidence: str
    platform: str
    iterations: int
    prompt_tokens: int
    generated_tokens: int
    prefill_tokens_per_second: float
    decode_tokens_per_second: float
    ttft_ms: float
    peak_rss_bytes: int
    file_read_bytes_per_token: float
    backend: str
    device: str
    dense_precision: str
    cuda_allocation: str
    cuda_weights: str
    cuda_batching: str
    cuda_resident_bytes: int
    kernel_nanoseconds: int
    host_to_device_bytes: int
    weight_h2d_bytes: int
    activation_h2d_bytes: int
    device_to_host_bytes: int
    peak_vram_bytes: int | None
    device_allocation_count: int
    device_free_count: int
    stream_synchronization_count: int
    weight_cache_hits: int
    weight_cache_misses: int
    weight_cache_bypasses: int
    resident_weight_bytes: int
    peak_resident_weight_bytes: int
    scratch_bytes: int
    peak_scratch_bytes: int
    grouped_projection_calls: int
    grouped_projection_members: int
    max_absolute_error: float | None
    max_relative_error: float | None
    kda_state_bytes: int
    mla_kv_bytes: int
    per_layer_nanoseconds: tuple[int, ...]

    def __post_init__(self) -> None:
        if self.scope not in {
            "synthetic-milestone-zero",
            "synthetic-milestone-one",
        }:
            raise ValueError("measured records must use a synthetic-milestone scope")
        if self.evidence != "measured":
            raise ValueError("benchmark evidence must be measured")


def write_results(record: BenchmarkRecord, json_path: Path, csv_path: Path) -> None:
    json_path.parent.mkdir(parents=True, exist_ok=True)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(record)
    json_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    csv_payload = dict(payload)
    csv_payload["per_layer_nanoseconds"] = ";".join(
        str(value) for value in record.per_layer_nanoseconds
    )
    with csv_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_payload.keys())
        writer.writeheader()
        writer.writerow(csv_payload)


def _run_process(
    artifact: Path,
    runner: Path,
    generated_tokens: int,
    output: Path,
    *,
    backend: str,
    dense_precision: str,
    cuda_allocation: str,
    cuda_weights: str,
    cuda_batching: str,
    cuda_resident_bytes: int,
    diagnostics: bool = False,
) -> tuple[dict, int, float]:
    command = [
        str(runner), "--model", str(artifact), "--prompt-ids", "1,7,3,9",
        "--generate", str(generated_tokens), "--mode", "incremental",
        "--backend", backend, "--dense-precision", dense_precision,
        "--cuda-allocation", cuda_allocation,
        "--cuda-weights", cuda_weights,
        "--cuda-batching", cuda_batching,
        "--cuda-resident-bytes", str(cuda_resident_bytes),
    ]
    if diagnostics:
        command.extend(["--diagnostics", "true"])
    command.extend(["--json", str(output)])
    started = time.perf_counter_ns()
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    observed_peak = 0
    observed = psutil.Process(process.pid)
    while process.poll() is None:
        try:
            observed_peak = max(observed_peak, observed.memory_info().rss)
        except psutil.Error:
            pass
        time.sleep(0.001)
    stdout, stderr = process.communicate()
    wall_ms = (time.perf_counter_ns() - started) / 1.0e6
    if process.returncode:
        raise RuntimeError(
            f"runner failed with {process.returncode}: "
            f"{stderr.decode(errors='replace')} {stdout.decode(errors='replace')}"
        )
    return json.loads(output.read_text(encoding="utf-8")), observed_peak, wall_ms


def _state_sizes(artifact: Path, context_tokens: int) -> tuple[int, int]:
    config = K3XReader.open(artifact).model_config
    values = struct.unpack_from("<20I", config)
    layer_count, kda_heads, kda_dim, conv_kernel = values[2:6]
    mla_heads, q_main, q_extra, value_dim = values[6], values[9], values[10], values[11]
    kda_layers = layer_count - 1
    kda_per_layer = (
        3 * (conv_kernel - 1) * kda_heads * kda_dim
        + kda_heads * kda_dim * kda_dim
    ) * 4
    mla_per_token = (mla_heads * (q_main + value_dim) + q_extra) * 4
    return kda_layers * kda_per_layer, context_tokens * mla_per_token


def _flatten_numbers(value: object) -> list[float]:
    if isinstance(value, list):
        flattened: list[float] = []
        for item in value:
            flattened.extend(_flatten_numbers(item))
        return flattened
    return [float(value)]


def _numerical_errors(reference: dict, candidate: dict) -> tuple[float, float]:
    if candidate["token_ids"] != reference["token_ids"]:
        raise RuntimeError("backend token sequence diverged from the CPU reference")
    reference_values: list[float] = []
    candidate_values: list[float] = []
    for key in ("prefill_layer_outputs", "prefill_logits", "prefill_state"):
        reference_values.extend(_flatten_numbers(reference[key]))
        candidate_values.extend(_flatten_numbers(candidate[key]))
    if len(reference_values) != len(candidate_values):
        raise RuntimeError("backend diagnostic shape diverged from the CPU reference")
    maximum_absolute = 0.0
    maximum_relative = 0.0
    for expected, actual in zip(reference_values, candidate_values, strict=True):
        absolute = abs(actual - expected)
        relative = absolute / max(abs(expected), 1.0e-30)
        maximum_absolute = max(maximum_absolute, absolute)
        maximum_relative = max(maximum_relative, relative)
    return maximum_absolute, maximum_relative


def benchmark_once(
    artifact: Path,
    runner: Path,
    warmup: int,
    iterations: int,
    *,
    backend: str = "cpu",
    dense_precision: str = "fp32",
    cuda_allocation: str = "per-operation",
    cuda_weights: str = "transient",
    cuda_batching: str = "scalar",
    cuda_resident_bytes: int = 0,
) -> BenchmarkRecord:
    if warmup < 0 or iterations <= 0:
        raise ValueError("warmup must be non-negative and iterations must be positive")
    artifact, runner = Path(artifact).resolve(), Path(runner).resolve()
    prompt_tokens, generated_tokens = 4, 6
    samples: list[dict] = []
    peaks: list[int] = []
    ttft_samples: list[float] = []
    with tempfile.TemporaryDirectory(prefix="k3x-benchmark-") as temporary:
        root = Path(temporary)
        for index in range(warmup + iterations):
            sample, peak, _ = _run_process(
                artifact,
                runner,
                generated_tokens,
                root / f"run-{index}.json",
                backend=backend,
                dense_precision=dense_precision,
                cuda_allocation=cuda_allocation,
                cuda_weights=cuda_weights,
                cuda_batching=cuda_batching,
                cuda_resident_bytes=cuda_resident_bytes,
            )
            _, ttft_peak, ttft = _run_process(
                artifact,
                runner,
                1,
                root / f"ttft-{index}.json",
                backend=backend,
                dense_precision=dense_precision,
                cuda_allocation=cuda_allocation,
                cuda_weights=cuda_weights,
                cuda_batching=cuda_batching,
                cuda_resident_bytes=cuda_resident_bytes,
            )
            if index >= warmup:
                samples.append(sample)
                peaks.append(max(peak, ttft_peak))
                ttft_samples.append(ttft)
        if backend == "cpu":
            max_absolute_error = 0.0
            max_relative_error = 0.0
        else:
            reference, _, _ = _run_process(
                artifact,
                runner,
                generated_tokens,
                root / "reference.json",
                backend="cpu",
                dense_precision="fp32",
                cuda_allocation="per-operation",
                cuda_weights="transient",
                cuda_batching="scalar",
                cuda_resident_bytes=0,
                diagnostics=True,
            )
            candidate, _, _ = _run_process(
                artifact,
                runner,
                generated_tokens,
                root / "candidate.json",
                backend=backend,
                dense_precision=dense_precision,
                cuda_allocation=cuda_allocation,
                cuda_weights=cuda_weights,
                cuda_batching=cuda_batching,
                cuda_resident_bytes=cuda_resident_bytes,
                diagnostics=True,
            )
            max_absolute_error, max_relative_error = _numerical_errors(
                reference, candidate
            )
    deterministic_fields = (
        "backend",
        "device",
        "dense_precision",
        "cuda_allocation",
        "cuda_weights",
        "cuda_batching",
        "cuda_resident_bytes",
        "device_allocation_count",
        "device_free_count",
        "stream_synchronization_count",
        "weight_cache_hits",
        "weight_cache_misses",
        "weight_cache_bypasses",
        "resident_weight_bytes",
        "peak_resident_weight_bytes",
        "scratch_bytes",
        "peak_scratch_bytes",
        "weight_h2d_bytes",
        "activation_h2d_bytes",
        "grouped_projection_calls",
        "grouped_projection_members",
    )
    if any(
        any(item[field] != samples[0][field] for field in deterministic_fields)
        for item in samples[1:]
    ):
        raise RuntimeError("runner metadata changed across benchmark samples")
    expected_options = (
        backend,
        dense_precision,
        cuda_allocation,
        cuda_weights,
        cuda_batching,
        cuda_resident_bytes,
    )
    option_fields = deterministic_fields[:1] + deterministic_fields[2:7]
    observed_options = tuple(samples[0][field] for field in option_fields)
    if observed_options != expected_options:
        raise RuntimeError("runner metadata did not match requested benchmark options")
    prefill_ns = statistics.median(item["prefill_nanoseconds"] for item in samples)
    decode_ns = statistics.median(item["decode_nanoseconds"] for item in samples)
    layer_count = len(samples[0]["per_layer_nanoseconds"])
    layer_ns = tuple(
        int(statistics.median(item["per_layer_nanoseconds"][index] for item in samples))
        for index in range(layer_count)
    )
    kda_state, mla_kv = _state_sizes(artifact, prompt_tokens + generated_tokens - 1)
    return BenchmarkRecord(
        scope="synthetic-milestone-one",
        evidence="measured",
        platform=f"{platform.system()} {platform.release()} / {platform.machine()}",
        iterations=iterations,
        prompt_tokens=prompt_tokens,
        generated_tokens=generated_tokens,
        prefill_tokens_per_second=prompt_tokens * 1.0e9 / prefill_ns,
        decode_tokens_per_second=(generated_tokens - 1) * 1.0e9 / decode_ns,
        ttft_ms=statistics.median(ttft_samples),
        peak_rss_bytes=max(peaks),
        file_read_bytes_per_token=statistics.median(
            item["read_bytes"] / generated_tokens for item in samples
        ),
        backend=samples[0]["backend"],
        device=samples[0]["device"],
        dense_precision=samples[0]["dense_precision"],
        cuda_allocation=samples[0]["cuda_allocation"],
        cuda_weights=samples[0]["cuda_weights"],
        cuda_batching=samples[0]["cuda_batching"],
        cuda_resident_bytes=samples[0]["cuda_resident_bytes"],
        kernel_nanoseconds=int(
            statistics.median(item["kernel_nanoseconds"] for item in samples)
        ),
        host_to_device_bytes=int(
            statistics.median(item["host_to_device_bytes"] for item in samples)
        ),
        weight_h2d_bytes=samples[0]["weight_h2d_bytes"],
        activation_h2d_bytes=samples[0]["activation_h2d_bytes"],
        device_to_host_bytes=int(
            statistics.median(item["device_to_host_bytes"] for item in samples)
        ),
        peak_vram_bytes=max(item["peak_vram_bytes"] for item in samples),
        device_allocation_count=samples[0]["device_allocation_count"],
        device_free_count=samples[0]["device_free_count"],
        stream_synchronization_count=samples[0]["stream_synchronization_count"],
        weight_cache_hits=samples[0]["weight_cache_hits"],
        weight_cache_misses=samples[0]["weight_cache_misses"],
        weight_cache_bypasses=samples[0]["weight_cache_bypasses"],
        resident_weight_bytes=samples[0]["resident_weight_bytes"],
        peak_resident_weight_bytes=samples[0]["peak_resident_weight_bytes"],
        scratch_bytes=samples[0]["scratch_bytes"],
        peak_scratch_bytes=samples[0]["peak_scratch_bytes"],
        grouped_projection_calls=samples[0]["grouped_projection_calls"],
        grouped_projection_members=samples[0]["grouped_projection_members"],
        max_absolute_error=max_absolute_error,
        max_relative_error=max_relative_error,
        kda_state_bytes=kda_state,
        mla_kv_bytes=mla_kv,
        per_layer_nanoseconds=layer_ns,
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument(
        "--backend", choices=("cpu", "cuda-dense", "cuda-custom"), default="cpu"
    )
    parser.add_argument(
        "--dense-precision", choices=("fp32", "bf16"), default="fp32"
    )
    parser.add_argument(
        "--cuda-allocation", choices=("per-operation", "reused"),
        default="per-operation",
    )
    parser.add_argument(
        "--cuda-weights", choices=("transient", "resident"), default="transient"
    )
    parser.add_argument(
        "--cuda-batching", choices=("scalar", "grouped"), default="scalar"
    )
    parser.add_argument("--cuda-resident-bytes", type=int, default=0)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    args = parser.parse_args()
    result = benchmark_once(
        args.artifact,
        args.runner,
        args.warmup,
        args.iterations,
        backend=args.backend,
        dense_precision=args.dense_precision,
        cuda_allocation=args.cuda_allocation,
        cuda_weights=args.cuda_weights,
        cuda_batching=args.cuda_batching,
        cuda_resident_bytes=args.cuda_resident_bytes,
    )
    write_results(result, args.json, args.csv)
    print(json.dumps(asdict(result), sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
