# 공식 Kimi K3의 영구 인메모리 다중 토큰 decode 처리량을 측정합니다.
from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
import time
from pathlib import Path

import torch

from k3x_converter.format import K3XError
from tools.official_in_memory_state import OfficialInMemoryState
from tools.official_runtime_context import OfficialRuntimeContext
from tools.run_official_head import run as run_head
from tools.run_official_layer0 import _write_json_atomic, run as run_layer0
from tools.run_official_remaining import run as run_remaining


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", type=Path, required=True)
    parser.add_argument("--object-dir", type=Path, required=True)
    parser.add_argument("--k3x-set", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--implementation-commit", required=True)
    parser.add_argument("--input-token-id", type=int, default=1)
    parser.add_argument("--generate", type=int, default=2)
    parser.add_argument("--q8-host-cache-bytes", type=int, default=0)
    parser.add_argument("--q8-device-cache-bytes", type=int, default=0)
    parser.add_argument("--mxfp4-host-cache-bytes", type=int, default=0)
    parser.add_argument("--mxfp4-device-cache-bytes", type=int, default=0)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if not torch.cuda.is_available() or args.generate < 2:
        raise K3XError("OFFICIAL_DECODE_BENCHMARK_UNAVAILABLE")
    context = OfficialRuntimeContext.create(
        topology_path=args.topology.resolve(),
        object_dir=args.object_dir.resolve(),
        k3x_set=args.k3x_set.resolve(),
        q8_host_cache_bytes=args.q8_host_cache_bytes,
        q8_device_cache_bytes=args.q8_device_cache_bytes,
        mxfp4_host_cache_bytes=args.mxfp4_host_cache_bytes,
        mxfp4_device_cache_bytes=args.mxfp4_device_cache_bytes,
    )
    state = OfficialInMemoryState()
    device = torch.device("cuda", torch.cuda.current_device())
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats(device)
    token_walls: list[float] = []
    input_tokens: list[int] = []
    current_token = args.input_token_id

    with tempfile.TemporaryDirectory(prefix="k3x-official-decode-") as temporary:
        temporary_path = Path(temporary)
        result_dir = temporary_path / "results"
        result_dir.mkdir()
        for token_index in range(args.generate):
            input_tokens.append(current_token)
            state.begin_token(current_token)
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            common = {
                "topology": args.topology.resolve(),
                "object_dir": args.object_dir.resolve(),
                "state_dir": temporary_path,
                "k3x_set": args.k3x_set.resolve(),
                "runtime_context": context,
                "in_memory_state": state,
                "direct_q8": True,
            }
            if run_layer0(
                argparse.Namespace(
                    **common,
                    output=result_dir / "layer-00.json",
                    token_id=current_token,
                )
            ) != 0:
                raise K3XError("OFFICIAL_DECODE_LAYER0")
            if run_remaining(
                argparse.Namespace(
                    **common,
                    result_dir=result_dir,
                    stop_layer=92,
                    execution_mode="in-process",
                )
            ) != 0:
                raise K3XError("OFFICIAL_DECODE_REMAINING")
            if run_head(
                argparse.Namespace(
                    **common,
                    output=result_dir / "head.json",
                )
            ) != 0:
                raise K3XError("OFFICIAL_DECODE_HEAD")
            torch.cuda.synchronize(device)
            wall = time.perf_counter() - started
            token_walls.append(wall)
            current_token = state.generated_tokens[-1]
            print(
                f"completed_token={token_index + 1}/{args.generate} "
                f"token_id={current_token} wall_seconds={wall:.6f}",
                flush=True,
            )

    decode_seconds = sum(token_walls[1:])
    decode_tokens = len(token_walls) - 1
    mla_lengths = [
        attention.length
        for attention in state.attention_states.values()
        if hasattr(attention, "length")
    ]
    record = {
        "format": "k3x-official-in-memory-decode-benchmark-v1",
        "implementation_commit": args.implementation_commit,
        "device": torch.cuda.get_device_name(device),
        "input_token_ids": input_tokens,
        "generated_token_ids": state.generated_tokens,
        "generated_logits_fp32": state.generated_logits,
        "token_wall_seconds": token_walls,
        "ttft_seconds": token_walls[0],
        "decode_tokens": decode_tokens,
        "decode_seconds": decode_seconds,
        "decode_tokens_per_second": decode_tokens / decode_seconds,
        "throughput_measured": True,
        "direct_q8": True,
        "natural_top_k": 16,
        "first_token_matches_b0050": state.generated_tokens[0] == 9689,
        "attention_state_count": len(state.attention_states),
        "maximum_mla_length": max(mla_lengths, default=0),
        "q8_cache": context.packed_q8_cache.snapshot(),
        "mxfp4_cache": context.packed_mxfp4_cache.snapshot(),
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(device),
        "peak_cuda_reserved_bytes": torch.cuda.max_memory_reserved(device),
        "physical_h2d_bytes_measured": False,
        "physical_nvme_bytes_measured": False,
        "coding_quality_measured": False,
    }
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    record["record_sha256"] = hashlib.sha256(encoded).hexdigest()
    _write_json_atomic(args.output.resolve(), record)
    print(json.dumps(record, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
