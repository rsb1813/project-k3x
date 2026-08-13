# 공개 Kimi K3 전문가 하나의 3비트 K3X 변환과 오차를 측정합니다.
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch
from safetensors.torch import save_file

from k3x_converter.official_quant3 import quantize_mxfp4_payload
from k3x_converter.safetensors_reader import inspect_shard, iter_tensor_chunks
from k3x_converter.writer import convert
from k3x_ref.mxfp4 import decode_mxfp4
from k3x_ref.ops import situ_glu
from k3x_ref.quant3 import decode_groupwise_3bit


def _payload(tensor) -> bytes:
    return b"".join(iter_tensor_chunks(tensor, 8 * 1024 * 1024))


def _metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict[str, float]:
    reference = reference.float().flatten()
    candidate = candidate.float().flatten()
    difference = candidate - reference
    reference_norm = torch.linalg.vector_norm(reference).item()
    candidate_norm = torch.linalg.vector_norm(candidate).item()
    denominator = max(reference_norm * candidate_norm, 1.0e-30)
    return {
        "cosine": float(torch.dot(reference, candidate).item() / denominator),
        "max_abs": float(torch.max(torch.abs(difference)).item()),
        "relative_l2": float(
            torch.linalg.vector_norm(difference).item() / max(reference_norm, 1.0e-30)
        ),
        "rmse": float(torch.sqrt(torch.mean(difference.square())).item()),
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _tensor_payload_sha256(tensors: dict[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(tensors):
        payload = tensors[name].contiguous().numpy().tobytes()
        digest.update(name.encode("utf-8") + b"\0")
        digest.update(len(payload).to_bytes(8, "little"))
        digest.update(payload)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, sort_keys=True, separators=(",", ":"))
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    source_manifest = json.loads(
        (args.source / "source-manifest.json").read_text(encoding="utf-8")
    )
    shard_name = next(iter(set(source_manifest["weight_map"].values())))
    source_shard = args.source / shard_name
    tensors = inspect_shard(source_shard)
    bases = sorted(source_manifest["packed_shapes"])
    by_role = {base.rsplit(".", 1)[-1]: base for base in bases}
    if set(by_role) != {"gate", "up", "down"}:
        raise RuntimeError("official expert role mismatch")

    args.output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(20260813)
    hidden = torch.randn(4, source_manifest["config"]["routed_latent_size"])
    native_outputs: dict[str, torch.Tensor] = {}
    quant3_outputs: dict[str, torch.Tensor] = {}
    quant3_tensors: dict[str, torch.Tensor] = {}
    matrix_metrics: dict[str, object] = {}

    for role in ("gate", "up"):
        base = by_role[role]
        rows, cols = source_manifest["packed_shapes"][base]
        packed = _payload(tensors[base + ".weight_packed"])
        scales = _payload(tensors[base + ".weight_scale"])
        native = decode_mxfp4(packed, scales, rows, cols)
        encoded = quantize_mxfp4_payload(packed, scales, rows=rows, cols=cols)
        quantized = decode_groupwise_3bit(encoded)
        native_outputs[role] = hidden @ native.transpose(0, 1)
        quant3_outputs[role] = hidden @ quantized.transpose(0, 1)
        matrix_metrics[role] = _metrics(native, quantized)
        quant3_tensors[base + ".weight_q3_packed"] = torch.frombuffer(
            bytearray(encoded.packed), dtype=torch.uint8
        ).clone()
        quant3_tensors[base + ".weight_q3_scale"] = torch.frombuffer(
            bytearray(encoded.scales_bf16), dtype=torch.uint8
        ).clone()

    beta = float(source_manifest["config"]["activation_situ_beta"])
    linear_beta = float(source_manifest["config"]["activation_situ_linear_beta"])
    native_activation = situ_glu(
        native_outputs["gate"], native_outputs["up"], beta, linear_beta
    )
    quant3_activation = situ_glu(
        quant3_outputs["gate"], quant3_outputs["up"], beta, linear_beta
    )

    base = by_role["down"]
    rows, cols = source_manifest["packed_shapes"][base]
    packed = _payload(tensors[base + ".weight_packed"])
    scales = _payload(tensors[base + ".weight_scale"])
    native = decode_mxfp4(packed, scales, rows, cols)
    encoded = quantize_mxfp4_payload(packed, scales, rows=rows, cols=cols)
    quantized = decode_groupwise_3bit(encoded)
    native_expert_output = native_activation @ native.transpose(0, 1)
    quant3_same_input = native_activation @ quantized.transpose(0, 1)
    quant3_expert_output = quant3_activation @ quantized.transpose(0, 1)
    matrix_metrics["down"] = _metrics(native, quantized)
    quant3_tensors[base + ".weight_q3_packed"] = torch.frombuffer(
        bytearray(encoded.packed), dtype=torch.uint8
    ).clone()
    quant3_tensors[base + ".weight_q3_scale"] = torch.frombuffer(
        bytearray(encoded.scales_bf16), dtype=torch.uint8
    ).clone()

    q3_source = args.output / "source"
    q3_source.mkdir(exist_ok=True)
    q3_shard = q3_source / "expert-q3.safetensors"
    save_file(quant3_tensors, q3_shard)
    quant3_shapes = {base: source_manifest["packed_shapes"][base] for base in bases}
    q3_manifest = {
        "format": "synthetic-k3-source-v1",
        "config": source_manifest["config"],
        "packed_shapes": {},
        "quant3_shapes": quant3_shapes,
        "weight_map": {name: q3_shard.name for name in quant3_tensors},
    }
    _write_json(q3_source / "source-manifest.json", q3_manifest)
    k3x_path = args.output / "official-expert-q3.k3x"
    report = convert(q3_source, k3x_path)
    if not report.completed:
        raise RuntimeError("K3X conversion did not complete")

    record = {
        "format": "k3x-official-expert-quant3-quality-v1",
        "quality_scope": "deterministic-random-normal-expert-proxy",
        "samples": hidden.shape[0],
        "seed": 20260813,
        "source_revision": source_manifest["source_provenance"]["resolved_revision"],
        "source_sha256": source_manifest["source_sha256"],
        "matrix_metrics": matrix_metrics,
        "projection_metrics": {
            role: _metrics(native_outputs[role], quant3_outputs[role])
            for role in ("gate", "up")
        },
        "down_same_input_metrics": _metrics(native_expert_output, quant3_same_input),
        "expert_output_metrics": _metrics(native_expert_output, quant3_expert_output),
        "native_payload_bytes": int(source_manifest["payload_bytes"]),
        "quant3_payload_bytes": sum(tensor.numel() for tensor in quant3_tensors.values()),
        "quant3_payload_sha256": _tensor_payload_sha256(quant3_tensors),
        "quant3_source_shard_bytes": q3_shard.stat().st_size,
        "k3x_bytes": k3x_path.stat().st_size,
        "k3x_sha256": _sha256(k3x_path),
    }
    _write_json(args.output / "quality.json", record)
    print(json.dumps(record, sort_keys=True, separators=(",", ":")))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
