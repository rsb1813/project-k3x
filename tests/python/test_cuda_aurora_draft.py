# Top-16 합성 artifact에서 CUDA persistent AURORA draft CLI 경계를 검증합니다.
import json
import os
import subprocess
from pathlib import Path

import pytest

from conftest import cpp_binary
from k3x_converter.writer import convert
from k3x_ref.config import SyntheticK3Config
from k3x_ref.fixtures import write_source_checkpoint


def _top16_artifact(tmp_path: Path) -> Path:
    config = SyntheticK3Config.default().replace(num_experts=24, top_k=16)
    source = tmp_path / "source-top16"
    write_source_checkpoint(source, config=config)
    artifact = tmp_path / "top16.k3x"
    convert(source, artifact, chunk_bytes=257)
    return artifact


def test_cpu_build_rejects_cuda_draft_without_fallback(
    tmp_path: Path,
) -> None:
    if Path(os.environ.get("K3X_BUILD_DIR", "build")).name == "build-cuda":
        pytest.skip("CPU unavailable behavior is exercised against build")
    artifact = _top16_artifact(tmp_path)
    output = tmp_path / "unexpected-fallback.json"
    result = subprocess.run(
        [
            str(cpp_binary("k3x_run")),
            "--model", str(artifact),
            "--prompt-ids", "1,7,3,9",
            "--generate", "6",
            "--mode", "incremental",
            "--speculative-mode", "aurora-persistent",
            "--speculative-block-size", "2",
            "--aurora-draft-k", "4",
            "--aurora-draft-backend", "cuda-custom",
            "--json", str(output),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 4
    assert result.stderr.strip() == (
        "BACKEND_UNAVAILABLE: CUDA backend is disabled at build time"
    )
    assert not output.exists()


def test_cuda_draft_matches_cpu_persistent_target_execution(
    tmp_path: Path,
) -> None:
    if Path(os.environ.get("K3X_BUILD_DIR", "build")).name != "build-cuda":
        pytest.skip("CUDA AURORA CLI parity is exercised only against build-cuda")
    artifact = _top16_artifact(tmp_path)
    common = [
        str(cpp_binary("k3x_run")),
        "--model", str(artifact),
        "--prompt-ids", "1,7,3,9",
        "--generate", "6",
        "--mode", "incremental",
        "--diagnostics", "true",
        "--backend", "cpu",
        "--speculative-mode", "aurora-persistent",
        "--speculative-verification", "token-major",
        "--speculative-block-size", "2",
        "--aurora-draft-k", "4",
    ]

    for policy in ("fixed", "adaptive"):
        results: dict[str, dict] = {}
        for draft_backend in ("cpu", "cuda-custom"):
            output = tmp_path / f"{policy}-{draft_backend}.json"
            subprocess.run(
                [
                    *common,
                    "--aurora-block-policy", policy,
                    "--aurora-draft-backend", draft_backend,
                    "--json", str(output),
                ],
                check=True,
            )
            results[draft_backend] = json.loads(
                output.read_text(encoding="utf-8")
            )
        cpu = results["cpu"]
        cuda = results["cuda-custom"]
        assert cpu["aurora_draft_backend"] == "cpu"
        assert cuda["aurora_draft_backend"] == "cuda-custom"
        assert cpu["draft_device"] == "CPU"
        assert cpu["draft_kernel_nanoseconds"] == 0
        assert cpu["draft_host_to_device_bytes"] == 0
        assert cpu["draft_peak_vram_bytes"] == 0
        assert cuda["draft_cuda_allocation"] == "reused"
        assert cuda["draft_cuda_weights"] == "transient"
        assert cuda["draft_cuda_batching"] == "grouped"
        assert cuda["draft_cuda_boundary"] == "ffn-block"
        assert cuda["draft_cuda_transfer"] == "synchronous"
        assert cuda["draft_cuda_moe_fusion"] == "none"
        assert cuda["draft_kernel_nanoseconds"] > 0
        assert cuda["draft_host_to_device_bytes"] > 0
        assert cuda["draft_weight_h2d_bytes"] > 0
        assert cuda["draft_activation_h2d_bytes"] > 0
        assert cuda["draft_device_to_host_bytes"] > 0
        assert cuda["draft_peak_vram_bytes"] > 0
        assert cuda["draft_device_allocation_count"] > 0
        assert cuda["draft_stream_synchronization_count"] > 0
        assert cuda["kernel_nanoseconds"] == 0
        assert cuda["host_to_device_bytes"] == 0
        assert cuda["peak_vram_bytes"] == 0
        for field in (
            "token_ids",
            "final_state",
            "routed_experts",
            "routed_k",
            "speculative_proposed_draft_tokens",
            "speculative_accepted_draft_tokens",
            "speculative_committed_tokens",
            "draft_proposal_calls",
            "draft_candidate_tokens",
            "draft_context_prefill_tokens",
            "draft_incremental_forward_calls",
            "draft_rollback_events",
            "draft_mla_positions_cropped",
            "draft_kda_checkpoint_bytes",
        ):
            assert cuda[field] == cpu[field]
