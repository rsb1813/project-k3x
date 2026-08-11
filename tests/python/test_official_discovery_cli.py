# 공식 가중치 discovery CLI의 dry-run과 증거 출력을 검증합니다.
from __future__ import annotations

import csv
import hashlib
import json
import struct
from pathlib import Path

import pytest

from k3x_converter.format import K3XError, OPTIONAL_STORAGE_FIXTURE
from k3x_converter.official_transport import HttpResponse, TransportStats
from tools.discover_official_kimi_k3 import main
from tools.verify_official_discovery import (
    CSV_FIELDS,
    canonical_record_sha256,
    summary_csv_row,
    verify_summary,
)


_COMMIT = "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"
_SHARD = "model-00002-of-000096.safetensors"
_SHARD_SIZE = 16_990_911_504
_HEADER_LENGTH = 818_696
_DATA_START = 818_704
_PAYLOAD_START = 1_268_562_960
_PAYLOAD_END = 1_286_110_224
_BASE = "language_model.model.layers.1.block_sparse_moe.experts.0"


def _git_blob_id(body: bytes) -> str:
    prefix = b"blob " + str(len(body)).encode() + b"\0"
    return hashlib.sha1(prefix + body).hexdigest()


def _config_body() -> bytes:
    value = {
        "model_type": "kimi_k3",
        "text_config": {
            "model_type": "kimi_linear",
            "vocab_size": 163_840,
            "num_hidden_layers": 93,
            "first_k_dense_replace": 1,
            "moe_layer_freq": 1,
            "num_experts": 896,
            "num_experts_per_token": 16,
            "num_shared_experts": 2,
            "hidden_size": 7_168,
            "routed_expert_hidden_size": 3_584,
            "moe_intermediate_size": 3_072,
            "moe_renormalize": True,
            "moe_router_activation_func": "sigmoid",
            "num_expert_group": 1,
            "topk_group": 1,
            "activation_situ_beta": 4.0,
            "activation_situ_linear_beta": 25.0,
            "routed_scaling_factor": 1.0,
            "latent_moe_use_norm": True,
            "rms_norm_eps": 1.0e-5,
        },
    }
    return json.dumps(value, separators=(",", ":")).encode()


def _index_body(shards: tuple[str, ...]) -> bytes:
    weight_map = {
        f"{_BASE}.{matrix}.{kind}": _SHARD
        for matrix in ("w1", "w2", "w3")
        for kind in ("weight_packed", "weight_scale")
    }
    weight_map.update(
        {f"unused.{index}": path for index, path in enumerate(shards) if path != _SHARD}
    )
    value = {
        "metadata": {"total_size": 1_560_860_324_864},
        "weight_map": weight_map,
    }
    return json.dumps(value, separators=(",", ":")).encode()


def _header_body() -> bytes:
    selected = [
        ("w1.weight_packed", [3072, 1792], [1_267_744_256, 1_273_249_280]),
        ("w1.weight_scale", [3072, 112], [1_273_249_280, 1_273_593_344]),
        ("w2.weight_packed", [3584, 1536], [1_273_593_344, 1_279_098_368]),
        ("w2.weight_scale", [3584, 96], [1_279_098_368, 1_279_442_432]),
        ("w3.weight_packed", [3072, 1792], [1_279_442_432, 1_284_947_456]),
        ("w3.weight_scale", [3072, 112], [1_284_947_456, 1_285_291_520]),
    ]
    value: dict[str, object] = {
        "before": {
            "dtype": "I16",
            "shape": [1],
            "data_offsets": [0, selected[0][2][0]],
        },
        **{
            f"{_BASE}.{suffix}": {
                "dtype": "U8",
                "shape": shape,
                "data_offsets": offsets,
            }
            for suffix, shape, offsets in selected
        },
        "after": {
            "dtype": "I16",
            "shape": [1],
            "data_offsets": [selected[-1][2][1], _SHARD_SIZE - _DATA_START],
        },
    }
    encoded = json.dumps(value, separators=(",", ":")).encode()
    return encoded + b" " * (_HEADER_LENGTH - len(encoded))


class _DiscoveryTransport:
    def __init__(self) -> None:
        self.config = _config_body()
        self.shards = tuple(
            f"model-{index:05d}-of-000096.safetensors" for index in range(1, 97)
        )
        self.index = _index_body(self.shards)
        self.header = _header_body()
        cycle = bytes(range(256))
        length = _PAYLOAD_END - _PAYLOAD_START
        self.payload = (cycle * ((length + 255) // 256))[:length]
        siblings: list[dict[str, object]] = [
            {
                "rfilename": "config.json",
                "size": len(self.config),
                "blobId": _git_blob_id(self.config),
            },
            {
                "rfilename": "model.safetensors.index.json",
                "size": len(self.index),
                "blobId": "2" * 40,
                "lfs": {
                    "size": len(self.index),
                    "sha256": hashlib.sha256(self.index).hexdigest(),
                },
            },
        ]
        for index, path in enumerate(self.shards):
            size = _SHARD_SIZE if path == _SHARD else 10_000 + index
            siblings.append(
                {
                    "rfilename": path,
                    "size": size,
                    "blobId": f"{index + 3:040x}",
                    "lfs": {"size": size, "sha256": f"{index + 3:064x}"},
                }
            )
        self.api = json.dumps(
            {
                "id": "moonshotai/Kimi-K3",
                "sha": _COMMIT,
                "private": False,
                "gated": False,
                "siblings": siblings,
            },
            separators=(",", ":"),
        ).encode()
        self.calls: list[str] = []
        self.response_bytes = 0
        self.maximum_response_bytes = 0
        self.payload_requested = False

    @property
    def stats(self) -> TransportStats:
        return TransportStats(
            len(self.calls), self.response_bytes, self.maximum_response_bytes
        )

    def get(
        self,
        url: str,
        *,
        headers: dict[str, str],
        max_bytes: int,
        timeout_seconds: float,
        expected_status: int = 200,
    ) -> HttpResponse:
        self.calls.append(url)
        response_headers: dict[str, str] = {}
        if "/api/models/" in url:
            body = self.api
        elif "model.safetensors.index.json" in url:
            body = self.index
        elif url.endswith("/config.json"):
            body = self.config
        else:
            requested = headers["Range"]
            if requested == "bytes=0-7":
                body = struct.pack("<Q", _HEADER_LENGTH)
                start, end = 0, 7
            elif requested == f"bytes=8-{_HEADER_LENGTH + 7}":
                body = self.header
                start, end = 8, _HEADER_LENGTH + 7
            elif requested == f"bytes={_PAYLOAD_START}-{_PAYLOAD_END - 1}":
                self.payload_requested = True
                body = self.payload
                start, end = _PAYLOAD_START, _PAYLOAD_END - 1
            else:
                raise AssertionError(requested)
            response_headers["content-range"] = f"bytes {start}-{end}/{_SHARD_SIZE}"
        assert len(body) <= max_bytes
        self.response_bytes += len(body)
        self.maximum_response_bytes = max(self.maximum_response_bytes, len(body))
        return HttpResponse(expected_status, url, response_headers, body)


def test_cli_dry_run_plans_real_shape_without_payload_access(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    summary_path = tmp_path / "dry-run.json"
    transport = _DiscoveryTransport()

    assert main(["--summary-json", str(summary_path)], transport=transport) == 0

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    stdout = json.loads(capsys.readouterr().out)
    assert stdout == summary
    assert summary["mode"] == "dry-run"
    assert summary["resolved_revision"] == _COMMIT
    assert summary["expert"]["payload_bytes"] == 17_547_264
    assert summary["traffic"]["header_bytes"] == 818_704
    assert summary["traffic"]["tensor_payload_bytes"] == 0
    assert summary["reader_valid"] is False
    assert transport.payload_requested is False


def test_cli_requires_explicit_live_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("K3X_TEST_OFFICIAL_DISCOVERY", raising=False)

    with pytest.raises(K3XError, match="OFFICIAL_LIVE_OPT_IN_REQUIRED"):
        main([])


def test_cli_materialization_requires_untracked_output_directory(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main(["--materialize-expert"], transport=_DiscoveryTransport())

    repository_results = Path(__file__).resolve().parents[2] / "results" / "forbidden"
    with pytest.raises(K3XError, match="OFFICIAL_OUTPUT_LOCATION"):
        main(
            ["--materialize-expert", "--output-dir", str(repository_results)],
            transport=_DiscoveryTransport(),
        )


def test_cli_materializes_and_writes_verifiable_json_csv(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    output = tmp_path / "artifacts"
    summary_json = tmp_path / "summary.json"
    summary_csv = tmp_path / "summary.csv"
    transport = _DiscoveryTransport()

    assert main(
        [
            "--materialize-expert",
            "--output-dir",
            str(output),
            "--summary-json",
            str(summary_json),
            "--summary-csv",
            str(summary_csv),
        ],
        transport=transport,
    ) == 0
    capsys.readouterr()

    summary = verify_summary(summary_json, summary_csv, strict_official=False)
    assert summary["mode"] == "materialize-expert"
    assert summary["traffic"]["tensor_payload_bytes"] == 17_547_264
    assert summary["reader_valid"] is True
    assert summary["optional_features"] == OPTIONAL_STORAGE_FIXTURE
    assert transport.payload_requested is True
    with summary_csv.open(newline="", encoding="utf-8") as stream:
        assert len(list(csv.DictReader(stream))) == 1


def test_verifier_rejects_consistently_rehashed_invalid_artifact_digest(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    summary_json = tmp_path / "summary.json"
    summary_csv = tmp_path / "summary.csv"
    main(
        [
            "--materialize-expert",
            "--output-dir",
            str(tmp_path / "artifacts"),
            "--summary-json",
            str(summary_json),
            "--summary-csv",
            str(summary_csv),
        ],
        transport=_DiscoveryTransport(),
    )
    capsys.readouterr()
    record = json.loads(summary_json.read_text(encoding="utf-8"))
    record["artifacts"]["payload_sha256"] = "not-a-digest"
    record.pop("record_sha256")
    record.pop("summary_csv_sha256")
    record["record_sha256"] = canonical_record_sha256(record)
    with summary_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerow(summary_csv_row(record))
    record["summary_csv_sha256"] = hashlib.sha256(summary_csv.read_bytes()).hexdigest()
    summary_json.write_text(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(K3XError, match="INVALID_OFFICIAL_EVIDENCE"):
        verify_summary(summary_json, summary_csv, strict_official=False)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda value: value.update(repository="other/model"),
        lambda value: value.update(requested_revision="latest"),
        lambda value: value.update(repository_bytes=1),
        lambda value: value.update(snapshot_sha256="0" * 64),
        lambda value: value["config"].update(git_blob_id="0" * 40),
        lambda value: value["index"].update(tensor_count=1),
        lambda value: value["expert"].update(shard_path="other.safetensors"),
        lambda value: value["artifacts"].update(payload_sha256="0" * 64),
        lambda value: value["artifacts"]["tensor_sha256"].update(
            {"model.layers.1.feed_forward.experts.0.down.weight_packed": "0" * 64}
        ),
    ],
)
def test_strict_verifier_binds_official_snapshot_and_layout_identity(
    tmp_path: Path, mutation
) -> None:
    root = Path(__file__).resolve().parents[2]
    record = json.loads(
        (root / "results/b0027-official-range/summary.json").read_text(
            encoding="utf-8"
        )
    )
    mutation(record)
    record.pop("record_sha256")
    record.pop("summary_csv_sha256")
    record["record_sha256"] = canonical_record_sha256(record)
    summary_csv = tmp_path / "summary.csv"
    with summary_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerow(summary_csv_row(record))
    record["summary_csv_sha256"] = hashlib.sha256(summary_csv.read_bytes()).hexdigest()
    summary_json = tmp_path / "summary.json"
    summary_json.write_text(
        json.dumps(record, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )

    with pytest.raises(K3XError, match="OFFICIAL_IDENTITY_MISMATCH"):
        verify_summary(summary_json, summary_csv)
