# B-0027 공식 range discovery JSON과 CSV 증거의 일치를 검증합니다.
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from pathlib import Path
from typing import Mapping

from k3x_converter.format import K3XError, OPTIONAL_STORAGE_FIXTURE


CSV_FIELDS = (
    "mode",
    "resolved_revision",
    "index_sha256",
    "index_bytes",
    "shard_lfs_sha256",
    "shard_bytes",
    "payload_start",
    "payload_end",
    "payload_bytes",
    "http_requests",
    "metadata_bytes",
    "header_bytes",
    "tensor_payload_bytes",
    "maximum_response_bytes",
    "reader_valid",
    "optional_features",
    "provenance",
    "full_shard_verified",
    "payload_sha256",
    "microshard_sha256",
    "k3x_root_sha256",
    "wall_seconds",
)
_FORBIDDEN_KEYS = {
    "decode_tok_s",
    "prefill_tok_s",
    "ttft",
    "gpu_utilization",
    "gpu_memory_bandwidth",
    "nvme_gb_per_token",
    "quality",
}
_COMMIT = "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"
_INDEX_SHA256 = "a1c5210650ce71d2d3ae9ec5a101ac4afd3cf4b10091be589853437eb967febd"
_SHARD_SHA256 = "26a3284e1d2cb567934ebef002e6a1813551d646739e8bcb1e9e3fe7f878e0f5"
_HEX64 = re.compile(r"[0-9a-f]{64}")


def canonical_record_sha256(record: Mapping[str, object]) -> str:
    value = dict(record)
    value.pop("record_sha256", None)
    value.pop("summary_csv_sha256", None)
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _scalar(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def summary_csv_row(record: Mapping[str, object]) -> dict[str, str]:
    index = record["index"]
    expert = record["expert"]
    traffic = record["traffic"]
    artifacts = record["artifacts"]
    assert isinstance(index, dict)
    assert isinstance(expert, dict)
    assert isinstance(traffic, dict)
    assert isinstance(artifacts, dict)
    values: dict[str, object] = {
        "mode": record["mode"],
        "resolved_revision": record["resolved_revision"],
        "index_sha256": index["sha256"],
        "index_bytes": index["bytes"],
        "shard_lfs_sha256": expert["shard_lfs_sha256"],
        "shard_bytes": expert["shard_bytes"],
        "payload_start": expert["payload_start"],
        "payload_end": expert["payload_end"],
        "payload_bytes": expert["payload_bytes"],
        "http_requests": traffic["http_requests"],
        "metadata_bytes": traffic["metadata_bytes"],
        "header_bytes": traffic["header_bytes"],
        "tensor_payload_bytes": traffic["tensor_payload_bytes"],
        "maximum_response_bytes": traffic["maximum_response_bytes"],
        "reader_valid": record["reader_valid"],
        "optional_features": record["optional_features"],
        "provenance": record["provenance"],
        "full_shard_verified": record["full_shard_verified"],
        "payload_sha256": artifacts.get("payload_sha256"),
        "microshard_sha256": artifacts.get("microshard_sha256"),
        "k3x_root_sha256": artifacts.get("k3x_root_sha256"),
        "wall_seconds": record["wall_seconds"],
    }
    return {key: _scalar(values[key]) for key in CSV_FIELDS}


def _load_json(path: Path) -> dict[str, object]:
    def reject_pairs(pairs: list[tuple[str, object]]) -> dict[str, object]:
        result: dict[str, object] = {}
        for key, value in pairs:
            if key in result:
                raise K3XError("INVALID_OFFICIAL_EVIDENCE")
            result[key] = value
        return result

    try:
        value = json.loads(
            path.read_bytes(),
            object_pairs_hook=reject_pairs,
            parse_constant=lambda _: (_ for _ in ()).throw(
                K3XError("INVALID_OFFICIAL_EVIDENCE")
            ),
        )
    except K3XError:
        raise
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise K3XError("INVALID_OFFICIAL_EVIDENCE") from error
    if not isinstance(value, dict):
        raise K3XError("INVALID_OFFICIAL_EVIDENCE")
    return value


def _reject_forbidden(value: object) -> None:
    if isinstance(value, dict):
        if _FORBIDDEN_KEYS.intersection(value):
            raise K3XError("FORBIDDEN_OFFICIAL_METRIC")
        for child in value.values():
            _reject_forbidden(child)
    elif isinstance(value, list):
        for child in value:
            _reject_forbidden(child)


def verify_summary(
    json_path: Path,
    csv_path: Path,
    *,
    strict_official: bool = True,
) -> dict[str, object]:
    json_path, csv_path = Path(json_path), Path(csv_path)
    record = _load_json(json_path)
    _reject_forbidden(record)
    if record.get("format") != "k3x-official-discovery-v1":
        raise K3XError("INVALID_OFFICIAL_EVIDENCE")
    if record.get("record_sha256") != canonical_record_sha256(record):
        raise K3XError("OFFICIAL_EVIDENCE_DIGEST_MISMATCH")
    csv_digest = hashlib.sha256(csv_path.read_bytes()).hexdigest()
    if record.get("summary_csv_sha256") != csv_digest:
        raise K3XError("OFFICIAL_EVIDENCE_DIGEST_MISMATCH")
    with csv_path.open(newline="", encoding="utf-8") as stream:
        reader = csv.DictReader(stream)
        rows = list(reader)
        if tuple(reader.fieldnames or ()) != CSV_FIELDS or len(rows) != 1:
            raise K3XError("INVALID_OFFICIAL_EVIDENCE")
    if rows[0] != summary_csv_row(record):
        raise K3XError("OFFICIAL_EVIDENCE_PARITY_MISMATCH")
    expert = record.get("expert")
    traffic = record.get("traffic")
    index = record.get("index")
    artifacts = record.get("artifacts")
    if (
        not isinstance(expert, dict)
        or not isinstance(traffic, dict)
        or not isinstance(index, dict)
        or not isinstance(artifacts, dict)
    ):
        raise K3XError("INVALID_OFFICIAL_EVIDENCE")
    tensor_digests = artifacts.get("tensor_sha256")
    if (
        any(
            not isinstance(artifacts.get(key), str)
            or _HEX64.fullmatch(artifacts[key]) is None
            for key in ("payload_sha256", "microshard_sha256", "k3x_root_sha256")
        )
        or not isinstance(tensor_digests, dict)
        or len(tensor_digests) != 6
        or any(
            not isinstance(name, str)
            or not name
            or not isinstance(digest, str)
            or _HEX64.fullmatch(digest) is None
            for name, digest in tensor_digests.items()
        )
    ):
        raise K3XError("INVALID_OFFICIAL_EVIDENCE")
    if (
        record.get("mode") != "materialize-expert"
        or record.get("resolved_revision") != _COMMIT
        or expert.get("payload_start") != 1_268_562_960
        or expert.get("payload_end") != 1_286_110_224
        or expert.get("payload_bytes") != 17_547_264
        or traffic.get("header_bytes") != 818_704
        or traffic.get("tensor_payload_bytes") != 17_547_264
        or record.get("reader_valid") is not True
        or record.get("optional_features") != OPTIONAL_STORAGE_FIXTURE
        or record.get("provenance") != "transport-pinned-range"
        or record.get("full_shard_verified") is not False
    ):
        raise K3XError("INVALID_OFFICIAL_EVIDENCE")
    if strict_official and (
        record.get("file_count") != 118
        or index.get("bytes") != 59_764_096
        or index.get("sha256") != _INDEX_SHA256
        or expert.get("shard_bytes") != 16_990_911_504
        or expert.get("shard_lfs_sha256") != _SHARD_SHA256
    ):
        raise K3XError("OFFICIAL_IDENTITY_MISMATCH")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("summary_json", type=Path)
    parser.add_argument("summary_csv", type=Path)
    args = parser.parse_args(argv)
    verify_summary(args.summary_json, args.summary_csv)
    print("B-0027 evidence verified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
