# 공식 Kimi K3 snapshot 메타데이터의 고정 신원을 검증합니다.
from __future__ import annotations

import json

import pytest

from k3x_converter.format import K3XError
from k3x_converter.official_source import discover_official_snapshot
from k3x_converter.official_transport import HttpResponse


COMMIT = "9f62e4e9fffbd0a83ddd60e1c209d828994b3569"
INDEX_SHA256 = "a1c5210650ce71d2d3ae9ec5a101ac4afd3cf4b10091be589853437eb967febd"
SHARD_SHA256 = "26a3284e1d2cb567934ebef002e6a1813551d646739e8bcb1e9e3fe7f878e0f5"


def _api_record() -> dict[str, object]:
    return {
        "id": "moonshotai/Kimi-K3",
        "sha": COMMIT,
        "private": False,
        "gated": False,
        "siblings": [
            {
                "rfilename": "config.json",
                "size": 2468,
                "blobId": "1" * 40,
            },
            {
                "rfilename": "model.safetensors.index.json",
                "size": 59_764_096,
                "blobId": "2" * 40,
                "lfs": {"size": 59_764_096, "sha256": INDEX_SHA256},
            },
            {
                "rfilename": "model-00002-of-000096.safetensors",
                "size": 16_990_911_504,
                "blobId": "3" * 40,
                "lfs": {"size": 16_990_911_504, "sha256": SHARD_SHA256},
            },
        ],
    }


class _FakeTransport:
    def __init__(self, body: bytes) -> None:
        self.body = body
        self.calls: list[tuple[str, int]] = []

    def get(
        self,
        url: str,
        *,
        headers: dict[str, str],
        max_bytes: int,
        timeout_seconds: float,
        expected_status: int = 200,
    ) -> HttpResponse:
        self.calls.append((url, max_bytes))
        return HttpResponse(200, url, {"content-type": "application/json"}, self.body)


def _discover(record: dict[str, object], observed_at: str = "2026-08-10T00:00:00Z"):
    payload = json.dumps(record, separators=(",", ":")).encode("utf-8")
    return discover_official_snapshot(_FakeTransport(payload), observed_at=observed_at)


def test_snapshot_binds_repository_revision_and_file_identities() -> None:
    snapshot = _discover(_api_record())

    assert snapshot.repository == "moonshotai/Kimi-K3"
    assert snapshot.requested_revision == "main"
    assert snapshot.resolved_revision == COMMIT
    assert snapshot.file_count == 3
    assert snapshot.files["model.safetensors.index.json"].size == 59_764_096
    assert snapshot.files["model.safetensors.index.json"].lfs_sha256 == INDEX_SHA256
    assert snapshot.files["model-00002-of-000096.safetensors"].size == 16_990_911_504
    assert snapshot.files["model-00002-of-000096.safetensors"].lfs_sha256 == SHARD_SHA256
    assert len(snapshot.canonical_sha256) == 64


def test_snapshot_digest_excludes_observation_time() -> None:
    first = _discover(_api_record(), "2026-08-10T00:00:00Z")
    second = _discover(_api_record(), "2026-08-10T01:00:00Z")

    assert first.observed_at != second.observed_at
    assert first.canonical_sha256 == second.canonical_sha256


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        (lambda value: value.update(sha="main"), "OFFICIAL_REVISION_DRIFT"),
        (lambda value: value.update(private=True), "OFFICIAL_REPOSITORY_ACCESS"),
        (lambda value: value.update(gated=True), "OFFICIAL_REPOSITORY_ACCESS"),
        (
            lambda value: value["siblings"][0].update(rfilename="../config.json"),
            "INVALID_OFFICIAL_FILE",
        ),
        (
            lambda value: value["siblings"][1].update(size=True),
            "INVALID_OFFICIAL_FILE",
        ),
        (
            lambda value: value["siblings"][1].pop("lfs"),
            "INVALID_OFFICIAL_FILE",
        ),
    ],
)
def test_snapshot_rejects_identity_drift(mutation, code: str) -> None:
    record = _api_record()
    mutation(record)

    with pytest.raises(K3XError, match=code):
        _discover(record)


def test_snapshot_rejects_duplicate_file_paths() -> None:
    record = _api_record()
    record["siblings"].append(dict(record["siblings"][0]))

    with pytest.raises(K3XError, match="INVALID_OFFICIAL_FILE"):
        _discover(record)


def test_snapshot_rejects_duplicate_json_keys_and_nonstandard_constants() -> None:
    duplicate = b'{"id":"moonshotai/Kimi-K3","id":"other"}'
    with pytest.raises(K3XError, match="INVALID_OFFICIAL_API"):
        discover_official_snapshot(_FakeTransport(duplicate))

    invalid_constant = b'{"id":NaN}'
    with pytest.raises(K3XError, match="INVALID_OFFICIAL_API"):
        discover_official_snapshot(_FakeTransport(invalid_constant))

