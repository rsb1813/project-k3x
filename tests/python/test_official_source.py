# 공식 Kimi K3 snapshot 메타데이터의 고정 신원을 검증합니다.
from __future__ import annotations

import json
import hashlib
import struct
from types import MappingProxyType

import pytest

from k3x_converter.format import K3XError
from k3x_converter.official_source import (
    OfficialFile,
    OfficialIndex,
    OfficialSnapshot,
    discover_official_snapshot,
    inspect_official_shard_header,
    load_official_config,
    load_official_index,
    plan_official_expert,
)
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


def _git_blob_id(body: bytes) -> str:
    prefix = b"blob " + str(len(body)).encode("ascii") + b"\0"
    return hashlib.sha1(prefix + body).hexdigest()


def _released_config() -> dict[str, object]:
    return {
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
            "activation_situ_beta": 4.0,
            "activation_situ_linear_beta": 25.0,
            "routed_scaling_factor": 1.0,
        },
    }


def _index_record(shards: tuple[str, ...]) -> dict[str, object]:
    return {
        "metadata": {"total_size": 1_560_860_324_864},
        "weight_map": {
            "model.layers.1.feed_forward.experts.0.w1.weight_packed": shards[0],
            "model.layers.1.feed_forward.experts.0.w1.weight_scale": shards[0],
            "model.layers.1.feed_forward.experts.0.w2.weight_packed": shards[0],
            "model.layers.1.feed_forward.experts.0.w2.weight_scale": shards[0],
            "model.layers.1.feed_forward.experts.0.w3.weight_packed": shards[0],
            "model.layers.1.feed_forward.experts.0.w3.weight_scale": shards[0],
            **{f"unused.{index}": path for index, path in enumerate(shards[1:], 1)},
        },
    }


def _snapshot_with_bodies(
    index_body: bytes, config_body: bytes
) -> OfficialSnapshot:
    shard_names = tuple(
        f"model-{index:05d}-of-000096.safetensors" for index in range(1, 97)
    )
    files = {
        "config.json": OfficialFile(
            "config.json", len(config_body), _git_blob_id(config_body), None
        ),
        "model.safetensors.index.json": OfficialFile(
            "model.safetensors.index.json",
            len(index_body),
            "2" * 40,
            hashlib.sha256(index_body).hexdigest(),
        ),
        **{
            path: OfficialFile(path, 1000 + index, f"{index + 4:040x}", f"{index + 4:064x}")
            for index, path in enumerate(shard_names)
        },
    }
    return OfficialSnapshot(
        "moonshotai/Kimi-K3",
        "main",
        COMMIT,
        "2026-08-10T00:00:00Z",
        MappingProxyType(files),
        len(files),
        sum(item.size for item in files.values()),
        "4" * 64,
    )


class _BodyTransport:
    def __init__(self, bodies: dict[str, bytes]) -> None:
        self.bodies = bodies
        self.calls: list[str] = []

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
        for suffix, body in self.bodies.items():
            if suffix in url:
                assert len(body) <= max_bytes
                return HttpResponse(expected_status, url, {}, body)
        raise AssertionError(f"unexpected URL: {url}")


def test_index_binds_lfs_digest_and_all_96_declared_shards() -> None:
    shards = tuple(
        f"model-{index:05d}-of-000096.safetensors" for index in range(1, 97)
    )
    index_body = json.dumps(_index_record(shards), separators=(",", ":")).encode()
    config_body = json.dumps(_released_config(), separators=(",", ":")).encode()
    snapshot = _snapshot_with_bodies(index_body, config_body)
    transport = _BodyTransport({"model.safetensors.index.json": index_body})

    index = load_official_index(snapshot, transport)

    assert index.total_size == 1_560_860_324_864
    assert index.tensor_count == 101
    assert index.shard_paths == shards
    assert index.sha256 == hashlib.sha256(index_body).hexdigest()


@pytest.mark.parametrize(
    ("mutation", "code"),
    [
        (lambda value: value.update(extra={}), "INVALID_OFFICIAL_INDEX"),
        (lambda value: value["metadata"].update(total_size=True), "INVALID_OFFICIAL_INDEX"),
        (
            lambda value: value["weight_map"].update(bad="../escape.safetensors"),
            "INVALID_OFFICIAL_INDEX",
        ),
        (
            lambda value: value["weight_map"].pop("unused.95"),
            "OFFICIAL_SHARD_SET_MISMATCH",
        ),
    ],
)
def test_index_rejects_schema_or_shard_set_drift(mutation, code: str) -> None:
    shards = tuple(
        f"model-{index:05d}-of-000096.safetensors" for index in range(1, 97)
    )
    record = _index_record(shards)
    mutation(record)
    index_body = json.dumps(record, separators=(",", ":")).encode()
    config_body = json.dumps(_released_config(), separators=(",", ":")).encode()
    snapshot = _snapshot_with_bodies(index_body, config_body)

    with pytest.raises(K3XError, match=code):
        load_official_index(
            snapshot, _BodyTransport({"model.safetensors.index.json": index_body})
        )


def test_index_rejects_api_lfs_digest_mismatch() -> None:
    shards = tuple(
        f"model-{index:05d}-of-000096.safetensors" for index in range(1, 97)
    )
    index_body = json.dumps(_index_record(shards), separators=(",", ":")).encode()
    config_body = json.dumps(_released_config(), separators=(",", ":")).encode()
    snapshot = _snapshot_with_bodies(index_body, config_body)
    files = dict(snapshot.files)
    item = files["model.safetensors.index.json"]
    files[item.path] = OfficialFile(item.path, item.size, item.blob_id, "0" * 64)
    bad = OfficialSnapshot(
        snapshot.repository,
        snapshot.requested_revision,
        snapshot.resolved_revision,
        snapshot.observed_at,
        MappingProxyType(files),
        snapshot.file_count,
        snapshot.repository_bytes,
        snapshot.canonical_sha256,
    )

    with pytest.raises(K3XError, match="OFFICIAL_INDEX_SHA256_MISMATCH"):
        load_official_index(
            bad, _BodyTransport({"model.safetensors.index.json": index_body})
        )


def test_config_binds_git_blob_and_released_text_dimensions() -> None:
    shards = tuple(
        f"model-{index:05d}-of-000096.safetensors" for index in range(1, 97)
    )
    index_body = json.dumps(_index_record(shards), separators=(",", ":")).encode()
    config_body = json.dumps(_released_config(), separators=(",", ":")).encode()
    snapshot = _snapshot_with_bodies(index_body, config_body)

    config = load_official_config(
        snapshot, _BodyTransport({"config.json": config_body})
    )

    assert config.git_blob_id == _git_blob_id(config_body)
    assert config.sha256 == hashlib.sha256(config_body).hexdigest()
    assert config.hidden_size == 7_168
    assert config.num_experts == 896
    assert config.top_k == 16


def test_config_rejects_blob_or_dimension_drift_before_any_range() -> None:
    shards = tuple(
        f"model-{index:05d}-of-000096.safetensors" for index in range(1, 97)
    )
    index_body = json.dumps(_index_record(shards), separators=(",", ":")).encode()
    config = _released_config()
    config["text_config"]["hidden_size"] = 7_167
    config_body = json.dumps(config, separators=(",", ":")).encode()
    snapshot = _snapshot_with_bodies(index_body, config_body)
    transport = _BodyTransport({"config.json": config_body})

    with pytest.raises(K3XError, match="OFFICIAL_CONFIG_MISMATCH"):
        load_official_config(snapshot, transport)
    assert len(transport.calls) == 1

    files = dict(snapshot.files)
    item = files["config.json"]
    files[item.path] = OfficialFile(item.path, item.size, "0" * 40, None)
    bad = OfficialSnapshot(
        snapshot.repository,
        snapshot.requested_revision,
        snapshot.resolved_revision,
        snapshot.observed_at,
        MappingProxyType(files),
        snapshot.file_count,
        snapshot.repository_bytes,
        snapshot.canonical_sha256,
    )
    with pytest.raises(K3XError, match="OFFICIAL_CONFIG_BLOB_MISMATCH"):
        load_official_config(bad, _BodyTransport({"config.json": config_body}))


_SHARD = "model-00002-of-000096.safetensors"
_SHARD_SIZE = 16_990_911_504
_HEADER_LENGTH = 818_696
_DATA_START = 818_704
_PAYLOAD_START = 1_268_562_960
_PAYLOAD_END = 1_286_110_224
_OFFICIAL_BASE = "language_model.model.layers.1.block_sparse_moe.experts.0"


def _official_header() -> bytes:
    selected = [
        ("w1.weight_packed", [3072, 1792], [1_267_744_256, 1_273_249_280]),
        ("w1.weight_scale", [3072, 112], [1_273_249_280, 1_273_593_344]),
        ("w2.weight_packed", [3584, 1536], [1_273_593_344, 1_279_098_368]),
        ("w2.weight_scale", [3584, 96], [1_279_098_368, 1_279_442_432]),
        ("w3.weight_packed", [3072, 1792], [1_279_442_432, 1_284_947_456]),
        ("w3.weight_scale", [3072, 112], [1_284_947_456, 1_285_291_520]),
    ]
    data_bytes = _SHARD_SIZE - _DATA_START
    header: dict[str, object] = {
        "before": {
            "dtype": "I16",
            "shape": [1],
            "data_offsets": [0, selected[0][2][0]],
        },
        **{
            f"{_OFFICIAL_BASE}.{suffix}": {
                "dtype": "U8",
                "shape": shape,
                "data_offsets": offsets,
            }
            for suffix, shape, offsets in selected
        },
        "after": {
            "dtype": "I16",
            "shape": [1],
            "data_offsets": [selected[-1][2][1], data_bytes],
        },
    }
    encoded = json.dumps(header, separators=(",", ":")).encode()
    assert len(encoded) < _HEADER_LENGTH
    return encoded + b" " * (_HEADER_LENGTH - len(encoded))


def _range_snapshot() -> OfficialSnapshot:
    index_body = b"{}"
    config_body = json.dumps(_released_config(), separators=(",", ":")).encode()
    snapshot = _snapshot_with_bodies(index_body, config_body)
    files = dict(snapshot.files)
    files[_SHARD] = OfficialFile(_SHARD, _SHARD_SIZE, "3" * 40, SHARD_SHA256)
    return OfficialSnapshot(
        snapshot.repository,
        snapshot.requested_revision,
        snapshot.resolved_revision,
        snapshot.observed_at,
        MappingProxyType(files),
        snapshot.file_count,
        snapshot.repository_bytes,
        snapshot.canonical_sha256,
    )


class _RangeTransport:
    def __init__(
        self,
        *,
        status: int = 206,
        wrong_content_range: bool = False,
        short_header: bool = False,
    ) -> None:
        self.status = status
        self.wrong_content_range = wrong_content_range
        self.short_header = short_header
        self.calls: list[str] = []

    def get(
        self,
        url: str,
        *,
        headers: dict[str, str],
        max_bytes: int,
        timeout_seconds: float,
        expected_status: int = 200,
    ) -> HttpResponse:
        value = headers["Range"]
        self.calls.append(value)
        if value == "bytes=0-7":
            body = struct.pack("<Q", _HEADER_LENGTH)
            content_range = f"bytes 0-7/{_SHARD_SIZE}"
        elif value == f"bytes=8-{_HEADER_LENGTH + 7}":
            body = _official_header()
            if self.short_header:
                body = body[:-1]
            content_range = f"bytes 8-{_HEADER_LENGTH + 7}/{_SHARD_SIZE}"
        else:
            raise AssertionError(value)
        if self.wrong_content_range:
            content_range = content_range.replace(f"/{_SHARD_SIZE}", "/1")
        return HttpResponse(
            self.status,
            url,
            {"content-range": content_range, "content-length": str(len(body))},
            body,
        )


def _expert_index() -> OfficialIndex:
    names = {
        f"{_OFFICIAL_BASE}.{matrix}.{kind}": _SHARD
        for matrix in ("w1", "w2", "w3")
        for kind in ("weight_packed", "weight_scale")
    }
    return OfficialIndex(
        1_560_860_324_864,
        MappingProxyType(names),
        (_SHARD,),
        len(names),
        INDEX_SHA256,
    )


def test_exact_header_ranges_produce_official_absolute_tensor_offsets() -> None:
    transport = _RangeTransport()

    header = inspect_official_shard_header(_range_snapshot(), _SHARD, transport)

    assert transport.calls == ["bytes=0-7", f"bytes=8-{_HEADER_LENGTH + 7}"]
    assert header.header_length == _HEADER_LENGTH
    assert header.data_start == _DATA_START
    assert header.tensors[f"{_OFFICIAL_BASE}.w1.weight_packed"].offset == _PAYLOAD_START
    assert header.tensors[f"{_OFFICIAL_BASE}.w3.weight_scale"].offset + 344_064 == _PAYLOAD_END


@pytest.mark.parametrize(
    ("kwargs", "code"),
    [
        ({"status": 200}, "OFFICIAL_HTTP_STATUS"),
        ({"wrong_content_range": True}, "OFFICIAL_CONTENT_RANGE_MISMATCH"),
        ({"short_header": True}, "OFFICIAL_RANGE_LENGTH_MISMATCH"),
    ],
)
def test_exact_header_ranges_reject_status_metadata_or_length_drift(
    kwargs: dict[str, object], code: str
) -> None:
    with pytest.raises(K3XError, match=code):
        inspect_official_shard_header(
            _range_snapshot(), _SHARD, _RangeTransport(**kwargs)
        )


def test_expert_plan_maps_official_w1_w2_w3_and_exact_contiguous_union() -> None:
    header = inspect_official_shard_header(
        _range_snapshot(), _SHARD, _RangeTransport()
    )

    plan = plan_official_expert(_expert_index(), header, layer_id=1, expert_id=0)

    assert plan.shard_path == _SHARD
    assert plan.payload_start == _PAYLOAD_START
    assert plan.payload_end == _PAYLOAD_END
    assert plan.payload_bytes == 17_547_264
    assert [tensor.role for tensor in plan.tensors] == [
        "gate",
        "gate",
        "down",
        "down",
        "up",
        "up",
    ]
    assert plan.tensors[0].canonical_name == (
        "model.layers.1.feed_forward.experts.0.gate.weight_packed"
    )
    assert plan.tensors[2].canonical_name == (
        "model.layers.1.feed_forward.experts.0.down.weight_packed"
    )
    assert plan.tensors[4].canonical_name == (
        "model.layers.1.feed_forward.experts.0.up.weight_packed"
    )


@pytest.mark.parametrize("failure", ("missing", "mixed_shard", "wrong_shape"))
def test_expert_plan_rejects_incomplete_ownership_or_shape_drift(failure: str) -> None:
    header = inspect_official_shard_header(
        _range_snapshot(), _SHARD, _RangeTransport()
    )
    index = _expert_index()
    weight_map = dict(index.weight_map)
    tensors = dict(header.tensors)
    if failure == "missing":
        weight_map.pop(f"{_OFFICIAL_BASE}.w1.weight_scale")
    elif failure == "mixed_shard":
        weight_map[f"{_OFFICIAL_BASE}.w2.weight_scale"] = "other.safetensors"
    else:
        item = tensors[f"{_OFFICIAL_BASE}.w3.weight_scale"]
        tensors[item.name] = type(item)(
            item.name, item.dtype, (3072, 111), item.offset, item.length
        )
    bad_index = OfficialIndex(
        index.total_size,
        MappingProxyType(weight_map),
        index.shard_paths,
        len(weight_map),
        index.sha256,
    )
    bad_header = type(header)(
        header.shard_path,
        header.file_size,
        header.header_length,
        header.data_start,
        MappingProxyType(tensors),
    )

    with pytest.raises(K3XError, match="INVALID_OFFICIAL_EXPERT"):
        plan_official_expert(bad_index, bad_header, layer_id=1, expert_id=0)
