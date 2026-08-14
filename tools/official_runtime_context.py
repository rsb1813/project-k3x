# 공식 K3X 상주 실행이 공유하는 검증된 메타데이터와 shard 저장소를 관리합니다.
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from k3x_converter.format import K3XError
from k3x_converter.fragment_set import (
    FragmentSetManifest,
    read_fragment_set_manifest,
)
from k3x_converter.fragment_tensor_store import K3XTensorStore
from k3x_converter.official_source import (
    OfficialConfig,
    OfficialIndex,
    OfficialShardHeader,
    OfficialSnapshot,
    discover_official_snapshot,
    inspect_official_shard_header,
    load_official_config,
    load_official_index,
)
from k3x_converter.official_transport import UrllibTransport


def load_official_topology(path: Path) -> dict[str, object]:
    record = json.loads(path.read_text(encoding="utf-8"))
    digest = record.pop("record_sha256", None)
    encoded = json.dumps(record, sort_keys=True, separators=(",", ":")).encode()
    if digest != hashlib.sha256(encoded).hexdigest():
        raise K3XError("OFFICIAL_TOPOLOGY_DIGEST")
    record["record_sha256"] = digest
    if record.get("format") != "k3x-official-topology-v2":
        raise K3XError("OFFICIAL_TOPOLOGY_FORMAT")
    return record


@dataclass
class OfficialRuntimeContext:
    topology: dict[str, object]
    object_dir: Path
    transport: UrllibTransport
    snapshot: OfficialSnapshot
    index: OfficialIndex
    config: OfficialConfig
    set_manifest: FragmentSetManifest | None
    _headers: dict[str, OfficialShardHeader] = field(default_factory=dict)
    _stores: dict[str, K3XTensorStore] = field(default_factory=dict)

    @classmethod
    def create(
        cls,
        *,
        topology_path: Path,
        object_dir: Path,
        k3x_set: Path | None,
    ) -> "OfficialRuntimeContext":
        topology = load_official_topology(topology_path.resolve())
        object_dir = object_dir.resolve()
        transport = UrllibTransport(
            cache_directory=object_dir / "official-metadata-cache"
        )
        snapshot = discover_official_snapshot(transport)
        index = load_official_index(snapshot, transport)
        config = load_official_config(snapshot, transport)
        if (
            topology["resolved_revision"] != snapshot.resolved_revision
            or topology["snapshot_sha256"] != snapshot.canonical_sha256
            or topology["index_sha256"] != index.sha256
            or topology["config_sha256"] != config.sha256
        ):
            raise K3XError("OFFICIAL_TOPOLOGY_SOURCE_DRIFT")
        manifest = (
            read_fragment_set_manifest(k3x_set)
            if k3x_set is not None
            else None
        )
        return cls(
            topology,
            object_dir,
            transport,
            snapshot,
            index,
            config,
            manifest,
        )

    @property
    def set_identity(self) -> str | None:
        return self.set_manifest.record_sha256 if self.set_manifest else None

    def header(self, source_shard: str) -> OfficialShardHeader:
        header = self._headers.get(source_shard)
        if header is None:
            header = inspect_official_shard_header(
                self.snapshot, source_shard, self.transport
            )
            self._headers[source_shard] = header
        return header

    def store(self, source_shard: str) -> K3XTensorStore:
        store = self._stores.get(source_shard)
        if store is not None:
            return store
        if self.set_manifest is None:
            raise K3XError("K3X_FRAGMENT_SET_REQUIRED")
        if not source_shard.endswith(".safetensors"):
            raise K3XError("INVALID_OFFICIAL_SHARD", source_shard)
        filename = source_shard.removesuffix(".safetensors") + ".k3x"
        fragment = next(
            (path for path in self.set_manifest.fragments if path.name == filename),
            None,
        )
        if fragment is None:
            raise K3XError("K3X_FRAGMENT_NOT_FOUND", filename)
        store = K3XTensorStore.open(
            [fragment], verify_root=False, verify_payload=False
        )
        self._stores[source_shard] = store
        return store
