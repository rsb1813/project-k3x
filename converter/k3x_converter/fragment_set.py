# 여러 K3X fragment를 복사 없이 봉인하는 set manifest를 작성합니다.
from __future__ import annotations

import hashlib
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from .format import K3XError, SUPERBLOCK_BYTES, Superblock


_SHA256 = re.compile(r"[0-9a-f]{64}")


@dataclass(frozen=True)
class FragmentSetManifest:
    path: Path
    plan_sha256: str
    fragments: tuple[Path, ...]
    record_sha256: str


def read_fragment_set_manifest(path: Path) -> FragmentSetManifest:
    path = Path(path).resolve(strict=True)
    try:
        lines = path.read_text(encoding="ascii").splitlines(keepends=True)
    except (UnicodeDecodeError, OSError) as exc:
        raise K3XError("INVALID_FRAGMENT_SET_MANIFEST") from exc
    if len(lines) < 3 or not all(line.endswith("\n") for line in lines):
        raise K3XError("INVALID_FRAGMENT_SET_MANIFEST")
    header = lines[0].removesuffix("\n").split("\t")
    if len(header) != 3 or header[0] != "K3XSET1" or _SHA256.fullmatch(header[1]) is None:
        raise K3XError("INVALID_FRAGMENT_SET_MANIFEST")
    try:
        count = int(header[2])
    except ValueError as exc:
        raise K3XError("INVALID_FRAGMENT_SET_MANIFEST") from exc
    if count <= 0 or count > 256 or len(lines) != count + 2:
        raise K3XError("INVALID_FRAGMENT_SET_COUNT")
    final = lines[-1].removesuffix("\n").split("\t")
    canonical = "".join(lines[:-1]).encode("ascii")
    digest = hashlib.sha256(canonical).hexdigest()
    if len(final) != 2 or final[0] != "SHA256" or final[1] != digest:
        raise K3XError("FRAGMENT_SET_SHA256_MISMATCH")
    fragments = []
    for line in lines[1:-1]:
        fields = line.removesuffix("\n").split("\t")
        if (
            len(fields) != 4
            or fields[0] != "FRAGMENT"
            or not fields[1]
            or fields[1] in {".", ".."}
            or any(separator in fields[1] for separator in ("/", "\\"))
            or _SHA256.fullmatch(fields[3]) is None
        ):
            raise K3XError("INVALID_FRAGMENT_SET_MANIFEST")
        try:
            expected_size = int(fields[2])
        except ValueError as exc:
            raise K3XError("INVALID_FRAGMENT_SET_MANIFEST") from exc
        fragment = path.parent / fields[1]
        if expected_size <= 0 or not fragment.is_file() or fragment.stat().st_size != expected_size:
            raise K3XError("INVALID_FRAGMENT_SET_ARTIFACT", fields[1])
        with fragment.open("rb") as stream:
            superblock = Superblock.decode(stream.read(SUPERBLOCK_BYTES))
        if (
            superblock.state != 1
            or superblock.file_length != expected_size
            or superblock.root_sha256.hex() != fields[3]
        ):
            raise K3XError("INVALID_FRAGMENT_SET_ARTIFACT", fields[1])
        fragments.append(fragment)
    if len(set(fragments)) != len(fragments):
        raise K3XError("DUPLICATE_FRAGMENT_SET_PATH")
    return FragmentSetManifest(path, header[1], tuple(fragments), digest)


def write_fragment_set_manifest(
    path: Path,
    fragments: Iterable[Path],
    *,
    plan_sha256: str,
) -> str:
    path = Path(path).resolve()
    if _SHA256.fullmatch(plan_sha256) is None:
        raise K3XError("INVALID_FRAGMENT_SET_PLAN_SHA256")
    resolved = [Path(fragment).resolve(strict=True) for fragment in fragments]
    if not resolved or len(resolved) > 256:
        raise K3XError("INVALID_FRAGMENT_SET_COUNT")
    if len(set(resolved)) != len(resolved):
        raise K3XError("DUPLICATE_FRAGMENT_SET_PATH")
    lines = [f"K3XSET1\t{plan_sha256}\t{len(resolved)}\n"]
    for fragment in resolved:
        if fragment.parent != path.parent or any(
            separator in fragment.name for separator in ("/", "\\", "\t", "\n")
        ):
            raise K3XError("INVALID_FRAGMENT_SET_PATH", str(fragment))
        with fragment.open("rb") as stream:
            superblock = Superblock.decode(stream.read(SUPERBLOCK_BYTES))
        if superblock.state != 1 or superblock.file_length != fragment.stat().st_size:
            raise K3XError("INVALID_FRAGMENT_SET_ARTIFACT", fragment.name)
        lines.append(
            f"FRAGMENT\t{fragment.name}\t{fragment.stat().st_size}\t"
            f"{superblock.root_sha256.hex()}\n"
        )
    canonical = "".join(lines).encode("ascii")
    record_sha256 = hashlib.sha256(canonical).hexdigest()
    payload = canonical + f"SHA256\t{record_sha256}\n".encode("ascii")
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    with partial.open("wb") as stream:
        stream.write(payload)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(partial, path)
    return record_sha256
