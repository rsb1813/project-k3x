# 여러 K3X fragment를 복사 없이 봉인하는 set manifest를 작성합니다.
from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Iterable

from .format import K3XError, SUPERBLOCK_BYTES, Superblock


_SHA256 = re.compile(r"[0-9a-f]{64}")


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
