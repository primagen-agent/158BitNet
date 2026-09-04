"""Stable identity helpers for binding memory artifacts to one GGUF."""
from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=8)
def _sha256_file_cached(path: str, size: int, mtime_ns: int) -> bytes:
    del size, mtime_ns
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.digest()


def sha256_file(path: str | Path) -> bytes:
    resolved = Path(path).expanduser().resolve()
    stat = resolved.stat()
    return _sha256_file_cached(
        str(resolved), stat.st_size, stat.st_mtime_ns)


def require_matching_sha256(
    artifact_name: str,
    expected: bytes | None,
    gguf_path: str | Path,
) -> bytes:
    if expected is None:
        raise ValueError(
            f"{artifact_name} is not bound to a backbone model; "
            "bind or retrain it before use")
    actual = sha256_file(gguf_path)
    if actual != expected:
        raise ValueError(
            f"{artifact_name} backbone SHA-256 mismatch: "
            f"trained={expected.hex()} loaded={actual.hex()}")
    return actual
