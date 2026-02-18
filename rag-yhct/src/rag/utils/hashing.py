"""Hashing utilities for fingerprints and stable IDs."""

from __future__ import annotations

import hashlib
from pathlib import Path


def sha1_fingerprint(file_path: str | Path) -> str:
    """Compute SHA-1 hex digest of a file. Returns 'sha1:<hex>'."""
    h = hashlib.sha1()
    with open(file_path, "rb") as f:
        while True:
            chunk = f.read(1 << 16)  # 64 KB
            if not chunk:
                break
            h.update(chunk)
    return f"sha1:{h.hexdigest()}"


def sha1_short(text: str, length: int = 12) -> str:
    """Return first *length* hex chars of SHA-1 of *text*."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:length]


def stable_point_id(value: str) -> int:
    """Produce a stable positive int from a string via SHA-1 (first 16 hex chars → int)."""
    hex16 = hashlib.sha1(value.encode("utf-8")).hexdigest()[:16]
    return int(hex16, 16)
