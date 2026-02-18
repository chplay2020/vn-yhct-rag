"""I/O utilities: JSONL read/write, directory helpers."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


def ensure_parent_dir(path: str | Path) -> Path:
    """Create parent directories if they don't exist."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    """Read a JSONL file and return list of dicts."""
    records: list[dict[str, Any]] = []
    p = Path(path)
    if not p.exists():
        logger.warning("JSONL file not found: %s", p)
        return records
    with open(p, encoding="utf-8") as f:
        for lineno, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                logger.error("Bad JSON at %s:%d — %s", p, lineno, exc)
    logger.info("Read %d records from %s", len(records), p)
    return records


def write_jsonl(records: list[dict[str, Any]], path: str | Path) -> None:
    """Write list of dicts as JSONL."""
    p = ensure_parent_dir(path)
    with open(p, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    logger.info("Wrote %d records to %s", len(records), p)
