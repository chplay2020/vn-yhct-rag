"""Scan data/raw/** and generate a sources YAML manifest.

Usage:
    python -m tools.generate_sources_yaml \
        --raw-dir data/raw \
        --out data/sources_full.yaml

Produces a YAML file compatible with data/sources.yaml schema.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
from pathlib import Path
from typing import Any

import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

ALLOWED_EXT = {".pdf", ".docx", ".jpg", ".jpeg", ".png", ".txt"}
EXCLUDE_DIRS = {"System", "__MACOSX", ".git", ".ipynb_checkpoints", "__pycache__"}

IMAGE_EXT = {".jpg", ".jpeg", ".png"}


def _stable_source_id(rel_path: str) -> str:
    """Deterministic source_id from the relative file path (first 12 hex of SHA-1)."""
    h = hashlib.sha1(rel_path.encode("utf-8")).hexdigest()[:12]
    return f"src_{h}"


def _doc_type(ext: str) -> str:
    if ext == ".pdf":
        return "pdf"
    if ext == ".docx":
        return "docx"
    if ext in IMAGE_EXT:
        return "image"
    if ext == ".txt":
        return "txt"
    return "unknown"


def _should_exclude(path: Path) -> bool:
    """Return True if any parent component is in EXCLUDE_DIRS."""
    for part in path.parts:
        if part in EXCLUDE_DIRS:
            return True
    return False


def scan_raw_dir(raw_dir: str, base_dir: str = ".") -> list[dict[str, Any]]:
    """Recursively scan raw_dir for allowed files, return list of source records."""
    root = Path(base_dir).resolve()
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        logger.error("Raw directory not found: %s", raw_dir)
        return []

    sources: list[dict[str, Any]] = []
    all_files = sorted(raw_path.rglob("*"))

    for fp in all_files:
        if not fp.is_file():
            continue
        ext = fp.suffix.lower()
        if ext not in ALLOWED_EXT:
            continue
        if _should_exclude(fp):
            continue
        # Zone.Identifier files (Windows metadata) — skip
        if ":Zone.Identifier" in fp.name or fp.name.endswith(":Zone.Identifier"):
            continue

        # Use relative path from project root for file_path (matches sources.yaml convention)
        rel = str(fp.relative_to(root)) if fp.is_relative_to(root) else str(fp)
        source_id = _stable_source_id(rel)
        title = fp.stem  # filename without extension

        sources.append({
            "source_id": source_id,
            "file_path": rel,
            "doc_type": _doc_type(ext),
            "title": title,
            "doc_language": "vi",
            "url": "",
        })

    return sources


def generate_yaml(sources: list[dict[str, Any]], out_path: str) -> None:
    """Write sources list as YAML manifest."""
    manifest = {"sources": sources}
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        yaml.dump(manifest, f, allow_unicode=True, default_flow_style=False, sort_keys=False)
    logger.info("Wrote %d sources to %s", len(sources), out_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate sources YAML from raw data directory")
    parser.add_argument("--raw-dir", default="data/raw", help="Root directory to scan")
    parser.add_argument("--out", default="data/sources_full.yaml", help="Output YAML path")
    parser.add_argument("--base-dir", default=".", help="Project base directory for relative paths")
    args = parser.parse_args()

    sources = scan_raw_dir(args.raw_dir, args.base_dir)
    if not sources:
        logger.warning("No files found in %s", args.raw_dir)
        return

    # Summary
    from collections import Counter
    ext_counts = Counter(s["doc_type"] for s in sources)
    logger.info("Found %d files: %s", len(sources), dict(ext_counts))

    generate_yaml(sources, args.out)


if __name__ == "__main__":
    main()
