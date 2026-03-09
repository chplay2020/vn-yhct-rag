#!/usr/bin/env python3
"""
Post-process raw TXT files (from convert_to_txt.py) → clean UTF-8 text.

Fixes applied:
  1. Form-feed \f (page separator) → blank line
  2. Collapse 3+ consecutive spaces on one line → single space
     (removes two-column PDF layout artifacts)
  3. Broken-word line-join: if a line ends with a bare Vietnamese
     syllable fragment (no space, short ending) and the next starts
     with a continuation, join them (conservative heuristic)
  4. normalize_vi_en() from rag.utils.text:
       • Unicode NFC
       • Remove control chars
       • Whitespace normalization
       • Word-glue fix patterns 5a–5f (Vietnamese/English)
       • Limit consecutive blank lines to 2

Usage:
    cd rag-yhct/
    python scripts/normalize_txt.py [--in-dir raw-txt] [--out-dir raw-txt-clean] [--force]
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Pre-processing helpers (run BEFORE normalize_vi_en)
# ---------------------------------------------------------------------------

# Collapse 3 or more consecutive spaces/tabs within a line to a single space.
_RE_MULTI_SPACE = re.compile(r"[ \t]{3,}")

# Vietnamese vowels (used for broken-word detection)
_VI_VOWELS = (
    "aáàảãạăắằẳẵặâấầẩẫậeéèẻẽẹêếềểễệiíìỉĩị"
    "oóòỏõọôốồổỗộơớờởỡợuúùủũụưứừửữựyýỳỷỹỵ"
    "AÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬEÉÈẺẼẸÊẾỀỂỄỆIÍÌỈĨỊ"
    "OÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢUÚÙỦŨỤƯỨỪỬỮỰYÝỲỶỸỴ"
)
# A "short fragment" at end of line: 1-2 chars, no space, all letters/vi-vowels.
# This matches syllable fragments like "bị", "ch", "ng" that got split by line break.
_RE_SHORT_LINE_FRAG = re.compile(
    r"^([\w" + re.escape(_VI_VOWELS) + r"]{1,3})$",
    re.UNICODE,
)


def _pre_normalize(text: str) -> str:
    """Pre-process raw TXT before passing to normalize_vi_en."""

    # 1. Replace form-feeds with blank lines
    text = text.replace("\f", "\n\n")

    # 2. Process line by line
    lines = text.split("\n")
    result: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]

        # 2a. Collapse multi-space (column layout artifact)
        line = _RE_MULTI_SPACE.sub(" ", line)
        line = line.strip()

        # 2b. Broken-word join:
        #     If this line is a very short fragment AND the next line
        #     starts immediately with a letter (no leading space/indent),
        #     join them without a separator so normalize_vi_en can fix glue.
        if (
            line
            and _RE_SHORT_LINE_FRAG.match(line)
            and i + 1 < len(lines)
            and lines[i + 1].strip()
            and not lines[i + 1].startswith(" ")
        ):
            # Peek next line
            next_stripped = lines[i + 1].strip()
            # Only join if next starts with lowercase/vi letter (continuation)
            if next_stripped and next_stripped[0].islower():
                result.append(line + next_stripped)
                i += 2
                continue

        result.append(line)
        i += 1

    return "\n".join(result)


# ---------------------------------------------------------------------------
# Normalize one file
# ---------------------------------------------------------------------------

def normalize_file(src: Path, dst: Path) -> bool:
    """Read *src* TXT, normalize, write UTF-8 to *dst*. Returns True on success."""
    try:
        text = src.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        # Try with replacement so we don't crash on rare bad bytes
        logger.warning("UTF-8 decode error in %s — using replace mode", src)
        text = src.read_text(encoding="utf-8", errors="replace")

    from rag.utils.text import normalize_vi_en  # local import for portability

    text = _pre_normalize(text)
    text = normalize_vi_en(text)

    if not text.strip():
        logger.warning("Empty content after normalize: %s — skipping", src)
        return False

    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text, encoding="utf-8")
    return True


# ---------------------------------------------------------------------------
# Walk directory
# ---------------------------------------------------------------------------

def normalize_directory(in_dir: Path, out_dir: Path, force: bool = False) -> None:
    """Walk *in_dir* for .txt files and write normalized versions to *out_dir*."""
    txt_files = sorted(p for p in in_dir.rglob("*.txt") if p.is_file())

    if not txt_files:
        logger.warning("No .txt files found in %s", in_dir)
        return

    logger.info("Found %d .txt file(s) in %s", len(txt_files), in_dir)
    ok = fail = skip = 0

    for src in txt_files:
        rel = src.relative_to(in_dir)
        dst = out_dir / rel

        if dst.exists() and not force:
            logger.debug("Skipping (exists): %s", dst)
            skip += 1
            continue

        logger.info("Normalizing: %s", rel)
        if normalize_file(src, dst):
            ok += 1
        else:
            fail += 1

    logger.info(
        "Done — normalized: %d | failed: %d | skipped (exists): %d",
        ok, fail, skip,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    here = Path(__file__).resolve().parent.parent  # project root (rag-yhct/)

    parser = argparse.ArgumentParser(
        description="Normalize raw TXT files (fix word-glue, encoding, layout artifacts)"
    )
    parser.add_argument(
        "--in-dir",
        type=Path,
        default=here / "data" / "raw-txt",
        help="Thư mục TXT đầu vào (mặc định: data/raw-txt/)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=here / "data" / "raw-txt-clean",
        help="Thư mục TXT đầu ra sau khi chuẩn hóa (mặc định: data/raw-txt-clean/)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Ghi đè file đã tồn tại trong out-dir",
    )
    args = parser.parse_args()

    in_dir: Path = args.in_dir.resolve()
    out_dir: Path = args.out_dir.resolve()

    if not in_dir.exists():
        logger.error("in-dir không tồn tại: %s", in_dir)
        sys.exit(1)

    logger.info("Đầu vào : %s", in_dir)
    logger.info("Đầu ra  : %s", out_dir)

    normalize_directory(in_dir, out_dir, force=args.force)


if __name__ == "__main__":
    main()
