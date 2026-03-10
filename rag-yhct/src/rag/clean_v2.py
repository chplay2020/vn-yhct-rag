"""Clean v2 — Post-chunk normalization for embeddings/retrieval.

Reads chunks_v1.jsonl, preserves original "text" for citations,
adds "text_norm" (normalized for embeddings), "clean_version", and "is_noise".

Usage:
    python -m rag.clean_v2 --in data/chunks/chunks_v1.jsonl --out data/chunks/chunks_v2.jsonl
    python -m rag.clean_v2 --in data/chunks/chunks_v1.jsonl --out data/chunks/chunks_v2.jsonl --debug
"""

from __future__ import annotations

import argparse
import logging
import re
import unicodedata
from collections import defaultdict
from typing import Any

from rag.utils.io import read_jsonl, write_jsonl, ensure_parent_dir

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")


# ---------------------------------------------------------------------------
# Encoding-noise / mojibake detection
# ---------------------------------------------------------------------------

_RE_CYRILLIC = re.compile(r"[\u0400-\u04FF]")
_MOJIBAKE_CHARS = frozenset("¿¶µ¹²³¼½¾")

# Thresholds — tune if needed
BAD_RATIO_THRESHOLD: float = 0.03    # 3 %
UFFFD_RATIO_THRESHOLD: float = 0.005  # 0.5 %


def _detect_encoding_noise(
    text: str,
) -> tuple[float, float, bool]:
    """Return (bad_ratio, ufffd_ratio, has_cyrillic) for *text*."""
    ufffd_count = text.count("\ufffd")
    text_no_space = re.sub(r"\s", "", text)
    length = max(1, len(text_no_space))

    bad_count = len(_RE_CYRILLIC.findall(text))
    bad_count += sum(1 for ch in text if ch in _MOJIBAKE_CHARS)

    bad_ratio = bad_count / length
    ufffd_ratio = ufffd_count / length
    has_cyrillic = bool(_RE_CYRILLIC.search(text))
    return bad_ratio, ufffd_ratio, has_cyrillic


# ---------------------------------------------------------------------------
# Noise detection for IMAGE OCR chunks
# ---------------------------------------------------------------------------

_RE_ONLY_PUNCT_SYMBOLS = re.compile(r"^[\W_]+$", re.UNICODE)
_RE_COMMON_NOISE = re.compile(r"^[\-–—_oO0\s]+$")


def _is_noise(text: str) -> bool:
    """Return True if the text is too short or matches noise patterns."""
    stripped = text.strip()
    if len(stripped) < 4:
        return True
    if _RE_ONLY_PUNCT_SYMBOLS.match(stripped):
        return True
    if _RE_COMMON_NOISE.match(stripped):
        return True
    return False


# ---------------------------------------------------------------------------
# Latin word-break repair (PDF line-break / OCR space artifacts)
# ---------------------------------------------------------------------------

# Pattern A: ASCII word split across line break
#   "Portulaca olera\ncea" -> "Portulaca oleracea"
#   Only merge when both parts are pure ASCII letters AND right group is a
#   complete word (not followed by more letters, including Vietnamese diacritics).
_RE_LATIN_LINE_BREAK = re.compile(
    r"([A-Za-z]{2,})\s*\n\s*([a-z]{2,})(?=[^a-zA-Z\u00C0-\u024F\u1E00-\u1EFF]|$)"
)

# Pattern B: Hyphenation line break
#   "cardiomy-\nopathy" -> "cardiomyopathy"
_RE_HYPHEN_LINE_BREAK = re.compile(
    r"([A-Za-z])-\n\s*([A-Za-z])"
)

# Pattern C: Capitalized short fragment + lowercase fragment in same line
#   "Eup horbia" -> "Euphorbia",  "Oc tober" -> "October"
#   Only when left part is 2-3 chars (too short to be a real word) to avoid
#   merging Vietnamese words ("Nanh heo") or genus+species ("Piper lolot").
_RE_CAP_FRAGMENT = re.compile(
    r"\b([A-Z][a-z]{1,2})\s+([a-z]{2,})\b"
)

# Pattern D: Two lowercase ASCII fragments inside "Tên khoa học:" context
#   "olera cea" -> "oleracea" (only within scientific-name context)
_RE_LOWER_FRAGMENT = re.compile(
    r"\b([a-z]{2,})\s+([a-z]{2,})\b"
)

# Pattern E: Pipe/hash inside word (extraction artifact)
#   "HƠ|P" -> "HƠP",  "THUỐ#C" -> "THUỐC"
_RE_PIPE_HASH_INSIDE = re.compile(
    r"(?<=[\wÀ-ỹ])[|#](?=[\wÀ-ỹ])"
)

# ---------------------------------------------------------------------------
# PDF-specific: unwrap soft line breaks between Latin letters
# ---------------------------------------------------------------------------

# PDF extractors produce hard newlines at visual wrap points.  These cause
# the h1-report metrics ``letter\nletter`` and ``sci-split`` to spike.
# Strategy: replace ``[A-Za-z]\n[A-Za-z]`` with a space.
# Uses a lookahead so chained breaks ``A\nB\nC`` become ``A B C``.
_RE_PDF_LETTER_NL_LETTER = re.compile(r"([A-Za-z])\n(?=[A-Za-z])")

# Zero-width characters
_RE_ZERO_WIDTH = re.compile(r"[\u200b\u200c\u200d\ufeff]")


def _is_pure_ascii_word(s: str) -> bool:
    """Check if string contains only ASCII letters."""
    return bool(s) and s.isascii() and s.isalpha()


def _merge_latin_line_breaks(text: str) -> str:
    """Merge Latin words split across line breaks."""
    def _replace(m: re.Match[str]) -> str:
        left, right = m.group(1), m.group(2)
        if _is_pure_ascii_word(left) and _is_pure_ascii_word(right):
            return left + right
        return m.group(0)
    return _RE_LATIN_LINE_BREAK.sub(_replace, text)


def _merge_hyphen_breaks(text: str) -> str:
    """Merge hyphenated line breaks: 'word-\\nrest' -> 'wordrest'."""
    def _replace(m: re.Match[str]) -> str:
        return m.group(1) + m.group(2)
    return _RE_HYPHEN_LINE_BREAK.sub(_replace, text)


def _merge_cap_fragments(text: str) -> str:
    """Merge 'Eup horbia' -> 'Euphorbia' style fragments.

    Only when left part is 2-3 ASCII chars (too short for a real word).
    """
    def _replace(m: re.Match[str]) -> str:
        left, right = m.group(1), m.group(2)
        if not (_is_pure_ascii_word(left) and _is_pure_ascii_word(right)):
            return m.group(0)
        return left + right
    return _RE_CAP_FRAGMENT.sub(_replace, text)


def _merge_scientific_name_fragments(text: str) -> str:
    """Merge lowercase fragments ONLY inside 'Tên khoa học:' context."""
    # Find "Tên khoa học:" and process only the substring after it until newline/period
    pattern = re.compile(r"(Tên khoa học\s*:\s*)([^\n.;]+)", re.IGNORECASE)

    def _fix_context(m: re.Match[str]) -> str:
        prefix = m.group(1)
        content = m.group(2)

        def _merge_lower(m2: re.Match[str]) -> str:
            left, right = m2.group(1), m2.group(2)
            if _is_pure_ascii_word(left) and _is_pure_ascii_word(right):
                return left + right
            return m2.group(0)

        fixed = _RE_LOWER_FRAGMENT.sub(_merge_lower, content)
        return prefix + fixed

    return pattern.sub(_fix_context, text)


# ---------------------------------------------------------------------------
# Main normalization function
# ---------------------------------------------------------------------------

def normalize_text_v2(text: str, doc_type: str = "") -> tuple[str, bool]:
    """Normalize text for embeddings/retrieval.

    Returns (text_norm, is_noise).
    """
    # IMAGE noise detection
    if doc_type == "image":
        if _is_noise(text):
            return "", True

    # --- Encoding-noise detection (on original text) ---
    bad_ratio, ufffd_ratio, _has_cyr = _detect_encoding_noise(text)
    encoding_noise = (
        bad_ratio > BAD_RATIO_THRESHOLD
        or ufffd_ratio > UFFFD_RATIO_THRESHOLD
    )

    # --- Sanitize: replace U+FFFD with space ---
    text = text.replace("\ufffd", " ")

    # A) Unicode NFKC
    norm = unicodedata.normalize("NFKC", text)

    # Remove zero-width chars
    norm = _RE_ZERO_WIDTH.sub("", norm)

    # Normalize newlines
    norm = norm.replace("\r\n", "\n").replace("\r", "\n")

    # B) Latin word-break repair
    norm = _merge_latin_line_breaks(norm)
    norm = _merge_hyphen_breaks(norm)
    norm = _merge_scientific_name_fragments(norm)  # before cap-fragments
    norm = _merge_cap_fragments(norm)

    # C) PDF-specific: unwrap soft line breaks between Latin letters
    #    "Thái\nLan" -> "Thái Lan", "bệnh\nParkin" -> "bệnh Parkin"
    if doc_type == "pdf":
        norm = _RE_PDF_LETTER_NL_LETTER.sub(r"\1 ", norm)

    # D) Remove pipe/hash inside words
    norm = _RE_PIPE_HASH_INSIDE.sub("", norm)

    # Collapse whitespace: multiple spaces -> single space per line
    lines = norm.split("\n")
    cleaned_lines: list[str] = []
    for line in lines:
        line = re.sub(r"[ \t]+", " ", line).strip()
        cleaned_lines.append(line)

    # Collapse 3+ consecutive blank lines to 2
    result_lines: list[str] = []
    blank_count = 0
    for line in cleaned_lines:
        if line == "":
            blank_count += 1
            if blank_count <= 2:
                result_lines.append(line)
        else:
            blank_count = 0
            result_lines.append(line)

    norm = "\n".join(result_lines).strip()

    if encoding_noise:
        return norm, True

    return norm, False


# ---------------------------------------------------------------------------
# Process all chunks
# ---------------------------------------------------------------------------

def process_chunks(records: list[dict[str, Any]], debug: bool = False) -> list[dict[str, Any]]:
    """Add text_norm, clean_version, is_noise to each chunk record."""
    # Metrics regexes (same as h1_report)
    re_lnl = re.compile(r"[A-Za-z]\n[A-Za-z]")
    re_sci = re.compile(r"[a-z]{2,}\n[a-z]{2,}")

    output: list[dict[str, Any]] = []
    noise_count = 0
    transforms: list[tuple[str, str]] = []

    before_lnl = 0
    before_sci = 0
    after_lnl = 0
    after_sci = 0

    # --- Encoding-noise trackers ---
    cyrillic_count = 0
    noise_encoding_count = 0
    file_bad_ratios: dict[str, list[float]] = defaultdict(list)

    for rec in records:
        text = rec.get("text", "")
        doc_type = rec.get("doc_type", "")

        # Count before (original text)
        if re_lnl.search(text):
            before_lnl += 1
        if re_sci.search(text):
            before_sci += 1

        # Encoding-noise stats (on original text)
        bad_ratio, ufffd_ratio, has_cyr = _detect_encoding_noise(text)
        if has_cyr:
            cyrillic_count += 1
        fp = rec.get("file_path") or rec.get("source_id") or "unknown"
        if bad_ratio > 0 or ufffd_ratio > 0:
            file_bad_ratios[fp].append(bad_ratio + ufffd_ratio)

        text_norm, is_noise_flag = normalize_text_v2(text, doc_type)

        if is_noise_flag and (bad_ratio > BAD_RATIO_THRESHOLD
                              or ufffd_ratio > UFFFD_RATIO_THRESHOLD):
            noise_encoding_count += 1

        # Count after (normalized text)
        if not is_noise_flag and re_lnl.search(text_norm):
            after_lnl += 1
        if not is_noise_flag and re_sci.search(text_norm):
            after_sci += 1

        # Build output record: preserve all original fields
        out = dict(rec)
        out["text_norm"] = text_norm
        out["clean_version"] = "v2"
        if is_noise_flag:
            out["is_noise"] = True
            noise_count += 1
        else:
            out["is_noise"] = False

        output.append(out)

        # Collect transforms for debug
        if debug and text != text_norm and not is_noise_flag:
            transforms.append((text[:120], text_norm[:120]))

    # --- Before / After stats ---
    logger.info("=== Clean v2 error stats (chunk-level) ===")
    pct_lnl = (1 - after_lnl / max(before_lnl, 1)) * 100
    pct_sci = (1 - after_sci / max(before_sci, 1)) * 100
    logger.info(
        "  letter\\nletter : %d -> %d  (reduced %.1f%%)",
        before_lnl, after_lnl, pct_lnl,
    )
    logger.info(
        "  sci-split      : %d -> %d  (reduced %.1f%%)",
        before_sci, after_sci, pct_sci,
    )

    # --- Encoding-noise stats ---
    logger.info("=== Encoding-noise stats ===")
    logger.info("  Chunks with Cyrillic chars : %d", cyrillic_count)
    logger.info("  Chunks marked is_noise (encoding) : %d", noise_encoding_count)
    logger.info("  Total is_noise (all reasons)       : %d", noise_count)
    # Top 5 files by worst bad_ratio
    if file_bad_ratios:
        file_avg = {
            fp: sum(vals) / len(vals)
            for fp, vals in file_bad_ratios.items()
        }
        top5 = sorted(file_avg.items(), key=lambda x: x[1], reverse=True)[:5]
        logger.info("  Top 5 files with highest avg bad_ratio:")
        for fp, avg in top5:
            logger.info("    %.4f  %s", avg, fp)

    if debug and transforms:
        logger.info("--- Top 20 transformations (before -> after) ---")
        for i, (before, after) in enumerate(transforms[:20]):
            logger.info("  [%d] BEFORE: %r", i + 1, before)
            logger.info("       AFTER:  %r", after)

    return output


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Clean v2 — post-chunk normalization")
    parser.add_argument("--in", dest="input_path", required=True, help="Input chunks JSONL (e.g. chunks_v1.jsonl)")
    parser.add_argument("--out", dest="output_path", required=True, help="Output chunks JSONL (e.g. chunks_v2.jsonl)")
    parser.add_argument("--debug", action="store_true", help="Show top 20 transformation examples")
    args = parser.parse_args()

    records = read_jsonl(args.input_path)
    logger.info("Input records: %d", len(records))

    output = process_chunks(records, debug=args.debug)

    noise_count = sum(1 for r in output if r.get("is_noise"))
    logger.info("Output records: %d", len(output))
    logger.info("Noise records (is_noise=true): %d", noise_count)

    ensure_parent_dir(args.output_path)
    write_jsonl(output, args.output_path)
    logger.info("Wrote %s", args.output_path)


if __name__ == "__main__":
    main()
