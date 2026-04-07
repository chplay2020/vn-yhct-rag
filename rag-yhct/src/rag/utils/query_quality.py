# pyright: reportUnknownMemberType=false
"""Query quality utilities — noise detection, duplicate/query normalisation.

Used by retrieval_ablation and hybrid_retriever for:
  • validating synthetic questions before evaluation
  • deduplicating chunk text in candidate lists
    • normalising user queries consistently across retrievers
"""

from __future__ import annotations

import json
import os
import re
import unicodedata
from collections import Counter

# ── noise patterns ────────────────────────────────────────────────────────

_RE_CYRILLIC = re.compile(r"[\u0400-\u04FF]")
_RE_CJK = re.compile(r"[\u4E00-\u9FFF\u3400-\u4DBF]")
_RE_LATIN_VIET = re.compile(
    r"[a-zA-Zàáảãạăắằẳẵặâấầẩẫậèéẻẽẹêếềểễệìíỉĩịòóỏõọôốồổỗộơớờởỡợ"
    r"ùúủũụưứừửữựỳýỷỹỵđĐ]",
)
_MOJIBAKE_CHARS = frozenset(
    "\u00bf\u00b6\u00b5\u00b9\u00b2\u00b3\u00bc\u00bd\u00be"
    "\ufffd\u0000\u001a"
)
_VAGUE_EN = re.compile(
    r"\b(this|that|the passage|it says|it is|they are|these|those)\b",
    re.IGNORECASE,
)

MIN_QUESTION_LEN = 12
MAX_CJK_RATIO = 0.15
MAX_MOJIBAKE_RATIO = 0.03


def is_query_noisy(text: str) -> bool:
    """Return True if *text* looks like a bad synthetic question.

    Criteria:
      • too short
      • contains Cyrillic characters
      • excessive mojibake / replacement chars
      • too many CJK characters (not Vietnamese)
      • mostly non-Latin/Vietnamese text
      • contains vague English pronouns / phrases
    """
    if len(text.strip()) < MIN_QUESTION_LEN:
        return True

    if _RE_CYRILLIC.search(text):
        return True

    text_ns = re.sub(r"\s", "", text)
    length = max(1, len(text_ns))

    # mojibake ratio
    bad = sum(1 for ch in text_ns if ch in _MOJIBAKE_CHARS)
    if bad / length > MAX_MOJIBAKE_RATIO:
        return True

    # CJK ratio (Chinese chars leak)
    cjk = len(_RE_CJK.findall(text_ns))
    if cjk / length > MAX_CJK_RATIO:
        return True

    # Latin/Vietnamese ratio — at least 40 % of chars should be Latin/Viet
    latin = len(_RE_LATIN_VIET.findall(text_ns))
    if latin / length < 0.40:
        return True

    # Vague English phrases
    if _VAGUE_EN.search(text):
        return True

    return False


# ── duplicate text normalisation ──────────────────────────────────────────

_RE_MULTISPACE = re.compile(r"\s+")
_RE_WORD = re.compile(r"\w+", re.UNICODE)

# chunks_path -> (size, mtime_ns, mapping)
_DIACRITIC_LEXICON_CACHE: dict[str, tuple[int, int, dict[str, str]]] = {}


def strip_vietnamese_diacritics(text: str) -> str:
    """Return accent-folded text for Vietnamese matching.

    Examples:
      "tác dụng cây ngải cứu" -> "tac dung cay ngai cuu"
      "điều trị" -> "dieu tri"
    """
    t = unicodedata.normalize("NFD", text)
    t = "".join(ch for ch in t if unicodedata.category(ch) != "Mn")
    t = t.replace("đ", "d").replace("Đ", "D")
    return unicodedata.normalize("NFC", t)


def normalize_query_for_retrieval(text: str) -> str:
    """Normalize a user query for retrieval.

    Steps:
      1) Unicode NFC normalization
      2) lowercase
      3) collapse repeated whitespace
      4) trim leading/trailing whitespace
    """
    t = unicodedata.normalize("NFC", text)
    t = t.lower()
    t = _RE_MULTISPACE.sub(" ", t).strip()
    return t


def normalize_query_no_diacritics(text: str) -> str:
    """Normalize query then fold Vietnamese diacritics for accent-insensitive match."""
    return strip_vietnamese_diacritics(normalize_query_for_retrieval(text))


def _file_signature(path: str) -> tuple[int, int]:
    st = os.stat(path)
    return int(st.st_size), int(st.st_mtime_ns)


def _build_diacritic_lexicon(chunks_path: str) -> dict[str, str]:
    """Build folded-token -> most common accented-token mapping from chunks JSONL."""
    folded_to_counter: dict[str, Counter[str]] = {}

    with open(chunks_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue

            text = str(rec.get("text_norm") or rec.get("text") or "")
            if not text:
                continue
            norm = normalize_query_for_retrieval(text)
            for tok in _RE_WORD.findall(norm):
                if len(tok) < 2:
                    continue
                folded = strip_vietnamese_diacritics(tok)
                if folded == tok:
                    continue
                bucket = folded_to_counter.setdefault(folded, Counter())
                bucket[tok] += 1

    mapping: dict[str, str] = {}
    for folded, counts in folded_to_counter.items():
        if not counts:
            continue
        mapping[folded] = counts.most_common(1)[0][0]
    return mapping


def restore_query_diacritics_from_corpus(
    text: str,
    *,
    chunks_path: str = "data/chunks/chunks_v2_full.jsonl",
) -> str:
    """Restore likely Vietnamese diacritics for a no-accent query.

    This uses corpus statistics: each folded token maps to its most frequent
    accented token seen in chunks data.
    """
    normalized = normalize_query_for_retrieval(text)
    folded_query = strip_vietnamese_diacritics(normalized)

    # If query already has diacritics (or no Vietnamese letters), keep it.
    if normalized != folded_query:
        return normalized

    if not os.path.exists(chunks_path):
        return normalized

    sig = _file_signature(chunks_path)
    cached = _DIACRITIC_LEXICON_CACHE.get(chunks_path)
    if cached is None or (cached[0], cached[1]) != sig:
        mapping = _build_diacritic_lexicon(chunks_path)
        _DIACRITIC_LEXICON_CACHE[chunks_path] = (sig[0], sig[1], mapping)
    else:
        mapping = cached[2]

    if not mapping:
        return normalized

    def _replace(match: re.Match[str]) -> str:
        tok = match.group(0)
        return mapping.get(tok, tok)

    return _RE_WORD.sub(_replace, normalized)


def explain_query_normalization(text: str) -> dict[str, str | bool]:
    """Return each transformation stage for notebook/debug visibility."""
    stage_nfc = unicodedata.normalize("NFC", text)
    stage_lower = stage_nfc.lower()
    stage_ws = _RE_MULTISPACE.sub(" ", stage_lower)
    normalized = stage_ws.strip()
    return {
        "raw": text,
        "nfc": stage_nfc,
        "lower": stage_lower,
        "whitespace_collapsed": stage_ws,
        "normalized": normalized,
        "changed": normalized != text,
    }


def normalize_for_dedup(text: str) -> str:
    """Normalise text for exact-duplicate detection.

    NFC → lowercase → collapse whitespace → strip.
    """
    return normalize_query_for_retrieval(text)
