# pyright: reportUnknownMemberType=false
"""Query quality utilities — noise detection, duplicate text normalisation.

Used by retrieval_ablation and hybrid_retriever for:
  • validating synthetic questions before evaluation
  • deduplicating chunk text in candidate lists
"""

from __future__ import annotations

import re
import unicodedata

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


def normalize_for_dedup(text: str) -> str:
    """Normalise text for exact-duplicate detection.

    NFC → lowercase → collapse whitespace → strip.
    """
    t = unicodedata.normalize("NFC", text).lower()
    t = _RE_MULTISPACE.sub(" ", t).strip()
    return t
