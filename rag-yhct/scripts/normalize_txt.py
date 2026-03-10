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
  3b. Latin/English fragment rejoin: merge spurious spaces within
      Latin/English words created by PDF text extraction.
      E.g. "Eucalyp tus" → "Eucalyptus", "Cym bopogon" → "Cymbopogon"
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


# ---------------------------------------------------------------------------
# Latin / English fragment rejoining  (PDF text-extraction artifact fix)
# ---------------------------------------------------------------------------
# PyMuPDF often inserts spurious spaces within Latin/English words based on
# character positioning gaps in the PDF.  This produces fragments like
# "Eucalyp tus" (should be "Eucalyptus") or "Cym bopogon" ("Cymbopogon").
#
# Strategy:
#   1. Find runs of consecutive pure-ASCII-alphabetic tokens on each line.
#   2. Within each run, merge short fragments (≤3 chars) with a neighbour:
#        • Prefer merging LEFT if the left neighbour looks incomplete
#          (ends with a consonant that rarely terminates a Latin word).
#        • Otherwise merge RIGHT.
#   3. Also merge tokens whose ending signals an incomplete word (e.g.
#      "Eucalyp" ends with 'p') with a short right neighbour (≤6 chars).
#   4. Iterate until stable.
# ---------------------------------------------------------------------------

# Consonant letters that very rarely end a Latin / English word.
# If a pure-ASCII token ends with one of these it is probably truncated.
_RARE_FINAL_CHARS = frozenset("bcfgjkpqvwz")
_ASCII_CONSONANTS = frozenset("bcdfghjklmnpqrstvwxyz")

# Trailing punctuation that can follow a Latin word in running text.
_TRAILING_PUNCT = frozenset(".,;:!?)]}")

# Common Vietnamese words that are pure ASCII (no diacritics).
# Used to prevent false merges with adjacent Latin words.
# Only LOWERCASE forms are checked — uppercase may be Latin genus names.
# Excludes words that commonly appear as Latin word fragments
# (in, an, et, ta, la, lon, nam, tan, ten, tim, tro, ba, etc.).
_VI_ASCII_WORDS = frozenset({
    "do", "va", "co", "tu", "ra", "vi", "ma", "mg",
    "no", "me", "bo", "bi", "di",
    "cay", "con", "hay", "hai", "lam", "cho", "cha", "cao",
    "nay", "ban", "ben", "chu", "dai", "dan", "dau",
    "don", "gam", "ghi", "gia", "goi", "hoa", "hoi", "hon", "hoc",
    "kem", "khi", "lai", "luc", "mat", "mau", "moi",
    "mot", "nha", "nho", "noi", "ong", "qua", "que", "qui",
    "sua", "thi", "thu",
    "tre", "vai", "vao", "voi", "von", "yen",
    "cung", "dung", "duoc", "giua", "khac", "nang", "nhau", "nhom",
    "phan", "rang", "sinh", "tang", "them", "tren", "trong",
    "vung",
})


def _is_ascii_alpha(s: str) -> bool:
    """Return *True* if *s* is non-empty and contains only ASCII letters."""
    return bool(s) and s.isascii() and s.isalpha()


def _strip_trailing_punct(s: str) -> tuple[str, str]:
    """Split *s* into (word, trailing_punctuation).

    >>> _strip_trailing_punct("tia.")
    ('tia', '.')
    >>> _strip_trailing_punct("hello")
    ('hello', '')
    """
    i = len(s)
    while i > 0 and s[i - 1] in _TRAILING_PUNCT:
        i -= 1
    return s[:i], s[i:]


# Botanical rank abbreviations that must NOT be merged with neighbours.
_RANK_ABBREVS = frozenset({"var", "subsp", "ssp", "sp", "cv", "aff", "cf"})

# Latin/English particles and conjunctions that should always stay separate.
_LATIN_PARTICLES = frozenset({"et", "ex", "vel", "nom"})


def _is_abbreviation(word: str, punct: str) -> bool:
    """True if *word* + *punct* looks like an abbreviation that should stay separate.

    Matches:
      • Author abbreviations: L., DC.  (≤2 chars + period)
      • All-caps abbreviations with period: NXB., TP.  (≤4 chars, uppercase + period)
      • Botanical rank words: var, var., subsp, subsp., sp, sp., etc.
      • Latin particles/conjunctions: et, ex, in, de, vel, nom
    """
    if len(word) <= 2 and "." in punct:
        return True
    if len(word) <= 4 and word.isupper() and "." in punct:
        return True
    if word.lower() in _RANK_ABBREVS:
        return True
    if word.lower() in _LATIN_PARTICLES:
        return True
    return False


# Consonant clusters that typically START a syllable (onset clusters).
# If a token ends with one of these, it was likely truncated mid-word.
_ONSET_CLUSTERS = frozenset({
    "bl", "br", "ch", "cl", "cr", "dr", "dw", "fl", "fr",
    "gl", "gn", "gr", "kn", "ph", "pl", "pr", "sc", "sh",
    "sk", "sl", "sm", "sn", "sp", "sq", "sw", "th", "tr",
    "tw", "wh", "wr",
})


def _looks_incomplete(token: str) -> bool:
    """Heuristic: does *token* look like a truncated Latin/English word?

    True when the token ends with a consonant that rarely terminates a word,
    OR when the last two characters form an onset cluster (e.g. "gl", "sc",
    "pl") — consonant pairs that typically start a syllable, signalling
    the word was split mid-syllable.
    """
    if not token:
        return False
    last = token[-1].lower()
    if last in _RARE_FINAL_CHARS:
        return True
    # Onset cluster at end: e.g. "Ophiogl" ending with "gl"
    if len(token) >= 2:
        pair = token[-2:].lower()
        if pair in _ONSET_CLUSTERS:
            return True
    return False


def _merge_ascii_group(group: list[str], *, last_is_eol: bool = False) -> list[str]:
    """Merge probable fragments in a run of consecutive ASCII-alpha tokens.

    If *last_is_eol* is True the last token sits at end-of-line and may
    continue on the next line — it must NOT be merged left (but CAN be
    merged right if a later pass adds tokens).

    Rules (applied iteratively):
      R1  Token ≤ 3 chars → merge LEFT if left neighbour looks incomplete,
          otherwise merge RIGHT (or LEFT if it is the last token).
      R2  Token that looks incomplete AND whose right neighbour is ≤ 6 chars
          → merge RIGHT (catches "Botryc" + "hium" → "Botrychium" after
          an earlier R1 pass merged "Bot" + "ryc").
      R3  Both > 3 AND both ≤ 6 AND at least one ≤ 4 → merge RIGHT.
          Catches "Amaran"(6) + "thus"(4) → "Amaranthus".
          Requires both > 3 so R1 gets priority for short tokens.
    """
    changed = True
    while changed:
        changed = False
        new: list[str] = []
        i = 0
        while i < len(group):
            tok = group[i]
            is_last = i == len(group) - 1

            # --- R1: short token (≤ 3 chars) ---
            if len(tok) <= 3:
                # If this is the last token and it sits at end-of-line,
                # do NOT merge it left — it likely continues on next line.
                if is_last and last_is_eol:
                    new.append(tok)
                    i += 1
                    continue

                # If this is the FIRST token (no left), only merge RIGHT
                # if the next token starts with lowercase (continuation
                # fragment, e.g. "Bot" + "ryc").  If next starts uppercase,
                # it is likely a separate word (e.g. "do" + "Cynarin").
                if not new:
                    if (
                        i + 1 < len(group)
                        and group[i + 1][0].islower()
                    ):
                        new.append(tok + group[i + 1])
                        i += 2
                        changed = True
                    else:
                        new.append(tok)
                        i += 1
                    continue

                # Direction: prefer LEFT if left looks incomplete, or if
                # RIGHT starts with uppercase (author/proper name boundary).
                right_upper = (
                    i + 1 < len(group) and group[i + 1][0].isupper()
                )
                if new and (_looks_incomplete(new[-1]) or right_upper):
                    # Merge LEFT
                    new[-1] += tok
                    changed = True
                elif i + 1 < len(group):
                    # Merge RIGHT
                    new.append(tok + group[i + 1])
                    i += 1          # skip the right token
                    changed = True
                elif new and _looks_incomplete(new[-1]):
                    # Last token, no right → merge LEFT only if left is
                    # incomplete (otherwise a standalone letter/label like
                    # "A", "B" should stay separate).
                    new[-1] += tok
                    changed = True
                else:
                    new.append(tok)
                i += 1
                continue

            # --- R2: incomplete token + right neighbour ≤ 8 chars ---
            if (
                _looks_incomplete(tok)
                and i + 1 < len(group)
                and len(group[i + 1]) <= 8
            ):
                new.append(tok + group[i + 1])
                i += 2
                changed = True
                continue

            # --- R3: both > 3, both ≤ 6, at least one ≤ 4,
            #         AND the left token looks incomplete ---
            if (
                i + 1 < len(group)
                and _looks_incomplete(tok)
                and len(tok) > 3
                and len(group[i + 1]) > 3
                and len(tok) <= 6
                and len(group[i + 1]) <= 6
                and (len(tok) <= 4 or len(group[i + 1]) <= 4)
            ):
                new.append(tok + group[i + 1])
                i += 2
                changed = True
                continue

            new.append(tok)
            i += 1

        group = new
    return group


# Regex to detect Latin author citations: (L.), (Thunb), (DC), etc.
_RE_AUTHOR_CITATION = re.compile(r"\([A-Z][A-Za-z]{0,10}\.?\)")


def _is_latin_context(line: str) -> bool:
    """Return True if *line* likely contains Latin/English scientific text.

    Prevents false-positive merging of Vietnamese ASCII-only words
    (e.g. "Gia Lai", "anh em", "Jrai Bahnar").
    """
    # Lines with ethnic language markers (/B/, /J/) are NOT Latin context
    if "/B/" in line or "/J/" in line:
        return False
    lower = line.lower()
    if "khoa học" in lower:
        return True
    if "thuộc họ" in lower:
        return True
    if _RE_AUTHOR_CITATION.search(line):
        return True
    return False


def _rejoin_latin_fragments(text: str) -> str:
    """Rejoin Latin/English word fragments created by PDF text extraction.

    Scans each line for runs of consecutive pure-ASCII-alphabetic tokens
    (also including tokens with trailing punctuation like ``tia.``) and
    merges short fragments with their neighbours.

    Guards against false positives on Vietnamese words:
      • On lines with clear Latin context (``Tên khoa học:``, ``thuộc họ``,
        or author citations like ``(L.)``), all groups of ≥2 tokens are
        eligible for merging.
      • On other lines, a group must contain at least one "anchor" token
        ≥ 7 chars (a length rarely seen in Vietnamese words) to be merged.

    Abbreviations (single/double-letter + period, e.g. "L.", "DC.") are
    excluded from merge groups to avoid corrupting author citations.
    """
    out_lines: list[str] = []
    for line in text.split("\n"):
        tokens = line.split(" ")
        if len(tokens) < 2:
            out_lines.append(line)
            continue

        # Skip lines with ethnic language markers entirely — these contain
        # Bahnar/Jrai names that should not be merged.
        if "/B/" in line or "/J/" in line:
            out_lines.append(line)
            continue

        latin_ctx = _is_latin_context(line)

        output: list[str] = []
        i = 0
        while i < len(tokens):
            # Check if token (possibly with trailing punct) is ASCII alpha
            word, punct = _strip_trailing_punct(tokens[i])

            # Skip non-alpha tokens, abbreviations, common Vietnamese
            # ASCII words, and single letters (author labels L/A/B, hybrid x).
            if (
                not _is_ascii_alpha(word)
                or _is_abbreviation(word, punct)
                or (word[0].islower() and word.lower() in _VI_ASCII_WORDS)
                or len(word) == 1
            ):
                output.append(tokens[i])
                i += 1
                continue

            # Collect a run of consecutive ASCII-alpha tokens.
            # A token WITH trailing punctuation (e.g. "calcium,") ends
            # the current group — the punctuation signals a word boundary.
            # Common Vietnamese ASCII words also break the run.
            group: list[str] = [word]
            trailing: list[str] = [punct]
            if not punct:
                # Only continue collecting if current token has no trailing punct
                i += 1
                while i < len(tokens):
                    w, p = _strip_trailing_punct(tokens[i])
                    if (
                        _is_ascii_alpha(w)
                        and not _is_abbreviation(w, p)
                        and not (w[0].islower() and w.lower() in _VI_ASCII_WORDS)
                        and len(w) != 1
                    ):
                        group.append(w)
                        trailing.append(p)
                        i += 1
                        if p:
                            # Token has trailing punct — end the group here
                            break
                    else:
                        break
            else:
                i += 1

            # Guard: only merge in Latin context, or if the group has a
            # long anchor (≥ 7 chars — unlikely to be a Vietnamese word).
            has_anchor = latin_ctx or any(len(t) >= 7 for t in group)

            # Detect if the group sits at the end of the line.
            # If so, the last token may continue on the next line and
            # should not be merged left.
            group_at_eol = i >= len(tokens)

            if len(group) >= 2 and has_anchor:
                group = _merge_ascii_group(group, last_is_eol=group_at_eol)
                # Re-attach trailing punctuation to the last merged token.
                # Only the LAST original token's punct is meaningful.
                last_punct = ""
                for p in reversed(trailing):
                    if p:
                        last_punct = p
                        break
                if last_punct and group:
                    group[-1] += last_punct
                output.extend(group)
            else:
                # No merge — output original tokens with their punct intact
                for g, t in zip(group, trailing):
                    output.append(g + t)

        out_lines.append(" ".join(output))
    return "\n".join(out_lines)


# ---------------------------------------------------------------------------
# Cross-line joining for Latin/English fragments
# ---------------------------------------------------------------------------

def _cross_line_join_latin(lines: list[str]) -> list[str]:
    """Join lines where a Latin/English word was split across a line break.

    If a line ends with a pure-ASCII-alpha token that looks incomplete
    (ends with a rare final consonant) and the next non-empty line starts
    with a lowercase ASCII-alpha token, concatenate the ending + beginning.
    """
    result: list[str] = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if (
            line
            and i + 1 < len(lines)
            and lines[i + 1].strip()
        ):
            last_tok = line.rsplit(" ", 1)[-1] if " " in line else line
            next_stripped = lines[i + 1].strip()
            first_tok = next_stripped.split(" ", 1)[0]

            if (
                _is_ascii_alpha(last_tok)
                and _looks_incomplete(last_tok)
                and _is_ascii_alpha(first_tok)
                and first_tok[0].islower()
            ):
                # Join: current line + first token of next line (no space)
                rest_of_next = next_stripped[len(first_tok):]
                joined = line + first_tok
                if rest_of_next:
                    # Remaining text of next line goes on a new line (or same)
                    joined += rest_of_next
                result.append(joined)
                i += 2
                continue
        result.append(line)
        i += 1
    return result


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

    # 2c. Cross-line join for Latin/English fragments
    #     (e.g. "Eucalyp\ntus" → "Eucalyptus")
    result = _cross_line_join_latin(result)

    # 2d. Rejoin intra-line Latin/English fragments
    #     (e.g. "Cym bopogon" → "Cymbopogon")
    text = _rejoin_latin_fragments("\n".join(result))

    return text


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
