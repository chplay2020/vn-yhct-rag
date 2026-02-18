"""Text normalization utilities."""

from __future__ import annotations

import re
import unicodedata

# ---------------------------------------------------------------------------
# Conservative Vietnamese word-boundary fix patterns
# ---------------------------------------------------------------------------
# Pattern 1: lowercase/vi-letter followed by Uppercase letter with NO space
#   e.g. "rễĐem" => "rễ Đem",  "hoặcSao" => "hoặc Sao"
_RE_LOWER_UPPER_GLUE = re.compile(
    r"(?<=[\p{Ll}])(?=[\p{Lu}])" if False else  # regex module style (fallback below)
    r"(?<=[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđ])"
    r"(?=[A-ZÁÀẢÃẠĂẮẰẲẴẶÂẤẦẨẪẬÉÈẺẼẸÊẾỀỂỄỆÍÌỈĨỊÓÒỎÕỌÔỐỒỔỖỘƠỚỜỞỠỢÚÙỦŨỤƯỨỪỬỮỰÝỲỶỸỴĐĐ])"
)

# Pattern 2: common Vietnamese word-boundary glue after punctuation
#   e.g.  ")hoặc" => ") hoặc",  ".Đem" => ". Đem"
_RE_PUNCT_LETTER_GLUE = re.compile(
    r"(?<=[.)\]!?;:])(?=[A-Za-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợúùủũụưứừửữựýỳỷỹỵđĐ])"
)

# Pattern 3: Vietnamese closed-syllable glue — a vowel + final consonant (c/m/n/p/t)
#   directly touching next syllable's initial consonant + vowel.
#   e.g. "hoặcsao" → "hoặc sao",  "đắplên" → "đắp lên"
_VIET_VOWEL_END = (
    r"[aáàảãạăắằẳẵặâấầẩẫậeéèẻẽẹêếềểễệiíìỉĩị"
    r"oóòỏõọôốồổỗộơớờởỡợuúùủũụưứừửữựyýỳỷỹỵ]"
)
_VIET_CONSONANT_START = r"[bcdfghjklmnpqrstvwxzđ]"
_RE_VIET_CLOSED_SYLLABLE_GLUE = re.compile(
    rf"(?<={_VIET_VOWEL_END}[cmnpt])(?={_VIET_CONSONANT_START}{{1,2}}{_VIET_VOWEL_END})",
    re.IGNORECASE,
)

# Pattern 4: Vietnamese open-syllable glue — vowel (with diacritic) directly
#   touching next syllable's initial consonant + vowel.
#   e.g. "rễđem" → "rễ đem"
#   Use ONLY Vietnamese diacritical vowels (not plain a,e,i,o,u,y) to avoid
#   English false positives like "education" → "educa tion".
_VIET_DIACRITIC_VOWEL = (
    r"[áàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợ"
    r"úùủũụưứừửữựýỳỷỹỵ]"
)
_RE_VIET_OPEN_SYLLABLE_GLUE = re.compile(
    rf"(?<={_VIET_DIACRITIC_VOWEL})(?={_VIET_CONSONANT_START}{{1,2}}{_VIET_VOWEL_END})",
    re.IGNORECASE,
)

# Pattern 5: plain-vowel + consonant + diacritical-vowel glue.
#   Catches "rửasạch" → "rửa sạch" where boundary char is plain "a"
#   but the NEXT syllable has a Vietnamese diacritical vowel.
_RE_VIET_PLAIN_DIAC_GLUE = re.compile(
    rf"(?<=[aeiouy])(?={_VIET_CONSONANT_START}{{1,2}}{_VIET_DIACRITIC_VOWEL})",
    re.IGNORECASE,
)


def normalize_vi_en(text: str) -> str:
    """Normalize text for Vietnamese / English:
    1) Unicode NFC
    2) Remove control chars (Cc/Cf) except newline/tab
    3) Remove U+FFFD replacement char "�"
    4) Normalize whitespace (CRLF→LF, collapse spaces, strip trailing per line)
    5) Fix lightweight OCR/convert glue errors (conservative)
    6) Cap consecutive blank lines at 2
    """
    # 1. Unicode NFC
    text = unicodedata.normalize("NFC", text)

    # 2. Remove control characters except \n \r \t
    def _remove_control(ch: str) -> str:
        if ch in ("\n", "\r", "\t"):
            return ch
        cat = unicodedata.category(ch)
        if cat.startswith("C"):  # Cc, Cf, Co, Cs
            return ""
        return ch

    text = "".join(_remove_control(c) for c in text)

    # 3. Replace U+FFFD replacement character with space (often was a space/separator)
    text = text.replace("\ufffd", " ")

    # 4. Normalize whitespace
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    cleaned_lines: list[str] = []
    for line in lines:
        line = re.sub(r"[ \t]+", " ", line).strip()
        cleaned_lines.append(line)

    # 5. Fix glue errors per line (conservative)
    fixed_lines: list[str] = []
    for line in cleaned_lines:
        # 5a. lowercase + Uppercase glue  (e.g. "rễĐem" → "rễ Đem")
        line = _RE_LOWER_UPPER_GLUE.sub(" ", line)
        # 5b. punctuation + letter glue   (e.g. ")hoặc" → ") hoặc")
        line = _RE_PUNCT_LETTER_GLUE.sub(" ", line)
        # 5c. Vietnamese closed-syllable glue (e.g. "hoặcsao" → "hoặc sao")
        line = _RE_VIET_CLOSED_SYLLABLE_GLUE.sub(" ", line)
        # 5d. Vietnamese open-syllable glue (e.g. "rễđem" → "rễ đem")
        line = _RE_VIET_OPEN_SYLLABLE_GLUE.sub(" ", line)
        # 5e. plain vowel + consonant + diacritical vowel (e.g. "rửasạch" → "rửa sạch")
        line = _RE_VIET_PLAIN_DIAC_GLUE.sub(" ", line)
        fixed_lines.append(line)

    # 6. Remove excessive blank lines (max 2 consecutive)
    result_lines: list[str] = []
    blank_count = 0
    for line in fixed_lines:
        if line == "":
            blank_count += 1
            if blank_count <= 2:
                result_lines.append(line)
        else:
            blank_count = 0
            result_lines.append(line)

    return "\n".join(result_lines).strip()
