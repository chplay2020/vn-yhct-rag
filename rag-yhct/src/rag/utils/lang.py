"""Language detection utility."""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def detect_language(text: str, fallback: str = "vi") -> str:
    """Detect language of text using langdetect. Returns 'vi' or 'en'.

    Falls back to *fallback* on any error or unsupported language.
    """
    try:
        from langdetect import detect  # type: ignore

        lang: str = detect(text)  # type: ignore
        if lang in ("vi", "en"):
            return lang
        return fallback
    except Exception:
        logger.debug("Language detection failed, using fallback=%s", fallback)
        return fallback
