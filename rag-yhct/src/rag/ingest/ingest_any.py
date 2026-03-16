"""B1 — Ingest any document (PDF / DOCX / IMAGE) → raw_passages.jsonl

Usage:
    python -m rag.ingest.ingest_any --config config/config.yaml
"""

from __future__ import annotations

import argparse
import logging
from collections import Counter
from pathlib import Path
from typing import Any, cast

import yaml
from tqdm import tqdm  # type: ignore

from rag.utils.hashing import sha1_fingerprint
from rag.utils.io import ensure_parent_dir, write_jsonl
from rag.utils.lang import detect_language

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# PDF ingestion
# ---------------------------------------------------------------------------

def _ingest_pdf(
    source: dict[str, Any],
    cfg_ingest: dict[str, Any],
    paper_sections: set[str],
) -> list[dict[str, Any]]:
    """Ingest a single PDF file using PyMuPDF (fitz), one record per page."""
    import fitz  # type: ignore  # PyMuPDF

    file_path = source["file_path"]
    source_id = source["source_id"]
    min_text_len = int(cfg_ingest.get("min_text_len_pdf", 30))
    doc_fp = sha1_fingerprint(file_path)
    manifest_lang = source.get("doc_language")

    try:
        doc = fitz.open(file_path)  # type: ignore
    except Exception as exc:
        logger.error("PyMuPDF failed to open %s: %s", file_path, exc)
        return []

    records: list[dict[str, Any]] = []
    total_pages = len(doc)  # type: ignore
    kept = 0

    for i, page in enumerate(cast(list[Any], doc)):  # type: ignore
        page_no = i + 1  # 1-based
        raw_text = page.get_text("text")
        text = str(raw_text).strip()

        if len(text) < min_text_len:
            continue

        kept += 1
        doc_lang = manifest_lang or detect_language(text)

        rec: dict[str, Any] = {
            "source_id": source_id,
            "doc_type": "pdf",
            "title": source.get("title", ""),
            "author": source.get("author", ""),
            "year": source.get("year"),
            "file_path": file_path,
            "url": source.get("url", ""),
            "page": page_no,
            "page_range": None,
            "section_heading": None,
            "span": None,
            "locator": f"p{page_no}",
            "doc_fingerprint": doc_fp,
            "doc_language": doc_lang,
            "text": text,
            "element_idx": page_no,
        }
        for key in ("doi", "journal"):
            if key in source:
                rec[key] = source[key]

        records.append(rec)

    doc.close()  # type: ignore
    logger.info(
        "[PDF] %s — %d pages total, %d pages kept (>=%d chars)",
        source_id, total_pages, kept, min_text_len,
    )
    return records


# ---------------------------------------------------------------------------
# DOCX ingestion
# ---------------------------------------------------------------------------

def _heading_level_from_element(el: Any) -> int | None:
    """Try to extract heading level (1-6) from unstructured element metadata/style."""
    meta = getattr(el, "metadata", None)
    if meta:
        # Some unstructured versions expose category_depth or parent_id-based depth
        depth = getattr(meta, "category_depth", None)
        if depth is not None:
            try:
                return max(1, int(depth))
            except (ValueError, TypeError):
                pass
        # Try the original Word style name, e.g. "Heading 2"
        style = getattr(meta, "element_style", None) or ""
        if not style:
            style = getattr(meta, "style_name", None) or ""
        import re as _re
        m = _re.search(r"[Hh]eading\s*(\d)", style)
        if m:
            return int(m.group(1))
    return None


def _ingest_docx(
    source: dict[str, Any],
    cfg_ingest: dict[str, Any],
    paper_sections: set[str],
) -> list[dict[str, Any]]:
    """Ingest a single DOCX file."""
    from unstructured.partition.docx import partition_docx  # type: ignore

    file_path = source["file_path"]
    source_id = source["source_id"]

    try:
        elements = partition_docx(filename=file_path)  # type: ignore
    except Exception as exc:
        logger.error("partition_docx failed for %s: %s", file_path, exc)
        return []

    min_text_len = int(cfg_ingest.get("min_text_len_docx", 30))
    doc_fp = sha1_fingerprint(file_path)
    manifest_lang = source.get("doc_language")

    records: list[dict[str, Any]] = []
    # heading_stack maps level -> heading text (ordered by level)
    heading_stack: dict[int, str] = {}
    current_heading: str | None = None
    para_idx = 0

    for elem_idx, el in enumerate(cast(list[Any], elements)):
        text = str(el).strip()
        if not text:
            continue

        cat = getattr(el, "category", "") or ""  # type: ignore
        class_name = type(el).__name__  # type: ignore

        # Heading detection
        is_title = cat == "Title" or "Title" in class_name
        if is_title and len(text) <= 160:
            norm_lower = text.strip().lower()
            matched = False
            for sec in paper_sections:
                if norm_lower == sec or norm_lower.startswith(sec):
                    current_heading = sec.title()
                    matched = True
                    break
            if not matched:
                current_heading = text.strip()

            # Try to get heading level for a proper stack
            level = _heading_level_from_element(el)
            if level is not None:
                # Remove all levels >= this one, then set
                heading_stack = {k: v for k, v in heading_stack.items() if k < level}
                heading_stack[level] = current_heading or text.strip()
            else:
                # No level info — treat as flat heading (reset stack to just this)
                heading_stack = {1: current_heading or text.strip()}

        if len(text) < min_text_len:
            continue

        para_idx += 1
        # Build heading_path from sorted stack levels
        if heading_stack:
            heading_path = " > ".join(heading_stack[k] for k in sorted(heading_stack))
        else:
            heading_path = current_heading
        doc_lang = manifest_lang or detect_language(text)

        rec: dict[str, Any] = {
            "source_id": source_id,
            "doc_type": "docx",
            "title": source.get("title", ""),
            "author": source.get("author", ""),
            "year": source.get("year"),
            "file_path": file_path,
            "url": source.get("url", ""),
            "page": None,
            "page_range": None,
            "section_heading": current_heading,
            "heading_path": heading_path,
            "span": None,
            "locator": f"para_{para_idx}",
            "doc_fingerprint": doc_fp,
            "doc_language": doc_lang,
            "text": text,
            "element_idx": elem_idx,
            "para_idx": para_idx,
        }
        for key in ("doi", "journal"):
            if key in source:
                rec[key] = source[key]

        records.append(rec)

    logger.info("[DOCX] %s — %d passages, %d headings detected",
                source_id, len(records), len({r["section_heading"] for r in records if r.get("section_heading")}))
    return records


# ---------------------------------------------------------------------------
# IMAGE ingestion (OCR)
# ---------------------------------------------------------------------------

def _ingest_image(
    source: dict[str, Any],
    cfg_ingest: dict[str, Any],
) -> list[dict[str, Any]]:
    """Ingest a single image via OCR (pytesseract)."""
    from PIL import Image  # type: ignore
    import pytesseract  # type: ignore

    file_path = source["file_path"]
    source_id = source["source_id"]
    ocr_cfg = cfg_ingest.get("ocr", {})
    ocr_lang = ocr_cfg.get("lang", "vie")
    min_conf = float(ocr_cfg.get("min_confidence", 0.55))
    min_text_len = int(cfg_ingest.get("min_text_len_image", 5))
    doc_fp = sha1_fingerprint(file_path)
    manifest_lang = source.get("doc_language", "vi")

    try:
        img = Image.open(file_path)  # type: ignore
        image_to_data = cast(Any, getattr(pytesseract, "image_to_data", None))
        output_type_dict = cast(Any, getattr(getattr(pytesseract, "Output", None), "DICT", None))
        if image_to_data is None or output_type_dict is None:
            logger.error("pytesseract API unavailable for %s", file_path)
            return []
        ocr_data = cast(dict[str, list[Any]], image_to_data(img, lang=ocr_lang, output_type=output_type_dict))
    except Exception as exc:
        logger.error("OCR failed for %s: %s", file_path, exc)
        return []

    records: list[dict[str, Any]] = []
    n_boxes = len(ocr_data.get("text", []))  # type: ignore
    low_conf_skipped = 0

    for i in range(n_boxes):
        text = (ocr_data["text"][i] or "").strip()  # type: ignore
        if not text:
            continue

        conf_raw = ocr_data["conf"][i]  # type: ignore
        try:
            conf = float(conf_raw) / 100.0  # type: ignore
        except (ValueError, TypeError):
            conf = 0.0

        if conf < min_conf:
            low_conf_skipped += 1
            continue

        if len(text) < min_text_len:
            continue

        x1 = ocr_data["left"][i]  # type: ignore
        y1 = ocr_data["top"][i]  # type: ignore
        x2 = x1 + ocr_data["width"][i]  # type: ignore
        y2 = y1 + ocr_data["height"][i]  # type: ignore

        rec: dict[str, Any] = {
            "source_id": source_id,
            "doc_type": "image",
            "title": source.get("title", ""),
            "author": source.get("author", ""),
            "year": source.get("year"),
            "file_path": file_path,
            "url": source.get("url", ""),
            "page": None,
            "page_range": None,
            "section_heading": None,
            "span": None,
            "locator": f"{source_id}_bbox_{x1}_{y1}_{x2}_{y2}",
            "doc_fingerprint": doc_fp,
            "doc_language": manifest_lang,
            "text": text,
            "bbox": [x1, y1, x2, y2],
            "ocr_engine": "tesseract",
            "ocr_confidence": round(conf, 4),
        }
        for key in ("doi", "journal"):
            if key in source:
                rec[key] = source[key]

        records.append(rec)

    logger.info(
        "[IMAGE] %s — %d boxes kept, %d low-conf skipped",
        source_id, len(records), low_conf_skipped,
    )
    return records


# ---------------------------------------------------------------------------
# TCVN3 garbled-text detection
# ---------------------------------------------------------------------------

# Characters that appear when a TCVN3-font PDF is extracted as Latin-1.
_TCVN3_MARKERS = frozenset(chr(cp) for cp in [
    0xA7, 0xA8, 0xAB, 0xAC, 0xAE,
    0xB5, 0xB6, 0xB7, 0xB8, 0xB9, 0xBB, 0xBE,
    0xC6, 0xC7, 0xC8, 0xC9, 0xCA, 0xCB, 0xCC, 0xCE, 0xCF,
    0xD0, 0xD1, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xD7, 0xD8,
    0xDC, 0xDD, 0xDE, 0xDF,
    0xE4, 0xE5, 0xE6, 0xE7, 0xE8, 0xE9, 0xEA, 0xEB, 0xEE, 0xEF,
    0xF1, 0xF5, 0xF6, 0xF7, 0xF8, 0xF9, 0xFA, 0xFB, 0xFE,
])
_TCVN3_STRONG = frozenset(chr(cp) for cp in [0xAE, 0xB8, 0xB5])  # ® ¸ µ


def _is_tcvn3_garbled(text: str, ratio_threshold: float = 0.02, strong_threshold: int = 10) -> bool:
    """Return True if *text* looks like TCVN3 garbled content (not proper Unicode)."""
    if not text:
        return False
    marker_count = sum(1 for ch in text if ch in _TCVN3_MARKERS)
    strong_count = sum(1 for ch in text if ch in _TCVN3_STRONG)
    ratio = marker_count / len(text)
    return ratio > ratio_threshold or strong_count > strong_threshold


# ---------------------------------------------------------------------------
# TXT ingestion
# ---------------------------------------------------------------------------

def _ingest_txt(
    source: dict[str, Any],
    cfg_ingest: dict[str, Any],
) -> list[dict[str, Any]]:
    """Ingest a pre-normalized TXT file. Each blank-line-separated paragraph = one passage."""
    file_path = source["file_path"]
    source_id = source["source_id"]
    min_text_len = int(cfg_ingest.get("min_text_len_txt", cfg_ingest.get("min_text_len_pdf", 30)))
    doc_fp = sha1_fingerprint(file_path)
    manifest_lang = source.get("doc_language")

    try:
        raw = Path(file_path).read_text(encoding="utf-8")
    except Exception as exc:
        logger.error("Cannot read TXT %s: %s", file_path, exc)
        return []

    if _is_tcvn3_garbled(raw):
        logger.warning("[TXT] SKIP (TCVN3 garbled): %s", file_path)
        return []

    # Split on blank lines to get paragraphs
    blocks = [b.strip() for b in raw.split("\n\n")]
    records: list[dict[str, Any]] = []
    para_idx = 0

    for block in blocks:
        text = " ".join(block.split())  # collapse inner newlines
        if len(text) < min_text_len:
            continue
        para_idx += 1
        doc_lang = manifest_lang or detect_language(text)
        rec: dict[str, Any] = {
            "source_id": source_id,
            "doc_type": "txt",
            "title": source.get("title", ""),
            "author": source.get("author", ""),
            "year": source.get("year"),
            "file_path": file_path,
            "url": source.get("url", ""),
            "page": None,
            "page_range": None,
            "section_heading": None,
            "span": None,
            "locator": f"para_{para_idx}",
            "doc_fingerprint": doc_fp,
            "doc_language": doc_lang,
            "text": text,
            "element_idx": para_idx,
        }
        for key in ("doi", "journal"):
            if key in source:
                rec[key] = source[key]
        records.append(rec)

    logger.info("[TXT] %s — %d paragraphs kept", source_id, len(records))
    return records


# ---------------------------------------------------------------------------
# Main ingest driver
# ---------------------------------------------------------------------------

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def _resolve_doc_type(source: dict[str, Any]) -> str:
    """Determine doc_type from manifest or file extension."""
    doc_type = str(source.get("doc_type", "")).lower()
    if doc_type in ("pdf", "docx", "image", "txt"):
        return doc_type
    ext = Path(source["file_path"]).suffix.lower()
    if ext == ".pdf":
        return "pdf"
    if ext == ".docx":
        return "docx"
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext == ".txt":
        return "txt"
    return doc_type


def run_ingest(config: dict[str, Any]) -> int:
    """Run B1 ingest for all sources in the manifest. Returns record count."""
    import os
    cfg_ingest = config["ingest"]
    manifest_path = os.environ.get("SOURCES_YAML") or cfg_ingest["sources_manifest"]
    output_path = cfg_ingest["output_jsonl"]
    allowed_ext = set(cfg_ingest.get("allowed_ext", [".pdf", ".docx", ".png", ".jpg", ".jpeg"]))

    paper_sections_list = cfg_ingest.get("paper_sections", [])
    paper_sections = {s.lower() for s in paper_sections_list}

    # Load manifest
    if not Path(manifest_path).exists():
        logger.warning("Manifest file not found: %s — nothing to ingest.", manifest_path)
        return 0

    with open(manifest_path, encoding="utf-8") as f:
        manifest = yaml.safe_load(f) or {}  # type: ignore
    sources: list[dict[str, Any]] = manifest.get("sources", [])  # type: ignore

    if not sources:
        logger.warning("No sources in manifest %s", manifest_path)
        return 0

    all_records: list[dict[str, Any]] = []

    for src in tqdm(sources, desc="Ingesting"):  # type: ignore
        file_path = src.get("file_path", "")
        if not Path(file_path).exists():
            logger.warning("File not found, skipping: %s", file_path)
            continue
        ext = Path(file_path).suffix.lower()
        if ext not in allowed_ext:
            logger.warning("Extension %s not allowed, skipping: %s", ext, file_path)
            continue

        doc_type = _resolve_doc_type(src)
        try:
            if doc_type == "pdf":
                recs = _ingest_pdf(src, cfg_ingest, paper_sections)
            elif doc_type == "docx":
                recs = _ingest_docx(src, cfg_ingest, paper_sections)
            elif doc_type == "image":
                recs = _ingest_image(src, cfg_ingest)
            elif doc_type == "txt":
                recs = _ingest_txt(src, cfg_ingest)
            else:
                logger.warning("Unknown doc_type '%s' for %s", doc_type, file_path)
                continue
            all_records.extend(recs)
        except Exception as exc:
            logger.error("Error processing %s: %s", file_path, exc, exc_info=True)

    # Write output
    ensure_parent_dir(output_path)
    write_jsonl(all_records, output_path)
    logger.info("B1 Ingest complete: %d total passages → %s", len(all_records), output_path)

    # doc_type + source_id counts
    type_counts: Counter[str] = Counter(r["doc_type"] for r in all_records)
    src_counts: Counter[str] = Counter(r["source_id"] for r in all_records)
    logger.info("  doc_type counts: %s", dict(type_counts))
    for sid, cnt in src_counts.most_common():
        logger.info("  source %-30s : %d passages", sid, cnt)
    return len(all_records)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="B1 — Ingest documents")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--output", default=None, help="Override output JSONL path")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.output:
        config["ingest"]["output_jsonl"] = args.output

    run_ingest(config)


if __name__ == "__main__":
    main()
