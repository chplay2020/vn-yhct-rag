"""B3 — Chunking by structure → chunks_v1.jsonl

Usage:
    python -m rag.chunk.chunk_by_structure --config config/config.yaml
"""

from __future__ import annotations

import argparse
import logging
import re
from collections import defaultdict
from typing import Any

import tiktoken  # type: ignore
import yaml  # type: ignore

from rag.utils.hashing import sha1_short
from rag.utils.io import read_jsonl, write_jsonl, ensure_parent_dir

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")


# ---------------------------------------------------------------------------
# Grouping helpers
# ---------------------------------------------------------------------------

def _group_key(rec: dict[str, Any]) -> tuple[str, str]:
    """Return (source_id, grouping_field)."""
    source_id = rec.get("source_id", "")
    doc_type = rec.get("doc_type", "")
    if doc_type == "docx":
        gk = rec.get("heading_path") or rec.get("section_heading") or "__no_heading__"
    else:
        gk = rec.get("section_heading") or "__no_heading__"
    return (source_id, gk)


def _sort_key(rec: dict[str, Any]) -> tuple[str, int, str]:
    """Sort key: (source_id, element_idx, locator)."""
    return (
        rec.get("source_id", ""),
        rec.get("element_idx", 0) or 0,
        rec.get("locator", ""),
    )


# ---------------------------------------------------------------------------
# DOCX paragraph-level chunking
# ---------------------------------------------------------------------------

def _chunk_docx_paragraphs(
    group_recs: list[dict[str, Any]],
    chunk_size: int,
    overlap: int,
    enc: Any,
) -> list[dict[str, Any]]:
    """Chunk DOCX paragraphs by accumulating para records up to chunk_size tokens.

    Returns list of chunk dicts with correct para_idx_min/max, element_idx_min/max,
    locator_context per chunk.
    """
    # sort by para_idx ascending
    group_recs.sort(key=lambda r: (r.get("para_idx") or 0, r.get("element_idx") or 0))

    first = group_recs[0]
    chunks: list[dict[str, Any]] = []

    # current accumulator
    current_paras: list[dict[str, Any]] = []
    current_tokens = 0

    def _token_len(text: str) -> int:
        return len(enc.encode(text))  # type: ignore

    def _finalize_chunk(paras: list[dict[str, Any]]) -> dict[str, Any]:
        """Build one chunk dict from accumulated paragraphs."""
        chunk_text = "\n\n".join(p["text"] for p in paras)
        para_idxs: list[int] = [p["para_idx"] for p in paras if p.get("para_idx") is not None]
        elem_idxs: list[int] = [p["element_idx"] for p in paras if p.get("element_idx") is not None]

        pi_min: int | None = min(para_idxs) if para_idxs else None
        pi_max: int | None = max(para_idxs) if para_idxs else None
        ei_min: int | None = min(elem_idxs) if elem_idxs else None
        ei_max: int | None = max(elem_idxs) if elem_idxs else None

        if pi_min is not None and pi_max is not None:
            if pi_min == pi_max:
                loc_ctx = f"para_{pi_min}"
            else:
                loc_ctx = f"para_{pi_min}-{pi_max}"
        else:
            loc_ctx = None

        chunk_id = f"{first['source_id']}:{sha1_short(chunk_text)}"
        return {
            "chunk_id": chunk_id,
            "text": chunk_text,
            "source_id": first.get("source_id"),
            "title": first.get("title"),
            "author": first.get("author"),
            "year": first.get("year"),
            "file_path": first.get("file_path"),
            "url": first.get("url"),
            "doc_type": "docx",
            "doc_language": first.get("doc_language"),
            "section_heading": first.get("section_heading"),
            "heading_path": first.get("heading_path"),
            "page_range": None,
            "doc_fingerprint": first.get("doc_fingerprint"),
            "locator_context": loc_ctx,
            "element_idx_min": ei_min,
            "element_idx_max": ei_max,
            "para_idx_min": pi_min,
            "para_idx_max": pi_max,
            "span": None,
        }

    for rec in group_recs:
        rec_tokens = _token_len(rec["text"])

        # If adding this para exceeds chunk_size, finalize current chunk first
        if current_paras and (current_tokens + rec_tokens) > chunk_size:
            chunks.append(_finalize_chunk(current_paras))

            # Overlap: keep trailing paragraphs whose total tokens ~ overlap
            overlap_paras: list[dict[str, Any]] = []
            overlap_tokens = 0
            for p in reversed(current_paras):
                pt = _token_len(p["text"])
                if overlap_tokens + pt > overlap:
                    break
                overlap_paras.insert(0, p)
                overlap_tokens += pt

            current_paras = overlap_paras
            current_tokens = overlap_tokens

        current_paras.append(rec)
        current_tokens += rec_tokens

    # Finalize last chunk
    if current_paras:
        chunks.append(_finalize_chunk(current_paras))

    return chunks


# ---------------------------------------------------------------------------
# PDF page-aware chunking
# ---------------------------------------------------------------------------

_RE_PAGE_NUM_LINE = re.compile(r"^\s*\d{1,4}\s*$")


def _strip_page_artifacts(text: str) -> str:
    """Lightly strip page-number-only lines and excessive blank lines."""
    lines = text.split("\n")
    cleaned: list[str] = []
    for ln in lines:
        # Drop lines that are only a page number (1-4 digits)
        if _RE_PAGE_NUM_LINE.match(ln):
            continue
        cleaned.append(ln)
    # Collapse runs of 3+ blank lines down to 2
    result: list[str] = []
    blank_count = 0
    for ln in cleaned:
        if ln.strip() == "":
            blank_count += 1
            if blank_count <= 2:
                result.append(ln)
        else:
            blank_count = 0
            result.append(ln)
    return "\n".join(result).strip()


def _chunk_pdf_pages(
    group_recs: list[dict[str, Any]],
    chunk_size: int,
    overlap: int,
    enc: Any,
) -> list[dict[str, Any]]:
    """Chunk PDF page records by accumulating pages up to *chunk_size* tokens.

    Each chunk tracks its own page_range and locator_context.
    """
    # Sort by page ascending
    group_recs.sort(key=lambda r: (r.get("page") or 0))

    first = group_recs[0]
    chunks: list[dict[str, Any]] = []

    current_pages: list[dict[str, Any]] = []
    current_tokens = 0

    def _token_len(text: str) -> int:
        return len(enc.encode(text))  # type: ignore

    def _finalize_chunk(pages: list[dict[str, Any]]) -> dict[str, Any]:
        """Build one chunk dict from accumulated page records."""
        chunk_text = "\n\n".join(_strip_page_artifacts(p["text"]) for p in pages)
        page_nos: list[int] = [p["page"] for p in pages if p.get("page") is not None]

        p_min: int | None = min(page_nos) if page_nos else None
        p_max: int | None = max(page_nos) if page_nos else None

        if p_min is not None and p_max is not None:
            if p_min == p_max:
                pr = str(p_min)
                loc_ctx = f"p{p_min}"
            else:
                pr = f"{p_min}-{p_max}"
                loc_ctx = f"p{p_min}-{p_max}"
        else:
            pr = None
            loc_ctx = None

        chunk_id = f"{first['source_id']}:{sha1_short(chunk_text)}"
        return {
            "chunk_id": chunk_id,
            "text": chunk_text,
            "source_id": first.get("source_id"),
            "title": first.get("title"),
            "author": first.get("author"),
            "year": first.get("year"),
            "file_path": first.get("file_path"),
            "url": first.get("url"),
            "doc_type": "pdf",
            "doc_language": first.get("doc_language"),
            "section_heading": None,
            "heading_path": None,
            "page_range": pr,
            "doc_fingerprint": first.get("doc_fingerprint"),
            "locator_context": loc_ctx,
            "element_idx_min": p_min,
            "element_idx_max": p_max,
            "span": None,
        }

    for rec in group_recs:
        rec_tokens = _token_len(rec["text"])

        # If adding this page exceeds chunk_size, finalize current chunk first
        if current_pages and (current_tokens + rec_tokens) > chunk_size:
            chunks.append(_finalize_chunk(current_pages))

            # Overlap: keep trailing pages whose total tokens ~ overlap
            overlap_pages: list[dict[str, Any]] = []
            overlap_tokens = 0
            for p in reversed(current_pages):
                pt = _token_len(p["text"])
                if overlap_tokens + pt > overlap:
                    break
                overlap_pages.insert(0, p)
                overlap_tokens += pt

            current_pages = overlap_pages
            current_tokens = overlap_tokens

        current_pages.append(rec)
        current_tokens += rec_tokens

    # Finalize last chunk
    if current_pages:
        chunks.append(_finalize_chunk(current_pages))

    return chunks


# ---------------------------------------------------------------------------
# Main chunking driver
# ---------------------------------------------------------------------------

def run_chunk(config: dict[str, Any]) -> None:
    """Run B3 chunking."""
    input_path = config["clean"]["output_jsonl"]
    output_path = config["index"]["input_chunks"]
    chunk_cfg = config["chunking"]
    chunk_size: int = chunk_cfg.get("chunk_size", 500)
    overlap: int = chunk_cfg.get("overlap", 100)

    records = read_jsonl(input_path)
    if not records:
        logger.warning("No records to chunk from %s", input_path)
        return

    # Sort
    records.sort(key=_sort_key)

    enc = tiktoken.get_encoding("cl100k_base")  # type: ignore
    chunks: list[dict[str, Any]] = []

    # Separate by doc_type
    image_records = [r for r in records if r.get("doc_type") == "image"]
    docx_records = [r for r in records if r.get("doc_type") == "docx"]
    pdf_records = [r for r in records if r.get("doc_type") == "pdf"]

    # --- Image chunks: 1:1 ---
    for rec in image_records:
        chunk_id = f"{rec['source_id']}:{sha1_short(rec['text'])}"
        chunk: dict[str, Any] = {
            "chunk_id": chunk_id,
            "text": rec["text"],
            "source_id": rec.get("source_id"),
            "title": rec.get("title"),
            "author": rec.get("author"),
            "year": rec.get("year"),
            "file_path": rec.get("file_path"),
            "url": rec.get("url"),
            "doc_type": "image",
            "doc_language": rec.get("doc_language"),
            "section_heading": None,
            "heading_path": None,
            "page_range": None,
            "doc_fingerprint": rec.get("doc_fingerprint"),
            "locator_context": rec.get("locator"),
            "bbox": rec.get("bbox"),
            "ocr_confidence": rec.get("ocr_confidence"),
        }
        chunks.append(chunk)

    # --- DOCX: paragraph-level chunking per group ---
    docx_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for rec in docx_records:
        docx_groups[_group_key(rec)].append(rec)

    for _, group_recs in docx_groups.items():
        docx_chunks = _chunk_docx_paragraphs(group_recs, chunk_size, overlap, enc)
        chunks.extend(docx_chunks)

    # --- PDF: page-aware chunking (accumulate pages up to chunk_size) ---
    pdf_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in pdf_records:
        pdf_groups[rec.get("source_id", "")].append(rec)

    for _, group_recs in pdf_groups.items():
        pdf_chunks = _chunk_pdf_pages(group_recs, chunk_size, overlap, enc)
        chunks.extend(pdf_chunks)

    ensure_parent_dir(output_path)
    write_jsonl(chunks, output_path)

    # --- Validation stats ---
    logger.info("B3 Chunk complete: %d total chunks → %s", len(chunks), output_path)

    # Count replacement chars remaining
    fffd_count = sum(1 for c in chunks if "\ufffd" in c.get("text", ""))
    logger.info("  Chunks with U+FFFD '�': %d / %d", fffd_count, len(chunks))

    # Show 3 example DOCX chunks with different para ranges
    docx_chunks_out = [c for c in chunks if c.get("doc_type") == "docx"]
    for i, ex in enumerate(docx_chunks_out[:3]):
        logger.info(
            "  Example DOCX chunk %d: chunk_id=%s  locator=%s  "
            "para_idx=%s-%s  elem_idx=%s-%s",
            i + 1,
            ex.get("chunk_id", "?"),
            ex.get("locator_context"),
            ex.get("para_idx_min"),
            ex.get("para_idx_max"),
            ex.get("element_idx_min"),
            ex.get("element_idx_max"),
        )

    # Show 3 example PDF chunks with page ranges
    pdf_chunks_out = [c for c in chunks if c.get("doc_type") == "pdf"]
    logger.info("  PDF chunks: %d", len(pdf_chunks_out))
    for i, ex in enumerate(pdf_chunks_out[:3]):
        logger.info(
            "  Example PDF chunk %d: chunk_id=%s  page_range=%s  "
            "locator=%s  elem_idx=%s-%s",
            i + 1,
            ex.get("chunk_id", "?"),
            ex.get("page_range"),
            ex.get("locator_context"),
            ex.get("element_idx_min"),
            ex.get("element_idx_max"),
        )

    # Verify no duplicate page_range across PDF chunks per source
    pdf_page_ranges = [c.get("page_range") for c in pdf_chunks_out]
    unique_ranges = len(set(pdf_page_ranges))
    logger.info("  PDF unique page_ranges: %d / %d chunks", unique_ranges, len(pdf_chunks_out))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="B3 — Chunk by structure")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    run_chunk(config)


if __name__ == "__main__":
    main()
