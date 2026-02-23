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
# PDF Strategy B: entry-based chunking
# ---------------------------------------------------------------------------

# Markers that start a new "entry" (numbered item or bullet)
_RE_ENTRY_START = re.compile(
    r"^(?:"
    r"\d{1,3}\s*[-–—.)\]]\s"    # "1- ", "2. ", "3) "
    r"|•\s"                      # bullet "• "
    r"|[A-Z][A-Z\s]{3,}:"       # all-caps field header like "TÊN PHỔ THÔNG:"
    r")",
    re.MULTILINE,
)

# Field headers that indicate structured content within an entry
_RE_FIELD_HEADER = re.compile(
    r"^(?:"
    r"Tên phổ thông|Tên khoa học|Bộ phận dùng|Tác dụng|Cách dùng"
    r"|Kiêng kỵ|Nơi thu thập|Tên địa phương|Tên khác|Mô tả"
    r"|Công dụng|Thành phần|Phân bố|Thu hái|Chế biến"
    r")\s*:",
    re.MULTILINE | re.IGNORECASE,
)


def _has_entry_markers(text: str) -> bool:
    """Check if concatenated text from a source has enough entry markers
    to warrant strategy-B splitting (at least 3 markers)."""
    return len(_RE_ENTRY_START.findall(text)) >= 3


def _split_into_entries(full_text: str) -> list[str]:
    """Split text into entries using entry-start markers.

    Each entry starts at a marker match and ends just before the next marker.
    """
    positions = [m.start() for m in _RE_ENTRY_START.finditer(full_text)]
    if not positions:
        return [full_text]

    entries: list[str] = []
    # Text before first entry marker is a preamble
    if positions[0] > 0:
        preamble = full_text[:positions[0]].strip()
        if preamble:
            entries.append(preamble)

    for i, start in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(full_text)
        entry_text = full_text[start:end].strip()
        if entry_text:
            entries.append(entry_text)

    return entries


def _chunk_pdf_entries(
    group_recs: list[dict[str, Any]],
    chunk_size: int,
    overlap: int,
    enc: Any,
) -> list[dict[str, Any]]:
    """Chunk PDF by entry markers (strategy B).

    Steps:
    1. Concatenate all pages (preserving page boundaries).
    2. Split by entry markers.
    3. Accumulate entries up to chunk_size tokens.
    """
    group_recs.sort(key=lambda r: (r.get("page") or 0))
    first = group_recs[0]

    def _token_len(text: str) -> int:
        return len(enc.encode(text))  # type: ignore

    # Build page-boundary map: (char_offset -> page_no)
    page_boundaries: list[tuple[int, int]] = []  # (start_char, page_no)
    full_parts: list[str] = []
    offset = 0
    for rec in group_recs:
        page_text = _strip_page_artifacts(rec["text"])
        page_no = rec.get("page") or 0
        page_boundaries.append((offset, page_no))
        full_parts.append(page_text)
        offset += len(page_text) + 2  # +2 for "\n\n" separator

    full_text = "\n\n".join(full_parts)
    entries = _split_into_entries(full_text)

    def _page_for_offset(char_pos: int) -> int:
        """Find which page a character offset belongs to."""
        result_page = page_boundaries[0][1] if page_boundaries else 1
        for boundary_offset, page_no in page_boundaries:
            if char_pos >= boundary_offset:
                result_page = page_no
            else:
                break
        return result_page

    # Build entry records with page info
    entry_records: list[dict[str, str | int]] = []
    search_start = 0
    for entry_text in entries:
        idx = full_text.find(entry_text, search_start)
        if idx < 0:
            idx = search_start
        page_start = _page_for_offset(idx)
        page_end = _page_for_offset(idx + len(entry_text))
        entry_records.append({
            "text": entry_text,
            "page_start": page_start,
            "page_end": page_end,
        })
        search_start = idx + len(entry_text)

    # Accumulate entries into chunks
    chunks: list[dict[str, Any]] = []
    current_entries: list[dict[str, str | int]] = []
    current_tokens = 0

    def _finalize(entries_acc: list[dict[str, str | int]]) -> dict[str, Any]:
        chunk_text = "\n\n".join(str(e["text"]) for e in entries_acc)
        pages: set[int] = set()
        for e in entries_acc:
            for p in range(int(e["page_start"]), int(e["page_end"]) + 1):
                pages.add(p)
        p_min: int | None = min(pages) if pages else None
        p_max: int | None = max(pages) if pages else None

        if p_min is not None and p_max is not None:
            pr = str(p_min) if p_min == p_max else f"{p_min}-{p_max}"
            loc = f"p{p_min}" if p_min == p_max else f"p{p_min}-{p_max}"
        else:
            pr, loc = None, None

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
            "locator_context": loc,
            "element_idx_min": p_min,
            "element_idx_max": p_max,
            "span": None,
            "chunk_strategy": "entry",
        }

    for erec in entry_records:
        et = _token_len(str(erec["text"]))
        if current_entries and (current_tokens + et) > chunk_size:
            chunks.append(_finalize(current_entries))
            # Overlap: keep last entry if within overlap budget
            overlap_entries: list[dict[str, str | int]] = []
            overlap_tokens = 0
            for e in reversed(current_entries):
                esize = _token_len(str(e["text"]))
                if overlap_tokens + esize > overlap:
                    break
                overlap_entries.insert(0, e)
                overlap_tokens += esize
            current_entries = overlap_entries
            current_tokens = overlap_tokens

        current_entries.append(erec)
        current_tokens += et

    if current_entries:
        chunks.append(_finalize(current_entries))

    return chunks


# ---------------------------------------------------------------------------
# Main chunking driver
# ---------------------------------------------------------------------------

def run_chunk(config: dict[str, Any]) -> int:
    """Run B3 chunking. Returns chunk count."""
    input_path = config["clean"]["output_jsonl"]
    output_path = config["index"]["input_chunks"]
    chunk_cfg = config["chunking"]
    chunk_size: int = chunk_cfg.get("chunk_size", 500)
    overlap: int = chunk_cfg.get("overlap", 100)

    records = read_jsonl(input_path)
    if not records:
        logger.warning("No records to chunk from %s", input_path)
        return 0

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

    # --- PDF: auto-select strategy (B=entry-based or A=page-accumulate) ---
    pdf_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in pdf_records:
        pdf_groups[rec.get("source_id", "")].append(rec)

    entry_strategy_count = 0
    page_strategy_count = 0
    for _sid, group_recs in pdf_groups.items():
        # Probe: concatenate first few pages to check for entry markers
        probe_text = "\n\n".join(r["text"] for r in sorted(group_recs, key=lambda r: r.get("page") or 0)[:10])
        if _has_entry_markers(probe_text):
            pdf_chunks = _chunk_pdf_entries(group_recs, chunk_size, overlap, enc)
            entry_strategy_count += 1
        else:
            pdf_chunks = _chunk_pdf_pages(group_recs, chunk_size, overlap, enc)
            page_strategy_count += 1
        chunks.extend(pdf_chunks)

    logger.info("  PDF strategy: %d sources entry-based (B), %d page-accum (A)",
                entry_strategy_count, page_strategy_count)

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
    return len(chunks)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="B3 — Chunk by structure")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--input", default=None, help="Override input JSONL path")
    parser.add_argument("--output", default=None, help="Override output JSONL path")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.input:
        config["clean"]["output_jsonl"] = args.input
    if args.output:
        config["index"]["input_chunks"] = args.output

    run_chunk(config)


if __name__ == "__main__":
    main()
