"""B3 — Chunking by structure → chunks_v2.jsonl (Parent–Child v2)

Fixes "parent quá to" by:
  - DOCX: sub-blocking by MAX_PARENT_TOKENS per heading group
  - PDF entry-strategy: each entry = 1 parent
  - PDF page-strategy: each PAGES_PER_PARENT pages = 1 parent
  - child_index resets per parent

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
# Category extraction
# ---------------------------------------------------------------------------

def parse_category(file_path: str | None) -> str:
    """Extract category from file_path.

    Expected layout: ``…/TT_YHCT/<CATEGORY>/…``
    Returns *CATEGORY* or ``"unknown"``.
    """
    if not file_path:
        return "unknown"
    s = file_path.replace("\\", "/")
    m = re.search(r"/TT_YHCT/([^/]+)/", s)
    return m.group(1) if m else "unknown"


# ---------------------------------------------------------------------------
# Parent-size constants
# ---------------------------------------------------------------------------

MAX_PARENT_TOKENS: int = 2000   # DOCX: max tokens per parent block
PAGES_PER_PARENT: int = 2       # PDF page strategy: pages per parent
HARD_CAP_CHILDREN: int = 80     # warn if any parent exceeds this


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
# Token-safe splitting
# ---------------------------------------------------------------------------

def _token_len(text: str, enc: Any) -> int:
    return len(enc.encode(text))  # type: ignore


def _split_text_by_tokens(text: str, max_tokens: int, enc: Any) -> list[str]:
    """Split text into parts, each <= max_tokens tokens (hard cap)."""
    if max_tokens <= 0:
        return [text]
    toks = enc.encode(text)  # type: ignore
    if len(toks) <= max_tokens:
        return [text]
    parts: list[str] = []
    for i in range(0, len(toks), max_tokens):
        piece = enc.decode(toks[i:i + max_tokens])  # type: ignore
        piece = piece.strip()
        if piece:
            parts.append(piece)
    return parts if parts else [text]


def _split_text_children(
    text: str,
    chunk_size: int,
    overlap: int,
    enc: Any,
) -> list[str]:
    """Split *text* into overlapping children of ~chunk_size tokens.

    Respects paragraph boundaries (``\\n\\n``) when possible.
    Falls back to token-level splitting for oversized paragraphs.
    Returns at least one non-empty string.
    """
    if not text or not text.strip():
        return [text] if text else [""]

    paragraphs = re.split(r"\n\n+", text)
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    if not paragraphs:
        return [text.strip()] if text.strip() else [""]

    # expand oversized paragraphs
    expanded: list[str] = []
    for p in paragraphs:
        if _token_len(p, enc) > chunk_size:
            expanded.extend(_split_text_by_tokens(p, chunk_size, enc))
        else:
            expanded.append(p)

    children: list[str] = []
    current: list[str] = []
    current_tokens = 0

    for para in expanded:
        pt = _token_len(para, enc)
        if current and (current_tokens + pt) > chunk_size:
            children.append("\n\n".join(current))
            # overlap: keep trailing parts
            overlap_parts: list[str] = []
            overlap_tokens = 0
            for prev in reversed(current):
                ptt = _token_len(prev, enc)
                if overlap_tokens + ptt > overlap:
                    break
                overlap_parts.insert(0, prev)
                overlap_tokens += ptt
            current = overlap_parts
            current_tokens = overlap_tokens

        current.append(para)
        current_tokens += pt

    if current:
        children.append("\n\n".join(current))

    return children if children else [text.strip()]


# ---------------------------------------------------------------------------
# PDF helpers
# ---------------------------------------------------------------------------

_RE_PAGE_NUM_LINE = re.compile(r"^\s*\d{1,4}\s*$")


def _strip_page_artifacts(text: str) -> str:
    """Lightly strip page-number-only lines and excessive blank lines."""
    lines = text.split("\n")
    cleaned: list[str] = []
    for ln in lines:
        if _RE_PAGE_NUM_LINE.match(ln):
            continue
        cleaned.append(ln)

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


_RE_ENTRY_START = re.compile(
    r"^(?:"
    r"\d{1,3}\s*[-\u2013\u2014.)\]]\s"
    r"|\u2022\s"
    r"|[A-Z][A-Z\s]{3,}:"
    r")",
    re.MULTILINE,
)

_RE_FIELD_HEADER = re.compile(
    r"^(?:"
    r"T\u00ean ph\u1ed5 th\u00f4ng|T\u00ean khoa h\u1ecdc|B\u1ed9 ph\u1eadn d\u00f9ng|T\u00e1c d\u1ee5ng|C\u00e1ch d\u00f9ng"
    r"|Ki\u00eang k\u1ef5|N\u01a1i thu th\u1eadp|T\u00ean \u0111\u1ecba ph\u01b0\u01a1ng|T\u00ean kh\u00e1c|M\u00f4 t\u1ea3"
    r"|C\u00f4ng d\u1ee5ng|Th\u00e0nh ph\u1ea7n|Ph\u00e2n b\u1ed1|Thu h\u00e1i|Ch\u1ebf bi\u1ebfn"
    r")\s*:",
    re.MULTILINE | re.IGNORECASE,
)


def _has_entry_markers(text: str) -> bool:
    return len(_RE_ENTRY_START.findall(text)) >= 3


def _split_into_entries(full_text: str) -> list[str]:
    positions = [m.start() for m in _RE_ENTRY_START.finditer(full_text)]
    if not positions:
        return [full_text]

    entries: list[str] = []
    if positions[0] > 0:
        preamble = full_text[: positions[0]].strip()
        if preamble:
            entries.append(preamble)

    for i, start in enumerate(positions):
        end = positions[i + 1] if i + 1 < len(positions) else len(full_text)
        entry_text = full_text[start:end].strip()
        if entry_text:
            entries.append(entry_text)

    return entries


# ---------------------------------------------------------------------------
# DOCX: paragraph-level chunking with sub-parent blocking
# ---------------------------------------------------------------------------

def _chunk_docx_group(
    group_recs: list[dict[str, Any]],
    chunk_size: int,
    overlap: int,
    max_parent_tokens: int,
    enc: Any,
    doc_type: str = "docx",
) -> list[dict[str, Any]]:
    """Chunk DOCX/TXT paragraph group into parent blocks then children.

    1. Expand oversized paragraphs.
    2. Block paragraphs by *max_parent_tokens* -> N parent blocks.
    3. For each block, split into child chunks (chunk_size / overlap).
    4. Set parent_id (with block_idx if N > 1), child_index.
    """
    # -- expand long paragraphs --
    expanded: list[dict[str, Any]] = []
    for rec in group_recs:
        txt = rec.get("text", "")
        nt = _token_len(txt, enc)
        if nt > chunk_size:
            parts = _split_text_by_tokens(txt, chunk_size, enc)
            for part in parts:
                r2 = dict(rec)
                r2["text"] = part
                expanded.append(r2)
        else:
            expanded.append(dict(rec))

    expanded.sort(key=lambda r: (
        r.get("para_idx") or 0,
        r.get("element_idx") or 0,
    ))

    if not expanded:
        return []

    first = expanded[0]
    source_id = first.get("source_id", "")
    heading_path = (
        first.get("heading_path")
        or first.get("section_heading")
        or "__no_heading__"
    )
    heading_hash = sha1_short(heading_path)

    # -- build parent blocks by token budget --
    blocks: list[list[dict[str, Any]]] = []
    cur_block: list[dict[str, Any]] = []
    cur_tok = 0

    for rec in expanded:
        rt = _token_len(rec["text"], enc)
        if cur_block and (cur_tok + rt) > max_parent_tokens:
            blocks.append(cur_block)
            cur_block = []
            cur_tok = 0
        cur_block.append(rec)
        cur_tok += rt

    if cur_block:
        blocks.append(cur_block)

    # -- for each block -> parent -> children --
    all_chunks: list[dict[str, Any]] = []
    multi = len(blocks) > 1

    for block_idx, block in enumerate(blocks):
        parent_text = "\n\n".join(r["text"] for r in block)

        if multi:
            parent_id = f"{source_id}:h:{heading_hash}:b{block_idx}"
            locator = f"{heading_path}#b{block_idx}"
        else:
            parent_id = f"{source_id}:h:{heading_hash}"
            locator = heading_path

        para_idxs: list[int] = [
            r["para_idx"] for r in block if isinstance(r.get("para_idx"), int)
        ]
        elem_idxs: list[int] = [
            r["element_idx"] for r in block if isinstance(r.get("element_idx"), int)
        ]
        pi_min = min(para_idxs) if para_idxs else None
        pi_max = max(para_idxs) if para_idxs else None
        ei_min = min(elem_idxs) if elem_idxs else None
        ei_max = max(elem_idxs) if elem_idxs else None

        child_texts = _split_text_children(parent_text, chunk_size, overlap, enc)

        for child_idx, child_text in enumerate(child_texts):
            chunk_id = f"{source_id}:{sha1_short(child_text)}"
            all_chunks.append({
                "chunk_id": chunk_id,
                "text": child_text,
                "source_id": source_id,
                "title": first.get("title"),
                "author": first.get("author"),
                "year": first.get("year"),
                "file_path": first.get("file_path"),
                "url": first.get("url"),
                "doc_type": doc_type,
                "doc_language": first.get("doc_language"),
                "section_heading": first.get("section_heading"),
                "heading_path": heading_path,
                "page_range": None,
                "doc_fingerprint": first.get("doc_fingerprint"),
                "locator_context": locator,
                "element_idx_min": ei_min,
                "element_idx_max": ei_max,
                "para_idx_min": pi_min,
                "para_idx_max": pi_max,
                "span": None,
                "parent_id": parent_id,
                "child_index": child_idx,
                "parent_locator": locator,
                "block_idx": block_idx if multi else None,
            })

    return all_chunks


# ---------------------------------------------------------------------------
# PDF entry-based chunking (each entry = 1 parent)
# ---------------------------------------------------------------------------

def _chunk_pdf_entries(
    group_recs: list[dict[str, Any]],
    chunk_size: int,
    overlap: int,
    enc: Any,
) -> list[dict[str, Any]]:
    """Chunk PDF by entry markers.  Each entry becomes its own parent.

    1. Build full text, detect entries via _split_into_entries.
    2. For each entry: parent_id = ``{source_id}:e:{entry_idx}``.
    3. Split entry_text -> child chunks; child_index resets per entry.
    """
    group_recs.sort(key=lambda r: (r.get("page") or 0))
    first = group_recs[0]
    source_id = first.get("source_id", "")

    # page-boundary map for locating entries
    page_boundaries: list[tuple[int, int]] = []
    full_parts: list[str] = []
    offset = 0
    for rec in group_recs:
        page_text = _strip_page_artifacts(rec.get("text", ""))
        page_no = rec.get("page") or 0
        page_boundaries.append((offset, page_no))
        full_parts.append(page_text)
        offset += len(page_text) + 2  # accounts for "\n\n"

    full_text = "\n\n".join(full_parts)
    entries = _split_into_entries(full_text)

    def _page_for_offset(char_pos: int) -> int:
        result_page = page_boundaries[0][1] if page_boundaries else 0
        for bo, pn in page_boundaries:
            if char_pos >= bo:
                result_page = pn
            else:
                break
        return result_page

    all_chunks: list[dict[str, Any]] = []
    search_start = 0

    for entry_idx, entry_text in enumerate(entries):
        idx = full_text.find(entry_text, search_start)
        if idx < 0:
            idx = search_start
        page_start = _page_for_offset(idx)
        page_end = _page_for_offset(idx + len(entry_text))
        search_start = idx + len(entry_text)

        parent_id = f"{source_id}:e:{entry_idx}"
        pr = str(page_start) if page_start == page_end else f"{page_start}-{page_end}"
        loc = f"entry_{entry_idx} p{pr}"

        child_texts = _split_text_children(entry_text, chunk_size, overlap, enc)

        for child_idx, child_text in enumerate(child_texts):
            chunk_id = f"{source_id}:{sha1_short(child_text)}"
            all_chunks.append({
                "chunk_id": chunk_id,
                "text": child_text,
                "source_id": source_id,
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
                "element_idx_min": page_start,
                "element_idx_max": page_end,
                "span": None,
                "chunk_strategy": "entry",
                "parent_id": parent_id,
                "child_index": child_idx,
                "parent_locator": f"entry {entry_idx} (pages {pr})",
                "entry_idx": entry_idx,
            })

    return all_chunks


# ---------------------------------------------------------------------------
# PDF page-based chunking (PAGES_PER_PARENT pages = 1 parent)
# ---------------------------------------------------------------------------

def _chunk_pdf_pages(
    group_recs: list[dict[str, Any]],
    chunk_size: int,
    overlap: int,
    pages_per_parent: int,
    enc: Any,
) -> list[dict[str, Any]]:
    """Chunk PDF by page groups.

    Each group of *pages_per_parent* consecutive pages becomes one parent.
    parent_id = ``{source_id}:p:{pmin}-{pmax}``.
    """
    group_recs.sort(key=lambda r: (r.get("page") or 0))
    first = group_recs[0]
    source_id = first.get("source_id", "")

    # clean page texts
    page_list: list[dict[str, Any]] = []
    for rec in group_recs:
        cleaned = _strip_page_artifacts(rec.get("text", ""))
        page_list.append({"text": cleaned, "page": rec.get("page") or 0})

    all_chunks: list[dict[str, Any]] = []

    for g_start in range(0, len(page_list), pages_per_parent):
        page_group = page_list[g_start: g_start + pages_per_parent]
        pmin = page_group[0]["page"]
        pmax = page_group[-1]["page"]
        parent_text = "\n\n".join(pg["text"] for pg in page_group if pg["text"])
        if not parent_text.strip():
            continue

        pr = str(pmin) if pmin == pmax else f"{pmin}-{pmax}"
        parent_id = f"{source_id}:p:{pmin}-{pmax}"
        loc = f"p{pr}"

        child_texts = _split_text_children(parent_text, chunk_size, overlap, enc)

        for child_idx, child_text in enumerate(child_texts):
            chunk_id = f"{source_id}:{sha1_short(child_text)}"
            all_chunks.append({
                "chunk_id": chunk_id,
                "text": child_text,
                "source_id": source_id,
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
                "element_idx_min": pmin,
                "element_idx_max": pmax,
                "span": None,
                "chunk_strategy": "page",
                "parent_id": parent_id,
                "child_index": child_idx,
                "parent_locator": f"pages {pr}",
            })

    return all_chunks


# ---------------------------------------------------------------------------
# Parent-Child: build parents JSONL + distribution stats
# ---------------------------------------------------------------------------

DEFAULT_PARENTS_PATH = "data/parents/parents_v2_full.jsonl"


def _assign_parent_child(
    chunks: list[dict[str, Any]],
    parents_path: str = DEFAULT_PARENTS_PATH,
) -> list[dict[str, Any]]:
    """Build parents JSONL from chunks (parent_id / child_index already set).

    Also logs children-per-parent distribution and warns about oversized parents.
    """
    # -- 1. Group by parent_id --
    parent_groups: dict[str, list[int]] = defaultdict(list)
    for idx, c in enumerate(chunks):
        pid = c.get("parent_id")
        if not pid:
            # fallback: should not happen after v2 chunking
            sid = c.get("source_id", "")
            pid = f"{sid}:unk:{sha1_short(c.get('chunk_id', ''))}"
            c["parent_id"] = pid
            c["child_index"] = 0
            c["parent_locator"] = ""
        parent_groups[pid].append(idx)

    # -- 2. Build parents JSONL --
    parents: list[dict[str, Any]] = []
    for pid, child_idxs in parent_groups.items():
        first_c = chunks[child_idxs[0]]
        parent_text = "\n\n".join(chunks[ci].get("text", "") for ci in child_idxs)
        dt = first_c.get("doc_type", "")

        parent_meta: dict[str, Any] = {}
        if dt == "docx":
            parent_meta["heading_path"] = first_c.get("heading_path")
            parent_meta["section_heading"] = first_c.get("section_heading")
            parent_meta["strategy"] = "docx_heading"
            if first_c.get("block_idx") is not None:
                parent_meta["block_idx"] = first_c["block_idx"]
        elif dt == "pdf":
            parent_meta["page_range"] = first_c.get("page_range")
            parent_meta["strategy"] = first_c.get("chunk_strategy", "page")
            if first_c.get("entry_idx") is not None:
                parent_meta["entry_idx"] = first_c["entry_idx"]
        elif dt == "txt":
            parent_meta["heading_path"] = first_c.get("heading_path")
            parent_meta["section_heading"] = first_c.get("section_heading")
            parent_meta["strategy"] = "txt_heading"
            if first_c.get("block_idx") is not None:
                parent_meta["block_idx"] = first_c["block_idx"]
        elif dt == "image":
            parent_meta["locator"] = first_c.get("locator_context")
            parent_meta["strategy"] = "image"
        else:
            parent_meta["strategy"] = "unknown"

        parents.append({
            "parent_id": pid,
            "source_id": first_c.get("source_id"),
            "doc_type": dt,
            "category": parse_category(first_c.get("file_path")),
            "title": first_c.get("title"),
            "author": first_c.get("author"),
            "year": first_c.get("year"),
            "file_path": first_c.get("file_path"),
            "url": first_c.get("url"),
            "parent_text": parent_text,
            "parent_meta": parent_meta,
            "children_count": len(child_idxs),
        })

    ensure_parent_dir(parents_path)
    write_jsonl(parents, parents_path)

    # -- 3. Stats --
    cc = sorted(p["children_count"] for p in parents)
    n = len(cc)
    if n:
        avg = sum(cc) / n
        p95 = cc[min(int(0.95 * n), n - 1)]
        mx = cc[-1]
        logger.info("Parent-Child stats:")
        logger.info("  parents_count          : %d", n)
        logger.info("  children_count (total) : %d", len(chunks))
        logger.info("  avg_children_per_parent: %.2f", avg)
        logger.info("  p95_children_per_parent: %d", p95)
        logger.info("  max_children_per_parent: %d", mx)
        logger.info("  parents JSONL          : %s", parents_path)

        if mx > HARD_CAP_CHILDREN:
            for p in parents:
                if p["children_count"] > HARD_CAP_CHILDREN:
                    meta = p.get("parent_meta", {})
                    logger.warning(
                        "Parent exceeds HARD_CAP (%d): parent_id=%s  "
                        "children=%d  file=%s  locator=%s",
                        HARD_CAP_CHILDREN,
                        p["parent_id"],
                        p["children_count"],
                        p.get("file_path", "?"),
                        meta.get("heading_path")
                        or meta.get("page_range")
                        or "?",
                    )

    return chunks


# ---------------------------------------------------------------------------
# Main chunking driver
# ---------------------------------------------------------------------------

def run_chunk(config: dict[str, Any]) -> int:
    """Run B3 chunking. Returns chunk count."""
    input_path = config["clean"]["output_jsonl"]
    output_path = config["index"]["input_chunks"]
    chunk_cfg = config["chunking"]
    chunk_size: int = int(chunk_cfg.get("chunk_size", 500))
    overlap: int = int(chunk_cfg.get("overlap", 100))

    records = read_jsonl(input_path)
    if not records:
        logger.warning("No records to chunk from %s", input_path)
        return 0

    records.sort(key=_sort_key)

    enc = tiktoken.get_encoding("cl100k_base")  # type: ignore
    chunks: list[dict[str, Any]] = []

    image_records = [r for r in records if r.get("doc_type") == "image"]
    docx_records = [r for r in records if r.get("doc_type") == "docx"]
    pdf_records = [r for r in records if r.get("doc_type") == "pdf"]
    txt_records = [r for r in records if r.get("doc_type") == "txt"]

    # --- Image: each image = its own parent ---
    for rec in image_records:
        source_id = rec.get("source_id", "")
        loc = rec.get("locator") or rec.get("file_path") or "img"
        parent_id = f"{source_id}:img:{sha1_short(loc)}"

        txt = rec.get("text", "")
        nt = _token_len(txt, enc)
        if nt > chunk_size:
            parts = _split_text_by_tokens(txt, chunk_size, enc)
            for pi, part in enumerate(parts):
                chunk_id = f"{source_id}:{sha1_short(part)}"
                chunks.append({
                    "chunk_id": chunk_id,
                    "text": part,
                    "source_id": source_id,
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
                    "locator_context": f"{loc}#part{pi}",
                    "bbox": rec.get("bbox"),
                    "ocr_confidence": rec.get("ocr_confidence"),
                    "chunk_strategy": "image_split",
                    "parent_id": parent_id,
                    "child_index": pi,
                    "parent_locator": loc,
                })
        else:
            chunk_id = f"{source_id}:{sha1_short(txt)}"
            chunks.append({
                "chunk_id": chunk_id,
                "text": txt,
                "source_id": source_id,
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
                "locator_context": loc,
                "bbox": rec.get("bbox"),
                "ocr_confidence": rec.get("ocr_confidence"),
                "parent_id": parent_id,
                "child_index": 0,
                "parent_locator": loc,
            })

    # --- DOCX: paragraph-level chunking with sub-parent blocking ---
    docx_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for rec in docx_records:
        docx_groups[_group_key(rec)].append(rec)

    for _, group_recs in docx_groups.items():
        docx_chunks = _chunk_docx_group(
            group_recs, chunk_size, overlap, MAX_PARENT_TOKENS, enc,
        )
        chunks.extend(docx_chunks)

    # --- TXT: paragraph-level chunking (reuse DOCX logic) ---
    txt_groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for rec in txt_records:
        txt_groups[_group_key(rec)].append(rec)

    for _, group_recs in txt_groups.items():
        txt_chunks = _chunk_docx_group(
            group_recs, chunk_size, overlap, MAX_PARENT_TOKENS, enc,
            doc_type="txt",
        )
        chunks.extend(txt_chunks)

    # --- PDF: auto-select strategy (entry-based or page-group) ---
    pdf_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for rec in pdf_records:
        pdf_groups[rec.get("source_id", "")].append(rec)

    entry_strategy_count = 0
    page_strategy_count = 0
    for _sid, group_recs in pdf_groups.items():
        probe_pages = sorted(group_recs, key=lambda r: r.get("page") or 0)[:10]
        probe_text = "\n\n".join(
            _strip_page_artifacts(r.get("text", "")) for r in probe_pages
        )

        if _has_entry_markers(probe_text):
            pdf_chunks = _chunk_pdf_entries(group_recs, chunk_size, overlap, enc)
            entry_strategy_count += 1
        else:
            pdf_chunks = _chunk_pdf_pages(
                group_recs, chunk_size, overlap, PAGES_PER_PARENT, enc,
            )
            page_strategy_count += 1
        chunks.extend(pdf_chunks)

    logger.info(
        "  PDF strategy: %d sources entry-based, %d page-group",
        entry_strategy_count,
        page_strategy_count,
    )

    # --- Assign category to every chunk ---
    for c in chunks:
        c["category"] = parse_category(c.get("file_path"))

    # --- Propagate is_noise from input records (source_id lookup) ---
    noisy_sources: set[str] = {
        r.get("source_id", "")
        for r in records
        if r.get("is_noise")
    }
    if noisy_sources:
        propagated = 0
        for c in chunks:
            if c.get("source_id", "") in noisy_sources:
                c["is_noise"] = True
                propagated += 1
        logger.info("  Propagated is_noise from input passages: %d chunks", propagated)

    # --- Parent-Child assignment (builds parents JSONL + stats) ---
    parents_path = config.get("parent_child", {}).get(
        "parents_path", DEFAULT_PARENTS_PATH,
    )
    chunks = _assign_parent_child(chunks, parents_path=parents_path)

    ensure_parent_dir(output_path)
    write_jsonl(chunks, output_path)

    logger.info("B3 Chunk complete: %d total chunks -> %s", len(chunks), output_path)

    fffd_count = sum(1 for c in chunks if "\ufffd" in c.get("text", ""))
    logger.info("  Chunks with U+FFFD: %d / %d", fffd_count, len(chunks))

    # Example chunks
    docx_chunks_out = [c for c in chunks if c.get("doc_type") == "docx"]
    for i, ex in enumerate(docx_chunks_out[:3]):
        logger.info(
            "  Example DOCX chunk %d: chunk_id=%s  locator=%s  "
            "para_idx=%s-%s  elem_idx=%s-%s  parent_id=%s  child_idx=%s",
            i + 1,
            ex.get("chunk_id", "?"),
            ex.get("locator_context"),
            ex.get("para_idx_min"),
            ex.get("para_idx_max"),
            ex.get("element_idx_min"),
            ex.get("element_idx_max"),
            ex.get("parent_id", "?"),
            ex.get("child_index"),
        )

    pdf_chunks_out = [c for c in chunks if c.get("doc_type") == "pdf"]
    logger.info("  PDF chunks: %d", len(pdf_chunks_out))
    for i, ex in enumerate(pdf_chunks_out[:3]):
        logger.info(
            "  Example PDF chunk %d: chunk_id=%s  page_range=%s  "
            "locator=%s  elem_idx=%s-%s  parent_id=%s  child_idx=%s  strategy=%s",
            i + 1,
            ex.get("chunk_id", "?"),
            ex.get("page_range"),
            ex.get("locator_context"),
            ex.get("element_idx_min"),
            ex.get("element_idx_max"),
            ex.get("parent_id", "?"),
            ex.get("child_index"),
            ex.get("chunk_strategy"),
        )

    return len(chunks)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="B3 -- Chunk by structure")
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
