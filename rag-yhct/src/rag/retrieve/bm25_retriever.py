# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
"""BM25 retriever over chunk texts.

Builds a BM25 index from chunks_v2_full.jsonl with persistent pickle cache.
Supports filtering by doc_type and noise exclusion.

Usage (CLI):
    PYTHONPATH=src uv run python -m rag.retrieve.bm25_retriever \
        --query "tác dụng của cây ngải cứu" --topk 10

    # Rebuild index from scratch:
    PYTHONPATH=src uv run python -m rag.retrieve.bm25_retriever \
        --build --chunks data/chunks/chunks_v2_full.jsonl
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import pickle
import re
import time
from pathlib import Path
from typing import Any, cast

from rank_bm25 import BM25Okapi  # type: ignore

from rag.utils.io import read_jsonl
from rag.utils.query_quality import normalize_query_no_diacritics

logger = logging.getLogger(__name__)

# ── defaults ───────────────────────────────────────────────────────────────
DEFAULT_CHUNKS_PATH = "data/chunks/chunks_v2_full.jsonl"
DEFAULT_INDEX_PATH = "data/indexes/bm25_chunks_v2_full.pkl"
DEFAULT_TOPK = 40
BM25_INDEX_VERSION = 3
CURATED_LEXICON_GROUPS = {"herbs", "symptoms", "treatments", "formulas", "patterns"}
_LEXICON_PATH = Path(__file__).with_name("yhct_domain_lexicon.json")
_domain_phrase_cache: list[list[str]] | None = None

# ── tokenization ───────────────────────────────────────────────────────────

_RE_WHITESPACE = re.compile(r"\s+")


def _load_domain_phrases() -> list[list[str]]:
    """Load curated YHCT phrases as accent-folded token sequences.

    Avoid auto-mined phrases here because they contain noisy/common fragments
    that can overpower BM25.
    """
    global _domain_phrase_cache  # noqa: PLW0603
    if _domain_phrase_cache is not None:
        return _domain_phrase_cache
    if not _LEXICON_PATH.exists():
        _domain_phrase_cache = []
        return _domain_phrase_cache
    try:
        raw = json.loads(_LEXICON_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to load BM25 domain lexicon %s: %s", _LEXICON_PATH, exc)
        _domain_phrase_cache = []
        return _domain_phrase_cache

    phrases: list[list[str]] = []
    if isinstance(raw, dict):
        for group, pairs in raw.items():
            if group not in CURATED_LEXICON_GROUPS or not isinstance(pairs, list):
                continue
            for item in pairs:
                if isinstance(item, list):
                    item_seq = cast(list[Any], item)
                elif isinstance(item, tuple):
                    item_seq = list(cast(tuple[Any, Any], item))
                else:
                    continue
                if len(item_seq) != 2:
                    continue
                folded = normalize_query_no_diacritics(str(item_seq[0]))
                toks = [t for t in folded.split() if t]
                if len(toks) >= 2:
                    phrases.append(toks)
    phrases.sort(key=len, reverse=True)
    _domain_phrase_cache = phrases
    return _domain_phrase_cache


def _add_domain_phrase_tokens(tokens: list[str]) -> list[str]:
    out = list(tokens)
    if len(tokens) < 2:
        return out
    for phrase in _load_domain_phrases():
        n = len(phrase)
        if n > len(tokens):
            continue
        phrase_token = "_".join(phrase)
        for i in range(0, len(tokens) - n + 1):
            if tokens[i:i + n] == phrase:
                out.append(phrase_token)
    return out


def tokenize_vi(text: str) -> list[str]:
    """Vietnamese tokenizer tolerant to missing diacritics and YHCT phrases.

    It normalizes Unicode/casing/whitespace, strips diacritics, then adds
    curated phrase tokens such as "ngai_cuu" alongside unigram tokens.
    """
    text = normalize_query_no_diacritics(text)
    text = _RE_WHITESPACE.sub(" ", text).strip()
    tokens = text.split()
    return _add_domain_phrase_tokens(tokens)


# ── index management ──────────────────────────────────────────────────────

def _file_checksum(path: str | Path) -> str:
    """Fast MD5 checksum of the first+last 64KB for change detection."""
    p = Path(path)
    h = hashlib.md5(usedforsecurity=False)
    size = p.stat().st_size
    with open(p, "rb") as f:
        h.update(f.read(65536))
        if size > 65536:
            f.seek(max(0, size - 65536))
            h.update(f.read(65536))
    h.update(str(size).encode())
    return h.hexdigest()


def build_bm25_index(
    chunks_path: str = DEFAULT_CHUNKS_PATH,
    index_path: str = DEFAULT_INDEX_PATH,
    force: bool = False,
) -> dict[str, Any]:
    """Build (or load cached) BM25 index.

    Returns dict with keys: bm25, docs, checksum
      - bm25: BM25Okapi instance
      - docs: list of metadata dicts (chunk_id, source_id, parent_id, etc.)
      - checksum: source file checksum
    """

    chunks_p = Path(chunks_path)
    index_p = Path(index_path)

    if not chunks_p.exists():
        raise FileNotFoundError(f"Chunks file not found: {chunks_p}")

    current_checksum = _file_checksum(chunks_p)

    # Try loading cached index
    if not force and index_p.exists():
        try:
            with open(index_p, "rb") as f:
                cached = pickle.load(f)  # noqa: S301
            if (
                cached.get("checksum") == current_checksum
                and int(cached.get("index_version", 0)) == BM25_INDEX_VERSION
            ):
                logger.info(
                    "Loaded cached BM25 index (%d docs) from %s",
                    len(cached["docs"]), index_p,
                )
                return cached
            logger.info("Chunks/tokenizer changed — rebuilding BM25 index")
        except Exception as exc:
            logger.warning("Failed to load BM25 cache (%s) — rebuilding", exc)

    # Build from scratch
    logger.info("Building BM25 index from %s ...", chunks_p)
    t0 = time.time()

    chunks = read_jsonl(chunks_path)
    tokenized_corpus: list[list[str]] = []
    docs: list[dict[str, Any]] = []

    for c in chunks:
        # Skip noise or empty text
        if c.get("is_noise"):
            continue
        text = c.get("text_norm") or c.get("text", "")
        if not text.strip():
            continue

        tokens = tokenize_vi(text)
        if not tokens:
            continue

        tokenized_corpus.append(tokens)
        docs.append({
            "chunk_id": c.get("chunk_id", ""),
            "source_id": c.get("source_id", ""),
            "parent_id": c.get("parent_id", ""),
            "child_index": c.get("child_index"),
            "doc_type": c.get("doc_type", ""),
            "category": c.get("category", ""),
            "is_noise": False,
            "text": text[:500],  # store snippet for display
        })

    bm25 = BM25Okapi(tokenized_corpus)

    result = {
        "bm25": bm25,
        "docs": docs,
        "checksum": current_checksum,
        "index_version": BM25_INDEX_VERSION,
    }

    # Save cache
    index_p.parent.mkdir(parents=True, exist_ok=True)
    with open(index_p, "wb") as f:
        pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
    elapsed = time.time() - t0
    logger.info(
        "BM25 index built: %d docs in %.1fs — cached at %s",
        len(docs), elapsed, index_p,
    )

    return result


# ── in-memory cache ───────────────────────────────────────────────────────

_bm25_cache: dict[str, Any] | None = None


def get_bm25_index(
    chunks_path: str = DEFAULT_CHUNKS_PATH,
    index_path: str = DEFAULT_INDEX_PATH,
    force: bool = False,
) -> dict[str, Any]:
    """Get BM25 index (cached in-memory across calls)."""
    global _bm25_cache  # noqa: PLW0603
    if _bm25_cache is not None and not force:
        return _bm25_cache
    _bm25_cache = build_bm25_index(chunks_path, index_path, force=force)
    return _bm25_cache


def invalidate_bm25_cache() -> None:
    """Force reload on next access."""
    global _bm25_cache  # noqa: PLW0603
    _bm25_cache = None


# ── retrieval ─────────────────────────────────────────────────────────────

def retrieve_bm25(
    query: str,
    *,
    topk: int = DEFAULT_TOPK,
    chunks_path: str = DEFAULT_CHUNKS_PATH,
    index_path: str = DEFAULT_INDEX_PATH,
    doc_type_filter: str | None = None,
) -> list[dict[str, Any]]:
    """Retrieve top-K chunks by BM25 score.

    Returns list of dicts with:
      chunk_id, bm25_score, source_id, parent_id, child_index, text, doc_type, category
    """
    idx = get_bm25_index(chunks_path, index_path)
    bm25: BM25Okapi = idx["bm25"]
    docs: list[dict[str, Any]] = idx["docs"]

    query_tokens = tokenize_vi(query)
    if not query_tokens:
        return []

    scores: list[float] = cast(list[float], bm25.get_scores(query_tokens))

    # Build (score, idx) pairs, apply filters
    scored: list[tuple[float, int]] = []
    for i, s in enumerate(scores):
        if s <= 0:
            continue
        if doc_type_filter and docs[i].get("doc_type") != doc_type_filter:
            continue
        scored.append((float(s), i))

    scored.sort(key=lambda x: x[0], reverse=True)
    scored = scored[:topk]

    results: list[dict[str, Any]] = []
    for rank, (score, doc_idx) in enumerate(scored):
        doc = docs[doc_idx]
        results.append({
            "rank": rank,
            "chunk_id": doc["chunk_id"],
            "bm25_score": round(score, 6),
            "source_id": doc["source_id"],
            "parent_id": doc["parent_id"],
            "child_index": doc["child_index"],
            "doc_type": doc["doc_type"],
            "category": doc["category"],
            "text": doc["text"],
        })

    return results


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    p = argparse.ArgumentParser(description="BM25 retriever")
    p.add_argument("--query", default=None, help="Query text to search")
    p.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    p.add_argument("--chunks", default=DEFAULT_CHUNKS_PATH)
    p.add_argument("--index", default=DEFAULT_INDEX_PATH)
    p.add_argument("--doc-type", default=None, help="Filter by doc_type")
    p.add_argument("--build", action="store_true", help="Force rebuild index only")
    args = p.parse_args()

    if args.build or args.query is None:
        build_bm25_index(args.chunks, args.index, force=True)
        if args.query is None:
            return

    t0 = time.time()
    results = retrieve_bm25(
        args.query,
        topk=args.topk,
        chunks_path=args.chunks,
        index_path=args.index,
        doc_type_filter=args.doc_type,
    )
    elapsed = time.time() - t0

    print(f"\nBM25 results for: {args.query!r}  ({elapsed:.2f}s, {len(results)} hits)\n")
    for r in results[:10]:
        text_preview = r["text"][:80].replace("\n", " ")
        print(
            f"  [{r['rank']:2d}] score={r['bm25_score']:.4f}  "
            f"chunk_id={r['chunk_id'][:30]}  "
            f"parent_id={r['parent_id'][:30]}  "
            f"text={text_preview}..."
        )


if __name__ == "__main__":
    main()
