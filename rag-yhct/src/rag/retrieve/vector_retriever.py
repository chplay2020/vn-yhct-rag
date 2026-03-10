# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
"""Vector retriever wrapper — consistent interface matching BM25 retriever output.

Wraps existing Qdrant vector search into a standardized result format
compatible with BM25 and hybrid fusion.

Usage (CLI):
    PYTHONPATH=src uv run python -m rag.retrieve.vector_retriever \
        --query "tác dụng của cây ngải cứu" --topk 10
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Any

import requests  # type: ignore
from qdrant_client import QdrantClient  # type: ignore

logger = logging.getLogger(__name__)

# ── defaults ───────────────────────────────────────────────────────────────
DEFAULT_COLLECTION = "yhct_chunks_v2_full_emb"
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "bge-m3"
DEFAULT_TOPK = 40


# ── embed query (reuse same logic as parent_child_retriever) ──────────────

def embed_query(
    text: str,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    model: str = DEFAULT_MODEL,
) -> list[float] | None:
    """Embed a single query text via Ollama."""
    try:
        resp = requests.post(
            f"{ollama_url}/api/embeddings",
            json={"model": model, "prompt": text},
            timeout=60,
        )
        if resp.status_code == 200:
            return resp.json().get("embedding")
        logger.warning("Ollama returned %d: %s", resp.status_code, resp.text[:120])
    except Exception as exc:
        logger.warning("Embed query failed: %s", exc)
    return None


# ── retrieval ─────────────────────────────────────────────────────────────

def retrieve_vector(
    query: str,
    *,
    topk: int = DEFAULT_TOPK,
    collection: str = DEFAULT_COLLECTION,
    qdrant_url: str = DEFAULT_QDRANT_URL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    model: str = DEFAULT_MODEL,
    doc_type_filter: str | None = None,
) -> list[dict[str, Any]]:
    """Retrieve top-K chunks by vector similarity.

    Returns list of dicts with:
      rank, chunk_id, vector_score, source_id, parent_id, child_index, text, doc_type, category
    """
    query_vec = embed_query(query, ollama_url, model)
    if query_vec is None:
        logger.error("Failed to embed query — cannot retrieve")
        return []

    return retrieve_vector_from_vec(
        query_vec,
        topk=topk,
        collection=collection,
        qdrant_url=qdrant_url,
        doc_type_filter=doc_type_filter,
    )


def retrieve_vector_from_vec(
    query_vec: list[float],
    *,
    topk: int = DEFAULT_TOPK,
    collection: str = DEFAULT_COLLECTION,
    qdrant_url: str = DEFAULT_QDRANT_URL,
    doc_type_filter: str | None = None,
) -> list[dict[str, Any]]:
    """Retrieve top-K chunks by pre-computed query vector.

    Useful when the caller already has the embedding (avoids double-embedding).
    """
    from qdrant_client.models import FieldCondition, Filter, MatchValue  # type: ignore

    client: Any = QdrantClient(url=qdrant_url)

    # Build optional filter
    query_filter = None
    conditions: list[FieldCondition] = []
    # Always exclude noise
    conditions.append(FieldCondition(key="is_noise", match=MatchValue(value=False)))
    if doc_type_filter:
        conditions.append(FieldCondition(key="doc_type", match=MatchValue(value=doc_type_filter)))
    if conditions:
        query_filter = Filter(must=conditions)  # type: ignore[arg-type]

    # Use query_points (latest qdrant-client API)
    response: Any = client.query_points(
        collection_name=collection,
        query=query_vec,
        limit=topk,
        with_payload=True,
        query_filter=query_filter,
    )
    results: Any = response.points

    output: list[dict[str, Any]] = []
    for rank, r in enumerate(results):
        payload = dict(getattr(r, "payload", {}) or {})
        text = payload.get("text_norm") or payload.get("text", "")
        output.append({
            "rank": rank,
            "chunk_id": payload.get("chunk_id", ""),
            "vector_score": round(float(r.score), 6),
            "source_id": payload.get("source_id", ""),
            "parent_id": payload.get("parent_id", ""),
            "child_index": payload.get("child_index"),
            "doc_type": payload.get("doc_type", ""),
            "category": payload.get("category", ""),
            "text": text[:500],
        })

    return output


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    p = argparse.ArgumentParser(description="Vector retriever")
    p.add_argument("--query", required=True, help="Query text to search")
    p.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    p.add_argument("--collection", default=DEFAULT_COLLECTION)
    p.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    p.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--doc-type", default=None, help="Filter by doc_type")
    args = p.parse_args()

    t0 = time.time()
    results = retrieve_vector(
        args.query,
        topk=args.topk,
        collection=args.collection,
        qdrant_url=args.qdrant_url,
        ollama_url=args.ollama_url,
        model=args.model,
        doc_type_filter=args.doc_type,
    )
    elapsed = time.time() - t0

    print(f"\nVector results for: {args.query!r}  ({elapsed:.2f}s, {len(results)} hits)\n")
    for r in results[:10]:
        text_preview = r["text"][:80].replace("\n", " ")
        print(
            f"  [{r['rank']:2d}] score={r['vector_score']:.4f}  "
            f"chunk_id={r['chunk_id'][:30]}  "
            f"parent_id={r['parent_id'][:30]}  "
            f"text={text_preview}..."
        )


if __name__ == "__main__":
    main()
