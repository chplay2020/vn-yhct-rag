"""B5 — Embed full data with Ollama BGE-M3 and upsert real vectors to Qdrant.

Usage:
    PYTHONPATH=src uv run python -m rag.embed.embed_full
    PYTHONPATH=src uv run python -m rag.embed.embed_full \
        --collection yhct_chunks_v2_full_emb \
        --chunks data/chunks/chunks_v2_full.jsonl \
        --embed-batch 16 --upsert-batch 64 --recreate

Output collection will have real bge-m3 vectors (dim=1024) instead of dummy 0.0.
"""

from __future__ import annotations

import argparse
import logging
import math
import statistics
import time
from typing import Any

import requests  # type: ignore
from qdrant_client import QdrantClient  # type: ignore
from qdrant_client.models import (  # type: ignore[import-untyped]
    Distance,  # type: ignore[no-redef]
    PointStruct,  # type: ignore[no-redef]
    VectorParams,  # type: ignore[no-redef]
)
from tqdm import tqdm  # type: ignore

from rag.utils.hashing import stable_point_id
from rag.utils.io import read_jsonl

logger = logging.getLogger(__name__)

# ── defaults ───────────────────────────────────────────────────────────────
DEFAULT_COLLECTION = "yhct_chunks_v2_full_emb"
DEFAULT_CHUNKS = "data/chunks/chunks_v2_full.jsonl"
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "bge-m3"
DEFAULT_EMBED_BATCH = 16
DEFAULT_UPSERT_BATCH = 64
DEFAULT_MIN_LEN = 30
DEFAULT_MAX_RETRIES = 3
DEFAULT_VECTOR_SIZE = 1024


# ── helpers ────────────────────────────────────────────────────────────────

def _embed_batch_ollama(
    texts: list[str],
    ollama_url: str,
    model: str,
    max_retries: int = 3,
) -> list[list[float] | None]:
    """Embed a list of texts one-by-one via Ollama /api/embeddings.

    Returns a list parallel to *texts*; failed entries are ``None``.
    """
    results: list[list[float] | None] = []
    for text in texts:
        embedding: list[float] | None = None
        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    f"{ollama_url}/api/embeddings",
                    json={"model": model, "prompt": text},
                    timeout=60,
                )
                if resp.status_code == 200:
                    embedding = resp.json().get("embedding")
                    break
                logger.warning(
                    "Ollama returned %d (attempt %d/%d): %s",
                    resp.status_code, attempt + 1, max_retries, resp.text[:120],
                )
            except Exception as exc:
                logger.warning(
                    "Embedding request failed (attempt %d/%d): %s",
                    attempt + 1, max_retries, exc,
                )
            time.sleep(0.5 * (attempt + 1))
        results.append(embedding)
    return results


def _vector_norm(vec: list[float]) -> float:
    return math.sqrt(sum(v * v for v in vec))


def _ensure_collection(
    client: Any,
    collection: str,
    vector_size: int,
    recreate: bool,
) -> None:
    """Create (or recreate) a Qdrant collection."""
    names = [c.name for c in client.get_collections().collections]
    if recreate and collection in names:
        logger.info("Dropping collection '%s' for recreate", collection)
        client.delete_collection(collection)
        names.remove(collection)
    if collection not in names:
        logger.info(
            "Creating collection '%s' (dim=%d, cosine)", collection, vector_size,
        )
        client.create_collection(  # type: ignore[union-attr]
            collection_name=collection,
            vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),  # type: ignore[attr-defined]
        )
    else:
        logger.info("Collection '%s' already exists — will upsert in-place", collection)


# ── main logic ─────────────────────────────────────────────────────────────

def run_embed(
    collection: str = DEFAULT_COLLECTION,
    chunks_path: str = DEFAULT_CHUNKS,
    qdrant_url: str = DEFAULT_QDRANT_URL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    model: str = DEFAULT_MODEL,
    embed_batch: int = DEFAULT_EMBED_BATCH,
    upsert_batch: int = DEFAULT_UPSERT_BATCH,
    min_len: int = DEFAULT_MIN_LEN,
    skip_noise: bool = True,
    max_retries: int = DEFAULT_MAX_RETRIES,
    recreate: bool = False,
    vector_size: int = DEFAULT_VECTOR_SIZE,
) -> dict[str, Any]:
    """Embed all chunks and upsert real vectors into *collection*.

    Returns a summary dict with keys: ``total``, ``embedded``, ``skipped``,
    ``norm_min``, ``norm_median``, ``norm_max``.
    """

    # ── 1. load chunks ────────────────────────────────────────────────
    chunks = read_jsonl(chunks_path)
    logger.info("Loaded %d chunks from %s", len(chunks), chunks_path)

    # Filter noise / too-short
    filtered: list[dict[str, Any]] = []
    for c in chunks:
        if skip_noise and c.get("is_noise"):
            continue
        text = (c.get("text_norm") or c.get("text", "")).strip()
        if len(text) < min_len:
            continue
        filtered.append(c)

    skipped = len(chunks) - len(filtered)
    if skipped:
        logger.info("Filtered out %d chunks (noise / too-short < %d chars)", skipped, min_len)
    logger.info("Chunks to embed: %d", len(filtered))

    if not filtered:
        logger.warning("Nothing to embed — aborting")
        return {"total": len(chunks), "embedded": 0, "skipped": skipped}

    # ── 2. Qdrant setup ──────────────────────────────────────────────
    client: Any = QdrantClient(url=qdrant_url)  # type: ignore[no-untyped-call]
    _ensure_collection(client, collection, vector_size, recreate)

    # ── 3. embed + upsert ────────────────────────────────────────────
    all_norms: list[float] = []
    points_buf: list[Any] = []
    total_upserted = 0
    failed_embed = 0

    progress = tqdm(range(0, len(filtered), embed_batch), desc="Embedding", unit="batch")
    for batch_start in progress:
        batch_chunks = filtered[batch_start : batch_start + embed_batch]
        texts = [
            (c.get("text_norm") or c.get("text", "")).strip()
            for c in batch_chunks
        ]

        embeddings = _embed_batch_ollama(texts, ollama_url, model, max_retries)

        for chunk, emb in zip(batch_chunks, embeddings):
            if emb is None:
                failed_embed += 1
                continue

            chunk_id: str = chunk.get("chunk_id", "")
            point_id = stable_point_id(chunk_id)

            # payload — keep text, text_norm, metadata
            payload: dict[str, Any] = {
                "chunk_id": chunk_id,
                "text": chunk.get("text", ""),
            }
            if chunk.get("text_norm"):
                payload["text_norm"] = chunk["text_norm"]
            skip_keys = {"chunk_id", "text", "text_norm"}
            for k, v in chunk.items():
                if k not in skip_keys:
                    payload[k] = v

            points_buf.append(
                PointStruct(id=point_id, vector=emb, payload=payload)
            )
            all_norms.append(_vector_norm(emb))

            # upsert when buffer full
            if len(points_buf) >= upsert_batch:
                client.upsert(collection_name=collection, points=points_buf)  # type: ignore[union-attr]
                total_upserted += len(points_buf)
                points_buf = []

        progress.set_postfix(upserted=total_upserted, failed=failed_embed)  # type: ignore[no-untyped-call]

    # flush remaining
    if points_buf:
        client.upsert(collection_name=collection, points=points_buf)  # type: ignore[union-attr]
        total_upserted += len(points_buf)

    # ── 4. summary ───────────────────────────────────────────────────
    if all_norms:
        norm_min = min(all_norms)
        norm_max = max(all_norms)
        norm_med = statistics.median(all_norms)
    else:
        norm_min = norm_max = norm_med = 0.0

    summary: dict[str, Any] = {
        "total": len(chunks),
        "embedded": total_upserted,
        "skipped": skipped,
        "failed_embed": failed_embed,
        "norm_min": round(norm_min, 6),
        "norm_median": round(norm_med, 6),
        "norm_max": round(norm_max, 6),
    }

    logger.info("=" * 60)
    logger.info("B5 Embed complete")
    logger.info("  Total chunks loaded : %d", summary["total"])
    logger.info("  Embedded & upserted : %d", summary["embedded"])
    logger.info("  Skipped (noise/short): %d", summary["skipped"])
    logger.info("  Failed embed         : %d", summary["failed_embed"])
    logger.info("  Vector norm min      : %.6f", summary["norm_min"])
    logger.info("  Vector norm median   : %.6f", summary["norm_median"])
    logger.info("  Vector norm max      : %.6f", summary["norm_max"])
    logger.info("=" * 60)

    return summary


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    p = argparse.ArgumentParser(
        description="B5 — Embed full data (Ollama BGE-M3) and upsert to Qdrant",
    )
    p.add_argument("--collection", default=DEFAULT_COLLECTION,
                    help=f"Qdrant collection name (default: {DEFAULT_COLLECTION})")
    p.add_argument("--chunks", default=DEFAULT_CHUNKS,
                    help=f"Input chunks JSONL (default: {DEFAULT_CHUNKS})")
    p.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    p.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    p.add_argument("--model", default=DEFAULT_MODEL,
                    help=f"Ollama model name (default: {DEFAULT_MODEL})")
    p.add_argument("--embed-batch", type=int, default=DEFAULT_EMBED_BATCH,
                    help=f"Texts per embed request batch (default: {DEFAULT_EMBED_BATCH})")
    p.add_argument("--upsert-batch", type=int, default=DEFAULT_UPSERT_BATCH,
                    help=f"Points per Qdrant upsert (default: {DEFAULT_UPSERT_BATCH})")
    p.add_argument("--min-len", type=int, default=DEFAULT_MIN_LEN,
                    help=f"Skip chunks shorter than this (default: {DEFAULT_MIN_LEN})")
    p.add_argument("--skip-noise", action="store_true", default=True,
                    help="Skip chunks flagged is_noise (default: True)")
    p.add_argument("--no-skip-noise", dest="skip_noise", action="store_false")
    p.add_argument("--max-retries", type=int, default=DEFAULT_MAX_RETRIES,
                    help=f"Retries per embed call (default: {DEFAULT_MAX_RETRIES})")
    p.add_argument("--recreate", action="store_true", default=False,
                    help="Drop and recreate collection before upserting")
    p.add_argument("--vector-size", type=int, default=DEFAULT_VECTOR_SIZE,
                    help=f"Vector dimension (default: {DEFAULT_VECTOR_SIZE})")
    args = p.parse_args()

    t0 = time.time()
    summary = run_embed(
        collection=args.collection,
        chunks_path=args.chunks,
        qdrant_url=args.qdrant_url,
        ollama_url=args.ollama_url,
        model=args.model,
        embed_batch=args.embed_batch,
        upsert_batch=args.upsert_batch,
        min_len=args.min_len,
        skip_noise=args.skip_noise,
        max_retries=args.max_retries,
        recreate=args.recreate,
        vector_size=args.vector_size,
    )
    elapsed = time.time() - t0
    logger.info("Total time: %.1fs (%.2f chunks/sec)",
                elapsed, summary["embedded"] / max(elapsed, 0.01))


if __name__ == "__main__":
    main()
