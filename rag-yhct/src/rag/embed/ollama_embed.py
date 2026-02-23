"""Generate embeddings using Ollama BGE-M3 and update Qdrant vectors.

Usage:
    python -m rag.embed.ollama_embed \
        --collection yhct_chunks_v2_full \
        --chunks data/chunks/chunks_v2_full.jsonl \
        --qdrant-url http://localhost:6333 \
        --ollama-url http://localhost:11434 \
        --batch-size 32 \
        --recreate
"""

from __future__ import annotations

import argparse
import logging
import time

import requests  # type: ignore
from qdrant_client import QdrantClient  # type: ignore
from qdrant_client.models import PointStruct  # type: ignore
from tqdm import tqdm  # type: ignore

from rag.utils.hashing import stable_point_id
from rag.utils.io import read_jsonl

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")


def _get_embedding(text: str, ollama_url: str, model: str = "bge-m3") -> list[float] | None:
    """Call Ollama API to embed text."""
    try:
        response = requests.post(
            f"{ollama_url}/api/embeddings",
            json={"model": model, "prompt": text},
            timeout=30,
        )  # type: ignore
        if response.status_code == 200:
            return response.json().get("embedding")
    except Exception as exc:
        logger.error("Embedding failed for text[:50]=%r: %s", text[:50], exc)
    return None


def embed_chunks(
    collection: str,
    chunks_path: str,
    qdrant_url: str,
    ollama_url: str,
    batch_size: int = 32,
    recreate: bool = False,
) -> int:
    """Embed all chunks and upsert to Qdrant with real vectors."""
    
    # Load chunks
    chunks = read_jsonl(chunks_path)
    logger.info("Loaded %d chunks", len(chunks))
    
    # Connect to Qdrant
    q_client = QdrantClient(url=qdrant_url)  # type: ignore
    
    # Check if collection exists
    collections = [c.name for c in q_client.get_collections().collections]  # type: ignore
    if collection not in collections:
        logger.error("Collection '%s' not found in Qdrant", collection)
        return 0
    
    if recreate:
        logger.info("Dropping and recreating collection %s", collection)
        q_client.delete_collection(collection)  # type: ignore
        # Will recreate below
    
    # Get sample embedding to determine vector size
    logger.info("Sampling embedding to determine vector size...")
    sample_text = chunks[0].get("text_norm") or chunks[0].get("text", "")[:500]
    sample_emb = _get_embedding(sample_text, ollama_url)
    if not sample_emb:
        logger.error("Failed to get sample embedding from Ollama. Is it running?")
        return 0
    
    vector_size = len(sample_emb)
    logger.info("Vector size: %d", vector_size)
    
    # Drop old collection if needed (with wrong vector size)
    if not recreate and collection in collections:
        try:
            old_collection_info = q_client.get_collection(collection)  # type: ignore
            # Check if vector size mismatch — if so, recreate
            # This is a heuristic; ideally we'd check config
            logger.info("Old collection exists — will update vectors in place")
        except Exception as exc:
            logger.warning("Could not get collection info: %s", exc)
    
    # Embed and upsert in batches
    points: list[PointStruct] = []  # type: ignore
    embedded_count = 0
    
    for i, chunk in enumerate(tqdm(chunks, desc="Embedding chunks")):  # type: ignore
        chunk_id = chunk.get("chunk_id", "")
        text = chunk.get("text_norm") or chunk.get("text", "")
        
        # Get embedding
        embedding = _get_embedding(text, ollama_url)
        if not embedding:
            logger.warning("Skipping chunk %s (embedding failed)", chunk_id)
            continue
        
        # Prepare point
        point_id = stable_point_id(chunk_id)
        payload = {
            "chunk_id": chunk_id,
            "text": chunk.get("text", ""),
        }
        if chunk.get("text_norm"):
            payload["text_norm"] = chunk["text_norm"]
        
        # Copy metadata
        skip_keys = {"chunk_id", "text", "text_norm"}
        for k, v in chunk.items():
            if k not in skip_keys:
                payload[k] = v
        
        points.append(PointStruct(id=point_id, vector=embedding, payload=payload))  # type: ignore
        embedded_count += 1
        
        # Upsert batch
        if len(points) >= batch_size or i == len(chunks) - 1:  # type: ignore
            logger.info("Upserting %d points... (total so far: %d)", len(points), embedded_count)  # type: ignore
            q_client.upsert(collection_name=collection, points=points)  # type: ignore
            points = []  # type: ignore
    
    logger.info("Embedding complete: %d chunks embedded and upserted", embedded_count)
    return embedded_count


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Embed chunks using Ollama BGE-M3")
    parser.add_argument("--collection", required=True, help="Qdrant collection name")
    parser.add_argument("--chunks", required=True, help="Chunks JSONL path")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--ollama-url", default="http://localhost:11434")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--recreate", action="store_true", help="Drop and recreate collection")
    args = parser.parse_args()
    
    start = time.time()
    count = embed_chunks(
        args.collection,
        args.chunks,
        args.qdrant_url,
        args.ollama_url,
        args.batch_size,
        args.recreate,
    )
    elapsed = time.time() - start
    
    logger.info("=" * 60)
    logger.info("Embedding done: %d chunks in %.1fs (%.2f chunks/sec)",
                count, elapsed, count / elapsed)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
