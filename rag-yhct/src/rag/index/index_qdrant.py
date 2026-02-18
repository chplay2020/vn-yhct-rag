"""B4 — Index chunks into Qdrant with dummy vectors.

Usage:
    python -m rag.index.index_qdrant --config config/config.yaml
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

import yaml
from qdrant_client import QdrantClient  # type: ignore
from qdrant_client.models import (  # type: ignore
    Distance,  # type: ignore
    PointStruct,  # type: ignore
    VectorParams,  # type: ignore
)

from rag.utils.hashing import stable_point_id
from rag.utils.io import read_jsonl

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")

BATCH_SIZE = 128

DISTANCE_MAP: dict[str, Any] = {
    "cosine": Distance.COSINE,  # type: ignore
    "euclid": Distance.EUCLID,  # type: ignore
    "dot": Distance.DOT,  # type: ignore
}


def _ensure_collection(
    client: Any,  # QdrantClient but using Any to avoid type issues
    collection: str,
    vector_size: int,
    distance: str,
    recreate: bool,
) -> None:
    """Create collection if needed; optionally recreate."""
    dist = DISTANCE_MAP.get(distance.lower(), Distance.COSINE)  # type: ignore
    collections = [c.name for c in client.get_collections().collections]  # type: ignore

    if recreate and collection in collections:
        logger.info("Recreating collection %s", collection)
        client.delete_collection(collection)  # type: ignore
        collections.remove(collection)  # type: ignore

    if collection not in collections:
        logger.info("Creating collection %s (vector_size=%d, distance=%s)", collection, vector_size, distance)
        client.create_collection(  # type: ignore
            collection_name=collection,
            vectors_config=VectorParams(size=vector_size, distance=dist),  # type: ignore
        )
    else:
        logger.info("Collection %s already exists", collection)


def run_index(config: dict[str, Any]) -> None:
    """Run B4 index into Qdrant."""
    q_cfg = config["qdrant"]
    idx_cfg = config["index"]

    url: str = q_cfg["url"]
    collection: str = q_cfg["collection"]
    vector_size: int = q_cfg["vector_size"]
    distance: str = q_cfg.get("distance", "cosine")
    recreate: bool = q_cfg.get("recreate", False)

    input_path = idx_cfg["input_chunks"]

    records = read_jsonl(input_path)
    if not records:
        logger.warning("No chunks to index from %s", input_path)
        return

    client = QdrantClient(url=url)  # type: ignore
    _ensure_collection(client, collection, vector_size, distance, recreate)  # type: ignore

    dummy_vector = [0.0] * vector_size
    total_upserted = 0

    # Batch upsert
    batch: list[Any] = []  # list[PointStruct] but using Any to avoid type issues
    for rec in records:
        chunk_id = rec.get("chunk_id", "")
        point_id = stable_point_id(chunk_id)

        payload = {
            "chunk_id": chunk_id,
            "text": rec.get("text", ""),
        }
        # Copy all metadata fields
        skip_keys = {"chunk_id", "text"}
        for k, v in rec.items():
            if k not in skip_keys:
                payload[k] = v

        batch.append(PointStruct(id=point_id, vector=dummy_vector, payload=payload))  # type: ignore

        if len(batch) >= BATCH_SIZE:
            client.upsert(collection_name=collection, points=batch)  # type: ignore
            total_upserted += len(batch)
            batch = []

    if batch:
        client.upsert(collection_name=collection, points=batch)  # type: ignore
        total_upserted += len(batch)

    logger.info("B4 Index complete: upserted %d points into collection '%s'", total_upserted, collection)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="B4 — Index into Qdrant")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    run_index(config)


if __name__ == "__main__":
    main()
