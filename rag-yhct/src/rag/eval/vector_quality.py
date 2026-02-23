"""Vector quality assessment — check if embeddings are good.

Usage:
    python -m rag.eval.vector_quality \
        --collection yhct_chunks_v2_full \
        --qdrant-url http://localhost:6333 \
        --chunks data/chunks/chunks_v2_full.jsonl \
        --sample-size 200 \
        --output data/reports/vector_quality.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any

import numpy as np  # type: ignore
from qdrant_client import QdrantClient  # type: ignore

from rag.utils.io import read_jsonl

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")


# ---------------------------------------------------------------------------
# Vector Statistics
# ---------------------------------------------------------------------------

def _vector_stats(vectors: list[list[float]]) -> dict[str, float | int]:
    """Compute basic stats on vectors (norms, dimensions)."""
    if not vectors:
        return {}
    
    arr = np.array(vectors)  # type: ignore
    norms = np.linalg.norm(arr, axis=1)  # type: ignore
    
    return {
        "vector_dim": int(arr.shape[1]),  # type: ignore
        "num_vectors": int(arr.shape[0]),  # type: ignore
        "norm_mean": float(np.mean(norms)),  # type: ignore
        "norm_std": float(np.std(norms)),  # type: ignore
        "norm_min": float(np.min(norms)),  # type: ignore
        "norm_max": float(np.max(norms)),  # type: ignore
    }


# ---------------------------------------------------------------------------
# Semantic Similarity Checks
# ---------------------------------------------------------------------------

def _cosine_similarity(v1: list[float], v2: list[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a = np.array(v1)
    b = np.array(v2)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _intra_source_similarity(
    client: Any,
    collection: str,
    chunk_source_map: dict[str, list[str]],
    sample_size: int = 100,
) -> dict[str, Any]:
    """Check if chunks from same source have similar vectors.
    
    Higher intra-source similarity (>0.7) suggests good clustering.
    """
    samples: list[tuple[str, list[str]]] = []
    for source_id, chunk_ids in chunk_source_map.items():
        if len(chunk_ids) >= 2:
            pair = random.sample(chunk_ids, 2)
            samples.append((source_id, pair))
            if len(samples) >= sample_size:
                break
    
    similarities: list[float] = []
    for source_id, (cid1, cid2) in samples:
        points = client.retrieve(collection, ids=[cid1, cid2], with_vectors=True)  # type: ignore
        if len(points) == 2:
            v1 = points[0].vector
            v2 = points[1].vector
            sim = _cosine_similarity(v1, v2)
            similarities.append(sim)
    
    if not similarities:
        return {"error": "No pairs found"}
    
    return {
        "sample_size": len(similarities),
        "mean_similarity": float(np.mean(similarities)),  # type: ignore
        "std_similarity": float(np.std(similarities)),  # type: ignore
        "min_similarity": float(np.min(similarities)),  # type: ignore
        "max_similarity": float(np.max(similarities)),  # type: ignore
        "high_similarity_ratio": float(sum(1 for s in similarities if s > 0.7) / len(similarities)),
    }


def _inter_source_dissimilarity(
    client: Any,
    collection: str,
    chunk_source_map: dict[str, list[str]],
    sample_size: int = 100,
) -> dict[str, Any]:
    """Check if chunks from different sources have dissimilar vectors.
    
    Lower inter-source similarity (<0.5) suggests good separation.
    """
    sources = list(chunk_source_map.keys())
    if len(sources) < 2:
        return {"error": "Need at least 2 sources"}
    
    similarities: list[float] = []
    for _ in range(sample_size):
        s1, s2 = random.sample(sources, 2)
        c1 = random.choice(chunk_source_map[s1])
        c2 = random.choice(chunk_source_map[s2])
        
        points = client.retrieve(collection, ids=[c1, c2])  # type: ignore
        if len(points) == 2:
            v1 = points[0].vector
            v2 = points[1].vector
            sim = _cosine_similarity(v1, v2)
            similarities.append(sim)
    
    if not similarities:
        return {"error": "No pairs found"}
    
    return {
        "sample_size": len(similarities),
        "mean_similarity": float(np.mean(similarities)),  # type: ignore
        "std_similarity": float(np.std(similarities)),  # type: ignore
        "min_similarity": float(np.min(similarities)),  # type: ignore
        "max_similarity": float(np.max(similarities)),  # type: ignore
        "low_similarity_ratio": float(sum(1 for s in similarities if s < 0.5) / len(similarities)),
    }


# ---------------------------------------------------------------------------
# Retrieval Quality (Mock Queries)
# ---------------------------------------------------------------------------

def _mock_retrieval_test(
    client: Any,
    collection: str,
    chunks: list[dict[str, Any]],
    sample_size: int = 50,
) -> dict[str, Any]:
    """Simulate retrieval by using chunk vectors as queries.
    
    For each sample chunk, retrieve its k-nearest neighbors and check
    if same-source chunks rank high.
    """
    if not chunks:
        return {"error": "No chunks"}
    
    sample_chunks = random.sample(chunks, min(sample_size, len(chunks)))
    source_ranks: list[int] = []
    
    for chunk in sample_chunks:
        source_id = chunk.get("source_id")
        chunk_id = chunk.get("chunk_id")
        if not chunk_id:
            continue
        
        # Retrieve this chunk's vector
        points = client.retrieve(collection, ids=[chunk_id])  # type: ignore
        if not points:
            continue
        query_vector = points[0].vector
        
        # Search for similar (top 10)
        results = client.search(
            collection_name=collection,
            query_vector=query_vector,
            limit=10,
        )  # type: ignore
        
        # Rank same-source chunks
        for rank, result in enumerate(results):
            retrieved_chunk_id = result.payload.get("chunk_id")
            # Find source_id of retrieved chunk
            for other_chunk in chunks:
                if other_chunk.get("chunk_id") == retrieved_chunk_id:
                    if other_chunk.get("source_id") == source_id:
                        source_ranks.append(rank)
                    break
    
    if not source_ranks:
        return {"error": "No relevant retrievals"}
    
    return {
        "sample_size": len(source_ranks),
        "mean_rank_of_same_source": float(np.mean(source_ranks)),  # type: ignore
        "median_rank": float(np.median(source_ranks)),  # type: ignore
        "within_top5_ratio": float(sum(1 for r in source_ranks if r < 5) / len(source_ranks)),
        "within_top10_ratio": float(sum(1 for r in source_ranks if r < 10) / len(source_ranks)),
    }


# ---------------------------------------------------------------------------
# Main Assessment
# ---------------------------------------------------------------------------

def assess_vector_quality(
    collection: str,
    qdrant_url: str,
    chunks_path: str,
    sample_size: int = 200,
) -> dict[str, Any]:
    """Comprehensive vector quality assessment."""
    report: dict[str, Any] = {"status": "ok"}
    
    client = QdrantClient(url=qdrant_url)  # type: ignore
    chunks = read_jsonl(chunks_path)
    
    logger.info("Loaded %d chunks", len(chunks))
    
    # --- Collection metadata ---
    collection_info = client.get_collection(collection)  # type: ignore
    point_count = client.count(collection, exact=True).count  # type: ignore
    
    # Extract vector size (try multiple attrs)
    vector_size: int | str = "unknown"
    if hasattr(collection_info, "config") and hasattr(collection_info.config, "vector"):  # type: ignore
        vector_size = collection_info.config.vector.size  # type: ignore
    elif hasattr(collection_info, "vectors_count"):  # type: ignore
        # vectors_count might be total, not dimension
        pass
    
    report["collection"] = {
        "name": collection,
        "point_count": point_count,
        "vector_size": vector_size,
    }
    
    # --- Vector statistics ---
    logger.info("Computing vector statistics...")
    # Scroll through collection to get sample vectors
    points_batch, _ = client.scroll(collection_name=collection, limit=min(sample_size, 500), with_vectors=True)  # type: ignore
    if points_batch:
        vectors: list[list[float]] = [p.vector for p in points_batch if p.vector]  # type: ignore
        report["vector_stats"] = _vector_stats(vectors)
    else:
        report["vector_stats"] = {"error": "No vectors found (collection may be empty or use dummy vectors)"}
    
    # --- Chunk mapping (source_id -> chunk_ids) ---
    logger.info("Building source-chunk mapping...")
    chunk_source_map: dict[str, list[str]] = {}
    for chunk in chunks:
        source_id = chunk.get("source_id", "unknown")
        chunk_id = chunk.get("chunk_id", "")
        if chunk_id:
            if source_id not in chunk_source_map:
                chunk_source_map[source_id] = []
            chunk_source_map[source_id].append(chunk_id)
    
    report["sources"] = {
        "unique_sources": len(chunk_source_map),
        "avg_chunks_per_source": float(np.mean([len(cids) for cids in chunk_source_map.values()])),  # type: ignore
    }
    
    # --- Intra-source similarity (should be high) ---
    logger.info("Testing intra-source similarity...")
    try:
        # Check if vectors are available
        test_points, _ = client.scroll(collection_name=collection, limit=1, with_vectors=True)  # type: ignore
        
        if test_points and test_points[0].vector and all(v != 0 for v in test_points[0].vector):  # type: ignore # pyright: ignore
            report["intra_source"] = _intra_source_similarity(client, collection, chunk_source_map, sample_size)
        else:
            report["intra_source"] = {"skipped": "Dummy vectors detected (all zeros)"}
    except Exception as exc:
        report["intra_source"] = {"error": str(exc)}
    
    # --- Inter-source dissimilarity (should be low) ---
    logger.info("Testing inter-source dissimilarity...")
    try:
        test_points, _ = client.scroll(collection_name=collection, limit=1, with_vectors=True)  # type: ignore
        if test_points and test_points[0].vector and all(v != 0 for v in test_points[0].vector):  # type: ignore # pyright: ignore
            report["inter_source"] = _inter_source_dissimilarity(client, collection, chunk_source_map, sample_size)
        else:
            report["inter_source"] = {"skipped": "Dummy vectors detected (all zeros)"}
    except Exception as exc:
        report["inter_source"] = {"error": str(exc)}
    
    # --- Mock retrieval test ---
    logger.info("Testing mock retrieval...")
    try:
        test_points, _ = client.scroll(collection_name=collection, limit=1, with_vectors=True)  # type: ignore
        if test_points and test_points[0].vector and all(v != 0 for v in test_points[0].vector):  # type: ignore # pyright: ignore
            report["retrieval"] = _mock_retrieval_test(client, collection, chunks, sample_size=50)
        else:
            report["retrieval"] = {"skipped": "Dummy vectors detected (all zeros)"}
    except Exception as exc:
        report["retrieval"] = {"error": str(exc)}
    
    # --- Health check ---
    logger.info("Overall quality assessment...")
    issues: list[str] = []
    norm_std = report["vector_stats"].get("norm_std", 0)
    if isinstance(norm_std, (int, float)) and norm_std > 2.0:  # type: ignore
        issues.append("Vector norm variation too high (vectors not normalized)")
    intra_sim = report.get("intra_source", {}).get("mean_similarity", 0)
    if isinstance(intra_sim, (int, float)) and intra_sim < 0.5:  # type: ignore
        issues.append("Intra-source similarity too low (poor clustering)")
    ret_ratio = report.get("retrieval", {}).get("within_top5_ratio", 0)
    if isinstance(ret_ratio, (int, float)) and ret_ratio < 0.6:  # type: ignore
        issues.append("Retrieval accuracy low (< 60% relevant in top 5)")
    
    report["assessment"] = {
        "status": "GOOD" if not issues else "WARNING",
        "issues": issues,
    }
    
    return report


def print_summary(report: dict[str, Any]) -> None:
    """Print human-readable summary."""
    print("\n" + "=" * 60)
    print("  VECTOR QUALITY ASSESSMENT")
    print("=" * 60)
    coll = report.get("collection", {})
    print(f"  Collection: {coll.get('name', '?')} ({coll.get('point_count', 0)} points)")
    print(f"  Vector dim: {report.get('vector_stats', {}).get('vector_dim', '?')}")
    
    stats = report.get("vector_stats", {})
    print(f"  Vector norms: mean={stats.get('norm_mean', 0):.3f}, std={stats.get('norm_std', 0):.3f}")
    
    sources = report.get("sources", {})
    print(f"  Sources: {sources.get('unique_sources', 0)} unique")
    
    intra = report.get("intra_source", {})
    if "error" not in intra:
        print(f"  Intra-source sim: {intra.get('mean_similarity', 0):.3f} (should be >0.7)")
        print(f"    High-sim ratio: {intra.get('high_similarity_ratio', 0):.1%}")
    
    inter = report.get("inter_source", {})
    if "error" not in inter:
        print(f"  Inter-source sim: {inter.get('mean_similarity', 0):.3f} (should be <0.5)")
        print(f"    Low-sim ratio: {inter.get('low_similarity_ratio', 0):.1%}")
    
    ret = report.get("retrieval", {})
    if "error" not in ret:
        print(f"  Retrieval: top-5 accuracy={ret.get('within_top5_ratio', 0):.1%}")
    
    assess = report.get("assessment", {})
    status = assess.get("status", "UNKNOWN")
    print(f"  Assessment: {status}")
    if assess.get("issues"):
        print("  Issues:")
        for issue in assess["issues"]:
            print(f"    - {issue}")
    
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Vector quality assessment")
    parser.add_argument("--collection", required=True, help="Qdrant collection name")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--chunks", required=True, help="Chunks JSONL path")
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--output", default=None, help="Save report JSON")
    args = parser.parse_args()
    
    report = assess_vector_quality(
        args.collection,
        args.qdrant_url,
        args.chunks,
        args.sample_size,
    )
    
    print_summary(report)
    
    if args.output:
        p = Path(args.output)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info("Report saved to %s", args.output)


if __name__ == "__main__":
    main()
