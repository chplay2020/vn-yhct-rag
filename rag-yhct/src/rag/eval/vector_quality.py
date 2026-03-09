# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""
Vector quality assessment — check if embeddings are good.

Usage:
    python -m rag.eval.vector_quality \
        --collection yhct_chunks_v2_full_emb \
        --qdrant-url http://localhost:6333 \
        --chunks data/chunks/chunks_v2_full.jsonl \
        --sample-size 200 \
        --group-by parent \
        --output data/reports/vector_quality.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np  # type: ignore
from qdrant_client import QdrantClient  # type: ignore

from rag.utils.io import read_jsonl
from rag.utils.hashing import stable_point_id

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _extract_vector(v: Any) -> Optional[List[float]]:
    """
    Qdrant may return:
      - vector: [..] (single)
      - vector: {"default":[..]} (named)
    """
    if v is None:
        return None
    if isinstance(v, list):
        return list(v)
    if isinstance(v, dict):
        # take first named vector if any
        for _vv in v.values():
            if isinstance(_vv, list):
                return list(_vv)
    return None


def _cosine_similarity(v1: List[float], v2: List[float]) -> float:
    a = np.array(v1, dtype=np.float64)
    b = np.array(v2, dtype=np.float64)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def _vector_stats(vectors: List[List[float]]) -> Dict[str, float | int]:
    if not vectors:
        return {}
    arr = np.array(vectors, dtype=np.float64)
    norms: Any = np.linalg.norm(arr, axis=1)
    return {
        "vector_dim": int(arr.shape[1]),
        "num_vectors": int(arr.shape[0]),
        "norm_mean": float(np.mean(norms)),
        "norm_std": float(np.std(norms)),
        "norm_min": float(np.min(norms)),
        "norm_max": float(np.max(norms)),
    }


def _build_maps(
    chunks: List[Dict[str, Any]],
    group_by: str,
) -> Tuple[Dict[int, str], Dict[int, str], Dict[str, List[int]]]:
    """
    Returns:
      point_id_to_source
      point_id_to_group (group = source_id or parent_id fallback)
      group_to_point_ids
    """
    pid_to_source: Dict[int, str] = {}
    pid_to_group: Dict[int, str] = {}
    group_to_pids: Dict[str, List[int]] = {}

    for c in chunks:
        cid = c.get("chunk_id")
        if not cid:
            continue
        pid = stable_point_id(str(cid))
        source_id = str(c.get("source_id", "unknown"))

        if group_by == "parent":
            parent_id = c.get("parent_id")
            group_id = str(parent_id) if parent_id else source_id  # fallback nếu chưa có parent_id
        else:
            group_id = source_id

        pid_to_source[pid] = source_id
        pid_to_group[pid] = group_id
        group_to_pids.setdefault(group_id, []).append(pid)

    return pid_to_source, pid_to_group, group_to_pids


# ---------------------------------------------------------------------------
# Similarity checks
# ---------------------------------------------------------------------------

def _intra_group_similarity(
    client: Any,
    collection: str,
    group_to_pids: Dict[str, List[int]],
    sample_size: int = 100,
) -> Dict[str, Any]:
    """
    Sample pairs within same group (source or parent) and compute cosine similarity.
    """
    groups = [g for g, ids in group_to_pids.items() if len(ids) >= 2]
    if not groups:
        return {"error": "No groups with >=2 points"}

    pairs: List[Tuple[int, int]] = []
    random.shuffle(groups)
    for g in groups:
        ids = group_to_pids[g]
        a, b = random.sample(ids, 2)
        pairs.append((a, b))
        if len(pairs) >= sample_size:
            break

    sims: List[float] = []
    for a, b in pairs:
        pts = client.retrieve(collection_name=collection, ids=[a, b], with_vectors=True)  # type: ignore
        if len(pts) != 2:
            continue
        v1 = _extract_vector(getattr(pts[0], "vector", None))
        v2 = _extract_vector(getattr(pts[1], "vector", None))
        if v1 is None or v2 is None:
            continue
        sims.append(_cosine_similarity(v1, v2))

    if not sims:
        return {"error": "No vector pairs retrieved"}

    return {
        "sample_size": len(sims),
        "mean_similarity": float(np.mean(sims)),
        "std_similarity": float(np.std(sims)),
        "min_similarity": float(np.min(sims)),
        "max_similarity": float(np.max(sims)),
        "high_similarity_ratio_0.7": float(sum(1 for s in sims if s > 0.7) / len(sims)),
    }


def _inter_group_dissimilarity(
    client: Any,
    collection: str,
    group_to_pids: Dict[str, List[int]],
    sample_size: int = 100,
) -> Dict[str, Any]:
    """
    Sample pairs from different groups and compute cosine similarity.
    """
    groups = [g for g, ids in group_to_pids.items() if len(ids) >= 1]
    if len(groups) < 2:
        return {"error": "Need at least 2 groups"}

    sims: List[float] = []
    for _ in range(sample_size):
        g1, g2 = random.sample(groups, 2)
        a = random.choice(group_to_pids[g1])
        b = random.choice(group_to_pids[g2])
        pts = client.retrieve(collection_name=collection, ids=[a, b], with_vectors=True)  # type: ignore
        if len(pts) != 2:
            continue
        v1 = _extract_vector(getattr(pts[0], "vector", None))
        v2 = _extract_vector(getattr(pts[1], "vector", None))
        if v1 is None or v2 is None:
            continue
        sims.append(_cosine_similarity(v1, v2))

    if not sims:
        return {"error": "No vector pairs retrieved"}

    return {
        "sample_size": len(sims),
        "mean_similarity": float(np.mean(sims)),
        "std_similarity": float(np.std(sims)),
        "min_similarity": float(np.min(sims)),
        "max_similarity": float(np.max(sims)),
        "low_similarity_ratio_0.5": float(sum(1 for s in sims if s < 0.5) / len(sims)),
    }


# ---------------------------------------------------------------------------
# Mock retrieval test (vector-as-query)
# ---------------------------------------------------------------------------

def _mock_retrieval_test(
    client: Any,
    collection: str,
    point_ids: List[int],
    pid_to_group: Dict[int, str],
    pid_to_source: Dict[int, str],
    sample_size: int = 50,
    k: int = 10,
) -> Dict[str, Any]:
    """
    Use a point's own vector as query (proxy sanity check).
    Measure rank of first "same group" hit (excluding itself).
    Also report hit@5/hit@10 for same group and same source.
    """
    if not point_ids:
        return {"error": "No point ids"}

    sample_pids = random.sample(point_ids, min(sample_size, len(point_ids)))

    first_rank_same_group: List[int] = []
    first_rank_same_source: List[int] = []
    hit5_group = 0
    hit10_group = 0
    hit5_source = 0
    hit10_source = 0
    used = 0

    for pid in sample_pids:
        pts = client.retrieve(collection_name=collection, ids=[pid], with_vectors=True)  # type: ignore
        if not pts:
            continue
        qv = _extract_vector(getattr(pts[0], "vector", None))
        if qv is None:
            continue

        results = client.search(collection_name=collection, query_vector=qv, limit=k + 1, with_payload=False)  # type: ignore

        my_group = pid_to_group.get(pid)
        my_source = pid_to_source.get(pid)

        rank_g = None
        rank_s = None

        # iterate results, skip itself if appears
        for rank, r in enumerate(results):
            rid = getattr(r, "id", None)
            if rid is None:
                continue
            if int(rid) == int(pid):
                continue

            if rank_g is None and pid_to_group.get(int(rid)) == my_group:
                rank_g = rank
            if rank_s is None and pid_to_source.get(int(rid)) == my_source:
                rank_s = rank

        if rank_g is not None:
            first_rank_same_group.append(rank_g)
            if rank_g < 5:
                hit5_group += 1
            if rank_g < 10:
                hit10_group += 1
        if rank_s is not None:
            first_rank_same_source.append(rank_s)
            if rank_s < 5:
                hit5_source += 1
            if rank_s < 10:
                hit10_source += 1

        used += 1

    if used == 0:
        return {"error": "No retrievals"}

    return {
        "sample_size": used,
        "k": k,
        "same_group": {
            "mean_first_rank": float(np.mean(first_rank_same_group)) if first_rank_same_group else None,
            "median_first_rank": float(np.median(first_rank_same_group)) if first_rank_same_group else None,
            "hit@5": float(hit5_group / used),
            "hit@10": float(hit10_group / used),
        },
        "same_source": {
            "mean_first_rank": float(np.mean(first_rank_same_source)) if first_rank_same_source else None,
            "median_first_rank": float(np.median(first_rank_same_source)) if first_rank_same_source else None,
            "hit@5": float(hit5_source / used),
            "hit@10": float(hit10_source / used),
        },
    }


# ---------------------------------------------------------------------------
# Main Assessment
# ---------------------------------------------------------------------------

def assess_vector_quality(
    collection: str,
    qdrant_url: str,
    chunks_path: str,
    sample_size: int = 200,
    group_by: str = "source",
) -> Dict[str, Any]:
    report: Dict[str, Any] = {"status": "ok"}

    client: Any = QdrantClient(url=qdrant_url)  # type: ignore
    chunks = read_jsonl(chunks_path)
    logger.info("Loaded %d chunks (from %s)", len(chunks), chunks_path)

    # Build maps (important: repo uses point_id=int(stable_point_id(chunk_id)))
    pid_to_source, pid_to_group, group_to_pids = _build_maps(chunks, group_by=group_by)
    all_pids = list(pid_to_group.keys())

    report["grouping"] = {
        "group_by": group_by,
        "unique_groups": len(group_to_pids),
        "avg_points_per_group": float(np.mean([len(v) for v in group_to_pids.values()])) if group_to_pids else 0.0,
    }

    # Collection metadata
    try:
        point_count = client.count(collection_name=collection, exact=True).count  # type: ignore
    except Exception:
        point_count = None

    report["collection"] = {
        "name": collection,
        "point_count_exact": point_count,
    }

    # Vector statistics (sample via scroll)
    logger.info("Computing vector statistics...")
    points_batch, _ = client.scroll(collection_name=collection, limit=min(sample_size, 500), with_vectors=True)  # type: ignore
    vectors: List[List[float]] = []
    for p in points_batch:  # type: ignore
        v = _extract_vector(getattr(p, "vector", None))
        if v is not None:
            vectors.append(v)
    report["vector_stats"] = _vector_stats(vectors) if vectors else {"error": "No vectors sampled"}

    # Intra / inter similarity by group
    logger.info("Testing intra-%s similarity...", group_by)
    report["intra_group"] = _intra_group_similarity(client, collection, group_to_pids, sample_size=min(sample_size, 200))

    logger.info("Testing inter-%s dissimilarity...", group_by)
    report["inter_group"] = _inter_group_dissimilarity(client, collection, group_to_pids, sample_size=min(sample_size, 200))

    # Mock retrieval
    logger.info("Testing mock retrieval (vector-as-query)...")
    report["retrieval"] = _mock_retrieval_test(
        client,
        collection,
        all_pids,
        pid_to_group=pid_to_group,
        pid_to_source=pid_to_source,
        sample_size=min(50, max(10, sample_size // 4)),
        k=10,
    )

    # Health check (very light heuristic)
    logger.info("Overall quality assessment...")
    issues: List[str] = []

    norm_std = report.get("vector_stats", {}).get("norm_std")
    if isinstance(norm_std, (int, float)) and norm_std > 2.0:
        issues.append("Vector norm variation too high (vectors may be unnormalized or inconsistent).")

    intra_sim = report.get("intra_group", {}).get("mean_similarity")
    if isinstance(intra_sim, (int, float)) and intra_sim < 0.45:
        issues.append("Intra-group similarity low (group clustering weak).")

    hit5 = report.get("retrieval", {}).get("same_group", {}).get("hit@5")
    if isinstance(hit5, (int, float)) and hit5 < 0.50:
        issues.append("Mock retrieval hit@5 (same group) is low (<50%).")

    report["assessment"] = {"status": "GOOD" if not issues else "WARNING", "issues": issues}
    return report


def print_summary(report: Dict[str, Any]) -> None:
    print("\n" + "=" * 60)
    print("  VECTOR QUALITY ASSESSMENT")
    print("=" * 60)

    coll = report.get("collection", {})
    grp = report.get("grouping", {})
    print(f"  Collection: {coll.get('name', '?')} (exact_count={coll.get('point_count_exact', '?')})")
    print(f"  Group-by: {grp.get('group_by')}  groups={grp.get('unique_groups')}  avg_points/group={grp.get('avg_points_per_group', 0):.1f}")

    stats = report.get("vector_stats", {})
    if "error" in stats:
        print(f"  Vector stats: {stats['error']}")
    else:
        print(f"  Vector dim: {stats.get('vector_dim', '?')}")
        print(f"  Norms: mean={stats.get('norm_mean', 0):.3f}, std={stats.get('norm_std', 0):.3f}, min={stats.get('norm_min', 0):.3f}, max={stats.get('norm_max', 0):.3f}")

    intra = report.get("intra_group", {})
    if "error" in intra:
        print(f"  Intra-group: {intra['error']}")
    else:
        print(f"  Intra-group sim: mean={intra.get('mean_similarity', 0):.3f}, high>0.7={intra.get('high_similarity_ratio_0.7', 0):.1%}")

    inter = report.get("inter_group", {})
    if "error" in inter:
        print(f"  Inter-group: {inter['error']}")
    else:
        print(f"  Inter-group sim: mean={inter.get('mean_similarity', 0):.3f}, low<0.5={inter.get('low_similarity_ratio_0.5', 0):.1%}")

    ret = report.get("retrieval", {})
    if "error" in ret:
        print(f"  Retrieval: {ret['error']}")
    else:
        sg = ret.get("same_group", {})
        ss = ret.get("same_source", {})
        print(f"  Retrieval same-group: hit@5={sg.get('hit@5', 0):.1%}, hit@10={sg.get('hit@10', 0):.1%}, median_first_rank={sg.get('median_first_rank')}")
        print(f"  Retrieval same-source: hit@5={ss.get('hit@5', 0):.1%}, hit@10={ss.get('hit@10', 0):.1%}, median_first_rank={ss.get('median_first_rank')}")

    assess = report.get("assessment", {})
    print(f"  Assessment: {assess.get('status', 'UNKNOWN')}")
    if assess.get("issues"):
        print("  Issues:")
        for issue in assess["issues"]:
            print(f"    - {issue}")
    print("=" * 60 + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Vector quality assessment")
    parser.add_argument("--collection", required=True, help="Qdrant collection name (use *_emb for real vectors)")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--chunks", required=True, help="Chunks JSONL path")
    parser.add_argument("--sample-size", type=int, default=200)
    parser.add_argument("--group-by", choices=["source", "parent"], default="source",
                        help="Evaluate clustering by source_id or parent_id (fallback to source_id if missing)")
    parser.add_argument("--output", default=None, help="Save report JSON")
    args = parser.parse_args()

    report = assess_vector_quality(
        collection=args.collection,
        qdrant_url=args.qdrant_url,
        chunks_path=args.chunks,
        sample_size=args.sample_size,
        group_by=args.group_by,
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