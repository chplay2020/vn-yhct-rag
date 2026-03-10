# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
"""Hybrid retriever — BM25 + Vector + RRF fusion.

Supports three retrieval modes: vector, bm25, hybrid_rrf.
Outputs uniform result format, optional debug JSON save,
and full parent-child compatibility.

Usage (CLI):
    # Hybrid RRF retrieval:
    PYTHONPATH=src uv run python -m rag.retrieve.hybrid_retriever \
        --query "tác dụng của cây ngải cứu" --mode hybrid_rrf

    # Vector-only:
    PYTHONPATH=src uv run python -m rag.retrieve.hybrid_retriever \
        --query "tác dụng của cây ngải cứu" --mode vector

    # BM25-only:
    PYTHONPATH=src uv run python -m rag.retrieve.hybrid_retriever \
        --query "tác dụng của cây ngải cứu" --mode bm25

    # Save debug JSON:
    PYTHONPATH=src uv run python -m rag.retrieve.hybrid_retriever \
        --query "tác dụng của cây ngải cứu" --mode hybrid_rrf --save-debug
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path
from typing import Any

import yaml  # type: ignore

from rag.retrieve.bm25_retriever import (
    DEFAULT_CHUNKS_PATH,
    DEFAULT_INDEX_PATH,
    retrieve_bm25,
)
from rag.retrieve.vector_retriever import (
    DEFAULT_COLLECTION,
    DEFAULT_MODEL,
    DEFAULT_OLLAMA_URL,
    DEFAULT_QDRANT_URL,
    retrieve_vector,
)
from rag.utils.query_quality import normalize_for_dedup

logger = logging.getLogger(__name__)

# ── defaults ───────────────────────────────────────────────────────────────
DEFAULT_MODE = "hybrid_rrf"
DEFAULT_TOPK_VECTOR = 40
DEFAULT_TOPK_BM25 = 40
DEFAULT_TOPK_FINAL = 40
DEFAULT_RRF_K = 60
DEFAULT_DEBUG_DIR = "data/reports/retrieval_debug"
DEFAULT_WINDOW_CENTERS = 2
DEFAULT_PARENT_SCORE_AGG = "max"  # "max" or "sum_top2"


# ── RRF fusion ─────────────────────────────────────────────────────────────

def reciprocal_rank_fusion(
    bm25_results: list[dict[str, Any]],
    vector_results: list[dict[str, Any]],
    *,
    k: int = DEFAULT_RRF_K,
    topk_final: int = DEFAULT_TOPK_FINAL,
) -> list[dict[str, Any]]:
    """Merge BM25 and vector ranked lists using Reciprocal Rank Fusion.

    Formula: RRF_score(doc) = sum(1 / (k + rank_i)) over all lists containing doc.

    Returns merged list sorted by fused score descending, deduped by chunk_id.
    """

    # chunk_id → accumulated info
    merged: dict[str, dict[str, Any]] = {}

    # Process BM25 results
    for r in bm25_results:
        cid = r["chunk_id"]
        rank = r["rank"]  # 0-based
        rrf_contrib = 1.0 / (k + rank + 1)  # rank+1 for 1-based in formula

        if cid not in merged:
            merged[cid] = {
                "chunk_id": cid,
                "fused_score": 0.0,
                "bm25_rank": None,
                "bm25_score": None,
                "vector_rank": None,
                "vector_score": None,
                "source_id": r.get("source_id", ""),
                "parent_id": r.get("parent_id", ""),
                "child_index": r.get("child_index"),
                "doc_type": r.get("doc_type", ""),
                "category": r.get("category", ""),
                "text": r.get("text", ""),
            }
        merged[cid]["fused_score"] += rrf_contrib
        merged[cid]["bm25_rank"] = rank
        merged[cid]["bm25_score"] = r.get("bm25_score")

    # Process vector results
    for r in vector_results:
        cid = r["chunk_id"]
        rank = r["rank"]  # 0-based
        rrf_contrib = 1.0 / (k + rank + 1)

        if cid not in merged:
            merged[cid] = {
                "chunk_id": cid,
                "fused_score": 0.0,
                "bm25_rank": None,
                "bm25_score": None,
                "vector_rank": None,
                "vector_score": None,
                "source_id": r.get("source_id", ""),
                "parent_id": r.get("parent_id", ""),
                "child_index": r.get("child_index"),
                "doc_type": r.get("doc_type", ""),
                "category": r.get("category", ""),
                "text": r.get("text", ""),
            }
        merged[cid]["fused_score"] += rrf_contrib
        merged[cid]["vector_rank"] = rank
        merged[cid]["vector_score"] = r.get("vector_score")

    # Sort by fused_score descending
    ranked = sorted(merged.values(), key=lambda x: x["fused_score"], reverse=True)
    ranked = ranked[:topk_final]

    # Assign final ranks and round scores
    for i, item in enumerate(ranked):
        item["rank"] = i
        item["fused_score"] = round(item["fused_score"], 6)

    return ranked


# ── duplicate suppression ─────────────────────────────────────────────────

def _deduplicate_results(
    results: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Remove results whose text is an exact duplicate (after normalisation).

    Keeps the highest-score (earliest-rank) version of each text.
    """
    seen_texts: set[str] = set()
    out: list[dict[str, Any]] = []
    for r in results:
        norm = normalize_for_dedup(r.get("text", ""))
        if not norm:
            out.append(r)
            continue
        if norm in seen_texts:
            continue
        seen_texts.add(norm)
        out.append(r)
    # Re-rank
    for i, item in enumerate(out):
        item["rank"] = i
    return out


# ── unified retrieval ─────────────────────────────────────────────────────

def retrieve(
    query: str,
    *,
    mode: str = DEFAULT_MODE,
    topk_vector: int = DEFAULT_TOPK_VECTOR,
    topk_bm25: int = DEFAULT_TOPK_BM25,
    topk_final: int = DEFAULT_TOPK_FINAL,
    rrf_k: int = DEFAULT_RRF_K,
    # vector params
    collection: str = DEFAULT_COLLECTION,
    qdrant_url: str = DEFAULT_QDRANT_URL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    model: str = DEFAULT_MODEL,
    # bm25 params
    chunks_path: str = DEFAULT_CHUNKS_PATH,
    index_path: str = DEFAULT_INDEX_PATH,
    # filters
    doc_type_filter: str | None = None,
    # debug
    save_debug: bool = False,
    debug_dir: str = DEFAULT_DEBUG_DIR,
    # dedup
    deduplicate: bool = True,
) -> list[dict[str, Any]]:
    """Unified retrieval API supporting vector, bm25, and hybrid_rrf modes.

    Returns a list of result dicts with uniform schema:
      rank, chunk_id, source_id, parent_id, child_index,
      bm25_score, vector_score, fused_score, text, doc_type, category
    """

    bm25_results: list[dict[str, Any]] = []
    vector_results: list[dict[str, Any]] = []

    if mode in ("bm25", "hybrid_rrf"):
        bm25_results = retrieve_bm25(
            query,
            topk=topk_bm25,
            chunks_path=chunks_path,
            index_path=index_path,
            doc_type_filter=doc_type_filter,
        )

    if mode in ("vector", "hybrid_rrf"):
        vector_results = retrieve_vector(
            query,
            topk=topk_vector,
            collection=collection,
            qdrant_url=qdrant_url,
            ollama_url=ollama_url,
            model=model,
            doc_type_filter=doc_type_filter,
        )

    # Build final results based on mode
    if mode == "vector":
        results = _normalize_vector_results(vector_results, topk_final)
    elif mode == "bm25":
        results = _normalize_bm25_results(bm25_results, topk_final)
    elif mode == "hybrid_rrf":
        results = reciprocal_rank_fusion(
            bm25_results, vector_results,
            k=rrf_k, topk_final=topk_final,
        )
    else:
        raise ValueError(f"Unknown retrieval mode: {mode!r}. Use: vector, bm25, hybrid_rrf")

    # Duplicate suppression
    if deduplicate:
        results = _deduplicate_results(results)

    # Save debug if requested
    if save_debug:
        _save_debug_json(
            query, mode, topk_final, results, debug_dir,
            bm25_raw=bm25_results, vector_raw=vector_results,
        )

    return results


def _normalize_vector_results(
    results: list[dict[str, Any]], topk: int,
) -> list[dict[str, Any]]:
    """Normalize vector results to uniform schema."""
    out = []
    for r in results[:topk]:
        out.append({
            "rank": r["rank"],
            "chunk_id": r["chunk_id"],
            "fused_score": r.get("vector_score"),
            "bm25_rank": None,
            "bm25_score": None,
            "vector_rank": r["rank"],
            "vector_score": r.get("vector_score"),
            "source_id": r.get("source_id", ""),
            "parent_id": r.get("parent_id", ""),
            "child_index": r.get("child_index"),
            "doc_type": r.get("doc_type", ""),
            "category": r.get("category", ""),
            "text": r.get("text", ""),
        })
    return out


def _normalize_bm25_results(
    results: list[dict[str, Any]], topk: int,
) -> list[dict[str, Any]]:
    """Normalize BM25 results to uniform schema."""
    out = []
    for r in results[:topk]:
        out.append({
            "rank": r["rank"],
            "chunk_id": r["chunk_id"],
            "fused_score": r.get("bm25_score"),
            "bm25_rank": r["rank"],
            "bm25_score": r.get("bm25_score"),
            "vector_rank": None,
            "vector_score": None,
            "source_id": r.get("source_id", ""),
            "parent_id": r.get("parent_id", ""),
            "child_index": r.get("child_index"),
            "doc_type": r.get("doc_type", ""),
            "category": r.get("category", ""),
            "text": r.get("text", ""),
        })
    return out


# ── debug save ────────────────────────────────────────────────────────────

def _save_debug_json(
    query: str,
    mode: str,
    topk: int,
    results: list[dict[str, Any]],
    debug_dir: str,
    *,
    bm25_raw: list[dict[str, Any]] | None = None,
    vector_raw: list[dict[str, Any]] | None = None,
) -> None:
    """Save retrieval results to debug JSON file."""
    import hashlib

    debug_path = Path(debug_dir)
    debug_path.mkdir(parents=True, exist_ok=True)

    # Create a short hash of the query for the filename
    q_hash = hashlib.md5(query.encode(), usedforsecurity=False).hexdigest()[:8]
    filename = f"{mode}_{q_hash}.json"
    filepath = debug_path / filename

    report: dict[str, Any] = {
        "query": query,
        "mode": mode,
        "top_k": topk,
        "results": results,
    }
    if bm25_raw is not None:
        report["bm25_candidates"] = [
            {"rank": r["rank"], "chunk_id": r["chunk_id"], "bm25_score": r.get("bm25_score")}
            for r in bm25_raw[:20]
        ]
    if vector_raw is not None:
        report["vector_candidates"] = [
            {"rank": r["rank"], "chunk_id": r["chunk_id"], "vector_score": r.get("vector_score")}
            for r in vector_raw[:20]
        ]

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    logger.info("Debug results saved to %s", filepath)


# ── config loader ─────────────────────────────────────────────────────────

def load_retrieval_config(config_path: str = "config/config.yaml") -> dict[str, Any]:
    """Load retrieval config from YAML, with defaults."""
    defaults: dict[str, Any] = {
        "mode": DEFAULT_MODE,
        "topk_vector": DEFAULT_TOPK_VECTOR,
        "topk_bm25": DEFAULT_TOPK_BM25,
        "topk_final": DEFAULT_TOPK_FINAL,
        "rrf_k": DEFAULT_RRF_K,
        "save_debug": False,
        "debug_dir": DEFAULT_DEBUG_DIR,
    }
    try:
        with open(config_path, encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        retrieval_cfg = cfg.get("retrieval", {})
        for key, default_val in defaults.items():
            if key not in retrieval_cfg:
                retrieval_cfg[key] = default_val
        return retrieval_cfg
    except FileNotFoundError:
        logger.warning("Config file not found: %s — using defaults", config_path)
        return defaults


# ── parent-child context builder integration ──────────────────────────────

def build_parent_child_context(
    results: list[dict[str, Any]],
    *,
    parents_path: str = "data/parents/parents_v2_full.jsonl",
    topk_parent: int = 4,
    window: int = 1,
    window_centers: int = DEFAULT_WINDOW_CENTERS,
    parent_score_agg: str = DEFAULT_PARENT_SCORE_AGG,
    token_budget: int = 3500,
    context_mode: str = "parent_child",
    deduplicate: bool = True,
) -> dict[str, Any]:
    """Build context from retrieval results with parent-child windowing support.

    Args:
        results: output from retrieve() in any mode
        parents_path: path to parents JSONL
        topk_parent: max parents to include
        window: ±W children around each selected center
        window_centers: top-M child centers per selected parent
        parent_score_agg: 'max' (default) or 'sum_top2'
        token_budget: max tokens for context
        context_mode: 'parent_child' or 'flat'
        deduplicate: suppress exact-duplicate text blocks

    Returns dict with: context, parents, children, tokens_used, mode, debug
    """
    import tiktoken  # type: ignore
    from rag.retrieve.parent_child_retriever import load_parents

    enc: Any = tiktoken.get_encoding("cl100k_base")
    debug_info: dict[str, Any] = {}

    if context_mode == "flat":
        # Flat mode: just concatenate child texts, with optional dedup
        context_parts: list[str] = []
        seen_texts: set[str] = set()
        tokens_used = 0
        for r in results:
            text = r.get("text", "")
            if deduplicate:
                norm = normalize_for_dedup(text)
                if norm in seen_texts:
                    continue
                seen_texts.add(norm)
            t_len = len(enc.encode(text))
            if tokens_used + t_len > token_budget:
                break
            context_parts.append(text)
            tokens_used += t_len
        return {
            "context": "\n\n---\n\n".join(context_parts),
            "parents": [],
            "children": results,
            "tokens_used": tokens_used,
            "mode": "flat",
            "debug": {},
        }

    # Parent-child mode: group by parent_id, window context
    parent_cache = load_parents(parents_path)

    # Group results by parent_id
    parent_groups: dict[str, list[dict[str, Any]]] = {}
    for r in results:
        pid = r.get("parent_id") or r.get("source_id", "unknown")
        parent_groups.setdefault(pid, []).append(r)

    # Score each parent
    parent_scores: list[tuple[str, float, list[dict[str, Any]]]] = []
    for pid, group in parent_groups.items():
        child_scores = sorted(
            [
                r.get("fused_score") or r.get("vector_score") or r.get("bm25_score") or 0.0
                for r in group
            ],
            reverse=True,
        )
        if parent_score_agg == "sum_top2":
            score = sum(child_scores[:2])
        else:  # default: max
            score = child_scores[0] if child_scores else 0.0
        parent_scores.append((pid, score, group))

    parent_scores.sort(key=lambda x: x[1], reverse=True)
    chosen_parents = parent_scores[:topk_parent]

    debug_info["parent_scores"] = [
        {"parent_id": pid, "score": round(sc, 6), "children_in_topk": len(g)}
        for pid, sc, g in parent_scores[:8]
    ]
    debug_info["selected_parent_ids"] = [pid for pid, _, _ in chosen_parents]

    # Build context with multi-center windowing
    context_blocks: list[str] = []
    seen_block_texts: set[str] = set()
    parent_results: list[dict[str, Any]] = []
    tokens_used = 0

    for pid, score, group_children in chosen_parents:
        parent_rec = parent_cache.get(pid)

        # Select top-M child centers sorted by score
        group_sorted = sorted(
            group_children,
            key=lambda c: c.get("fused_score") or c.get("vector_score") or c.get("bm25_score") or 0.0,
            reverse=True,
        )
        centers = group_sorted[:window_centers]
        center_indices: list[int] = [
            c["child_index"] for c in centers
            if isinstance(c.get("child_index"), int)
        ]

        title = (parent_rec or centers[0]).get("title", "")
        doc_type = (parent_rec or centers[0]).get("doc_type", "")
        header = f"[{doc_type}] {title} (parent_id={pid})"

        if parent_rec and parent_rec.get("parent_text"):
            parent_text = parent_rec["parent_text"]
            children_count = parent_rec.get("children_count", 1)
            if center_indices and children_count > 1:
                parts = parent_text.split("\n\n")
                # Union windows from all centers
                selected: set[int] = set()
                for ci in center_indices:
                    lo = max(0, ci - window)
                    hi = min(len(parts), ci + window + 1)
                    selected.update(range(lo, hi))
                # Sort by position and join
                window_text = "\n\n".join(
                    parts[idx] for idx in sorted(selected) if idx < len(parts)
                )
            else:
                window_text = parent_text
        else:
            window_text = "\n\n".join(c.get("text", "") for c in group_children)

        block = f"### {header}\n\n{window_text}"

        # Deduplicate context blocks
        if deduplicate:
            norm_block = normalize_for_dedup(window_text)
            if norm_block in seen_block_texts:
                continue
            seen_block_texts.add(norm_block)

        block_tokens = len(enc.encode(block))
        if tokens_used + block_tokens > token_budget:
            remaining = token_budget - tokens_used
            if remaining <= 50:
                break
            toks = enc.encode(block)[:remaining]
            block = enc.decode(toks)
            block_tokens = remaining

        context_blocks.append(block)
        tokens_used += block_tokens

        parent_results.append({
            "parent_id": pid,
            "score": score,
            "title": title,
            "doc_type": doc_type,
            "children_in_topk": len(group_children),
            "center_indices": center_indices,
        })

    debug_info["final_context_tokens"] = tokens_used
    debug_info["selected_child_ranges"] = [
        {"parent_id": pr["parent_id"], "centers": pr["center_indices"]}
        for pr in parent_results
    ]

    return {
        "context": "\n\n---\n\n".join(context_blocks),
        "parents": parent_results,
        "children": results[:10],
        "tokens_used": tokens_used,
        "mode": "parent_child",
        "debug": debug_info,
    }


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    p = argparse.ArgumentParser(description="Hybrid retriever (vector + BM25 + RRF)")
    p.add_argument("--query", required=True, help="Query text")
    p.add_argument("--mode", default=DEFAULT_MODE, choices=["vector", "bm25", "hybrid_rrf"])
    p.add_argument("--topk-vector", type=int, default=DEFAULT_TOPK_VECTOR)
    p.add_argument("--topk-bm25", type=int, default=DEFAULT_TOPK_BM25)
    p.add_argument("--topk-final", type=int, default=DEFAULT_TOPK_FINAL)
    p.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K)
    p.add_argument("--config", default="config/config.yaml", help="Config YAML path")
    # vector params
    p.add_argument("--collection", default=DEFAULT_COLLECTION)
    p.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    p.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    p.add_argument("--model", default=DEFAULT_MODEL)
    # bm25 params
    p.add_argument("--chunks", default=DEFAULT_CHUNKS_PATH)
    p.add_argument("--bm25-index", default=DEFAULT_INDEX_PATH)
    # filters
    p.add_argument("--doc-type", default=None, help="Filter by doc_type")
    # debug
    p.add_argument("--save-debug", action="store_true", default=False)
    p.add_argument("--debug-dir", default=DEFAULT_DEBUG_DIR)
    # parent-child context
    p.add_argument("--build-context", action="store_true", default=False,
                    help="Also build parent-child context from results")
    p.add_argument("--context-mode", default="parent_child", choices=["parent_child", "flat"])
    p.add_argument("--parents", default="data/parents/parents_v2_full.jsonl")
    p.add_argument("--topk-parent", type=int, default=4)
    p.add_argument("--window", type=int, default=1)
    p.add_argument("--window-centers", type=int, default=DEFAULT_WINDOW_CENTERS,
                    help="Top-M child centers per parent (default 2)")
    p.add_argument("--parent-score-agg", default=DEFAULT_PARENT_SCORE_AGG,
                    choices=["max", "sum_top2"])
    p.add_argument("--token-budget", type=int, default=3500)
    p.add_argument("--no-dedup", action="store_true", default=False,
                    help="Disable duplicate text suppression")
    args = p.parse_args()

    # Optionally load defaults from config
    cfg = load_retrieval_config(args.config)
    mode = args.mode or cfg.get("mode", DEFAULT_MODE)
    topk_vector = args.topk_vector or cfg.get("topk_vector", DEFAULT_TOPK_VECTOR)
    topk_bm25 = args.topk_bm25 or cfg.get("topk_bm25", DEFAULT_TOPK_BM25)
    topk_final = args.topk_final or cfg.get("topk_final", DEFAULT_TOPK_FINAL)
    rrf_k = args.rrf_k or cfg.get("rrf_k", DEFAULT_RRF_K)
    save_debug = args.save_debug or cfg.get("save_debug", False)
    debug_dir = args.debug_dir or cfg.get("debug_dir", DEFAULT_DEBUG_DIR)

    t0 = time.time()
    results = retrieve(
        args.query,
        mode=mode,
        topk_vector=topk_vector,
        topk_bm25=topk_bm25,
        topk_final=topk_final,
        rrf_k=rrf_k,
        collection=args.collection,
        qdrant_url=args.qdrant_url,
        ollama_url=args.ollama_url,
        model=args.model,
        chunks_path=args.chunks,
        index_path=args.bm25_index,
        doc_type_filter=args.doc_type,
        save_debug=save_debug,
        debug_dir=debug_dir,
        deduplicate=not args.no_dedup,
    )
    elapsed = time.time() - t0

    print(f"\n{'='*60}")
    print(f"  Mode: {mode}  |  Results: {len(results)}  |  Time: {elapsed:.2f}s")
    print(f"{'='*60}\n")

    for r in results[:10]:
        text_preview = r.get("text", "")[:80].replace("\n", " ")
        scores: list[str] = []
        if r.get("bm25_score") is not None:
            scores.append(f"bm25={r['bm25_score']:.4f}")
        if r.get("vector_score") is not None:
            scores.append(f"vec={r['vector_score']:.4f}")
        if r.get("fused_score") is not None and mode == "hybrid_rrf":
            scores.append(f"rrf={r['fused_score']:.6f}")
        score_str = "  ".join(scores)
        print(
            f"  [{r['rank']:2d}] {score_str}  "
            f"chunk={r['chunk_id'][:30]}  "
            f"text={text_preview}..."
        )

    # Optionally build parent-child context
    if args.build_context:
        print(f"\n{'─'*60}")
        print("Building parent-child context...\n")
        ctx = build_parent_child_context(
            results,
            parents_path=args.parents,
            topk_parent=args.topk_parent,
            window=args.window,
            window_centers=args.window_centers,
            parent_score_agg=args.parent_score_agg,
            token_budget=args.token_budget,
            context_mode=args.context_mode,
            deduplicate=not args.no_dedup,
        )
        print(f"  Context mode: {ctx['mode']}  |  Tokens: {ctx['tokens_used']}")
        if ctx["parents"]:
            print("  Parents:")
            for pr in ctx["parents"]:
                print(f"    - {pr['parent_id']}  score={pr['score']:.4f}  "
                      f"children_in_topk={pr['children_in_topk']}  "
                      f"centers={pr.get('center_indices', [])}")
        if ctx.get("debug"):
            print(f"\n  Debug info:")
            for dk, dv in ctx["debug"].items():
                print(f"    {dk}: {dv}")
        print(f"\n--- CONTEXT (first 1500 chars) ---\n{ctx['context'][:1500]}")
        if len(ctx["context"]) > 1500:
            print(f"\n... (truncated, total {len(ctx['context'])} chars)")

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
