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
import re
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
from rag.retrieve.answerability_gate import (
    DEFAULT_GATE_TOPK,
    run_answerability_gate,
)
from rag.generate.answer_generator import (
    DEFAULT_ANSWER_MAX_TOKENS,
    DEFAULT_ANSWER_MODEL,
    DEFAULT_ANSWER_TEMPERATURE,
    generate_structured_answer,
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
DEFAULT_GATE_DEBUG = False
DEFAULT_CONTEXT_FOCUS = "focused"
DEFAULT_MAX_CONTEXT_PARENTS = 3
DEFAULT_ANSWER_DEBUG = False
OFFTOPIC_MARKERS = (
    "mẹ đừng đánh con",
    "csgt",
    "vượt đèn đỏ",
    "đi chơi về muộn",
)


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


def _save_gate_debug_json(
    query: str,
    mode: str,
    results: list[dict[str, Any]],
    gate_decision: dict[str, Any],
    debug_dir: str,
    context_payload: dict[str, Any] | None = None,
) -> None:
    """Save gate decision and selected evidence for debugging."""
    import hashlib

    debug_path = Path(debug_dir)
    debug_path.mkdir(parents=True, exist_ok=True)

    q_hash = hashlib.md5(query.encode(), usedforsecurity=False).hexdigest()[:8]
    filename = f"{mode}_{q_hash}_gate.json"
    filepath = debug_path / filename

    payload: dict[str, Any] = {
        "query": query,
        "mode": mode,
        "hybrid_results": results,
        "gate_pass": gate_decision.get("pass"),
        "reason": gate_decision.get("reason"),
        "predicted_citation_count": gate_decision.get("predicted_citation_count"),
        "selected_evidence": gate_decision.get("selected_evidence"),
        "gate_features": gate_decision.get("gate_features"),
    }
    if context_payload is not None:
        payload["context_payload"] = {
            "mode": context_payload.get("mode"),
            "tokens_used": context_payload.get("tokens_used"),
            "parents": context_payload.get("parents"),
            "debug": context_payload.get("debug"),
        }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    logger.info("Gate debug results saved to %s", filepath)


def _save_answer_debug_json(
    query: str,
    mode: str,
    results: list[dict[str, Any]],
    gate_decision: dict[str, Any] | None,
    context_payload: dict[str, Any] | None,
    answer_payload: dict[str, Any],
    raw_model_output: str | None,
    *,
    model_name: str,
    debug_dir: str,
) -> None:
    """Save end-to-end retrieval + gate + context + answer debug JSON."""
    import hashlib

    debug_path = Path(debug_dir)
    debug_path.mkdir(parents=True, exist_ok=True)

    q_hash = hashlib.md5(query.encode(), usedforsecurity=False).hexdigest()[:8]
    filename = f"{mode}_{q_hash}_answer.json"
    filepath = debug_path / filename

    payload: dict[str, Any] = {
        "query": query,
        "mode": mode,
        "model": model_name,
        "hybrid_results": results,
        "gate_result": gate_decision,
        "selected_evidence": (
            list((gate_decision or {}).get("selected_evidence", []))
            if gate_decision is not None
            else []
        ),
        "context": {
            "mode": (context_payload or {}).get("mode"),
            "tokens_used": (context_payload or {}).get("tokens_used"),
            "debug": (context_payload or {}).get("debug"),
        },
        "answer": answer_payload,
        "raw_model_output": raw_model_output,
    }

    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    logger.info("Answer debug results saved to %s", filepath)


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
    query_text: str = "",
    parents_path: str = "data/parents/parents_v2_full.jsonl",
    topk_parent: int = 4,
    window: int = 1,
    window_centers: int = DEFAULT_WINDOW_CENTERS,
    parent_score_agg: str = DEFAULT_PARENT_SCORE_AGG,
    token_budget: int = 3500,
    context_mode: str = "parent_child",
    context_focus: str = DEFAULT_CONTEXT_FOCUS,
    deduplicate: bool = True,
    selected_evidence: list[dict[str, Any]] | None = None,
    context_from_gate: bool = False,
    max_context_parents: int = DEFAULT_MAX_CONTEXT_PARENTS,
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
    context_source = "gate_selected_evidence" if context_from_gate else "retrieval_topk"
    final_answer_evidence: list[dict[str, Any]] = []
    final_evidence_chunk_ids: set[str] = set()

    def _append_final_evidence(items: list[dict[str, Any]]) -> None:
        for item in items:
            cid = str(item.get("chunk_id") or "")
            if cid and cid in final_evidence_chunk_ids:
                continue
            if cid:
                final_evidence_chunk_ids.add(cid)
            final_answer_evidence.append(item)

    def _merge_ranges(ranges: list[tuple[int, int]]) -> list[list[int]]:
        if not ranges:
            return []
        ordered = sorted(ranges, key=lambda x: (x[0], x[1]))
        out: list[list[int]] = [[ordered[0][0], ordered[0][1]]]
        for start, end in ordered[1:]:
            last = out[-1]
            if start <= last[1]:
                last[1] = max(last[1], end)
            else:
                out.append([start, end])
        return out

    def _extract_query_terms(query: str) -> list[str]:
        stopwords = {
            "cua", "của", "la", "là", "va", "và", "cay", "cây", "thuoc", "thuốc",
            "tac", "tác", "dung", "dụng", "cho", "nhu", "như", "gi", "gì", "cac", "các",
            "nhung", "những", "ve", "về", "tu", "từ", "den", "đến", "mot", "một",
        }
        raw = [t.strip() for t in re.findall(r"\w+", query.lower()) if t.strip()]
        return [t for t in raw if len(t) >= 2 and t not in stopwords]

    def _score_of(item: dict[str, Any]) -> float:
        return float(
            item.get("fused_score")
            or item.get("vector_score")
            or item.get("bm25_score")
            or 0.0
        )

    def _dominant_topic_term(
        terms: list[str],
        evidence: list[dict[str, Any]],
    ) -> str | None:
        if not terms:
            return None
        if not evidence:
            return terms[0]
        counts: dict[str, int] = {t: 0 for t in terms}
        for item in evidence:
            text = str(item.get("text", "")).lower()
            for term in terms:
                if term in text:
                    counts[term] += 1
        best_term, best_count = max(counts.items(), key=lambda kv: kv[1])
        return best_term if best_count > 0 else terms[0]

    query_terms = _extract_query_terms(query_text)
    evidence_for_context = (
        list(selected_evidence or [])
        if context_from_gate and selected_evidence
        else list(results)
    )
    dominant_term = _dominant_topic_term(query_terms, evidence_for_context)

    if context_mode == "flat":
        # Flat mode: just concatenate child texts, with optional dedup
        context_parts: list[str] = []
        seen_texts: set[str] = set()
        tokens_used = 0
        for r in evidence_for_context:
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
            _append_final_evidence([r])
            tokens_used += t_len
        return {
            "context": "\n\n---\n\n".join(context_parts),
            "parents": [],
            "children": evidence_for_context,
            "final_answer_evidence": final_answer_evidence,
            "tokens_used": tokens_used,
            "mode": "flat",
            "debug": {
                "context_source": context_source,
                "selected_parent_ids": [],
                "selected_child_centers": [],
                "merged_child_ranges": [],
                "trimmed_context_spans": [],
                "per_parent_context_contribution": [],
                "final_context_token_count": tokens_used,
                "final_answer_chunk_ids": [
                    str(x.get("chunk_id") or "") for x in final_answer_evidence
                ],
            },
        }

    # Parent-child mode: group by parent_id, window context
    parent_cache = load_parents(parents_path)

    # Group results by parent_id
    parent_groups: dict[str, list[dict[str, Any]]] = {}
    for r in evidence_for_context:
        pid = r.get("parent_id") or r.get("source_id", "unknown")
        parent_groups.setdefault(pid, []).append(r)

    # Score each parent
    parent_scores: list[tuple[str, float, list[dict[str, Any]]]] = []
    for pid, group in parent_groups.items():
        child_scores = sorted([_score_of(r) for r in group], reverse=True)
        if parent_score_agg == "sum_top2":
            score = sum(child_scores[:2])
        else:  # default: max
            score = child_scores[0] if child_scores else 0.0
        parent_scores.append((pid, score, group))

    parent_scores.sort(key=lambda x: x[1], reverse=True)
    parent_limit = max(1, min(topk_parent, max_context_parents))

    ranked_candidates: list[dict[str, Any]] = []
    for pid, score, group in parent_scores:
        sorted_group = sorted(group, key=_score_of, reverse=True)
        support_count = len(sorted_group)
        top1_score = _score_of(sorted_group[0]) if sorted_group else 0.0
        sum_top2 = sum(_score_of(x) for x in sorted_group[:2])
        parent_rec = parent_cache.get(pid)
        parent_preview = str((parent_rec or {}).get("parent_text", "")).lower()
        evidence_preview = " ".join(str(x.get("text", "")) for x in sorted_group[:2]).lower()
        preview_text = parent_preview if parent_preview else evidence_preview
        overlap_hits = sum(1 for t in query_terms if t in preview_text)
        overlap_ratio = (overlap_hits / len(query_terms)) if query_terms else 0.0
        has_dominant_term = bool(dominant_term and dominant_term in preview_text)
        off_topic_markers = [m for m in OFFTOPIC_MARKERS if m in preview_text]
        relevance_score = (
            score
            + 0.02 * min(3, support_count - 1)
            + 0.03 * overlap_ratio
            + (0.04 if has_dominant_term else -0.04)
            - (0.1 if off_topic_markers else 0.0)
        )

        exclude_reasons: list[str] = []
        if overlap_hits == 0 and not has_dominant_term:
            exclude_reasons.append("no_query_overlap")
        if support_count == 1 and top1_score < 0.75 * (parent_scores[0][1] if parent_scores else 0.0):
            exclude_reasons.append("single_weak_support")
        if off_topic_markers and support_count <= 1:
            exclude_reasons.append("off_topic_marker")

        ranked_candidates.append({
            "parent_id": pid,
            "base_score": round(score, 6),
            "relevance_score": round(relevance_score, 6),
            "support_count": support_count,
            "sum_top2": round(sum_top2, 6),
            "query_overlap_ratio": round(overlap_ratio, 6),
            "has_dominant_term": has_dominant_term,
            "off_topic_markers": off_topic_markers,
            "exclude_reasons": exclude_reasons,
            "group": group,
        })

    ranked_candidates.sort(key=lambda x: x["relevance_score"], reverse=True)
    filtered_out: list[dict[str, Any]] = []
    kept_candidates: list[dict[str, Any]] = []
    for cand in ranked_candidates:
        reasons = list(cand.get("exclude_reasons", []))
        if reasons:
            filtered_out.append({
                "parent_id": cand["parent_id"],
                "reasons": reasons,
                "relevance_score": cand["relevance_score"],
                "support_count": cand["support_count"],
            })
            continue
        kept_candidates.append(cand)

    if not kept_candidates:
        kept_candidates = ranked_candidates[:parent_limit]

    effective_limit = parent_limit
    if len(kept_candidates) > parent_limit:
        top_rel = float(kept_candidates[0]["relevance_score"])
        strong_extra = [
            c for c in kept_candidates[parent_limit:]
            if c["support_count"] >= 2 and float(c["relevance_score"]) >= 0.9 * top_rel
        ]
        if strong_extra:
            effective_limit = min(parent_limit + 1, len(kept_candidates))

    selected_candidates = kept_candidates[:effective_limit]
    chosen_parents = [
        (str(c["parent_id"]), float(c["relevance_score"]), list(c["group"]))
        for c in selected_candidates
    ]

    debug_info["context_source"] = context_source
    debug_info["context_from_gate"] = context_from_gate
    debug_info["query_terms"] = query_terms
    debug_info["dominant_term"] = dominant_term
    debug_info["candidate_parent_count"] = len(ranked_candidates)
    debug_info["filtered_parent_count"] = len(filtered_out)
    debug_info["max_context_parents"] = parent_limit
    debug_info["effective_context_parent_limit"] = effective_limit
    debug_info["candidate_parents_before_filter"] = [
        {
            "parent_id": c["parent_id"],
            "base_score": c["base_score"],
            "relevance_score": c["relevance_score"],
            "support_count": c["support_count"],
            "query_overlap_ratio": c["query_overlap_ratio"],
            "has_dominant_term": c["has_dominant_term"],
            "off_topic_markers": c["off_topic_markers"],
        }
        for c in ranked_candidates
    ]
    debug_info["filtered_out_parents"] = filtered_out
    debug_info["selected_parent_ids"] = [pid for pid, _, _ in chosen_parents]
    debug_info["final_parent_count"] = len(chosen_parents)
    debug_info["selected_child_centers"] = []
    debug_info["merged_child_ranges"] = []
    debug_info["trimmed_context_spans"] = []
    debug_info["per_parent_context_contribution"] = []

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
        best_child_idx = group_sorted[0].get("child_index") if group_sorted else None

        title = (parent_rec or centers[0]).get("title", "")
        doc_type = (parent_rec or centers[0]).get("doc_type", "")
        header = f"[{doc_type}] {title} (parent_id={pid})"

        if parent_rec and parent_rec.get("parent_text"):
            parent_text = parent_rec["parent_text"]
            parts = parent_text.split("\n\n")
            ranges: list[tuple[int, int]] = []
            for ci in center_indices:
                lo = max(0, ci - window)
                hi = min(len(parts), ci + window + 1)
                ranges.append((lo, hi))

            if not ranges and isinstance(best_child_idx, int):
                lo = max(0, best_child_idx - window)
                hi = min(len(parts), best_child_idx + window + 1)
                ranges.append((lo, hi))

            if not ranges and query_terms:
                query_hit_indices = [
                    idx for idx, part in enumerate(parts)
                    if any(term in part.lower() for term in query_terms)
                ]
                for idx in query_hit_indices[:max(1, window_centers)]:
                    lo = max(0, idx - window)
                    hi = min(len(parts), idx + window + 1)
                    ranges.append((lo, hi))

            merged_ranges = _merge_ranges(ranges)
            if not merged_ranges:
                merged_ranges = [[0, min(len(parts), max(1, 2 * window + 1))]]

            effective_focus = "focused" if context_from_gate else context_focus
            if effective_focus == "broad" and merged_ranges:
                merged_ranges = [[merged_ranges[0][0], len(parts)]]

            if effective_focus == "focused" and query_terms:
                candidate_indices: list[int] = []
                for start, end in merged_ranges:
                    candidate_indices.extend(range(start, end))
                hit_indices = [
                    idx
                    for idx in candidate_indices
                    if idx < len(parts)
                    and any(term in parts[idx].lower() for term in query_terms)
                ]
                if hit_indices:
                    hit_lo = min(hit_indices)
                    hit_hi = max(hit_indices) + 1
                    merged_ranges = _merge_ranges([
                        (hit_lo, min(len(parts), hit_hi + min(1, window)))
                    ])

            span_texts: list[str] = []
            for start, end in merged_ranges:
                if 0 <= start < end <= len(parts):
                    span_texts.append("\n\n".join(parts[start:end]))
            window_text = "\n\n".join(span_texts)
        else:
            merged_ranges = None
            window_text = "\n\n".join(c.get("text", "") for c in group_children)

        evidence_in_ranges: list[dict[str, Any]] = []
        for child in group_children:
            child_idx = child.get("child_index")
            if merged_ranges is None:
                evidence_in_ranges.append(child)
                continue
            if not isinstance(child_idx, int):
                continue
            if any(start <= child_idx < end for start, end in merged_ranges):
                evidence_in_ranges.append(child)
        if not evidence_in_ranges:
            evidence_in_ranges = list(group_children)

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
        _append_final_evidence(evidence_in_ranges)

        preview = window_text[:220].replace("\n", " ")
        debug_info["selected_child_centers"].append({"parent_id": pid, "centers": center_indices})
        debug_info["merged_child_ranges"].append({"parent_id": pid, "ranges": merged_ranges})
        debug_info["trimmed_context_spans"].append({"parent_id": pid, "spans": merged_ranges})
        debug_info["per_parent_context_contribution"].append({
            "parent_id": pid,
            "tokens": block_tokens,
            "preview": preview,
        })

        parent_results.append({
            "parent_id": pid,
            "score": score,
            "title": title,
            "doc_type": doc_type,
            "children_in_topk": len(group_children),
            "center_indices": center_indices,
            "best_child_index": best_child_idx,
            "merged_child_ranges": merged_ranges,
            "trimmed_context_spans": merged_ranges,
            "context_tokens": block_tokens,
        })

    debug_info["final_context_tokens"] = tokens_used
    debug_info["final_context_token_count"] = tokens_used
    debug_info["final_answer_chunk_ids"] = [
        str(x.get("chunk_id") or "") for x in final_answer_evidence
    ]

    return {
        "context": "\n\n---\n\n".join(context_blocks),
        "parents": parent_results,
        "children": evidence_for_context[:10],
        "final_answer_evidence": final_answer_evidence,
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
    # answerability gate
    p.add_argument("--use-gate", action="store_true", default=False,
                    help="Run answerability gate on top of hybrid_rrf results")
    p.add_argument("--gate-topk", type=int, default=DEFAULT_GATE_TOPK,
                    help="Top-K hybrid results consumed by gate")
    p.add_argument("--gate-debug", action="store_true", default=DEFAULT_GATE_DEBUG,
                    help="Save gate decision debug JSON (also implied by --save-debug)")
    # parent-child context
    p.add_argument("--build-context", action="store_true", default=False,
                    help="Also build parent-child context from results")
    p.add_argument("--context-mode", default="parent_child", choices=["parent_child", "flat"])
    p.add_argument("--context-focus", default=DEFAULT_CONTEXT_FOCUS, choices=["focused", "broad"],
                    help="focused: only merged hit windows (default); broad: expand first hit to parent tail")
    p.add_argument("--parents", default="data/parents/parents_v2_full.jsonl")
    p.add_argument("--topk-parent", type=int, default=4)
    p.add_argument("--window", type=int, default=1)
    p.add_argument("--window-centers", type=int, default=DEFAULT_WINDOW_CENTERS,
                    help="Top-M child centers per parent (default 2)")
    p.add_argument("--parent-score-agg", default=DEFAULT_PARENT_SCORE_AGG,
                    choices=["max", "sum_top2"])
    p.add_argument("--token-budget", type=int, default=3500)
    p.add_argument("--max-context-parents", type=int, default=DEFAULT_MAX_CONTEXT_PARENTS,
                    help="Maximum parent sections in final context (default 3)")
    # local answer generation
    p.add_argument("--generate-answer", action="store_true", default=False,
                    help="Generate grounded answer from context using local Ollama LLM")
    p.add_argument("--answer-model", default=DEFAULT_ANSWER_MODEL,
                    help="Ollama model name for answer generation")
    p.add_argument("--answer-max-tokens", type=int, default=DEFAULT_ANSWER_MAX_TOKENS,
                    help="Maximum generated tokens for answer generation")
    p.add_argument("--answer-temperature", type=float, default=DEFAULT_ANSWER_TEMPERATURE,
                    help="Sampling temperature for answer generation")
    p.add_argument("--answer-debug", action="store_true", default=DEFAULT_ANSWER_DEBUG,
                    help="Save answer generation debug JSON")
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

    gate_decision: dict[str, Any] | None = None
    context_payload: dict[str, Any] | None = None
    answer_payload: dict[str, Any] | None = None
    raw_model_output: str | None = None

    if args.use_gate:
        if mode != "hybrid_rrf":
            logger.warning("--use-gate is intended for --mode=hybrid_rrf, current mode=%s", mode)
        gate_decision = run_answerability_gate(
            args.query,
            results,
            gate_topk=args.gate_topk,
        )
        print(f"\n{'─'*60}")
        print("Answerability Gate")
        print(
            f"  pass={gate_decision['pass']}  "
            f"predicted_citation_count={gate_decision['predicted_citation_count']}"
        )
        print(f"  reason: {gate_decision['reason']}")
        gf = gate_decision.get("gate_features", {})
        print(
            "  features: "
            f"top1={gf.get('top1_score')}  "
            f"top2={gf.get('top2_score')}  "
            f"gap={gf.get('top1_top2_gap')}  "
            f"evidence={gf.get('evidence_count')}  "
            f"parents={gf.get('distinct_parent_count')}  "
            f"sources={gf.get('distinct_source_count')}"
        )
        selected = gate_decision.get("selected_evidence", [])
        if selected:
            print("  selected_evidence:")
            for ev in selected:
                print(
                    f"    - rank={ev.get('rank')}  "
                    f"chunk={str(ev.get('chunk_id', ''))[:30]}  "
                    f"score={ev.get('fused_score')}"
                )

    should_build_context = bool(args.build_context or args.generate_answer)
    if args.generate_answer and not args.build_context:
        logger.info("--generate-answer enabled without --build-context; auto-building focused context.")

    if should_build_context:
        context_from_gate = bool(args.use_gate and gate_decision and gate_decision.get("selected_evidence"))
        selected_for_context = (
            list(gate_decision.get("selected_evidence", []))
            if gate_decision is not None
            else []
        )
        context_payload = build_parent_child_context(
            results,
            query_text=args.query,
            parents_path=args.parents,
            topk_parent=args.topk_parent,
            window=args.window,
            window_centers=args.window_centers,
            parent_score_agg=args.parent_score_agg,
            token_budget=args.token_budget,
            context_mode=args.context_mode,
            context_focus=args.context_focus,
            deduplicate=not args.no_dedup,
            selected_evidence=selected_for_context,
            context_from_gate=context_from_gate,
            max_context_parents=max(1, args.max_context_parents),
        )

    if args.use_gate and (save_debug or args.gate_debug) and gate_decision is not None:
        _save_gate_debug_json(
            args.query,
            mode,
            results,
            gate_decision,
            debug_dir,
            context_payload=context_payload,
        )

    # Optionally print parent-child context summary
    if should_build_context:
        print(f"\n{'─'*60}")
        print("Building parent-child context...\n")
        ctx = context_payload if context_payload is not None else {
            "mode": "none",
            "tokens_used": 0,
            "parents": [],
            "context": "",
            "debug": {},
        }
        print(f"  Context mode: {ctx['mode']}  |  Tokens: {ctx['tokens_used']}")
        ctx_debug = ctx.get("debug", {}) if isinstance(ctx.get("debug"), dict) else {}
        print(
            "  Context summary: "
            f"source={ctx_debug.get('context_source')}  "
            f"filtered_parents={ctx_debug.get('filtered_parent_count')}  "
            f"final_parents={ctx_debug.get('final_parent_count')}"
        )
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
        ctx_text_raw = ctx.get("context", "")
        ctx_text = ctx_text_raw if isinstance(ctx_text_raw, str) else ""
        print(f"\n--- CONTEXT (first 1500 chars) ---\n{ctx_text[:1500]}")
        if len(ctx_text) > 1500:
            print(f"\n... (truncated, total {len(ctx_text)} chars)")

    if args.generate_answer:
        if gate_decision is None:
            gate_decision = {
                "pass": True,
                "reason": "Answerability gate was not enabled; generation proceeds with available context.",
                "predicted_citation_count": 0,
                "selected_evidence": [],
                "gate_features": {},
            }

        selected_for_answer: list[dict[str, Any]] = []
        if context_payload is not None:
            raw_final = context_payload.get("final_answer_evidence", [])
            if isinstance(raw_final, list):
                selected_for_answer = [x for x in raw_final if isinstance(x, dict)]
        if not selected_for_answer:
            selected_for_answer = list(gate_decision.get("selected_evidence", []))
        if not selected_for_answer:
            selected_for_answer = results[: max(1, min(5, len(results)))]

        focused_context = ""
        if context_payload is not None:
            focused_context = str(context_payload.get("context", ""))

        answer_payload, raw_model_output = generate_structured_answer(
            query=args.query,
            gate_decision=gate_decision,
            focused_context=focused_context,
            selected_evidence=selected_for_answer,
            retrieval_results=selected_for_answer,
            ollama_url=args.ollama_url,
            model=args.answer_model,
            max_tokens=max(64, args.answer_max_tokens),
            temperature=max(0.0, args.answer_temperature),
        )

        print(f"\n{'─'*60}")
        print("Generated Answer")
        print(f"\nAnswer:\n{answer_payload.get('answer', '')}")

        key_concepts = answer_payload.get("key_concepts", [])
        if isinstance(key_concepts, list):
            print("\nKey concepts:")
            if key_concepts:
                for concept in key_concepts:
                    print(f"  - {concept}")
            else:
                print("  - (none)")

        print("\nEvidence summary:")
        evidence = answer_payload.get("evidence", [])
        if isinstance(evidence, list) and evidence:
            for ev in evidence:
                if not isinstance(ev, dict):
                    continue
                cite = ev.get("citation_id", "E?")
                snippet_raw = ev.get("snippet", "")
                snippet = snippet_raw if isinstance(snippet_raw, str) else ""
                snippet = snippet.replace("\n", " ")[:120]
                print(
                    f"  [{cite}] chunk={ev.get('chunk_id')}  parent={ev.get('parent_id')}  "
                    f"score={ev.get('score')}  title={ev.get('title') or ''}"
                )
                print(f"       snippet={snippet}")
        else:
            print("  - (no evidence)")

        print(f"\nLimits:\n{answer_payload.get('limits', '')}")
        print(f"\nSafety note:\n{answer_payload.get('safety_note', '')}")

    if (save_debug or args.answer_debug) and args.generate_answer and answer_payload is not None:
        _save_answer_debug_json(
            args.query,
            mode,
            results,
            gate_decision,
            context_payload,
            answer_payload,
            raw_model_output,
            model_name=args.answer_model,
            debug_dir=debug_dir,
        )

    print(f"\n{'='*60}")


if __name__ == "__main__":
    main()
