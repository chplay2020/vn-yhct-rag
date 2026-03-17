# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""Parent–Child retrieval context builder.

Implements HƯỚNG 1:
  1. Retrieve topK CHILD vectors from yhct_chunks_v2_full_emb
  2. Group by parent_id → score parents
  3. Select topP parents
  4. Fetch parent_text from parents JSONL (cached)
  5. Build windowed context around best child in each parent
  6. Trim to token budget

Usage (CLI demo):
    PYTHONPATH=src uv run python -m rag.retrieve.parent_child_retriever \
        --query "tác dụng của cây ngải cứu" \
        --parents data/parents/parents_v2_full.jsonl

Usage (library):
    from rag.retrieve.parent_child_retriever import retrieve_context
    ctx = retrieve_context(query_text, ...)
"""

from __future__ import annotations

import argparse
import logging
import re
import time
from typing import Any

import requests  # type: ignore
import tiktoken  # type: ignore
from qdrant_client import QdrantClient  # type: ignore

from rag.utils.io import read_jsonl

logger = logging.getLogger(__name__)

# ── defaults ───────────────────────────────────────────────────────────────
DEFAULT_COLLECTION = "yhct_chunks_v2_full_emb"
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_MODEL = "bge-m3"
DEFAULT_PARENTS_PATH = "data/parents/parents_v2_full.jsonl"
DEFAULT_TOPK_CHILD = 20
DEFAULT_TOPK_PARENT = 4
DEFAULT_WINDOW = 1  # ±W children around best child
DEFAULT_TOKEN_BUDGET = 3500
DEFAULT_WINDOW_CENTERS = 2
DEFAULT_CONTEXT_FOCUS = "focused"


# ── parent cache ───────────────────────────────────────────────────────────

_parent_cache: dict[str, dict[str, Any]] | None = None


def load_parents(parents_path: str = DEFAULT_PARENTS_PATH) -> dict[str, dict[str, Any]]:
    """Load parents JSONL into a dict keyed by parent_id. Cached in-memory."""
    global _parent_cache  # noqa: PLW0603
    if _parent_cache is not None:
        return _parent_cache
    records = read_jsonl(parents_path)
    cache: dict[str, dict[str, Any]] = {}
    for rec in records:
        pid = rec.get("parent_id")
        if pid:
            cache[pid] = rec
    _parent_cache = cache
    logger.info("Loaded %d parents from %s (cached)", len(cache), parents_path)
    return cache


def invalidate_parent_cache() -> None:
    """Force reload on next access."""
    global _parent_cache  # noqa: PLW0603
    _parent_cache = None


# ── embed query ────────────────────────────────────────────────────────────

def _embed_query(
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


def _score_of_child(item: dict[str, Any]) -> float:
    return float(item.get("_score", 0.0) or 0.0)


def _merge_ranges(ranges: list[tuple[int, int]]) -> list[list[int]]:
    """Merge overlapping/touching [start, end) child index ranges."""
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


def _extract_query_terms(query_text: str) -> list[str]:
    stopwords = {
        "cua", "của", "la", "là", "va", "và", "cay", "cây", "thuoc", "thuốc",
        "tac", "tác", "dung", "dụng", "cho", "nhu", "như", "gi", "gì", "cac", "các",
        "nhung", "những", "ve", "về", "tu", "từ", "den", "đến", "mot", "một",
    }
    raw = [t.strip() for t in re.findall(r"\w+", query_text.lower()) if t.strip()]
    return [t for t in raw if len(t) >= 2 and t not in stopwords]


# ── core retrieval ─────────────────────────────────────────────────────────

def retrieve_context(
    query_text: str,
    *,
    collection: str = DEFAULT_COLLECTION,
    qdrant_url: str = DEFAULT_QDRANT_URL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    model: str = DEFAULT_MODEL,
    parents_path: str = DEFAULT_PARENTS_PATH,
    topk_child: int = DEFAULT_TOPK_CHILD,
    topk_parent: int = DEFAULT_TOPK_PARENT,
    window: int = DEFAULT_WINDOW,
    window_centers: int = DEFAULT_WINDOW_CENTERS,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    context_focus: str = DEFAULT_CONTEXT_FOCUS,
    parent_child_enabled: bool = True,
) -> dict[str, Any]:
    """Retrieve context for a query using parent–child strategy.

    Returns a dict with:
      - ``context``: final context string for LLM
      - ``parents``: list of chosen parent records with score
      - ``children``: list of retrieved child payloads with score
      - ``tokens_used``: approximate token count
      - ``mode``: "parent_child" or "flat_fallback"

    If parent_child_enabled is False, returns flat top-K child texts.
    """

    enc: Any = tiktoken.get_encoding("cl100k_base")  # type: ignore

    # ── 1. Embed query ────────────────────────────────────────────────
    query_vec = _embed_query(query_text, ollama_url, model)
    if query_vec is None:
        logger.error("Failed to embed query — cannot retrieve")
        return {"context": "", "parents": [], "children": [], "tokens_used": 0, "mode": "error"}

    # ── 2. Search topK children ───────────────────────────────────────
    client: Any = QdrantClient(url=qdrant_url)  # type: ignore
    results: Any = client.search(
        collection_name=collection,
        query_vector=query_vec,
        limit=topk_child,
        with_payload=True,
    )

    children: list[dict[str, Any]] = []
    for r in results:
        payload = dict(getattr(r, "payload", {}) or {})
        payload["_score"] = float(getattr(r, "score", 0.0))
        payload["_point_id"] = getattr(r, "id", None)
        children.append(payload)

    if not children:
        return {"context": "", "parents": [], "children": [], "tokens_used": 0, "mode": "no_results"}

    # ── 3. Flat fallback if parent_child disabled or missing parent_id ──
    if not parent_child_enabled or not any(c.get("parent_id") for c in children):
        context_parts: list[str] = []
        tokens_used = 0
        for c in children:
            text = c.get("text_norm") or c.get("text", "")
            t_len = len(enc.encode(text))
            if tokens_used + t_len > token_budget:
                break
            context_parts.append(text)
            tokens_used += t_len
        return {
            "context": "\n\n---\n\n".join(context_parts),
            "parents": [],
            "children": children,
            "tokens_used": tokens_used,
            "mode": "flat_fallback",
            "debug": {
                "selected_parent_ids": [],
                "selected_child_centers": [],
                "merged_child_ranges": [],
                "trimmed_context_spans": [],
                "per_parent_context_contribution": [],
                "final_context_token_count": tokens_used,
            },
        }

    # ── 4. Group children by parent_id ────────────────────────────────
    parent_groups: dict[str, list[dict[str, Any]]] = {}
    for c in children:
        pid = c.get("parent_id") or c.get("source_id", "unknown")
        parent_groups.setdefault(pid, []).append(c)

    # Score each parent: max child score
    parent_scores: list[tuple[str, float, list[dict[str, Any]]]] = []
    for pid, group in parent_groups.items():
        best_score = max(c["_score"] for c in group)
        parent_scores.append((pid, best_score, group))

    parent_scores.sort(key=lambda x: x[1], reverse=True)
    chosen_parents = parent_scores[:topk_parent]

    # ── 5. Load parent texts ──────────────────────────────────────────
    parent_cache = load_parents(parents_path)

    # ── 6. Build context with windowing ───────────────────────────────
    context_blocks: list[str] = []
    parent_results: list[dict[str, Any]] = []
    tokens_used = 0
    debug_info: dict[str, Any] = {
        "selected_parent_ids": [pid for pid, _, _ in chosen_parents],
        "selected_child_centers": [],
        "merged_child_ranges": [],
        "trimmed_context_spans": [],
        "per_parent_context_contribution": [],
    }
    query_terms = _extract_query_terms(query_text)

    for pid, score, group_children in chosen_parents:
        parent_rec = parent_cache.get(pid)

        # Top-M child centers in this parent (by child score)
        sorted_children = sorted(group_children, key=_score_of_child, reverse=True)
        centers = sorted_children[:max(1, window_centers)]
        center_indices: list[int] = [
            int(c["child_index"])
            for c in centers
            if isinstance(c.get("child_index"), int)
        ]
        best_child = sorted_children[0]
        best_child_idx = best_child.get("child_index")

        # Build header
        title = (parent_rec or best_child).get("title", "")
        doc_type = (parent_rec or best_child).get("doc_type", "")
        locator = best_child.get("parent_locator", "")
        header = f"[{doc_type}] {title}"
        if locator:
            header += f" | {locator}"
        header += f" (parent_id={pid})"

        # Get context text via windowing
        if parent_rec and parent_rec.get("parent_text"):
            parent_text = parent_rec["parent_text"]
            parts = parent_text.split("\n\n")
            range_candidates: list[tuple[int, int]] = []

            if center_indices and len(parts) > 1:
                for ci in center_indices:
                    lo = max(0, ci - window)
                    hi = min(len(parts), ci + window + 1)
                    range_candidates.append((lo, hi))
            elif best_child_idx is not None and isinstance(best_child_idx, int):
                lo = max(0, best_child_idx - window)
                hi = min(len(parts), best_child_idx + window + 1)
                range_candidates.append((lo, hi))

            merged_ranges = _merge_ranges(range_candidates)
            if not merged_ranges:
                merged_ranges = [[0, min(len(parts), max(1, 2 * window + 1))]]

            if context_focus != "focused" and len(merged_ranges) > 0:
                merged_ranges = [
                    [merged_ranges[0][0], max(merged_ranges[0][1], len(parts))]
                ]

            if context_focus == "focused" and query_terms:
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

            # Focus mode trims unrelated leading/trailing text by selecting only merged spans.
            span_texts: list[str] = []
            for start, end in merged_ranges:
                if 0 <= start < end <= len(parts):
                    span_texts.append("\n\n".join(parts[start:end]))

            window_text = "\n\n".join(span_texts)
        else:
            # Fallback: concatenate child texts from this group
            merged_ranges = None
            window_text = "\n\n".join(
                c.get("text_norm") or c.get("text", "") for c in group_children
            )

        # Trim to fit budget
        block = f"### {header}\n\n{window_text}"
        block_tokens = len(enc.encode(block))
        if tokens_used + block_tokens > token_budget:
            # Truncate block to fit remaining budget
            remaining = token_budget - tokens_used
            if remaining <= 50:
                break
            toks = enc.encode(block)[:remaining]
            block = enc.decode(toks)
            block_tokens = remaining

        context_blocks.append(block)
        tokens_used += block_tokens

        selected_ranges = merged_ranges if parent_rec and parent_rec.get("parent_text") else None
        preview = window_text[:220].replace("\n", " ")
        debug_info["selected_child_centers"].append({
            "parent_id": pid,
            "centers": center_indices,
        })
        debug_info["merged_child_ranges"].append({
            "parent_id": pid,
            "ranges": selected_ranges,
        })
        debug_info["trimmed_context_spans"].append({
            "parent_id": pid,
            "spans": selected_ranges,
        })
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
            "locator": locator,
            "children_in_topk": len(group_children),
            "best_child_index": best_child_idx,
            "selected_child_centers": center_indices,
            "merged_child_ranges": selected_ranges,
            "trimmed_context_spans": selected_ranges,
            "context_tokens": block_tokens,
        })

    context = "\n\n---\n\n".join(context_blocks)
    debug_info["final_context_token_count"] = tokens_used

    return {
        "context": context,
        "parents": parent_results,
        "children": children[:10],  # return top10 for inspection
        "tokens_used": tokens_used,
        "mode": "parent_child",
        "debug": debug_info,
    }


# ── CLI demo ───────────────────────────────────────────────────────────────

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )

    p = argparse.ArgumentParser(description="Parent–Child retrieval demo")
    p.add_argument("--query", required=True, help="Query text")
    p.add_argument("--collection", default=DEFAULT_COLLECTION)
    p.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    p.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    p.add_argument("--model", default=DEFAULT_MODEL)
    p.add_argument("--parents", default=DEFAULT_PARENTS_PATH, help="Parents JSONL path")
    p.add_argument("--topk-child", type=int, default=DEFAULT_TOPK_CHILD)
    p.add_argument("--topk-parent", type=int, default=DEFAULT_TOPK_PARENT)
    p.add_argument("--window", type=int, default=DEFAULT_WINDOW)
    p.add_argument("--window-centers", type=int, default=DEFAULT_WINDOW_CENTERS)
    p.add_argument("--token-budget", type=int, default=DEFAULT_TOKEN_BUDGET)
    p.add_argument("--context-focus", choices=["focused", "broad"], default=DEFAULT_CONTEXT_FOCUS,
                    help="focused: only merged hit windows; broad: expand first window toward parent body")
    p.add_argument("--no-parent-child", action="store_true", default=False,
                    help="Disable parent-child mode (flat child-only)")
    args = p.parse_args()

    t0 = time.time()
    result = retrieve_context(
        args.query,
        collection=args.collection,
        qdrant_url=args.qdrant_url,
        ollama_url=args.ollama_url,
        model=args.model,
        parents_path=args.parents,
        topk_child=args.topk_child,
        topk_parent=args.topk_parent,
        window=args.window,
        window_centers=args.window_centers,
        token_budget=args.token_budget,
        context_focus=args.context_focus,
        parent_child_enabled=not args.no_parent_child,
    )
    elapsed = time.time() - t0

    print("=" * 60)
    print(f"  Mode: {result['mode']}  |  Tokens: {result['tokens_used']}  |  Time: {elapsed:.2f}s")
    print("=" * 60)

    if result["parents"]:
        print("\nChosen Parents:")
        for pr in result["parents"]:
            print(f"  - {pr['parent_id']}  score={pr['score']:.4f}  "
                  f"children_in_topk={pr['children_in_topk']}  "
                  f"best_child_idx={pr['best_child_index']}  "
                  f"ranges={pr.get('merged_child_ranges')}")

    print("\nTop Children:")
    for c in result["children"][:5]:
        text_preview = (c.get("text_norm") or c.get("text", ""))[:80].replace("\n", " ")
        print(f"  - score={c['_score']:.4f}  parent_id={c.get('parent_id', '?')}  "
              f"child_idx={c.get('child_index')}  text={text_preview}...")

    print("\n--- CONTEXT ---")
    print(result["context"][:2000])
    if len(result["context"]) > 2000:
        print(f"\n... (truncated, total {len(result['context'])} chars)")
    print("=" * 60)


if __name__ == "__main__":
    main()
