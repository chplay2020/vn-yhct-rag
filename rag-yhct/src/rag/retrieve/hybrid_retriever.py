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
from typing import Any, cast

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
from rag.utils.query_quality import (
    normalize_for_dedup,
    normalize_query_for_retrieval,
    restore_query_diacritics_tokenwise_from_corpus,
    strip_vietnamese_diacritics,
)

logger = logging.getLogger(__name__)

# ── defaults ───────────────────────────────────────────────────────────────
DEFAULT_MODE = "hybrid_rrf"
DEFAULT_TOPK_VECTOR = 40
DEFAULT_TOPK_BM25 = 40
DEFAULT_TOPK_FINAL = 40
DEFAULT_RRF_K = 60  #Thong so k
DEFAULT_DEBUG_DIR = "data/reports/retrieval_debug"
DEFAULT_WINDOW_CENTERS = 2
DEFAULT_PARENT_SCORE_AGG = "max"  # "max" or "sum_top2"
DEFAULT_GATE_DEBUG = False
DEFAULT_CONTEXT_FOCUS = "focused"
DEFAULT_MAX_CONTEXT_PARENTS = 3
DEFAULT_ANSWER_DEBUG = False
DEFAULT_RESTORE_TRIGGER_NON_ACCENT_RATIO = 0.50
DEFAULT_GENERAL_RESTORE_TRIGGER_NON_ACCENT_RATIO = 0.25
OFFTOPIC_MARKERS = (
    "mẹ đừng đánh con",
    "csgt",
    "vượt đèn đỏ",
    "đi chơi về muộn",
)


# ── query preparation (safe accent restoration) ───────────────────────────

_RE_WORD = re.compile(r"\w+", re.UNICODE)
_LEXICON_PATH = Path(__file__).with_name("yhct_domain_lexicon.json")
_yhct_lexicon_cache: list[dict[str, str]] | None = None


def _is_latin_token(token: str) -> bool:
    return any("a" <= ch <= "z" for ch in token.lower())


def strip_accents_for_matching(text: str) -> str:
    """Return normalized, accent-folded text for phrase matching."""
    return strip_vietnamese_diacritics(normalize_query_for_retrieval(text))


def _tokenize_latin_candidates(text: str) -> list[str]:
    return [t for t in _RE_WORD.findall(text.lower()) if _is_latin_token(t)]


def _non_accented_token_stats(text: str) -> dict[str, float | int]:
    """Count accented/non-accented candidate tokens for restore trigger."""
    tokens = _tokenize_latin_candidates(text)
    accented = sum(1 for t in tokens if t != strip_vietnamese_diacritics(t))
    non_accented = max(0, len(tokens) - accented)
    total = len(tokens)
    ratio = (float(non_accented) / float(total)) if total else 0.0
    return {
        "accented_token_count": accented,
        "non_accented_token_count": non_accented,
        "total_candidate_tokens": total,
        "non_accented_token_ratio": round(ratio, 6),
    }


def should_trigger_restore(
    text: str,
    *,
    threshold: float = DEFAULT_RESTORE_TRIGGER_NON_ACCENT_RATIO,
) -> dict[str, float | int | bool]:
    """Return restore trigger decision and token-ratio stats."""
    stats = _non_accented_token_stats(text)
    ratio = float(stats["non_accented_token_ratio"])
    total = int(stats["total_candidate_tokens"])
    return {
        **stats,
        "restore_triggered": bool(total >= 2 and ratio >= threshold),
        "restore_trigger_threshold": float(threshold),
    }


def _confidence_band(confidence: float) -> str:
    if confidence >= 0.65:
        return "high"
    if confidence >= 0.35:
        return "medium"
    if confidence > 0.0:
        return "low"
    return "none"


def _score_lexicon_restore(matches: list[dict[str, Any]], changed: bool) -> float:
    if not matches or not changed:
        return 0.0

    def _safe_ngram(match: dict[str, Any]) -> int:
        val = match.get("ngram", 0)
        return int(val) if isinstance(val, int | float | str) else 0

    unigram_hits = sum(1 for m in matches if _safe_ngram(m) <= 1)
    bigram_hits = sum(1 for m in matches if _safe_ngram(m) == 2)
    trigram_hits = sum(1 for m in matches if _safe_ngram(m) >= 3)

    score = 0.10
    score += 0.12 * min(unigram_hits, 3)
    score += 0.30 * min(bigram_hits, 3)
    score += 0.42 * min(trigram_hits, 2)

    # Phrase-level lexicon evidence should be trusted more than isolated tokens.
    if trigram_hits > 0:
        score = max(score, 0.70)
    elif bigram_hits > 0:
        score = max(score, 0.36)

    return round(max(0.0, min(1.0, score)), 6)


def should_run_general_restorer(
    text_after_lexicon: str,
    *,
    restore_triggered: bool,
    threshold: float = DEFAULT_GENERAL_RESTORE_TRIGGER_NON_ACCENT_RATIO,
) -> dict[str, float | int | bool]:
    """Return trigger decision for second-stage general Vietnamese restoration."""
    stats = _non_accented_token_stats(text_after_lexicon)
    ratio = float(stats["non_accented_token_ratio"])
    total = int(stats["total_candidate_tokens"])
    return {
        **stats,
        "general_restore_triggered": bool(restore_triggered and total >= 2 and ratio >= threshold),
        "general_restore_trigger_threshold": float(threshold),
    }


def restore_general_vietnamese(
    text_after_lexicon: str,
    *,
    chunks_path: str,
) -> str:
    """General-purpose accent restoration for common Vietnamese words.

    This must run after lexicon restoration so domain phrases are already protected.
    """
    return restore_query_diacritics_tokenwise_from_corpus(
        text_after_lexicon,
        chunks_path=chunks_path,
    )


def _score_general_restore_candidate(
    before: str,
    after: str,
) -> dict[str, Any]:
    """Score safety/confidence for general restore candidate."""
    before_norm = normalize_query_for_retrieval(before)
    after_norm = normalize_query_for_retrieval(after)
    before_folded = strip_vietnamese_diacritics(before_norm)
    after_folded = strip_vietnamese_diacritics(after_norm)

    suspicious = after_folded != before_folded
    changed = after_norm != before_norm

    src_tokens = _RE_WORD.findall(before_norm)
    dst_tokens = _RE_WORD.findall(after_norm)
    token_count = max(1, len(src_tokens))
    changed_tokens = sum(1 for a, b in zip(src_tokens, dst_tokens, strict=False) if a != b)
    if len(dst_tokens) != len(src_tokens):
        changed_tokens += abs(len(dst_tokens) - len(src_tokens))

    before_stats = _non_accented_token_stats(before_norm)
    after_stats = _non_accented_token_stats(after_norm)
    src_non = int(before_stats["non_accented_token_count"])
    dst_non = int(after_stats["non_accented_token_count"])
    total = max(1, int(before_stats["total_candidate_tokens"]))

    improved_tokens = max(0, src_non - dst_non)
    improvement_ratio = float(improved_tokens) / float(total)
    change_ratio = float(changed_tokens) / float(token_count)

    score = 0.0
    if suspicious or not changed:
        score = 0.0
    else:
        score = 0.20 + (0.65 * improvement_ratio) + (0.15 * min(change_ratio, 0.4))
        if improved_tokens == 0:
            score -= 0.15
        if changed_tokens > max(1, int(0.6 * token_count)):
            score -= 0.20

        # Strong win: most remaining non-accented tokens were restored safely.
        if dst_non <= max(1, total // 4) and improved_tokens >= 2:
            score = max(score, 0.72)
        elif improved_tokens >= 1 and dst_non < src_non:
            score = max(score, 0.45)

    confidence = round(max(0.0, min(1.0, score)), 6)
    return {
        "candidate": after_norm,
        "suspicious": suspicious,
        "changed": changed,
        "changed_tokens": changed_tokens,
        "token_count": token_count,
        "improved_non_accented_tokens": improved_tokens,
        "general_restore_confidence": confidence,
        "general_restore_conf_band": _confidence_band(confidence),
    }


def merge_restore_results(candidates: list[str]) -> list[str]:
    """Deduplicate while preserving order for vector query variants."""
    out: list[str] = []
    for item in candidates:
        val = normalize_query_for_retrieval(item)
        if not val:
            continue
        if val in out:
            continue
        out.append(val)
    return out


def choose_final_restored_query(
    *,
    query_after_normalize: str,
    query_after_lexicon_restore: str,
    query_after_general_restore: str,
    lexicon_restore_changed: bool,
    general_restore_triggered: bool,
    general_restore_changed: bool,
    general_restore_confidence: float,
    general_restore_suspicious: bool,
) -> dict[str, Any]:
    """Pick safest final restored query and vector branch variants."""
    base_restored = query_after_lexicon_restore if lexicon_restore_changed else query_after_normalize
    final_query = base_restored
    final_restore_source = "lexicon" if lexicon_restore_changed else "normalized"

    vector_candidates = [query_after_normalize]
    if lexicon_restore_changed:
        vector_candidates.append(query_after_lexicon_restore)

    if general_restore_triggered and general_restore_changed and not general_restore_suspicious:
        band = _confidence_band(general_restore_confidence)
        if band == "high":
            final_query = query_after_general_restore
            final_restore_source = "general"
            vector_candidates = [query_after_normalize, query_after_general_restore]
        elif band == "medium":
            vector_candidates.append(query_after_general_restore)

    vector_queries = merge_restore_results(vector_candidates)
    dual_query_enabled = len(vector_queries) > 1

    return {
        "query_final_restored": final_query,
        "final_restore_source": final_restore_source,
        "vector_queries_used": vector_queries,
        "dual_query_enabled": dual_query_enabled,
    }


def _load_yhct_domain_lexicon() -> list[dict[str, str]]:
    global _yhct_lexicon_cache  # noqa: PLW0603
    if _yhct_lexicon_cache is not None:
        return _yhct_lexicon_cache
    if not _LEXICON_PATH.exists():
        _yhct_lexicon_cache = []
        return _yhct_lexicon_cache

    try:
        raw = json.loads(_LEXICON_PATH.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed loading YHCT lexicon from %s: %s", _LEXICON_PATH, exc)
        _yhct_lexicon_cache = []
        return _yhct_lexicon_cache

    entries: list[dict[str, str]] = []
    if not isinstance(raw, dict):
        _yhct_lexicon_cache = []
        return _yhct_lexicon_cache

    raw_dict = cast(dict[str, Any], raw)
    for group_name, pairs in raw_dict.items():
        if not isinstance(pairs, list):
            continue
        typed_pairs = cast(list[Any], pairs)
        for item in typed_pairs:
            if isinstance(item, list):
                item_seq = cast(list[Any], item)
            elif isinstance(item, tuple):
                item_seq = list(cast(tuple[Any, Any], item))
            else:
                continue
            if len(item_seq) != 2:
                continue
            left = item_seq[0]
            right = item_seq[1]
            folded = normalize_query_for_retrieval(str(left))
            accented = normalize_query_for_retrieval(str(right))
            if not folded or not accented:
                continue
            entries.append({
                "group": str(group_name),
                "folded": folded,
                "accented": accented,
            })

    entries.sort(key=lambda x: len(x["folded"]), reverse=True)
    _yhct_lexicon_cache = entries
    return _yhct_lexicon_cache


def restore_with_domain_lexicon(
    text: str,
    *,
    accentless_for_matching: str | None = None,
) -> tuple[str, list[dict[str, Any]]]:
    """Apply longest-match phrase replacements using YHCT lexicon."""
    normalized = normalize_query_for_retrieval(text)
    folded_text = accentless_for_matching or strip_accents_for_matching(normalized)
    out_tokens = normalized.split(" ") if normalized else []
    folded_tokens = folded_text.split(" ") if folded_text else []
    if len(out_tokens) != len(folded_tokens):
        # Fallback safety: keep positional alignment for phrase replacement.
        out_tokens = normalize_query_for_retrieval(folded_text).split(" ") if folded_text else []
        folded_tokens = list(out_tokens)

    matches: list[dict[str, Any]] = []
    occupied_positions: set[int] = set()

    for ent in _load_yhct_domain_lexicon():
        folded_phrase = str(ent["folded"])
        accented_phrase = str(ent["accented"])
        if not folded_phrase:
            continue
        folded_phrase_tokens = [t for t in folded_phrase.split(" ") if t]
        accented_phrase_tokens = [t for t in accented_phrase.split(" ") if t]
        ngram_len = len(folded_phrase_tokens)
        if not ngram_len or len(out_tokens) < ngram_len:
            continue

        phrase_hits = 0
        i = 0
        while i <= len(folded_tokens) - ngram_len:
            span = set(range(i, i + ngram_len))
            if span & occupied_positions:
                i += 1
                continue

            if folded_tokens[i:i + ngram_len] == folded_phrase_tokens:
                replacement_tokens = list(accented_phrase_tokens)
                out_tokens[i:i + ngram_len] = replacement_tokens
                folded_tokens[i:i + ngram_len] = folded_phrase_tokens

                occupied_positions.update(range(i, i + len(replacement_tokens)))
                phrase_hits += 1
                i += len(replacement_tokens)
                continue
            i += 1

        if phrase_hits > 0:
            matches.append({
                "group": ent["group"],
                "folded": folded_phrase,
                "accented": accented_phrase,
                "ngram": ngram_len,
                "hits": phrase_hits,
            })

    out = " ".join(out_tokens)
    return normalize_query_for_retrieval(out), matches


def score_restore_candidates(
    query_after_normalize: str,
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    """Phrase-aware scoring over restore candidates.

    Uses conservative checks and boosts phrase evidence (uni/bi/tri+gram).
    """
    base_folded = strip_vietnamese_diacritics(query_after_normalize)
    best: dict[str, Any] = {
        "candidate": query_after_normalize,
        "method": "none",
        "score": 0.0,
        "confidence": 0.0,
        "suspicious": False,
        "changed_tokens": 0,
        "token_count": max(1, len(_RE_WORD.findall(query_after_normalize))),
        "domain_match_count": 0,
        "domain_unigram_hits": 0,
        "domain_bigram_hits": 0,
        "domain_trigram_hits": 0,
        "domain_lexicon_matches": [],
    }

    for cand in candidates:
        candidate = normalize_query_for_retrieval(str(cand.get("candidate", "")))
        method = str(cand.get("method", "none"))
        raw_matches = cand.get("domain_lexicon_matches", [])
        domain_matches: list[dict[str, Any]] = [
            m for m in raw_matches if isinstance(m, dict)
        ]
        candidate_folded = strip_vietnamese_diacritics(candidate)
        suspicious = candidate_folded != base_folded

        src_tokens = _RE_WORD.findall(query_after_normalize)
        dst_tokens = _RE_WORD.findall(candidate)
        token_count = max(1, len(src_tokens))
        changed_tokens = sum(
            1 for a, b in zip(src_tokens, dst_tokens, strict=False) if a != b
        )
        change_ratio = float(changed_tokens) / float(token_count)

        def _safe_ngram(match: dict[str, Any]) -> int:
            val = match.get("ngram", 0)
            return int(val) if isinstance(val, int | float | str) else 0

        unigram_hits = sum(1 for m in domain_matches if _safe_ngram(m) <= 1)
        bigram_hits = sum(1 for m in domain_matches if _safe_ngram(m) == 2)
        trigram_hits = sum(1 for m in domain_matches if _safe_ngram(m) >= 3)

        phrase_boost = 0.0
        phrase_boost += 0.08 * min(unigram_hits, 3)
        phrase_boost += 0.18 * min(bigram_hits, 3)
        phrase_boost += 0.28 * min(trigram_hits, 3)
        method_boost = 0.05 if method == "domain_lexicon" else 0.02

        score = 0.0
        if suspicious:
            score = -1.0
        else:
            score = (0.35 * change_ratio) + phrase_boost + method_boost
            if changed_tokens == 0:
                score -= 0.08
            if change_ratio < 0.15 and not domain_matches:
                score -= 0.10

        confidence = max(0.0, min(1.0, score))

        evaluated: dict[str, Any] = {
            "candidate": candidate,
            "method": method,
            "score": round(float(score), 6),
            "confidence": round(float(confidence), 6),
            "suspicious": suspicious,
            "changed_tokens": changed_tokens,
            "token_count": token_count,
            "domain_match_count": len(domain_matches),
            "domain_unigram_hits": unigram_hits,
            "domain_bigram_hits": bigram_hits,
            "domain_trigram_hits": trigram_hits,
            "domain_lexicon_matches": domain_matches,
        }

        if (float(evaluated["score"]) > float(best["score"])):
            best = evaluated

    return best


def prepare_hybrid_queries(
    query: str,
    *,
    chunks_path: str = DEFAULT_CHUNKS_PATH,
) -> dict[str, Any]:
    """Prepare safe query variants for hybrid retrieval.

    Policy:
      - keep original + normalized variants
      - attempt corpus accent restoration only for mostly non-accented Vietnamese
      - use conservative heuristics to decide whether restored query is safe enough
      - keep BM25 on stable normalized query; use restored query mainly for vector branch
    """
    query_original = query
    query_after_normalize = normalize_query_for_retrieval(query)
    query_accentless_for_matching = strip_accents_for_matching(query_after_normalize)
    query_after_lexicon_restore = query_after_normalize
    query_after_general_restore = query_after_normalize
    query_after_corpus_restore = query_after_normalize
    query_final_restored = query_after_normalize
    restore_changed = False
    restore_confidence = 0.0
    restore_method = "none"
    restore_candidates: list[dict[str, Any]] = []
    domain_lexicon_matches: list[dict[str, Any]] = []
    lexicon_restore_changed = False
    lexicon_restore_confidence = 0.0
    vector_queries_used: list[str] = [query_after_normalize]
    bm25_queries_used: list[str] = [query_after_normalize]
    dual_query_enabled = False
    restore_conf_band = "none"
    general_restore_triggered = False
    general_restore_changed = False
    general_restore_confidence = 0.0
    general_restore_conf_band = "none"
    final_restore_source = "normalized"

    trigger = should_trigger_restore(query_after_normalize)
    restore_triggered = bool(trigger["restore_triggered"])

    if restore_triggered:
        query_after_lexicon_restore, domain_lexicon_matches = restore_with_domain_lexicon(
            query_after_normalize,
            accentless_for_matching=query_accentless_for_matching,
        )
        lexicon_restore_changed = query_after_lexicon_restore != query_after_normalize
        lexicon_restore_confidence = _score_lexicon_restore(
            domain_lexicon_matches,
            lexicon_restore_changed,
        )

        general_trigger_info = should_run_general_restorer(
            query_after_lexicon_restore,
            restore_triggered=restore_triggered,
        )
        general_restore_triggered = bool(general_trigger_info["general_restore_triggered"])

        general_eval: dict[str, Any] = {
            "candidate": query_after_lexicon_restore,
            "suspicious": False,
            "changed": False,
            "changed_tokens": 0,
            "token_count": max(1, len(_RE_WORD.findall(query_after_lexicon_restore))),
            "improved_non_accented_tokens": 0,
            "general_restore_confidence": 0.0,
            "general_restore_conf_band": "none",
        }
        if general_restore_triggered:
            query_after_general_restore = restore_general_vietnamese(
                query_after_lexicon_restore,
                chunks_path=chunks_path,
            )
            general_eval = _score_general_restore_candidate(
                query_after_lexicon_restore,
                query_after_general_restore,
            )
            general_restore_changed = bool(general_eval["changed"])
            general_restore_confidence = float(general_eval["general_restore_confidence"])
            general_restore_conf_band = str(general_eval["general_restore_conf_band"])
        else:
            query_after_general_restore = query_after_lexicon_restore

        # Keep legacy alias for notebooks/tests expecting corpus naming.
        query_after_corpus_restore = query_after_general_restore

        chosen = choose_final_restored_query(
            query_after_normalize=query_after_normalize,
            query_after_lexicon_restore=query_after_lexicon_restore,
            query_after_general_restore=query_after_general_restore,
            lexicon_restore_changed=lexicon_restore_changed,
            general_restore_triggered=general_restore_triggered,
            general_restore_changed=general_restore_changed,
            general_restore_confidence=general_restore_confidence,
            general_restore_suspicious=bool(general_eval["suspicious"]),
        )
        query_final_restored = str(chosen["query_final_restored"])
        final_restore_source = str(chosen["final_restore_source"])
        vector_queries_used = [str(x) for x in cast(list[Any], chosen["vector_queries_used"])]
        dual_query_enabled = bool(chosen["dual_query_enabled"])

        restore_changed = query_final_restored != query_after_normalize
        if final_restore_source == "general":
            restore_method = "general_corpus"
            restore_confidence = general_restore_confidence
        elif final_restore_source == "lexicon":
            restore_method = "domain_lexicon"
            restore_confidence = lexicon_restore_confidence
        else:
            restore_method = "none"
            restore_confidence = 0.0
        restore_conf_band = _confidence_band(restore_confidence) if restore_changed else "none"

        restore_candidates = [
            {
                "candidate": query_after_normalize,
                "method": "none",
                "domain_lexicon_matches": [],
            },
            {
                "candidate": query_after_lexicon_restore,
                "method": "domain_lexicon",
                "domain_lexicon_matches": domain_lexicon_matches,
                "confidence": round(float(lexicon_restore_confidence), 6),
            },
            {
                "candidate": query_after_general_restore,
                "method": "general_corpus",
                "domain_lexicon_matches": domain_lexicon_matches,
                "confidence": round(float(general_restore_confidence), 6),
                "suspicious": bool(general_eval["suspicious"]),
            },
        ]

        # Keep BM25 stable (already accent-insensitive).
        bm25_queries_used = [query_after_normalize]

    # Branch B: query already has enough diacritics (or too short/noisy for restore).
    # Keep normalized query as primary path (single-query retrieval) via defaults above.

    query_after_restore = query_final_restored  # backward compatibility alias
    restore_confidence_band = restore_conf_band  # backward compatibility alias

    # Backward compatibility fields used by old tests/notebooks.
    vector_query_used = vector_queries_used[-1]
    bm25_query_used = bm25_queries_used[0]

    return {
        "query_original": query_original,
        "query_after_normalize": query_after_normalize,
        "query_accentless_for_matching": query_accentless_for_matching,
        "query_after_lexicon_restore": query_after_lexicon_restore,
        "query_after_general_restore": query_after_general_restore,
        "query_after_corpus_restore": query_after_corpus_restore,
        "query_final_restored": query_final_restored,
        "query_after_restore": query_after_restore,
        "accented_token_count": int(trigger["accented_token_count"]),
        "non_accented_token_count": int(trigger["non_accented_token_count"]),
        "total_candidate_tokens": int(trigger["total_candidate_tokens"]),
        "non_accented_token_ratio": float(trigger["non_accented_token_ratio"]),
        "restore_triggered": restore_triggered,
        "restore_trigger_threshold": float(trigger["restore_trigger_threshold"]),
        "restore_changed": restore_changed,
        "restore_confidence": round(float(restore_confidence), 6),
        "restore_method": restore_method,
        "restore_confidence_band": restore_confidence_band,
        "restore_conf_band": restore_conf_band,
        "restore_candidates": restore_candidates,
        "domain_lexicon_matches": domain_lexicon_matches,
        "lexicon_restore_changed": lexicon_restore_changed,
        "lexicon_restore_confidence": round(float(lexicon_restore_confidence), 6),
        "general_restore_triggered": general_restore_triggered,
        "general_restore_changed": general_restore_changed,
        "general_restore_confidence": round(float(general_restore_confidence), 6),
        "general_restore_conf_band": general_restore_conf_band,
        "final_restore_source": final_restore_source,
        "vector_queries_used": vector_queries_used,
        "bm25_queries_used": bm25_queries_used,
        "dual_query_enabled": dual_query_enabled,
        "vector_query_used": vector_query_used,
        "bm25_query_used": bm25_query_used,
    }


def _merge_multi_query_branch_results(
    branch_results: list[tuple[str, list[dict[str, Any]]]],
    *,
    score_key: str,
    topk: int,
) -> list[dict[str, Any]]:
    """Merge per-query branch results by chunk_id, keep best score, and rerank."""
    merged: dict[str, dict[str, Any]] = {}
    for query_text, results in branch_results:
        for r in results:
            cid = str(r.get("chunk_id") or "")
            if not cid:
                continue
            score = float(r.get(score_key) or 0.0)
            existing = merged.get(cid)
            if existing is None:
                merged[cid] = dict(r)
                merged[cid]["query_used"] = query_text
                merged[cid]["query_variants"] = [query_text]
                continue

            query_variants = list(existing.get("query_variants", []))
            if query_text not in query_variants:
                query_variants.append(query_text)
            existing["query_variants"] = query_variants

            existing_score = float(existing.get(score_key) or 0.0)
            if score > existing_score:
                keep_variants = list(existing.get("query_variants", []))
                merged[cid] = dict(r)
                merged[cid]["query_used"] = query_text
                merged[cid]["query_variants"] = keep_variants

    out = sorted(merged.values(), key=lambda x: float(x.get(score_key) or 0.0), reverse=True)
    out = out[:topk]
    for i, item in enumerate(out):
        item["rank"] = i
    return out


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

    query_debug = prepare_hybrid_queries(query, chunks_path=chunks_path)
    bm25_queries = [str(q) for q in query_debug.get("bm25_queries_used", []) if str(q).strip()]
    vector_queries = [str(q) for q in query_debug.get("vector_queries_used", []) if str(q).strip()]
    if not bm25_queries:
        bm25_queries = [str(query_debug["bm25_query_used"])]
    if not vector_queries:
        vector_queries = [str(query_debug["vector_query_used"])]

    bm25_results: list[dict[str, Any]] = []
    vector_results: list[dict[str, Any]] = []

    if mode in ("bm25", "hybrid_rrf"):
        bm25_runs: list[tuple[str, list[dict[str, Any]]]] = []
        for bm25_query in bm25_queries:
            bm25_runs.append((
                bm25_query,
                retrieve_bm25(
                    bm25_query,
                    topk=topk_bm25,
                    chunks_path=chunks_path,
                    index_path=index_path,
                    doc_type_filter=doc_type_filter,
                ),
            ))
        bm25_results = _merge_multi_query_branch_results(
            bm25_runs,
            score_key="bm25_score",
            topk=topk_bm25,
        )

    if mode in ("vector", "hybrid_rrf"):
        vector_runs: list[tuple[str, list[dict[str, Any]]]] = []
        for vector_query in vector_queries:
            vector_runs.append((
                vector_query,
                retrieve_vector(
                    vector_query,
                    topk=topk_vector,
                    collection=collection,
                    qdrant_url=qdrant_url,
                    ollama_url=ollama_url,
                    model=model,
                    chunks_path=chunks_path,
                    use_llm_diacritic_fallback=False,
                    doc_type_filter=doc_type_filter,
                ),
            ))
        vector_results = _merge_multi_query_branch_results(
            vector_runs,
            score_key="vector_score",
            topk=topk_vector,
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
            query_debug=query_debug,
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
    query_debug: dict[str, Any] | None = None,
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
    if query_debug is not None:
        report["query_debug"] = query_debug

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
