from __future__ import annotations

from collections import Counter
import re
from typing import Any, cast

DEFAULT_GATE_TOPK = 5
DEFAULT_MIN_STRONG_EVIDENCE = 2
DEFAULT_STRONG_REL_RATIO = 0.85
DEFAULT_STRONG_ABS_THRESHOLD = 0.015
DEFAULT_MAX_CITATIONS = 4

ABSOLUTE_CLAIM_PATTERNS = (
    r"chữa\s*khỏi\s*hoàn\s*toàn",
    r"chua\s*khoi\s*hoan\s*toan",
    r"mọi\s*loại",
    r"moi\s*loai",
    r"100\s*%",
    r"dứt\s*điểm",
    r"dut\s*diem",
    r"chắc\s*chắn\s*khỏi",
    r"chac\s*chan\s*khoi",
)

ABSOLUTE_SUPPORT_PATTERNS = (
    re.compile(r"chữa\s*khỏi\s*hoàn\s*toàn", re.IGNORECASE),
    re.compile(r"chua\s*khoi\s*hoan\s*toan", re.IGNORECASE),
    re.compile(r"mọi\s*loại", re.IGNORECASE),
    re.compile(r"moi\s*loai", re.IGNORECASE),
    re.compile(r"100\s*%", re.IGNORECASE),
    re.compile(r"dứt\s*điểm", re.IGNORECASE),
    re.compile(r"dut\s*diem", re.IGNORECASE),
    re.compile(r"chắc\s*chắn\s*khỏi", re.IGNORECASE),
    re.compile(r"chac\s*chan\s*khoi", re.IGNORECASE),
)

ABSOLUTE_COUNTERSIGNALS = (
    re.compile(r"hỗ\s*trợ", re.IGNORECASE),
    re.compile(r"ho\s*tro", re.IGNORECASE),
    re.compile(r"giảm", re.IGNORECASE),
    re.compile(r"giam", re.IGNORECASE),
    re.compile(r"cải\s*thiện", re.IGNORECASE),
    re.compile(r"cai\s*thien", re.IGNORECASE),
    re.compile(r"điều\s*trị", re.IGNORECASE),
    re.compile(r"dieu\s*tri", re.IGNORECASE),
    re.compile(r"ca\s*bệnh", re.IGNORECASE),
    re.compile(r"ca\s*benh", re.IGNORECASE),
    re.compile(r"có\s*thể", re.IGNORECASE),
    re.compile(r"co\s*the", re.IGNORECASE),
)


def _to_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return 0.0


def _score_of(item: dict[str, Any]) -> float:
    """Prefer fused score; fallback to mode-specific scores."""
    if item.get("fused_score") is not None:
        return _to_float(item.get("fused_score"))
    if item.get("vector_score") is not None:
        return _to_float(item.get("vector_score"))
    if item.get("bm25_score") is not None:
        return _to_float(item.get("bm25_score"))
    return 0.0


def _is_strong(
    score: float,
    top1_score: float,
    *,
    strong_rel_ratio: float,
    strong_abs_threshold: float,
) -> bool:
    # Treat "strong" as crossing both relative and absolute floors to avoid
    # weak-tail evidence being counted just because one threshold is permissive.
    rel_floor = top1_score * strong_rel_ratio if top1_score > 0.0 else 0.0
    required_floor = max(rel_floor, strong_abs_threshold)
    return score >= required_floor


def _has_absolute_claim_pattern(query: str) -> bool:
    q = query.lower()
    return any(re.search(pat, q, re.IGNORECASE) for pat in ABSOLUTE_CLAIM_PATTERNS)


def _absolute_claim_supported(evidence_items: list[dict[str, Any]]) -> bool:
    if not evidence_items:
        return False

    for item in evidence_items:
        text = str(item.get("text", "") or "")
        if not text:
            continue
        # Conservative support: evidence must explicitly contain absolute claim language.
        support_hits = sum(1 for pat in ABSOLUTE_SUPPORT_PATTERNS if pat.search(text))
        if support_hits >= 2:
            return True
    return False


def _absolute_countersignal_hits(evidence_items: list[dict[str, Any]]) -> int:
    hits = 0
    for item in evidence_items:
        text = str(item.get("text", "") or "")
        if not text:
            continue
        if any(pat.search(text) for pat in ABSOLUTE_COUNTERSIGNALS):
            hits += 1
    return hits


def run_answerability_gate(
    query: str,
    hybrid_results: list[dict[str, Any]],
    *,
    gate_topk: int = DEFAULT_GATE_TOPK,
    min_strong_evidence: int = DEFAULT_MIN_STRONG_EVIDENCE,
    strong_rel_ratio: float = DEFAULT_STRONG_REL_RATIO,
    strong_abs_threshold: float = DEFAULT_STRONG_ABS_THRESHOLD,
    max_citations: int = DEFAULT_MAX_CITATIONS,
    context_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Answerability gate on top of retrieved evidence.

    Returns structured decision dict:
      - pass: bool
      - reason: str
      - predicted_citation_count: int
      - selected_evidence: list[dict]
      - gate_features: dict
    """

    topk = max(1, gate_topk)
    ranked = list(hybrid_results[:topk])

    if not ranked:
        has_absolute = _has_absolute_claim_pattern(query)
        return {
            "pass": False,
            "reason": "Không tìm thấy bằng chứng truy xuất phù hợp.",
            "predicted_citation_count": 0,
            "selected_evidence": [],
            "gate_features": {
                "top1_score": 0.0,
                "top2_score": 0.0,
                "top1_top2_gap": 0.0,
                "evidence_count": 0,
                "distinct_parent_count": 0,
                "distinct_source_count": 0,
                "same_parent_support": False,
                "same_source_support": False,
                "gate_topk": topk,
                "min_strong_evidence": min_strong_evidence,
                "strong_rel_ratio": strong_rel_ratio,
                "strong_abs_threshold": strong_abs_threshold,
                "has_absolute_claim_pattern": has_absolute,
                "absolute_claim_supported": False,
                "absolute_countersignal_count": 0,
                "claim_mismatch_flag": has_absolute,
            },
        }

    scores = [_score_of(r) for r in ranked]
    top1_score = scores[0] if scores else 0.0
    top2_score = scores[1] if len(scores) > 1 else 0.0

    strong_pairs: list[tuple[dict[str, Any], float]] = []
    for item, score in zip(ranked, scores, strict=False):
        if _is_strong(
            score,
            top1_score,
            strong_rel_ratio=strong_rel_ratio,
            strong_abs_threshold=strong_abs_threshold,
        ):
            strong_pairs.append((item, score))

    strong_items = [item for item, _ in strong_pairs]
    strong_scores = [score for _, score in strong_pairs]

    parent_ids = [str(x.get("parent_id", "")) for x in strong_items if x.get("parent_id")]
    source_ids = [str(x.get("source_id", "")) for x in strong_items if x.get("source_id")]

    parent_counts = Counter(parent_ids)
    source_counts = Counter(source_ids)

    parent_score_sum: dict[str, float] = {}
    source_score_sum: dict[str, float] = {}
    for item, score in strong_pairs:
        pid_raw = item.get("parent_id")
        sid_raw = item.get("source_id")
        if pid_raw:
            pid = str(pid_raw)
            parent_score_sum[pid] = parent_score_sum.get(pid, 0.0) + score
        if sid_raw:
            sid = str(sid_raw)
            source_score_sum[sid] = source_score_sum.get(sid, 0.0) + score

    same_parent_support = any(v >= 2 for v in parent_counts.values())
    same_source_support = any(v >= 2 for v in source_counts.values())
    top_support_parent_id: str | None = (
        max(parent_score_sum.keys(), key=lambda pid: parent_score_sum.get(pid, 0.0))
        if parent_score_sum
        else None
    )
    top_support_source_id: str | None = (
        max(source_score_sum.keys(), key=lambda sid: source_score_sum.get(sid, 0.0))
        if source_score_sum
        else None
    )

    evidence_count = len(strong_items)
    distinct_parent_count = len(parent_counts)
    distinct_source_count = len(source_counts)
    evidence_score_sum = float(sum(strong_scores)) if strong_scores else 0.0
    evidence_score_mean = (evidence_score_sum / evidence_count) if evidence_count > 0 else 0.0

    strongest_parent_count = parent_counts.get(top_support_parent_id, 0) if top_support_parent_id else 0
    evidence_parent_concentration = (
        (strongest_parent_count / evidence_count) if evidence_count > 0 else 0.0
    )

    predicted_citation_count = min(max_citations, evidence_count)

    agreement = (
        same_parent_support
        or same_source_support
        or (evidence_count >= 3 and evidence_parent_concentration >= 0.67)
    )

    has_absolute_claim_pattern = _has_absolute_claim_pattern(query)
    evidence_for_claim = strong_items if strong_items else ranked
    absolute_claim_supported = _absolute_claim_supported(evidence_for_claim)
    absolute_countersignal_count = _absolute_countersignal_hits(evidence_for_claim)
    claim_mismatch_flag = bool(
        has_absolute_claim_pattern
        and (
            (not absolute_claim_supported)
            or absolute_countersignal_count > 0
        )
    )

    passed = (
        evidence_count >= max(2, min_strong_evidence)
        and predicted_citation_count >= 2
        and agreement
    )
    if claim_mismatch_flag:
        passed = False

    if claim_mismatch_flag:
        reason = (
            "Truy vấn chứa khẳng định tuyệt đối/phổ quát nhưng bằng chứng truy xuất không "
            "hỗ trợ rõ ràng mức độ chắc chắn đó."
        )
    elif passed:
        reason = "Có đủ bằng chứng mạnh và nhất quán từ truy xuất hybrid."
    elif evidence_count < min_strong_evidence:
        reason = "Chưa đủ số lượng đoạn bằng chứng mạnh."
    else:
        reason = "Bằng chứng thiếu độ hội tụ nhất quán (cùng parent/source)."

    selected_evidence: list[dict[str, Any]] = []
    for idx, item in enumerate(strong_items):
        if idx >= max_citations:
            break
        selected_evidence.append({
            "rank": item.get("rank"),
            "chunk_id": item.get("chunk_id"),
            "parent_id": item.get("parent_id"),
            "source_id": item.get("source_id"),
            "fused_score": item.get("fused_score"),
            "vector_score": item.get("vector_score"),
            "bm25_score": item.get("bm25_score"),
            "text": item.get("text", ""),
        })

    gate_features: dict[str, Any] = {
        "top1_score": round(top1_score, 6),
        "top2_score": round(top2_score, 6),
        "top1_top2_gap": round(top1_score - top2_score, 6),
        "evidence_count": evidence_count,
        "distinct_parent_count": distinct_parent_count,
        "distinct_source_count": distinct_source_count,
        "same_parent_support": same_parent_support,
        "same_source_support": same_source_support,
        "top_support_parent_id": top_support_parent_id,
        "top_support_source_id": top_support_source_id,
        "evidence_score_sum": round(evidence_score_sum, 6),
        "evidence_score_mean": round(evidence_score_mean, 6),
        "evidence_parent_concentration": round(evidence_parent_concentration, 6),
        "gate_topk": topk,
        "min_strong_evidence": min_strong_evidence,
        "strong_rel_ratio": strong_rel_ratio,
        "strong_abs_threshold": strong_abs_threshold,
        "has_absolute_claim_pattern": has_absolute_claim_pattern,
        "absolute_claim_supported": absolute_claim_supported,
        "absolute_countersignal_count": absolute_countersignal_count,
        "claim_mismatch_flag": claim_mismatch_flag,
    }

    if context_info is not None:
        raw_parents = context_info.get("parents")
        parents = cast(list[Any], raw_parents) if isinstance(raw_parents, list) else []
        gate_features["context_parent_count"] = len(parents)
        gate_features["context_mode"] = context_info.get("mode")
        gate_features["context_tokens_used"] = context_info.get("tokens_used")

    return {
        "pass": passed,
        "reason": reason,
        "predicted_citation_count": predicted_citation_count,
        "selected_evidence": selected_evidence,
        "gate_features": gate_features,
    }
