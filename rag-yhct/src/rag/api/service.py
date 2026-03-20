from __future__ import annotations

from typing import Any, cast

from rag.generate.answer_generator import ABSTAIN_ANSWER, SAFETY_NOTE_VI
from rag.ui.pipeline_runner import run_demo_pipeline


def _as_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return cast(dict[str, Any], value)
    return {}


def _as_dict_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    items = cast(list[Any], value)
    out: list[dict[str, Any]] = []
    for item in items:
        if isinstance(item, dict):
            out.append(cast(dict[str, Any], item))
    return out


def run_rag_pipeline(
    *,
    query: str,
    mode: str,
    use_gate: bool,
    build_context: bool,
    generate_answer: bool,
) -> dict[str, Any]:
    """Run existing Python RAG pipeline and normalize output for API clients."""

    raw = run_demo_pipeline(
        query=query,
        retrieval_mode=mode,
        use_gate=use_gate,
        build_context=build_context,
        generate_answer=generate_answer,
    )

    answer_result = _as_dict(raw.get("answer_result"))
    status = _as_dict(raw.get("status"))
    context_result = _as_dict(raw.get("context_result"))
    context_debug = _as_dict(context_result.get("debug"))

    answer_text = str(answer_result.get("answer") or "")
    if not answer_text and bool(status.get("abstained")):
        answer_text = ABSTAIN_ANSWER

    return {
        "query": str(raw.get("query") or query),
        "mode": str(mode),
        "answer": answer_text,
        "key_concepts": [str(x) for x in answer_result.get("key_concepts", []) if x is not None],
        "limits": str(answer_result.get("limits") or ""),
        "safety_note": str(answer_result.get("safety_note") or SAFETY_NOTE_VI),
        "abstained": bool(status.get("abstained")),
        "gate_result": _as_dict(raw.get("gate_result")) or None,
        "evidence": _as_dict_list(raw.get("evidence_panel")),
        "retrieval_results": _as_dict_list(raw.get("retrieval_results")),
        "context_debug": context_debug,
    }
