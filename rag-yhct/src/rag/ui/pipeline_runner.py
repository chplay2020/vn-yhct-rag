# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
from __future__ import annotations

from typing import Any

from rag.generate.answer_generator import (
    ABSTAIN_ANSWER,
    generate_structured_answer,
)
from rag.retrieve.answerability_gate import DEFAULT_GATE_TOPK, run_answerability_gate
from rag.retrieve.hybrid_retriever import (
    build_parent_child_context,
    load_retrieval_config,
    retrieve,
)


def _score_of(item: dict[str, Any]) -> float:
    for key in ("score", "fused_score", "vector_score", "bm25_score"):
        value = item.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except Exception:
            continue
    return 0.0


def _compact_text(value: Any, *, max_chars: int = 360) -> str:
    text = str(value or "").replace("\n", " ").strip()
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3].rstrip()}..."


def build_evidence_panel_items(
    evidence: list[dict[str, Any]] | None,
    *,
    retrieval_results: list[dict[str, Any]] | None = None,
    fallback_limit: int = 8,
) -> list[dict[str, Any]]:
    """Normalize evidence objects for UI rendering."""

    retrieval = list(retrieval_results or [])
    base = list(evidence or [])
    if not base:
        base = retrieval[:fallback_limit]

    by_chunk: dict[str, dict[str, Any]] = {}
    for item in retrieval:
        cid = str(item.get("chunk_id") or "")
        if cid and cid not in by_chunk:
            by_chunk[cid] = item

    out: list[dict[str, Any]] = []
    for idx, item in enumerate(base):
        cid = str(item.get("chunk_id") or "")
        merged = dict(by_chunk.get(cid, {}))
        merged.update(item)

        citation_id = str(merged.get("citation_id") or f"E{idx + 1}")
        out.append(
            {
                "citation_id": citation_id,
                "snippet": _compact_text(merged.get("snippet") or merged.get("text", "")),
                "chunk_id": merged.get("chunk_id"),
                "parent_id": merged.get("parent_id"),
                "score": round(_score_of(merged), 6),
                "title": merged.get("title"),
                "page_range": merged.get("page_range") or merged.get("parent_locator"),
                "section_heading": merged.get("section_heading") or merged.get("doc_type"),
                "file_path": merged.get("file_path") or merged.get("source_id"),
            }
        )
    return out


def run_demo_pipeline(
    *,
    query: str,
    retrieval_mode: str,
    use_gate: bool,
    build_context: bool,
    generate_answer: bool,
    config_path: str = "config/config.yaml",
    context_mode: str = "parent_child",
    context_focus: str = "focused",
    gate_topk: int = DEFAULT_GATE_TOPK,
    answer_max_tokens: int = 512,
    answer_temperature: float = 0.2,
) -> dict[str, Any]:
    """Run retrieval pipeline for Streamlit UI and return structured result."""

    result: dict[str, Any] = {
        "query": query,
        "mode": {
            "retrieval_mode": retrieval_mode,
            "use_gate": bool(use_gate),
            "build_context": bool(build_context),
            "generate_answer": bool(generate_answer),
        },
        "retrieval_results": [],
        "gate_result": None,
        "context_result": None,
        "answer_result": None,
        "raw_model_output": None,
        "evidence_panel": [],
        "errors": [],
        "status": {
            "retrieval_mode": retrieval_mode,
            "gate": "N/A",
            "predicted_citation_count": None,
            "context_built": False,
            "final_context_tokens": None,
            "answer_generated": False,
            "abstained": False,
        },
        "debug": {
            "selected_evidence": [],
            "selected_parent_ids": [],
            "filtered_out_parents": [],
            "final_answer_chunk_ids": [],
        },
    }

    cfg = load_retrieval_config(config_path)

    try:
        retrieval_results = retrieve(
            query,
            mode=retrieval_mode,
            topk_vector=int(cfg.get("topk_vector", 40)),
            topk_bm25=int(cfg.get("topk_bm25", 40)),
            topk_final=int(cfg.get("topk_final", 40)),
            rrf_k=int(cfg.get("rrf_k", 60)),
            collection=str(cfg.get("collection") or "yhct_chunks_v2_full"),
            qdrant_url=str(cfg.get("qdrant_url") or "http://localhost:6333"),
            ollama_url=str(cfg.get("ollama_url") or "http://localhost:11434"),
            model=str(cfg.get("embed_model") or "nomic-embed-text"),
            chunks_path=str(cfg.get("chunks_path") or "data/chunks/chunks_v2_full.jsonl"),
            index_path=str(cfg.get("index_path") or "data/indexes/bm25_v2.pkl"),
            deduplicate=True,
            save_debug=False,
        )
        result["retrieval_results"] = retrieval_results
    except Exception as exc:
        result["errors"].append(f"Lỗi retrieval: {exc}")
        return result

    gate_result: dict[str, Any] | None = None
    if use_gate:
        try:
            gate_result = run_answerability_gate(query, result["retrieval_results"], gate_topk=gate_topk)
            result["gate_result"] = gate_result
            result["status"]["gate"] = "PASS" if gate_result.get("pass") else "FAIL"
            result["status"]["predicted_citation_count"] = gate_result.get("predicted_citation_count")
            result["debug"]["selected_evidence"] = list(gate_result.get("selected_evidence", []))
        except Exception as exc:
            result["errors"].append(f"Lỗi answerability gate: {exc}")
            result["status"]["gate"] = "ERROR"
    else:
        result["status"]["gate"] = "OFF"

    context_result: dict[str, Any] | None = None
    if build_context:
        try:
            selected_for_context = list((gate_result or {}).get("selected_evidence", []))
            context_result = build_parent_child_context(
                result["retrieval_results"],
                query_text=query,
                parents_path=str(cfg.get("parents_path") or "data/parents/parents_v2_full.jsonl"),
                topk_parent=int(cfg.get("topk_parent", 4)),
                window=int(cfg.get("window", 1)),
                window_centers=int(cfg.get("window_centers", 2)),
                parent_score_agg=str(cfg.get("parent_score_agg") or "max"),
                token_budget=int(cfg.get("token_budget", 3500)),
                context_mode=context_mode,
                context_focus=context_focus,
                deduplicate=True,
                selected_evidence=selected_for_context,
                context_from_gate=bool(use_gate and selected_for_context),
                max_context_parents=int(cfg.get("max_context_parents", 3)),
            )
            result["context_result"] = context_result
            result["status"]["context_built"] = True
            result["status"]["final_context_tokens"] = context_result.get("tokens_used")

            debug_ctx = context_result.get("debug", {}) if isinstance(context_result.get("debug"), dict) else {}
            result["debug"]["selected_parent_ids"] = list(debug_ctx.get("selected_parent_ids", []))
            result["debug"]["filtered_out_parents"] = list(debug_ctx.get("filtered_out_parents", []))
            result["debug"]["final_answer_chunk_ids"] = list(debug_ctx.get("final_answer_chunk_ids", []))
        except Exception as exc:
            result["errors"].append(f"Lỗi build context: {exc}")

    if generate_answer:
        try:
            if gate_result is None:
                gate_for_answer = {
                    "pass": True,
                    "reason": "Gate tắt; tiếp tục sinh câu trả lời với bằng chứng hiện có.",
                    "predicted_citation_count": 0,
                    "selected_evidence": [],
                    "gate_features": {},
                }
            else:
                gate_for_answer = gate_result

            selected_for_answer = []
            if isinstance(context_result, dict):
                raw_final = context_result.get("final_answer_evidence", [])
                if isinstance(raw_final, list):
                    selected_for_answer = [x for x in raw_final if isinstance(x, dict)]
            if not selected_for_answer:
                selected_for_answer = list((gate_for_answer or {}).get("selected_evidence", []))
            if not selected_for_answer:
                selected_for_answer = result["retrieval_results"][:5]

            focused_context = ""
            if isinstance(context_result, dict):
                focused_context = str(context_result.get("context", ""))

            answer_result, raw_model_output = generate_structured_answer(
                query=query,
                gate_decision=gate_for_answer,
                focused_context=focused_context,
                selected_evidence=selected_for_answer,
                retrieval_results=selected_for_answer,
                ollama_url=str(cfg.get("ollama_url") or "http://localhost:11434"),
                model=str(cfg.get("answer_model") or "qwen2.5:7b-instruct"),
                max_tokens=max(64, int(answer_max_tokens)),
                temperature=max(0.0, float(answer_temperature)),
            )

            result["answer_result"] = answer_result
            result["raw_model_output"] = raw_model_output
            result["status"]["answer_generated"] = True
            answer_text = str((answer_result or {}).get("answer") or "").strip()
            result["status"]["abstained"] = answer_text == ABSTAIN_ANSWER
        except Exception as exc:
            result["errors"].append(f"Lỗi sinh câu trả lời: {exc}")

    if result["answer_result"] and isinstance(result["answer_result"], dict):
        result["evidence_panel"] = build_evidence_panel_items(
            list(result["answer_result"].get("evidence", [])),
            retrieval_results=result["retrieval_results"],
        )
    elif gate_result is not None:
        result["evidence_panel"] = build_evidence_panel_items(
            list(gate_result.get("selected_evidence", [])),
            retrieval_results=result["retrieval_results"],
        )
    else:
        result["evidence_panel"] = build_evidence_panel_items(
            result["retrieval_results"][:6],
            retrieval_results=result["retrieval_results"],
        )

    if not generate_answer and use_gate and gate_result is not None and not gate_result.get("pass"):
        result["status"]["abstained"] = True

    return result
