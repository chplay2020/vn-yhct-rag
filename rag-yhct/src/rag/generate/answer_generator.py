from __future__ import annotations

import json
import re
from typing import Any
from typing import cast

import requests  # type: ignore

DEFAULT_ANSWER_MODEL = "qwen2.5:7b-instruct"
DEFAULT_ANSWER_MAX_TOKENS = 512
DEFAULT_ANSWER_TEMPERATURE = 0.2
DEFAULT_ANSWER_TIMEOUT_S = 120

ABSTAIN_ANSWER = "Không đủ căn cứ trong tài liệu hiện có."
SAFETY_NOTE_VI = (
    "Thông tin này chỉ nhằm mục đích tham khảo từ tài liệu đã truy xuất, "
    "không thay thế chẩn đoán hoặc chỉ định điều trị của bác sĩ."
)


def _score_of(item: dict[str, Any]) -> float:
    for key in ("fused_score", "vector_score", "bm25_score", "score"):
        value = item.get(key)
        if value is None:
            continue
        try:
            return float(value)
        except Exception:
            continue
    return 0.0


def _compact_text(value: Any, max_chars: int = 300) -> str:
    text = str(value or "").replace("\n", " ").strip()
    text = re.sub(r"\s+", " ", text)
    if len(text) <= max_chars:
        return text
    return f"{text[: max_chars - 3].rstrip()}..."


def _build_evidence_items(
    selected_evidence: list[dict[str, Any]] | None,
    *,
    retrieval_results: list[dict[str, Any]] | None,
    fallback_limit: int = 4,
) -> list[dict[str, Any]]:
    selected = list(selected_evidence or [])
    if not selected:
        selected = list(retrieval_results or [])[:fallback_limit]

    by_chunk: dict[str, dict[str, Any]] = {}
    for r in retrieval_results or []:
        cid = str(r.get("chunk_id") or "")
        if cid and cid not in by_chunk:
            by_chunk[cid] = r

    out: list[dict[str, Any]] = []
    for idx, item in enumerate(selected):
        cid = str(item.get("chunk_id") or "")
        merged = dict(by_chunk.get(cid, {}))
        merged.update(item)

        evidence_item: dict[str, Any] = {
            "citation_id": f"E{idx + 1}",
            "chunk_id": merged.get("chunk_id"),
            "parent_id": merged.get("parent_id"),
            "title": merged.get("title"),
            "file_path": merged.get("file_path") or merged.get("source_id"),
            "page_range": merged.get("page_range") or merged.get("parent_locator"),
            "section_heading": merged.get("section_heading") or merged.get("doc_type"),
            "score": round(_score_of(merged), 6),
            "snippet": _compact_text(merged.get("text", ""), max_chars=360),
        }
        out.append(evidence_item)

    return out


def _extract_json(text: str) -> dict[str, Any] | None:
    try:
        loaded = json.loads(text)
        if isinstance(loaded, dict):
            return cast(dict[str, Any], loaded)
    except Exception:
        pass

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            loaded = json.loads(text[start : end + 1])
            if isinstance(loaded, dict):
                return cast(dict[str, Any], loaded)
        except Exception:
            return None
    return None


def _enforce_citations(answer: str, available_citations: list[str]) -> str:
    if not answer.strip() or not available_citations:
        return answer.strip()

    default_cite = f"[{available_citations[0]}]"
    lines = answer.strip().splitlines()
    fixed: list[str] = []
    citation_pattern = re.compile(r"\[E\d+\]")

    def _normalize_citation_spacing(text: str) -> str:
        # Ensure markers like "[E1]Nó" become "[E1] Nó".
        normalized = re.sub(r"(\[E\d+\])(?=\S)", r"\1 ", text)
        normalized = re.sub(r"[ \t]+(\[E\d+\])", r" \1", normalized)
        return re.sub(r"[ \t]{2,}", " ", normalized).strip()

    def _with_sentence_citations(text: str) -> str:
        # Add citation to each sentence-like claim when missing.
        parts = re.split(r"([.!?]+\s*)", text)
        rebuilt: list[str] = []
        i = 0
        while i < len(parts):
            segment = parts[i]
            tail = parts[i + 1] if i + 1 < len(parts) else ""
            combined = f"{segment}{tail}" if tail else segment
            stripped = combined.strip()
            if stripped and not citation_pattern.search(combined):
                combined = f"{combined.rstrip()} {default_cite}"
            rebuilt.append(combined)
            i += 2
        joined = "".join(rebuilt).strip()
        return _normalize_citation_spacing(joined)

    for line in lines:
        raw = line.rstrip()
        stripped = raw.strip()
        if not stripped:
            fixed.append(raw)
            continue

        has_citation = bool(citation_pattern.search(stripped))
        bullet_like = stripped.startswith("-") or stripped.startswith("*") or stripped.startswith("•")

        if bullet_like and not has_citation:
            fixed.append(f"{raw} {default_cite}")
            continue

        if re.search(r"[A-Za-zÀ-ỹà-ỹ0-9]", stripped):
            fixed.append(_with_sentence_citations(raw))
            continue

        fixed.append(raw)

    return _normalize_citation_spacing("\n".join(fixed).strip())


def _format_evidence_for_prompt(evidence_items: list[dict[str, Any]]) -> str:
    rows: list[str] = []
    for ev in evidence_items:
        rows.append(
            " | ".join(
                [
                    f"[{ev.get('citation_id')}]",
                    f"chunk_id={ev.get('chunk_id')}",
                    f"parent_id={ev.get('parent_id')}",
                    f"title={ev.get('title') or ''}",
                    f"file_path={ev.get('file_path') or ''}",
                    f"page_range={ev.get('page_range') or ''}",
                    f"section_heading={ev.get('section_heading') or ''}",
                    f"score={ev.get('score')}",
                    f"snippet={ev.get('snippet') or ''}",
                ]
            )
        )
    return "\n".join(rows)


def _abstain_response(
    *,
    limits: str,
    evidence_items: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "answer": ABSTAIN_ANSWER,
        "key_concepts": [],
        "evidence": evidence_items,
        "limits": limits.strip(),
        "safety_note": SAFETY_NOTE_VI,
    }


def generate_structured_answer(
    *,
    query: str,
    gate_decision: dict[str, Any] | None,
    focused_context: str,
    selected_evidence: list[dict[str, Any]] | None,
    retrieval_results: list[dict[str, Any]] | None = None,
    ollama_url: str = "http://localhost:11434",
    model: str = DEFAULT_ANSWER_MODEL,
    max_tokens: int = DEFAULT_ANSWER_MAX_TOKENS,
    temperature: float = DEFAULT_ANSWER_TEMPERATURE,
    timeout_s: int = DEFAULT_ANSWER_TIMEOUT_S,
) -> tuple[dict[str, Any], str | None]:
    """Generate grounded answer from context and evidence metadata.

    Returns (structured_answer, raw_model_output).
    """

    evidence_items = _build_evidence_items(
        selected_evidence,
        retrieval_results=retrieval_results,
    )
    available_citations = [str(ev.get("citation_id")) for ev in evidence_items if ev.get("citation_id")]

    gate_pass = bool((gate_decision or {}).get("pass", True))
    gate_reason = str((gate_decision or {}).get("reason") or "")

    if not gate_pass:
        limits = (
            "Bộ lọc answerability cho thấy bằng chứng truy xuất chưa đủ mạnh để trả lời chắc chắn. "
            f"Chi tiết: {gate_reason or 'Không có đủ bằng chứng mạnh và đồng thuận.'}"
        )
        return _abstain_response(limits=limits, evidence_items=evidence_items), None

    if not focused_context.strip() or not evidence_items:
        limits = "Không đủ ngữ cảnh hoặc bằng chứng đã chọn để tạo câu trả lời đáng tin cậy."
        return _abstain_response(limits=limits, evidence_items=evidence_items), None

    prompt = f"""
Bạn là trợ lý trả lời y học cổ truyền tiếng Việt, cực kỳ thận trọng và chỉ được dùng bằng chứng đã cung cấp.

YÊU CẦU BẮT BUỘC:
1) Chỉ dùng thông tin từ CONTEXT và EVIDENCE dưới đây, tuyệt đối không thêm kiến thức ngoài.
2) Mỗi ý chính trong câu trả lời phải có trích dẫn [E#] tương ứng.
3) Nếu thiếu bằng chứng hoặc còn mơ hồ, phải nói rõ giới hạn.
4) Không chẩn đoán bệnh, không kê đơn, không đưa chỉ định điều trị nguy hiểm.
5) Trả lời ngắn gọn, có căn cứ, tiếng Việt rõ ràng.

ĐỊNH DẠNG TRẢ VỀ: CHỈ MỘT JSON object hợp lệ, không markdown, không text thừa.
Schema JSON:
{{
  "answer": "string",
  "key_concepts": ["string"],
  "limits": "string"
}}

QUERY:
{query}

EVIDENCE MAP (dùng marker [E#] đúng như dưới):
{_format_evidence_for_prompt(evidence_items)}

CONTEXT:
{focused_context}
""".strip()

    payload: dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "format": "json",
        "options": {
            "temperature": float(temperature),
            "num_predict": int(max_tokens),
        },
    }

    raw_model_output: str | None = None
    try:
        resp = requests.post(f"{ollama_url}/api/generate", json=payload, timeout=timeout_s)
        if resp.status_code != 200:
            limits = f"Không gọi được mô hình cục bộ (HTTP {resp.status_code})."
            return _abstain_response(limits=limits, evidence_items=evidence_items), None

        body = resp.json()
        raw_model_output = str(body.get("response") or "").strip()
        parsed = _extract_json(raw_model_output)
        if parsed is None:
            limits = "Mô hình trả về sai định dạng JSON nên hệ thống không thể xác minh câu trả lời an toàn."
            return _abstain_response(limits=limits, evidence_items=evidence_items), raw_model_output

        answer = str(parsed.get("answer") or "").strip()
        key_concepts_raw = parsed.get("key_concepts")
        if isinstance(key_concepts_raw, list):
            key_concepts_any = cast(list[Any], key_concepts_raw)
            key_concepts = [str(x).strip() for x in key_concepts_any if str(x).strip()]
        else:
            key_concepts = []
        limits = str(parsed.get("limits") or "").strip()

        if not answer:
            answer = ABSTAIN_ANSWER
            if not limits:
                limits = "Mô hình không tạo được câu trả lời hợp lệ từ ngữ cảnh đã truy xuất."

        answer = _enforce_citations(answer, available_citations)
        if not limits:
            limits = "Câu trả lời bị giới hạn trong phạm vi các đoạn tài liệu đã truy xuất."

        structured: dict[str, Any] = {
            "answer": answer,
            "key_concepts": key_concepts[:8],
            "evidence": evidence_items,
            "limits": limits,
            "safety_note": SAFETY_NOTE_VI,
        }
        return structured, raw_model_output

    except Exception as exc:
        limits = f"Lỗi khi sinh câu trả lời từ mô hình cục bộ: {exc}"
        return _abstain_response(limits=limits, evidence_items=evidence_items), raw_model_output
