# pyright: reportMissingImports=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportOptionalMemberAccess=false
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import streamlit as st

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from rag.generate.answer_generator import ABSTAIN_ANSWER, SAFETY_NOTE_VI
from rag.ui.pipeline_runner import run_demo_pipeline


st.set_page_config(page_title="RAG YHCT - Demo Bằng Chứng", layout="wide")

st.title("RAG YHCT - Demo Trả Lời Có Bằng Chứng")
st.caption("Luồng hiện tại: Hybrid RRF -> Answerability Gate -> Focused Context -> Local LLM + Citations")

if "query_input" not in st.session_state:
    st.session_state["query_input"] = ""
if "demo_result" not in st.session_state:
    st.session_state["demo_result"] = None

st.subheader("A. Truy vấn")
query = st.text_input("Nhập câu hỏi", key="query_input", placeholder="Ví dụ: tác dụng của cây ngải cứu")

example_queries = [
    "tác dụng của cây ngải cứu",
    "tác dụng của cây sả",
    "thuốc nào chữa khỏi hoàn toàn mọi loại ung thư",
]
example_cols = st.columns(3)
for idx, sample in enumerate(example_queries):
    if example_cols[idx].button(sample, key=f"example_{idx}"):
        st.session_state["query_input"] = sample
        st.rerun()

st.subheader("B. Chế độ demo")
mode_col1, mode_col2 = st.columns(2)
with mode_col1:
    retrieval_mode = st.selectbox(
        "Retrieval mode",
        options=["vector", "bm25", "hybrid_rrf"],
        index=2,
    )
    use_gate = st.checkbox("Dùng Answerability Gate", value=True)
with mode_col2:
    build_context = st.checkbox("Xây dựng context", value=True)
    generate_answer = st.checkbox("Sinh câu trả lời", value=True)

adv_col1, adv_col2 = st.columns(2)
with adv_col1:
    gate_topk = st.slider("Gate Top-K", min_value=1, max_value=10, value=5)
with adv_col2:
    answer_max_tokens = st.slider("Số token trả lời tối đa", min_value=128, max_value=1024, value=512, step=64)

run_clicked = st.button("Chạy truy vấn", type="primary")

if run_clicked:
    q = st.session_state.get("query_input", "").strip()
    if not q:
        st.warning("Vui lòng nhập câu hỏi trước khi chạy.")
    else:
        with st.spinner("Đang chạy retrieval/gate/context/generation..."):
            result = run_demo_pipeline(
                query=q,
                retrieval_mode=retrieval_mode,
                use_gate=use_gate,
                build_context=build_context,
                generate_answer=generate_answer,
                gate_topk=gate_topk,
                answer_max_tokens=answer_max_tokens,
                answer_temperature=0.2,
            )
        st.session_state["demo_result"] = result

result: dict[str, Any] | None = st.session_state.get("demo_result")
if not result:
    st.info("Chưa có kết quả. Hãy nhập câu hỏi và bấm Chạy truy vấn.")
    st.stop()
assert result is not None

errors = list(result.get("errors", []))
for err in errors:
    st.error(err)

status = result.get("status", {}) if isinstance(result.get("status"), dict) else {}
gate_result = result.get("gate_result") if isinstance(result.get("gate_result"), dict) else None
context_result = result.get("context_result") if isinstance(result.get("context_result"), dict) else None
answer_result = result.get("answer_result") if isinstance(result.get("answer_result"), dict) else None

st.subheader("C. Kết quả chính")
if answer_result:
    answer_text = str(answer_result.get("answer") or "").strip()
    if answer_text == ABSTAIN_ANSWER or bool(status.get("abstained")):
        st.warning("Không đủ căn cứ trong tài liệu hiện có.")
    st.markdown("### Câu trả lời")
    st.write(answer_text or "(Không có nội dung trả lời)")

    st.markdown("### Key concepts")
    key_concepts = answer_result.get("key_concepts")
    if isinstance(key_concepts, list) and key_concepts:
        for concept in key_concepts:
            st.write(f"- {concept}")
    else:
        st.write("- (Không có)")

    st.markdown("### Limits")
    st.info(str(answer_result.get("limits") or "Chưa có mô tả giới hạn."))

    st.markdown("### Safety note")
    st.caption(str(answer_result.get("safety_note") or SAFETY_NOTE_VI))
else:
    if bool(status.get("abstained")):
        st.warning("Không đủ căn cứ trong tài liệu hiện có.")
    if not generate_answer:
        st.info("Chế độ sinh câu trả lời đang tắt. Hệ thống chỉ hiển thị bằng chứng và trạng thái pipeline.")

st.subheader("D. Evidence panel")
evidence_items = result.get("evidence_panel") if isinstance(result.get("evidence_panel"), list) else []
if not evidence_items:
    st.info("Không có bằng chứng để hiển thị.")
else:
    for idx, item in enumerate(evidence_items, start=1):
        if not isinstance(item, dict):
            continue
        cite = str(item.get("citation_id") or f"E{idx}")
        snippet = str(item.get("snippet") or "")
        with st.expander(f"{cite} | chunk={item.get('chunk_id')}"):
            st.write(snippet or "(Không có snippet)")
            st.write(f"chunk_id: {item.get('chunk_id')}")
            st.write(f"parent_id: {item.get('parent_id')}")
            st.write(f"score: {item.get('score')}")
            st.write(f"title: {item.get('title') or '(N/A)'}")
            st.write(f"page_range: {item.get('page_range') or '(N/A)'}")
            st.write(f"section_heading: {item.get('section_heading') or '(N/A)'}")
            st.write(f"file_path: {item.get('file_path') or '(N/A)'}")

st.subheader("E. Trạng thái gate / pipeline")
status_col1, status_col2, status_col3 = st.columns(3)
status_col1.metric("Retrieval mode", str(status.get("retrieval_mode") or "N/A"))
status_col2.metric("Gate", str(status.get("gate") or "N/A"))
status_col3.metric("Predicted citation", str(status.get("predicted_citation_count") or "N/A"))

status_col4, status_col5, status_col6 = st.columns(3)
status_col4.metric("Context built", "Yes" if status.get("context_built") else "No")
status_col5.metric("Context tokens", str(status.get("final_context_tokens") or "N/A"))
status_col6.metric("Answer generated", "Yes" if status.get("answer_generated") else "No")

if bool(status.get("abstained")):
    st.warning("Abstained: Yes")
else:
    st.success("Abstained: No")

if gate_result:
    gate_pass = bool(gate_result.get("pass"))
    if gate_pass:
        st.success("Gate PASS")
    else:
        st.warning("Gate FAIL")

    gf = gate_result.get("gate_features") if isinstance(gate_result.get("gate_features"), dict) else {}
    st.write(f"reason: {gate_result.get('reason')}")
    st.write(f"top1/top2/gap: {gf.get('top1_score')} / {gf.get('top2_score')} / {gf.get('top1_top2_gap')}")
    st.write(
        "evidence/parent/source: "
        f"{gf.get('evidence_count')} / {gf.get('distinct_parent_count')} / {gf.get('distinct_source_count')}"
    )

st.subheader("F. Raw debug / Advanced")
with st.expander("Xem debug chi tiết", expanded=False):
    debug = result.get("debug") if isinstance(result.get("debug"), dict) else {}
    debug_payload = {
        "raw_selected_evidence": debug.get("selected_evidence", []),
        "selected_parent_ids": debug.get("selected_parent_ids", []),
        "filtered_out_parents": debug.get("filtered_out_parents", []),
        "final_answer_chunk_ids": debug.get("final_answer_chunk_ids", []),
        "context_debug": (context_result or {}).get("debug", {}),
        "raw_model_output": result.get("raw_model_output"),
    }
    st.json(debug_payload)

    raw_result = {
        "query": result.get("query"),
        "mode": result.get("mode"),
        "retrieval_results_count": len(result.get("retrieval_results") or []),
        "gate_result": result.get("gate_result"),
        "context_result": {
            "mode": (context_result or {}).get("mode"),
            "tokens_used": (context_result or {}).get("tokens_used"),
            "parents": (context_result or {}).get("parents"),
        },
        "answer_result": result.get("answer_result"),
        "errors": result.get("errors", []),
    }
    st.code(json.dumps(raw_result, ensure_ascii=False, indent=2), language="json")
