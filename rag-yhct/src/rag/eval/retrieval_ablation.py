# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
"""Retrieval ablation — compare vector-only, BM25-only, and hybrid_rrf.

Generates synthetic questions from sampled chunks, retrieves with all 3 modes,
and reports Hit@K by chunk_id / parent_id / source_id + MRR.

Supports three question-generation modes:
  llm      — require Ollama; fail fast if unavailable
  fallback — deterministic keyword-based Vietnamese questions (no LLM)
  auto     — try LLM first, fallback on failure (default)

Reproducible question sets:
  --save-question-set <path>  save generated questions to JSON for reuse
  --question-set <path>       load a saved question set (skip generation)
  --generate-only             generate + save questions, then exit

Usage:
    # Original usage (unchanged)
    PYTHONPATH=src uv run python -m rag.eval.retrieval_ablation \
        --chunks data/chunks/chunks_v2_full.jsonl \
        --sample-size 30 --topk 10 --question-mode auto \
        --output data/reports/retrieval_ablation.json

    # Generate and save a fixed question set
    PYTHONPATH=src uv run python -m rag.eval.retrieval_ablation \
        --chunks data/chunks/chunks_v2_full.jsonl \
        --sample-size 100 --seed 42 --question-mode auto \
        --save-question-set data/eval/question_sets/qset_100_seed42.json \
        --generate-only

    # Run ablation from a saved question set
    PYTHONPATH=src uv run python -m rag.eval.retrieval_ablation \
        --question-set data/eval/question_sets/qset_100_seed42.json \
        --topk 10 --output data/reports/retrieval_ablation_qset.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, cast

import requests  # type: ignore

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
    embed_query,
    retrieve_vector_from_vec,
)
from rag.retrieve.hybrid_retriever import reciprocal_rank_fusion
from rag.utils.io import read_jsonl
from rag.utils.query_quality import is_query_noisy

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

# ── defaults ───────────────────────────────────────────────────────────────
DEFAULT_SAMPLE_SIZE = 30
DEFAULT_TOPK = 10
DEFAULT_TOPK_BM25 = 40
DEFAULT_TOPK_VECTOR = 40
DEFAULT_RRF_K = 60
DEFAULT_MIN_TEXT_LEN = 80
DEFAULT_SEED = 42
DEFAULT_GEN_MODEL = "qwen2.5"
DEFAULT_MAX_RETRIES = 3
QUESTION_MODES = ("llm", "fallback", "auto")

# Noise filters for chunk screening
_RE_CYRILLIC = re.compile(r"[\u0400-\u04FF]")
_MOJIBAKE_CHARS = frozenset("\u00bf\u00b6\u00b5\u00b9\u00b2\u00b3\u00bc\u00bd\u00be")
_BAD_RATIO_THRESHOLD = 0.03

# Vietnamese fallback question templates — {kw} is replaced with a keyword
_FALLBACK_TEMPLATES = [
    "Công dụng của {kw} là gì?",
    "Đặc điểm của {kw} là gì?",
    "{kw} có tác dụng gì trong y học cổ truyền?",
    "Bộ phận dùng của {kw} là gì?",
    "Cách sử dụng {kw} như thế nào?",
    "{kw} được dùng để chữa bệnh gì?",
    "Thành phần chính của {kw} gồm những gì?",
]


# ── Ollama health check ───────────────────────────────────────────────────

def _check_ollama_chat(ollama_url: str, model: str) -> bool:
    """Probe the Ollama /api/generate endpoint with a tiny prompt.

    Returns True if the model responds successfully, False otherwise.
    Logs the exact HTTP status on failure so the user knows what broke.
    """
    try:
        resp = requests.post(
            f"{ollama_url}/api/generate",
            json={"model": model, "prompt": "Xin chào", "stream": False},
            timeout=30,
        )
        if resp.status_code == 200:
            return True
        logger.error(
            "Ollama health-check FAILED: POST %s/api/generate → HTTP %d\n"
            "  model=%s   response_body=%s",
            ollama_url, resp.status_code, model, resp.text[:300],
        )
    except Exception as exc:
        logger.error("Ollama health-check FAILED: %s", exc)
    return False


# ── helpers ────────────────────────────────────────────────────────────────

def _is_clean(c: dict[str, Any], min_text_len: int) -> bool:
    """Check if chunk is suitable for evaluation."""
    text = c.get("text_norm") or c.get("text", "")
    if len(text) < min_text_len:
        return False
    if c.get("is_noise"):
        return False
    if not c.get("chunk_id"):
        return False
    if "\ufffd" in text:
        return False
    if _RE_CYRILLIC.search(text):
        return False
    text_ns = re.sub(r"\s", "", text)
    length = max(1, len(text_ns))
    bad = len(_RE_CYRILLIC.findall(text))
    bad += sum(1 for ch in text if ch in _MOJIBAKE_CHARS)
    if bad / length > _BAD_RATIO_THRESHOLD:
        return False
    return True


def _extract_keywords(text: str, n: int = 5) -> list[str]:
    """Extract N longest meaningful words as keyword hints."""
    words = re.findall(
        r"[a-záàảãạăắằẳẵặâấầẩẫậéèẻẽẹêếềểễệíìỉĩịóòỏõọôốồổỗộơớờởỡợ"
        r"úùủũụưứừửữựýỳỷỹỵđ]{3,}",
        text.lower(),
    )
    seen: set[str] = set()
    unique: list[str] = []
    for w in words:
        if w not in seen:
            seen.add(w)
            unique.append(w)
    unique.sort(key=len, reverse=True)
    return unique[:n]


# ── fallback question generator ───────────────────────────────────────────

def _generate_fallback_question(text: str, index: int = 0) -> str | None:
    """Create a deterministic Vietnamese question from the chunk text.

    Uses keyword extraction + rotating templates.  *index* is typically the
    sample counter so that consecutive chunks get different template styles.
    """
    keywords = _extract_keywords(text, n=5)
    if not keywords:
        return None
    # Pick template deterministically based on index
    tpl = _FALLBACK_TEMPLATES[index % len(_FALLBACK_TEMPLATES)]
    question = tpl.format(kw=keywords[0])
    if is_query_noisy(question):
        # Try with the second keyword if first one produces noise
        if len(keywords) >= 2:
            question = tpl.format(kw=keywords[1])
        if is_query_noisy(question):
            return None
    return question


# ── LLM question generator ───────────────────────────────────────────────

def _generate_question_llm(
    text: str,
    ollama_url: str,
    model: str = DEFAULT_GEN_MODEL,
    max_retries: int = DEFAULT_MAX_RETRIES,
) -> tuple[str | None, bool]:
    """Generate a question via Ollama LLM with validation and retry.

    Returns (question, used_llm).
    If HTTP 404 is returned, raises RuntimeError immediately (fail-fast)
    rather than silently retrying — 404 means the model is not loaded.
    """
    keywords = _extract_keywords(text, n=5)
    kw_hint = ", ".join(keywords[:4]) if keywords else ""

    prompt = (
        "Bạn là chuyên gia y học cổ truyền Việt Nam.\n"
        "Dựa trên đoạn văn bản dưới đây, hãy đặt MỘT câu hỏi ngắn gọn bằng tiếng Việt "
        "mà câu trả lời nằm hoàn toàn trong đoạn văn.\n\n"
        "YÊU CẦU:\n"
        "- Viết bằng tiếng Việt, KHÔNG dùng tiếng Anh hay ngôn ngữ khác.\n"
        "- Câu hỏi phải ngắn (dưới 30 từ), cụ thể, rõ ràng.\n"
        f"- Phải sử dụng ít nhất 1-2 từ khóa từ đoạn văn (ví dụ: {kw_hint}).\n"
        "- KHÔNG dùng các từ mơ hồ: \"đoạn văn\", \"bài viết\", \"nội dung trên\", "
        "\"this\", \"that\", \"it\".\n"
        "- Chỉ trả lời bằng câu hỏi, KHÔNG giải thích.\n\n"
        f"Đoạn văn:\n{text[:1500]}\n\nCâu hỏi:"
    )

    for attempt in range(max_retries):
        try:
            resp = requests.post(
                f"{ollama_url}/api/generate",
                json={"model": model, "prompt": prompt, "stream": False},
                timeout=120,
            )
            if resp.status_code == 404:
                raise RuntimeError(
                    f"Ollama returned HTTP 404 for model '{model}' at "
                    f"{ollama_url}/api/generate — model is not loaded. "
                    f"Run: ollama pull {model}"
                )
            if resp.status_code != 200:
                logger.warning(
                    "Question gen returned HTTP %d (attempt %d/%d)",
                    resp.status_code, attempt, max_retries,
                )
                continue
            answer = resp.json().get("response", "").strip()
            if not answer:
                continue
            question = answer.split("\n")[0].strip()
            if not is_query_noisy(question):
                return question, True
            logger.debug("Noisy LLM question (attempt %d): %s", attempt, question[:80])
        except RuntimeError:
            raise
        except Exception as exc:
            logger.warning("Question gen failed (attempt %d): %s", attempt, exc)

    return None, False


# ── unified question generator ────────────────────────────────────────────

def _generate_question(
    text: str,
    ollama_url: str,
    model: str,
    question_mode: str,
    index: int = 0,
) -> tuple[str | None, str]:
    """Generate a question according to *question_mode*.

    Returns (question, source) where source is "llm", "fallback", or "failed".
    """
    if question_mode == "fallback":
        q = _generate_fallback_question(text, index)
        return (q, "fallback") if q else (None, "failed")

    if question_mode in ("llm", "auto"):
        try:
            q, used_llm = _generate_question_llm(text, ollama_url, model)
            if q and used_llm:
                return q, "llm"
        except RuntimeError as exc:
            if question_mode == "llm":
                raise
            # auto mode — log and switch to fallback
            logger.warning("LLM unavailable, switching to fallback: %s", exc)

        # auto mode: try fallback
        if question_mode == "auto":
            q = _generate_fallback_question(text, index)
            if q:
                return q, "fallback"

    return None, "failed"


# ── question set save / load ───────────────────────────────────────────────

def save_question_set(
    questions: list[dict[str, Any]],
    path: str,
    *,
    seed: int,
    sample_size: int,
    question_mode: str,
    chat_model: str,
    embed_model: str,
    chunks_path: str,
) -> None:
    """Save a generated question set to *path* in a stable JSON format."""
    payload = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "seed": seed,
            "sample_size": sample_size,
            "question_mode": question_mode,
            "chat_model": chat_model,
            "embed_model": embed_model,
            "chunks_path": chunks_path,
        },
        "questions": questions,
    }
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    logger.info("Saved %d questions → %s", len(questions), out)


def load_question_set(
    path: str,
    chunks: list[dict[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load and validate a question set from *path*.

    Returns (questions, meta).  Validates required fields and tries to infer
    missing source_id / parent_id from *chunks* if provided.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Question set not found: {p}")

    with open(p, encoding="utf-8") as f:
        data = json.load(f)

    if not isinstance(data, dict) or "questions" not in data:
        raise ValueError(f"Malformed question set file: missing 'questions' key in {p}")

    raw_meta = data.get("meta", {})
    meta: dict[str, Any] = cast(dict[str, Any], raw_meta if isinstance(raw_meta, dict) else {})

    raw_questions_untyped = data["questions"]
    if not isinstance(raw_questions_untyped, list):
        raise ValueError(f"Question set is empty or not a list in {p}")
    raw_questions = cast(list[Any], raw_questions_untyped)
    if len(raw_questions) == 0:
        raise ValueError(f"Question set is empty or not a list in {p}")

    questions: list[dict[str, Any]] = []
    for i, item in enumerate(raw_questions):
        if not isinstance(item, dict):
            raise ValueError(f"Question at index {i} must be an object")
        questions.append(cast(dict[str, Any], item))

    # Build lookup from chunks for inference
    chunk_lookup: dict[str, dict[str, Any]] = {}
    if chunks:
        for c in chunks:
            cid = c.get("chunk_id")
            if cid:
                chunk_lookup[cid] = c

    for i, q in enumerate(questions):
        if not q.get("question"):
            raise ValueError(f"Question at index {i} missing required field 'question'")
        if not q.get("expected_chunk_id"):
            raise ValueError(f"Question at index {i} missing required field 'expected_chunk_id'")

        # Infer missing fields from chunk metadata
        cid = q["expected_chunk_id"]
        if chunk_lookup and cid in chunk_lookup:
            cm = chunk_lookup[cid]
            if not q.get("expected_source_id"):
                q["expected_source_id"] = cm.get("source_id", "")
            if not q.get("expected_parent_id"):
                q["expected_parent_id"] = cm.get("parent_id", "")

        # Ensure defaults
        q.setdefault("expected_source_id", "")
        q.setdefault("expected_parent_id", "")
        q.setdefault("question_source", "unknown")
        q.setdefault("is_noisy_question", False)

    logger.info("Loaded question set: %s  (%d questions)", p, len(questions))
    if meta:
        meta_seed = cast(object, meta.get("seed"))
        meta_sample_size = cast(object, meta.get("sample_size"))
        meta_question_mode = cast(object, meta.get("question_mode"))
        logger.info("  meta: seed=%s  sample_size=%s  question_mode=%s",
                     meta_seed, meta_sample_size, meta_question_mode)

    return questions, meta


# ── scoring helpers ───────────────────────────────────────────────────────

def _score_results(
    results: list[dict[str, Any]],
    target_chunk_id: str,
    target_parent_id: str,
    target_source_id: str,
    topk: int,
) -> dict[str, Any]:
    """Score a single retrieval result list for one query."""
    hit_chunk_at: int | None = None
    hit_parent_at: int | None = None
    hit_source_at: int | None = None

    for rank, r in enumerate(results[:topk]):
        cid = r.get("chunk_id", "")
        pid = r.get("parent_id", "")
        sid = r.get("source_id", "")

        if cid == target_chunk_id and hit_chunk_at is None:
            hit_chunk_at = rank
        if target_parent_id and pid == target_parent_id and hit_parent_at is None:
            hit_parent_at = rank
        if sid == target_source_id and hit_source_at is None:
            hit_source_at = rank

    return {
        "hit_chunk_at": hit_chunk_at,
        "hit_parent_at": hit_parent_at,
        "hit_source_at": hit_source_at,
    }


# ── main ablation ─────────────────────────────────────────────────────────

def run_ablation(
    chunks_path: str = DEFAULT_CHUNKS_PATH,
    collection: str = DEFAULT_COLLECTION,
    qdrant_url: str = DEFAULT_QDRANT_URL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    embed_model: str = DEFAULT_MODEL,
    gen_model: str = DEFAULT_GEN_MODEL,
    index_path: str = DEFAULT_INDEX_PATH,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    topk: int = DEFAULT_TOPK,
    topk_bm25: int = DEFAULT_TOPK_BM25,
    topk_vector: int = DEFAULT_TOPK_VECTOR,
    rrf_k: int = DEFAULT_RRF_K,
    min_text_len: int = DEFAULT_MIN_TEXT_LEN,
    seed: int = DEFAULT_SEED,
    question_mode: str = "auto",
    question_set_path: str | None = None,
    save_question_set_path: str | None = None,
    generate_only: bool = False,
) -> dict[str, Any] | None:
    """Run retrieval ablation: compare vector, bm25, hybrid_rrf.

    Returns report with per-mode metrics and per-query detail.
    If *generate_only* is True, saves the question set and returns None.
    """
    from_saved = question_set_path is not None

    # ── Load saved question set ───────────────────────────────────────────
    if from_saved:
        chunks_for_infer = read_jsonl(chunks_path) if Path(chunks_path).exists() else None
        loaded_questions, loaded_meta = load_question_set(
            question_set_path, chunks=chunks_for_infer,
        )
        # Prepare the eval_items list that mirrors the generation path
        eval_items: list[dict[str, Any]] = []
        llm_questions = 0
        fallback_questions = 0
        noisy_queries = 0
        for q in loaded_questions:
            query_is_noisy = q.get("is_noisy_question", False)
            if query_is_noisy:
                noisy_queries += 1
            qs = q.get("question_source", "unknown")
            if qs == "llm":
                llm_questions += 1
            elif qs == "fallback":
                fallback_questions += 1
            eval_items.append({
                "question": q["question"],
                "chunk_id": q["expected_chunk_id"],
                "parent_id": q.get("expected_parent_id", ""),
                "source_id": q.get("expected_source_id", ""),
                "q_source": qs,
                "query_is_noisy": query_is_noisy,
            })
        failed_gen = 0
        sample_size_actual = len(eval_items)
        logger.info("=" * 70)
        logger.info("ABLATION CONFIG  (from saved question set)")
        logger.info("  question_set : %s", question_set_path)
        logger.info("  questions    : %d", len(eval_items))
        logger.info("  embed_model  : %s", embed_model)
        if loaded_meta:
            logger.info("  meta.seed    : %s", loaded_meta.get("seed"))
            logger.info("  meta.mode    : %s", loaded_meta.get("question_mode"))
            logger.info("  meta.chat    : %s", loaded_meta.get("chat_model"))
        logger.info("=" * 70)
    else:
        # ── Original generation path ──────────────────────────────────────
        if question_mode not in QUESTION_MODES:
            raise ValueError(f"question_mode must be one of {QUESTION_MODES}, got '{question_mode}'")

        logger.info("=" * 70)
        logger.info("ABLATION CONFIG")
        logger.info("  ollama_url   : %s", ollama_url)
        logger.info("  chat_model   : %s", gen_model)
        logger.info("  embed_model  : %s", embed_model)
        logger.info("  chat_endpoint: %s/api/generate", ollama_url)
        logger.info("  question_mode: %s", question_mode)
        logger.info("=" * 70)

        # Ollama health check (for llm / auto modes)
        if question_mode in ("llm", "auto"):
            llm_ok = _check_ollama_chat(ollama_url, gen_model)
            if not llm_ok:
                if question_mode == "llm":
                    logger.error(
                        "FATAL: --question-mode=llm but Ollama chat model '%s' "
                        "is not reachable at %s. Aborting.",
                        gen_model, ollama_url,
                    )
                    sys.exit(1)
                else:
                    logger.warning(
                        "Ollama chat model '%s' not reachable — "
                        "auto mode will use fallback questions for ALL samples.",
                        gen_model,
                    )

        random.seed(seed)

        chunks = read_jsonl(chunks_path)
        candidates = [c for c in chunks if _is_clean(c, min_text_len)]
        logger.info("Candidates for sampling: %d / %d", len(candidates), len(chunks))

        sample = random.sample(candidates, min(sample_size, len(candidates)))
        logger.info("Sampled %d chunks (seed=%d)", len(sample), seed)

        # Generate questions
        eval_items = []
        saveable_questions: list[dict[str, Any]] = []
        failed_gen = 0
        llm_questions = 0
        fallback_questions = 0
        noisy_queries = 0

        for i, chunk in enumerate(sample):
            text = chunk.get("text_norm") or chunk.get("text", "")
            chunk_id = chunk["chunk_id"]
            parent_id = chunk.get("parent_id", "")
            source_id = chunk.get("source_id", "")

            question, q_source = _generate_question(
                text, ollama_url, gen_model, question_mode, index=i,
            )
            if not question:
                failed_gen += 1
                continue

            if q_source == "llm":
                llm_questions += 1
            elif q_source == "fallback":
                fallback_questions += 1

            query_is_noisy = is_query_noisy(question)
            if query_is_noisy:
                noisy_queries += 1

            eval_items.append({
                "question": question,
                "chunk_id": chunk_id,
                "parent_id": parent_id,
                "source_id": source_id,
                "q_source": q_source,
                "query_is_noisy": query_is_noisy,
            })

            saveable_questions.append({
                "question": question,
                "expected_chunk_id": chunk_id,
                "expected_source_id": source_id,
                "expected_parent_id": parent_id,
                "question_source": q_source,
                "chunk_text_preview": text[:200],
                "is_noisy_question": query_is_noisy,
            })

            if (i + 1) % 10 == 0:
                logger.info("Question generation: %d/%d (llm=%d fb=%d failed=%d)",
                            i + 1, len(sample), llm_questions, fallback_questions, failed_gen)

        # Save question set if requested
        if save_question_set_path:
            save_question_set(
                saveable_questions, save_question_set_path,
                seed=seed, sample_size=len(sample),
                question_mode=question_mode, chat_model=gen_model,
                embed_model=embed_model, chunks_path=chunks_path,
            )

        # Generate-only mode: save and exit
        if generate_only:
            if not save_question_set_path:
                logger.error("--generate-only requires --save-question-set")
                sys.exit(1)
            logger.info("Generate-only mode: %d questions saved. Exiting.", len(saveable_questions))
            return None

        sample_size_actual = len(sample)
        loaded_meta = None

    # ── Retrieval evaluation ──────────────────────────────────────────────
    modes = ["vector", "bm25", "hybrid_rrf"]

    # Accumulators: mode → metric_name → k → count
    hits: dict[str, dict[str, dict[int, int]]] = {
        m: {"chunk": {}, "parent": {}, "source": {}} for m in modes
    }
    mrr_sum: dict[str, dict[str, float]] = {
        m: {"chunk": 0.0, "parent": 0.0, "source": 0.0} for m in modes
    }

    total_evaluated = 0
    failed_embed = 0
    detail: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []
    sample_questions: list[dict[str, str]] = []

    for i, item in enumerate(eval_items):
        question = item["question"]
        chunk_id = item["chunk_id"]
        parent_id = item["parent_id"]
        source_id = item["source_id"]
        q_source = item["q_source"]
        query_is_noisy = item["query_is_noisy"]

        # Save sample questions (first 20)
        if len(sample_questions) < 20:
            sample_questions.append({
                "chunk_id": chunk_id,
                "question": question,
                "source": q_source,
            })

        if query_is_noisy:
            pass  # already counted

        # 2. Embed question (shared across modes that need it)
        q_vec = embed_query(question, ollama_url, embed_model)
        if not q_vec:
            failed_embed += 1
            continue

        total_evaluated += 1
        query_detail: dict[str, Any] = {
            "chunk_id": chunk_id,
            "parent_id": parent_id,
            "source_id": source_id,
            "question": question,
            "question_source": q_source,
            "query_noisy": query_is_noisy,
            "modes": {},
        }

        # 3. Retrieve with all 3 modes
        vec_results = retrieve_vector_from_vec(
            q_vec, topk=topk_vector, collection=collection, qdrant_url=qdrant_url,
        )
        bm25_results = retrieve_bm25(
            question, topk=topk_bm25, chunks_path=chunks_path, index_path=index_path,
        )
        hybrid_results = reciprocal_rank_fusion(
            bm25_results, vec_results, k=rrf_k, topk_final=topk,
        )

        mode_results = {
            "vector": vec_results,
            "bm25": bm25_results,
            "hybrid_rrf": hybrid_results,
        }

        for mode_name in modes:
            results = mode_results[mode_name]
            scores = _score_results(results, chunk_id, parent_id, source_id, topk)

            # Accumulate hit@K
            for entity, hit_at_key in [("chunk", "hit_chunk_at"), ("parent", "hit_parent_at"), ("source", "hit_source_at")]:
                hit_rank = scores[hit_at_key]
                if hit_rank is not None:
                    for k_val in range(hit_rank + 1, topk + 1):
                        hits[mode_name][entity][k_val] = hits[mode_name][entity].get(k_val, 0) + 1
                    # MRR contribution
                    mrr_sum[mode_name][entity] += 1.0 / (hit_rank + 1)

            top_cids = [r.get("chunk_id", "") for r in results[:topk]]
            top_pids = [r.get("parent_id", "") for r in results[:topk]]
            top_sids = [r.get("source_id", "") for r in results[:topk]]

            query_detail["modes"][mode_name] = {
                "hit_chunk_at": scores["hit_chunk_at"],
                "hit_parent_at": scores["hit_parent_at"],
                "hit_source_at": scores["hit_source_at"],
                "top5": [
                    {"rank": r.get("rank"), "chunk_id": r.get("chunk_id", "")[:40]}
                    for r in results[:5]
                ],
                "top_chunk_ids": top_cids[:5],
                "top_parent_ids": top_pids[:5],
                "top_source_ids": top_sids[:5],
            }

            # Collect failure diagnostics for missed chunk_id
            if scores["hit_chunk_at"] is None:
                failures.append({
                    "query": question,
                    "question_source": q_source,
                    "expected_chunk_id": chunk_id,
                    "expected_source_id": source_id,
                    "expected_parent_id": parent_id,
                    "query_noisy": query_is_noisy,
                    "mode": mode_name,
                    "top_retrieved_chunk_ids": top_cids[:10],
                    "top_retrieved_source_ids": top_sids[:10],
                    "top_retrieved_parent_ids": top_pids[:10],
                })

        detail.append(query_detail)

        if (i + 1) % 5 == 0:
            logger.info(
                "Progress: %d/%d evaluated=%d (llm=%d fb=%d)  "
                "vec_hit@5=%.0f%%  bm25_hit@5=%.0f%%  hybrid_hit@5=%.0f%%",
                i + 1, len(eval_items), total_evaluated,
                llm_questions, fallback_questions,
                (hits["vector"]["chunk"].get(5, 0) / max(total_evaluated, 1)) * 100,
                (hits["bm25"]["chunk"].get(5, 0) / max(total_evaluated, 1)) * 100,
                (hits["hybrid_rrf"]["chunk"].get(5, 0) / max(total_evaluated, 1)) * 100,
            )

    # Build summary
    k_values = [1, 3, 5, 10]
    n = max(total_evaluated, 1)

    summary: dict[str, Any] = {
        "config": {
            "sample_size": sample_size_actual,
            "total_evaluated": total_evaluated,
            "question_mode": question_mode,
            "llm_questions": llm_questions,
            "fallback_questions": fallback_questions,
            "failed_question_gen": failed_gen,
            "failed_embed": failed_embed,
            "noisy_queries": noisy_queries,
            "topk": topk,
            "topk_bm25": topk_bm25,
            "topk_vector": topk_vector,
            "rrf_k": rrf_k,
            "seed": seed,
            "ollama_url": ollama_url,
            "chat_model": gen_model,
            "embed_model": embed_model,
        },
        "metrics": {},
        "sample_questions": sample_questions,
        "detail": detail,
        "failures": failures,
    }

    # Include question set provenance when running from a saved set
    if from_saved:
        summary["question_set"] = {
            "question_set_path": question_set_path,
            "question_count": len(eval_items),
            "generation_meta": loaded_meta if loaded_meta else None,
        }

    for mode_name in modes:
        mode_metrics: dict[str, Any] = {}
        for k_val in k_values:
            if k_val > topk:
                continue
            mode_metrics[f"hit@{k_val}"] = {
                "chunk_id": round(hits[mode_name]["chunk"].get(k_val, 0) / n, 4),
                "parent_id": round(hits[mode_name]["parent"].get(k_val, 0) / n, 4),
                "source_id": round(hits[mode_name]["source"].get(k_val, 0) / n, 4),
            }
        mode_metrics["mrr"] = {
            "chunk_id": round(mrr_sum[mode_name]["chunk"] / n, 4),
            "parent_id": round(mrr_sum[mode_name]["parent"] / n, 4),
            "source_id": round(mrr_sum[mode_name]["source"] / n, 4),
        }
        summary["metrics"][mode_name] = mode_metrics

    # Print summary
    logger.info("=" * 70)
    logger.info(
        "RETRIEVAL ABLATION SUMMARY  (%d evaluated, seed=%d, mode=%s)",
        total_evaluated, seed, question_mode,
    )
    logger.info(
        "  Questions: %d LLM + %d fallback | %d failed | %d noisy",
        llm_questions, fallback_questions, failed_gen, noisy_queries,
    )
    logger.info("=" * 70)
    header = f"{'':>20s}  {'vector':>10s}  {'bm25':>10s}  {'hybrid_rrf':>10s}"
    logger.info(header)
    logger.info("-" * 60)
    for k_val in k_values:
        if k_val > topk:
            continue
        for entity in ["chunk_id", "parent_id", "source_id"]:
            row = f"  Hit@{k_val} {entity:>10s}"
            for mode_name in modes:
                val = summary["metrics"][mode_name][f"hit@{k_val}"][entity]
                row += f"  {val*100:9.1f}%"
            logger.info(row)
        logger.info("")

    for entity in ["chunk_id", "parent_id", "source_id"]:
        row = f"  MRR   {entity:>10s}"
        for mode_name in modes:
            val = summary["metrics"][mode_name]["mrr"][entity]
            row += f"  {val:9.4f}"
        logger.info(row)

    logger.info("=" * 70)

    return summary


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Retrieval ablation: vector vs BM25 vs hybrid")
    p.add_argument("--chunks", default=DEFAULT_CHUNKS_PATH)
    p.add_argument("--collection", default=DEFAULT_COLLECTION)
    p.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    p.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    p.add_argument("--embed-model", default=DEFAULT_MODEL)
    p.add_argument("--gen-model", "--chat-model", default=DEFAULT_GEN_MODEL,
                    dest="gen_model", help="Ollama generation/chat model name")
    p.add_argument("--bm25-index", default=DEFAULT_INDEX_PATH)
    p.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    p.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    p.add_argument("--topk-bm25", type=int, default=DEFAULT_TOPK_BM25)
    p.add_argument("--topk-vector", type=int, default=DEFAULT_TOPK_VECTOR)
    p.add_argument("--rrf-k", type=int, default=DEFAULT_RRF_K)
    p.add_argument("--min-text-len", type=int, default=DEFAULT_MIN_TEXT_LEN)
    p.add_argument("--seed", type=int, default=DEFAULT_SEED)
    p.add_argument("--question-mode", choices=QUESTION_MODES, default="auto",
                    help="llm=require Ollama; fallback=deterministic; auto=try LLM then fallback")
    p.add_argument("--output", default="data/reports/retrieval_ablation.json")
    # ── question-set reproducibility flags ────────────────────────────────
    p.add_argument("--question-set", default=None, metavar="PATH",
                    help="Load a saved question set (skip generation)")
    p.add_argument("--save-question-set", default=None, metavar="PATH",
                    help="Save generated questions to JSON for later reuse")
    p.add_argument("--generate-only", action="store_true", default=False,
                    help="Generate and save questions, then exit (no retrieval)")
    args = p.parse_args()

    # Validation
    if args.generate_only and not args.save_question_set:
        p.error("--generate-only requires --save-question-set")
    if args.question_set and args.save_question_set:
        p.error("--question-set and --save-question-set are mutually exclusive")
    if args.question_set and args.generate_only:
        p.error("--question-set and --generate-only are mutually exclusive")

    t0 = time.time()
    report = run_ablation(
        chunks_path=args.chunks,
        collection=args.collection,
        qdrant_url=args.qdrant_url,
        ollama_url=args.ollama_url,
        embed_model=args.embed_model,
        gen_model=args.gen_model,
        index_path=args.bm25_index,
        sample_size=args.sample_size,
        topk=args.topk,
        topk_bm25=args.topk_bm25,
        topk_vector=args.topk_vector,
        rrf_k=args.rrf_k,
        min_text_len=args.min_text_len,
        seed=args.seed,
        question_mode=args.question_mode,
        question_set_path=args.question_set,
        save_question_set_path=args.save_question_set,
        generate_only=args.generate_only,
    )
    elapsed = time.time() - t0
    logger.info("Total time: %.1fs", elapsed)

    if report is not None and args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info("Report saved to %s", out)


if __name__ == "__main__":
    main()
