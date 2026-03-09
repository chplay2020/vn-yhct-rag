# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false
"""Synthetic retrieval evaluation — sample chunks, generate questions, search, score.

Evaluates Hit@K by parent_id and source_id.

Usage:
    PYTHONPATH=src uv run python -m rag.eval.synth_retrieval_eval \
        --collection yhct_chunks_v2_full_emb \
        --chunks data/chunks/chunks_v2_full.jsonl \
        --sample-size 30 --topk 10 \
        --output data/reports/synth_retrieval_eval.json
"""

from __future__ import annotations

import argparse
import json
import logging
import random
import time
from pathlib import Path
from typing import Any

import requests  # type: ignore
from qdrant_client import QdrantClient  # type: ignore

from rag.utils.io import read_jsonl

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

# ── defaults ───────────────────────────────────────────────────────────────
DEFAULT_COLLECTION = "yhct_chunks_v2_full_emb"
DEFAULT_QDRANT_URL = "http://localhost:6333"
DEFAULT_OLLAMA_URL = "http://localhost:11434"
DEFAULT_EMBED_MODEL = "bge-m3"
DEFAULT_GEN_MODEL = "qwen2.5"
DEFAULT_SAMPLE_SIZE = 30
DEFAULT_TOPK = 10
DEFAULT_MIN_TEXT_LEN = 80


# ── helpers ────────────────────────────────────────────────────────────────

def _generate_question(
    text: str,
    ollama_url: str,
    model: str = DEFAULT_GEN_MODEL,
) -> str | None:
    """Generate a question about the text using Ollama chat."""
    prompt = (
        "Dựa trên đoạn văn bản y học cổ truyền sau, hãy đặt MỘT câu hỏi ngắn gọn "
        "mà câu trả lời nằm trong đoạn văn. Chỉ trả lời bằng câu hỏi, không giải thích.\n\n"
        f"Đoạn văn:\n{text[:1500]}\n\nCâu hỏi:"
    )
    try:
        resp = requests.post(
            f"{ollama_url}/api/generate",
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=120,
        )
        if resp.status_code == 200:
            answer = resp.json().get("response", "").strip()
            # Take first line only
            return answer.split("\n")[0].strip() if answer else None
        logger.warning("Question gen returned %d", resp.status_code)
    except Exception as exc:
        logger.warning("Question gen failed: %s", exc)
    return None


def _embed_text(
    text: str,
    ollama_url: str,
    model: str = DEFAULT_EMBED_MODEL,
) -> list[float] | None:
    """Embed a single text via Ollama."""
    try:
        resp = requests.post(
            f"{ollama_url}/api/embeddings",
            json={"model": model, "prompt": text},
            timeout=60,
        )
        if resp.status_code == 200:
            return resp.json().get("embedding")
    except Exception as exc:
        logger.warning("Embed failed: %s", exc)
    return None


# ── main eval ──────────────────────────────────────────────────────────────

def run_synth_eval(
    collection: str = DEFAULT_COLLECTION,
    chunks_path: str = "data/chunks/chunks_v2_full.jsonl",
    qdrant_url: str = DEFAULT_QDRANT_URL,
    ollama_url: str = DEFAULT_OLLAMA_URL,
    embed_model: str = DEFAULT_EMBED_MODEL,
    gen_model: str = DEFAULT_GEN_MODEL,
    sample_size: int = DEFAULT_SAMPLE_SIZE,
    topk: int = DEFAULT_TOPK,
    min_text_len: int = DEFAULT_MIN_TEXT_LEN,
) -> dict[str, Any]:
    """Run synthetic retrieval evaluation.

    1. Sample N chunks with sufficient text
    2. Generate a question for each
    3. Embed question → search topK
    4. Score hit@K by chunk_id, parent_id, source_id
    """
    chunks = read_jsonl(chunks_path)

    # Filter candidate chunks
    candidates = [
        c for c in chunks
        if len(c.get("text_norm") or c.get("text", "")) >= min_text_len
        and not c.get("is_noise")
        and c.get("chunk_id")
    ]
    logger.info("Candidates for sampling: %d / %d", len(candidates), len(chunks))

    sample = random.sample(candidates, min(sample_size, len(candidates)))
    logger.info("Sampled %d chunks for evaluation", len(sample))

    client: Any = QdrantClient(url=qdrant_url)  # type: ignore

    # Metrics accumulators
    hit_chunk: dict[int, int] = {}  # k -> count
    hit_parent: dict[int, int] = {}
    hit_source: dict[int, int] = {}
    total_evaluated = 0
    failed_gen = 0
    failed_embed = 0
    results_detail: list[dict[str, Any]] = []

    for i, chunk in enumerate(sample):
        text = chunk.get("text_norm") or chunk.get("text", "")
        chunk_id = chunk["chunk_id"]
        parent_id = chunk.get("parent_id", "")
        source_id = chunk.get("source_id", "")

        # 1. Generate question
        question = _generate_question(text, ollama_url, gen_model)
        if not question:
            failed_gen += 1
            continue

        # 2. Embed question
        q_vec = _embed_text(question, ollama_url, embed_model)
        if not q_vec:
            failed_embed += 1
            continue

        # 3. Search
        search_results: Any = client.search(
            collection_name=collection,
            query_vector=q_vec,
            limit=topk,
            with_payload=True,
        )

        # 4. Score
        found_chunk = False
        found_parent = False
        found_source = False
        result_items: list[dict[str, Any]] = []

        for rank, r in enumerate(search_results):
            payload = getattr(r, "payload", {}) or {}
            r_chunk_id = payload.get("chunk_id", "")
            r_parent_id = payload.get("parent_id", "")
            r_source_id = payload.get("source_id", "")

            if r_chunk_id == chunk_id and not found_chunk:
                found_chunk = True
                for k_val in range(rank + 1, topk + 1):
                    hit_chunk[k_val] = hit_chunk.get(k_val, 0) + 1

            if parent_id and r_parent_id == parent_id and not found_parent:
                found_parent = True
                for k_val in range(rank + 1, topk + 1):
                    hit_parent[k_val] = hit_parent.get(k_val, 0) + 1

            if r_source_id == source_id and not found_source:
                found_source = True
                for k_val in range(rank + 1, topk + 1):
                    hit_source[k_val] = hit_source.get(k_val, 0) + 1

            result_items.append({
                "rank": rank,
                "score": float(getattr(r, "score", 0)),
                "chunk_id": r_chunk_id,
                "parent_id": r_parent_id,
                "source_id": r_source_id,
            })

        total_evaluated += 1
        results_detail.append({
            "chunk_id": chunk_id,
            "parent_id": parent_id,
            "source_id": source_id,
            "question": question,
            "hit_chunk": found_chunk,
            "hit_parent": found_parent,
            "hit_source": found_source,
            "top_results": result_items[:5],
        })

        if (i + 1) % 5 == 0:
            logger.info(
                "Progress: %d/%d  (hit_chunk@%d=%.0f%%, hit_parent@%d=%.0f%%)",
                i + 1, len(sample),
                topk, (hit_chunk.get(topk, 0) / max(total_evaluated, 1)) * 100,
                topk, (hit_parent.get(topk, 0) / max(total_evaluated, 1)) * 100,
            )

    # Build summary
    k_values = [1, 3, 5, 10]
    summary: dict[str, Any] = {
        "total_sampled": len(sample),
        "total_evaluated": total_evaluated,
        "failed_question_gen": failed_gen,
        "failed_embed": failed_embed,
        "topk": topk,
        "metrics": {},
    }

    for k_val in k_values:
        if k_val > topk:
            continue
        n = max(total_evaluated, 1)
        summary["metrics"][f"hit@{k_val}"] = {
            "chunk_id": round(hit_chunk.get(k_val, 0) / n, 4),
            "parent_id": round(hit_parent.get(k_val, 0) / n, 4),
            "source_id": round(hit_source.get(k_val, 0) / n, 4),
        }

    summary["detail"] = results_detail

    logger.info("=" * 60)
    logger.info("Synthetic Retrieval Eval Summary")
    logger.info("  Evaluated: %d / %d sampled", total_evaluated, len(sample))
    for k_val in k_values:
        if k_val > topk:
            continue
        m = summary["metrics"][f"hit@{k_val}"]
        logger.info(
            "  Hit@%d: chunk=%.1f%%  parent=%.1f%%  source=%.1f%%",
            k_val, m["chunk_id"] * 100, m["parent_id"] * 100, m["source_id"] * 100,
        )
    logger.info("=" * 60)

    return summary


# ── CLI ────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(description="Synthetic retrieval evaluation")
    p.add_argument("--collection", default=DEFAULT_COLLECTION)
    p.add_argument("--chunks", default="data/chunks/chunks_v2_full.jsonl")
    p.add_argument("--qdrant-url", default=DEFAULT_QDRANT_URL)
    p.add_argument("--ollama-url", default=DEFAULT_OLLAMA_URL)
    p.add_argument("--embed-model", default=DEFAULT_EMBED_MODEL)
    p.add_argument("--gen-model", default=DEFAULT_GEN_MODEL)
    p.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    p.add_argument("--topk", type=int, default=DEFAULT_TOPK)
    p.add_argument("--min-text-len", type=int, default=DEFAULT_MIN_TEXT_LEN)
    p.add_argument("--output", default=None, help="Save report JSON")
    args = p.parse_args()

    t0 = time.time()
    report = run_synth_eval(
        collection=args.collection,
        chunks_path=args.chunks,
        qdrant_url=args.qdrant_url,
        ollama_url=args.ollama_url,
        embed_model=args.embed_model,
        gen_model=args.gen_model,
        sample_size=args.sample_size,
        topk=args.topk,
        min_text_len=args.min_text_len,
    )
    elapsed = time.time() - t0
    logger.info("Total time: %.1fs", elapsed)

    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        logger.info("Report saved to %s", out)


if __name__ == "__main__":
    main()
