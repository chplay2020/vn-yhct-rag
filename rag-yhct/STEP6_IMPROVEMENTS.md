# STEP 6: Hybrid Retrieval + Fusion — Stability & Thesis-Ready Patches

**Date**: March 10, 2026  
**Goal**: Make evaluation cleaner, more stable, and faithful to real retrieval objectives  
**Status**: ✅ Complete — all patches applied, validated, tested

---

## Files Changed

| File | Change | Purpose |
|------|--------|---------|
| `src/rag/utils/query_quality.py` | **Created** | Query-noise detection, duplicate text normalization |
| `src/rag/eval/retrieval_ablation.py` | **Patched** | Tighter QA prompt, validation/retry, failure diagnostics |
| `src/rag/retrieve/hybrid_retriever.py` | **Patched** | Multi-center windowing, dedup, parent scoring, debug |
| `src/rag/retrieve/vector_retriever.py` | **Fixed** | qdrant-client API compatibility (query_points) |

---

## Part A: Improved QA Generation & Validation (retrieval_ablation.py)

### Problem
Synthetic QA generator produced noisy/vague/multilingual questions → higher metric variance, inflated false-negatives.

### Solution

#### 1. Tighter Prompt (Vietnamese-only, keyword-grounded)
```python
"- Viết bằng tiếng Việt, KHÔNG dùng tiếng Anh hay ngôn ngữ khác.
- Câu hỏi phải ngắn (dưới 30 từ), cụ thể, rõ ràng.
- Phải sử dụng ít nhất 1-2 từ khóa từ đoạn văn (ví dụ: {kw_hint}).
- KHÔNG dùng các từ mơ hồ: \"đoạn văn\", \"bài viết\", \"nội dung trên\", \"this\", \"that\", \"it\"."
```

#### 2. Query Validation + Retry
- `is_query_noisy()`: rejects Cyrillic, mojibake, too-short (<12 chars), non-Latin/Viet, vague-English
- Retry up to 3 times on failure
- Fallback: simple keyword-grounded question like `"{keyword} có tác dụng gì trong y học cổ truyền?"`

#### 3. Failure Diagnostics
Each failed query stores:
- `query`: the attempted question
- `expected_chunk_id`, `expected_source_id`, `expected_parent_id`: ground truth
- `top_retrieved_*`: actual top-10 retrieved IDs by all 3 modes
- `query_noisy`: boolean flag for audit

### Output Changes
```json
{
  "config": {
    "total_evaluated": 30,
    "failed_question_gen": 2,
    "noisy_queries": 1,
    "failed_embed": 0
  },
  "detail": [
    {
      "query_noisy": false,
      "modes": {
        "vector": {
          "hit_chunk_at": 2,
          "top_chunk_ids": ["chunk_a", "chunk_b", "chunk_c"],
          "top_source_ids": ["src_1", "src_2", "src_3"]
        }
      }
    }
  ],
  "failures": [
    {
      "query": "...",
      "expected_chunk_id": "...",
      "expected_source_id": "...",
      "top_retrieved_chunk_ids": [...],
      "query_noisy": true
    }
  ]
}
```

### Metric Impact
- **Stability**: ↑ Noisy queries eliminated
- **Fairness**: ↑ Keyword-grounded questions match natural retrieval task
- **Auditability**: ↑ Explicit failure logs enable targeted debugging

---

## Part B: Multi-Center Windowing & Deduplication (hybrid_retriever.py)

### Problem
1. When a parent has multiple matching children, only best child's context was used → missed relevant context
2. Exact-duplicate text across chunks inflated Hit@K → misleading rank metrics
3. Limited parent scoring options (only max child score)

### Solution

#### 1. Multi-Center Windowing
```python
window_centers = 2  # default: top-2 child centers per parent
window = 1          # default: ±1 fragment around each center
# Union all window fragments, sort by position
```

**Why**: With avg 4.65 children/parent (p95=6), many parents have 2+ matching chunks. This captures richer context.

#### 2. Duplicate Suppression
```python
def _deduplicate_results(results):
    # NFC normalize + lowercase + collapse whitespace
    seen_texts = {normalize_for_dedup(r["text"]) for r in results}
    # Keep highest-score version of each unique text
    # re-rank output
```

**Why**: Exact duplicates (e.g., same text in multiple chunks) shouldn't occupy separate rank slots. Otherwise Hit@5 reports finding 2 "novel" chunks when really it's the same text twice.

#### 3. Parent Score Aggregation
```python
parent_score_agg = "max"      # default: max child score
# or: "sum_top2"             # sum of top-2 child scores
```

**Why**: Allows tuning context assembly based on task (re-ranking vs fusion).

#### 4. Enhanced Debug Output
```python
debug_info: dict = {
    "parent_scores": [
        {"parent_id": "p1", "score": 0.95, "children_in_topk": 3},
        ...
    ],
    "selected_parent_ids": ["p1", "p2"],
    "selected_child_ranges": [
        {"parent_id": "p1", "centers": [0, 2]},  # windows at indices 0 and 2
        ...
    ],
    "final_context_tokens": 2847,
}
```

### CLI Parameters
```bash
--window-centers 2            # top-M child centers per parent
--window 1                    # ±W fragments
--parent-score-agg max        # "max" or "sum_top2"
--no-dedup                    # disable dedup if needed
```

### Metric Impact
- **Context Quality**: ↑ Multi-child parents capture wider context
- **Rank Validity**: ↑ Dedup prevents duplicate-text rank slots
- **Flexibility**: ↑ Configurable scoring & debug visibility

---

## Part C: qdrant-client API Compatibility Fix (vector_retriever.py)

### Problem
New qdrant-client (v2.5+) changed API: `client.search()` → `client.query_points()`

### Solution
```python
# Adapt to latest qdrant-client API
response = client.query_points(
    collection_name=collection,
    query=query_vec,              # was: query_vector
    limit=topk,
    with_payload=True,
    query_filter=query_filter,
)
results = response.points
```

### Impact
- **Compatibility**: ✅ Works with latest qdrant-client
- **All Modes**: vector-only, bm25-only, hybrid_rrf all functional

---

## Execution Commands

### Build BM25 Index (one-time)
```bash
cd rag-yhct
PYTHONPATH=src uv run python -m rag.retrieve.bm25_retriever --build \
    --chunks data/chunks/chunks_v2_full.jsonl
```

### Single Query — Vector Only
```bash
PYTHONPATH=src uv run python -m rag.retrieve.hybrid_retriever \
    --query "tác dụng của cây ngải cứu" \
    --mode vector \
    --topk-final 10
```

### Single Query — Hybrid RRF with Multi-Center Context
```bash
PYTHONPATH=src uv run python -m rag.retrieve.hybrid_retriever \
    --query "tác dụng của cây ngải cứu" \
    --mode hybrid_rrf \
    --build-context \
    --window-centers 2 \
    --window 1 \
    --save-debug \
    --debug-dir data/reports/retrieval_debug
```

### Retrieval Ablation (Tight QA + Validation)
```bash
PYTHONPATH=src uv run python -m rag.eval.retrieval_ablation \
    --chunks data/chunks/chunks_v2_full.jsonl \
    --sample-size 100 \
    --topk 10 \
    --seed 42 \
    --output data/reports/retrieval_ablation_v2.json
```

---

## Validation

✅ **All tests passed**:
- `query_quality` helpers: noise detection, dedup (unit tests)
- `bm25_retriever`: import, BM25 indexing
- `vector_retriever`: vector search with query_points API
- `hybrid_retriever`: RRF fusion, multi-center windowing, dedup
- `retrieval_ablation`: QA generation with retry, failure collection
- **End-to-end**: 3-mode ablation on real data (vector, bm25, hybrid_rrf)

---

## Why These Changes Improve Metric Stability

| Issue | Before | After | Mechanism |
|-------|--------|-------|-----------|
| **Noisy questions** | Vague/multilingual QA → false-negative hits | Validation rejects bad questions | `is_query_noisy()` filtering + retry |
| **Unfair retrieval** | Generic questions don't reflect real task | Keyword-grounded fallback | Questions reuse terms from source |
| **Rank inflation** | Duplicates occupy separate ranks | Dedup keeps best version | Normalized text dedup |
| **Missed context** | Multi-child parents use only 1 child | Multi-center windows | Union ±W fragments around top-M centers |
| **Audit gap** | Failures invisible | Full diagnostic logs | `report["failures"]` with top_retrieved_* |

---

## Next Steps (Optional)

1. **Run full ablation**: `sample-size 200+` to get stable estimates
2. **Compare pre/post**: Re-run old ablation, compare Hit@K distributions
3. **Fine-tune parameters**: adjust `window_centers`, `window`, `rrf_k` per corpus
4. **Integrate dissertation**: Lock this as STEP 6 baseline before moving to STEP 7 (reranking/LLM)

---

## Code References

- [query_quality.py](src/rag/utils/query_quality.py) — noise detection, dedup
- [retrieval_ablation.py](src/rag/eval/retrieval_ablation.py) — improved QA generation (lines 1–450)
- [hybrid_retriever.py](src/rag/retrieve/hybrid_retriever.py) — multi-center windowing, dedup (lines 50–520)
- [vector_retriever.py](src/rag/retrieve/vector_retriever.py) — query_points API (lines 100–130)
