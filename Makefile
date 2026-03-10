.PHONY: clean-v2 index-v2 test-clean-v2 pipeline-v2 \
       bm25-build retrieve-vector retrieve-bm25 retrieve-hybrid retrieval-ablation

# --- Clean v2: normalize chunks for embeddings ---
clean-v2:
	cd rag-yhct && PYTHONPATH=src uv run python -m rag.clean_v2 \
		--in data/chunks/chunks_v1.jsonl \
		--out data/chunks/chunks_v2.jsonl

# --- Index v2: upsert chunks_v2 into yhct_chunks_v2 ---
index-v2:
	cd rag-yhct && PYTHONPATH=src uv run python -m rag.index.index_qdrant \
		--config config/config.yaml \
		--input data/chunks/chunks_v2.jsonl \
		--collection yhct_chunks_v2 \
		--recreate

# --- Run tests for clean_v2 ---
test-clean-v2:
	cd rag-yhct && PYTHONPATH=src uv run python -m pytest tests/test_clean_v2.py -v

# --- Full v2 pipeline: clean then index ---
pipeline-v2: clean-v2 index-v2

# --- BM25: build index ---
bm25-build:
	cd rag-yhct && PYTHONPATH=src uv run python -m rag.retrieve.bm25_retriever \
		--build --chunks data/chunks/chunks_v2_full.jsonl

# --- Retrieval: vector-only ---
retrieve-vector:
	cd rag-yhct && PYTHONPATH=src uv run python -m rag.retrieve.hybrid_retriever \
		--query "$(QUERY)" --mode vector --save-debug

# --- Retrieval: bm25-only ---
retrieve-bm25:
	cd rag-yhct && PYTHONPATH=src uv run python -m rag.retrieve.hybrid_retriever \
		--query "$(QUERY)" --mode bm25 --save-debug

# --- Retrieval: hybrid RRF ---
retrieve-hybrid:
	cd rag-yhct && PYTHONPATH=src uv run python -m rag.retrieve.hybrid_retriever \
		--query "$(QUERY)" --mode hybrid_rrf --save-debug

# --- Retrieval ablation eval ---
retrieval-ablation:
	cd rag-yhct && PYTHONPATH=src uv run python -m rag.eval.retrieval_ablation \
		--chunks data/chunks/chunks_v2_full.jsonl \
		--output data/reports/retrieval_ablation.json

# --- Retrieval ablation: fallback-only (no LLM needed) ---
retrieval-ablation-fallback:
	cd rag-yhct && PYTHONPATH=src uv run python -m rag.eval.retrieval_ablation \
		--chunks data/chunks/chunks_v2_full.jsonl \
		--question-mode fallback --sample-size 30 \
		--output data/reports/retrieval_ablation_fallback.json

# --- Retrieval ablation: auto (try LLM, fallback if broken) ---
retrieval-ablation-auto:
	cd rag-yhct && PYTHONPATH=src uv run python -m rag.eval.retrieval_ablation \
		--chunks data/chunks/chunks_v2_full.jsonl \
		--question-mode auto --sample-size 30 \
		--output data/reports/retrieval_ablation_auto.json
