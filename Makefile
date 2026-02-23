.PHONY: clean-v2 index-v2 test-clean-v2 pipeline-v2

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
