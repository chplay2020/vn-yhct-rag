## Vector Quality Assessment & Embedding Guide

### Current Status: Dummy Vectors
Vector collection hiện dùng **dummy vectors** (all zeros). Để có real embeddings, làm theo các bước dưới.

---

## 1️⃣ Cách kiểm tra vector quality

### Chạy assessment trên vectors hiện tại:
```bash
cd rag-yhct
export PYTHONPATH=src
uv run python -m rag.eval.vector_quality \
  --collection yhct_chunks_v2_full \
  --chunks data/chunks/chunks_v2_full.jsonl \
  --output data/reports/vector_quality.json
```

### Metrics được kiểm tra:
| Metric | Mục đích | Mức tốt |
|--------|----------|---------|
| **Intra-source similarity** | Chunks từ cùng source có gần nhau? | > 0.7 |
| **Inter-source dissimilarity** | Chunks từ khác source có xa? | < 0.5 |
| **Vector norms** | Độ đều của magnitude | std < 1.0 |
| **Retrieval accuracy** | % relevant chunks trong top-5 | > 60% |

---

## 2️⃣ Thêm real embeddings từ Ollama BGE-M3

### Prerequisites:
```bash
# Cài Ollama (nếu chưa có)
# https://ollama.ai

# Chạy bge-m3 model (nếu chưa):
ollama pull bge-m3:latest
ollama run bge-m3:latest  # Để nó chạy background

# Hoặc dùng API:
ollama serve  # Terminal khác
```

### Embedding chunks:
```bash
# Generate embeddings for chunks_v2_full.jsonl
export PYTHONPATH=src
uv run python -m rag.embed.ollama_embed \
  --collection yhct_chunks_v2_full \
  --chunks data/chunks/chunks_v2_full.jsonl \
  --qdrant-url http://localhost:6333 \
  --ollama-url http://localhost:11434 \
  --batch-size 32

# Speed: ~10-20 chunks/sec (depend trên GPU)
# Thời gian: ~10 phút cho 11k chunks
```

### Sau embedding, chạy lại quality check:
```bash
uv run python -m rag.eval.vector_quality \
  --collection yhct_chunks_v2_full \
  --chunks data/chunks/chunks_v2_full.jsonl \
  --output data/reports/vector_quality_after_embed.json
```

---

## 3️⃣ Duyệt report JSON

```bash
# Xem chi tiết
cat data/reports/vector_quality.json | jq '.assessment'

# Hoặc đọc dễ hơn:
python3 << 'EOF'
import json
with open('data/reports/vector_quality.json') as f:
    r = json.load(f)
    print(f"Status: {r['assessment']['status']}")
    if r['assessment']['issues']:
        print("Issues:")
        for issue in r['assessment']['issues']:
            print(f"  - {issue}")
EOF
```

---

## 4️⃣ Tối ưu embedding quality (nâng cao)

Nếu quality vẫn thấp sau embedding, thử:

### A) Khác chunking strategy
```bash
# PDF chunks có thể dùng strategy B (entry-based) tự động
# Nếu document có markers như "1-", "•", "Tên khoa học:"
# Xem: src/rag/chunk/chunk_by_structure.py lines 280-450
```

### B) Khác model embedding
```bash
# BGE-M3: tốt cho multilingual + Vietnamese
# LLMs khác có thể cài qua Ollama:
ollama pull mistral
ollama pull llama2-chinese

# Sửa trong script: model="mistral" instead of "bge-m3"
```

### C) Preprocessing text tốt hơn
- Kiểm tra `data/reports/h1_full_report.json` → error_samples
- Fix letter\nletter, sci-split bằng update clean_v2.py
- Re-run chunk + embed

### D) Cosine vs. Euclidean distance
```bash
# Sửa config.yaml:
qdrant:
  distance: cosine  # hoặc "euclid"
```

---

## 5️⃣ Diagnose issues

### Vector norm = 0 (dummy vectors)
→ Chạy embedding step (section 2)

### Intra-source sim < 0.5
→ Chunks của cùng document quá khác nhau
   - Check chunk_size (config.yaml: 500 tokens tốt cho hầu hết)
   - Check nếu document quá dài/diverse

### Inter-source sim > 0.7
→ Chunks từ doc khác nhau quá giống
   - Có thể documents liên quan (ví dụ: 2 cuốn cùng tác giả)
   - Hoặc model embedding chưa phân biệt tốt

### Retrieval accuracy < 60%
→ Có thể khớp với inter-source similarity cao
   - Cân nhắc clustering (HNSW) hoặc hybrid search

---

## 6️⃣ Full pipeline với embedding

Cập nhật `run_full_h1.py` để tự động embed (sắp tới):
```bash
PYTHONPATH=src uv run python -m rag.pipeline.run_full_h1 --with-embedding
```

---

## 📊 Expected Results

### Sau embedding với bge-m3:
```
Intra-source sim:  0.75 - 0.85  ✓ Good
Inter-source sim:  0.30 - 0.50  ✓ Good  
Vector norms:      1.0          ✓ Normalized
Retrieval top-5:   60 - 80%     ✓ Good
```

### Nếu kém hơn:
- Check clustering đặc trưng (genre, domain riêng)
- Có thể dùng custom model fine-tuned trên Vietnamese TCM texts

---

## 🔍 Kiểm tra nhanh

```bash
# 1. Vector stats
python3 -c "
from rag.utils.io import read_jsonl
chunks = read_jsonl('data/chunks/chunks_v2_full.jsonl')
print(f'Total chunks: {len(chunks)}')
print(f'Unique sources: {len(set(c[\"source_id\"] for c in chunks))}')
print(f'Noise: {sum(1 for c in chunks if c.get(\"is_noise\"))}')
"

# 2. Qdrant status
python3 -c "
from qdrant_client import QdrantClient
client = QdrantClient('http://localhost:6333')
info = client.get_collection('yhct_chunks_v2_full')
print(f'Points: {info.points_count}')
print(f'Status OK')
"
```

---

**Câu hỏi?** Liên hệ với phần embedding hoặc quality metrics.
