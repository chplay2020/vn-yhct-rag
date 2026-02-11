# RAG-YHCT — RAG Pipeline cho Y Học Cổ Truyền

Pipeline tự động: Ingest → Clean → Chunk → Index vào Qdrant.

## Yêu cầu

- Python 3.11
- [uv](https://docs.astral.sh/uv/) (package manager)
- Docker & Docker Compose (cho Qdrant + chạy 1 lệnh)
- Tesseract OCR + language pack `vie` (nếu chạy local)

## Cài đặt local (với uv)

```bash
# 1. Cài dependencies
uv sync

# 2. Lock & export
uv lock
uv export --format requirements-txt -o requirements.lock

# 3. Cài tesseract (Ubuntu/WSL)
sudo apt-get install -y tesseract-ocr tesseract-ocr-vie poppler-utils
```

## Chạy từng bước (local)

Đặt tài liệu vào `data/raw/`, cập nhật `data/sources.yaml`, rồi:

```bash
export PYTHONPATH=src

# B1 — Ingest
uv run python -m rag.ingest.ingest_any --config config/config.yaml

# B2 — Clean
uv run python -m rag.clean.clean_normalize --config config/config.yaml

# B3 — Chunk
uv run python -m rag.chunk.chunk_by_structure --config config/config.yaml

# B4 — Index (cần Qdrant đang chạy ở localhost:6333)
uv run python -m rag.index.index_qdrant --config config/config.yaml
```

## Chạy bằng Docker (1 lệnh)

```bash
bash scripts/run.sh
```

Lệnh trên chạy `docker compose up --build`, sẽ:

1. Khởi Qdrant container
2. Build app image (cài Python deps + Tesseract)
3. Chạy pipeline B1 → B4 tuần tự

## Cấu trúc thư mục

```
rag-yhct/
├── config/config.yaml          # Cấu hình pipeline
├── data/
│   ├── raw/                    # Đặt file gốc (PDF/DOCX/IMAGE) ở đây
│   ├── ingest/                 # Output B1
│   ├── clean/                  # Output B2
│   ├── chunks/                 # Output B3
│   └── sources.yaml            # Manifest nguồn tài liệu
├── src/rag/
│   ├── ingest/ingest_any.py    # B1 — Ingest
│   ├── clean/clean_normalize.py# B2 — Clean
│   ├── chunk/chunk_by_structure.py # B3 — Chunk
│   ├── index/index_qdrant.py   # B4 — Index
│   ├── pipeline/run_step0_4.py # Pipeline runner
│   └── utils/                  # IO, text, lang, hashing utilities
├── scripts/run.sh
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
└── .python-version
```

## Cấu hình

Chỉnh sửa `config/config.yaml` để thay đổi:

- `ingest.strategy_pdf`: strategy cho Unstructured (`hi_res`, `fast`, …)
- `chunking.chunk_size` / `overlap`: kích thước chunk (tokens)
- `qdrant.url` / `collection`: endpoint Qdrant
- `qdrant.vector_size`: kích thước vector (dummy = all zeros ở B4)
- `qdrant.recreate`: `true` để xoá + tạo lại collection

## Lưu ý

- B4 dùng **dummy vectors** (all zeros). Embedding thật sẽ được thêm ở B5.
- Pipeline không crash nếu 1 file lỗi — log exception rồi tiếp tục.
- Hỗ trợ cả tiếng Việt và tiếng Anh (paper khoa học) qua `doc_language` field.
