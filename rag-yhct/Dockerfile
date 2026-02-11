FROM python:3.11-slim

# System deps for PDF/OCR
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    tesseract-ocr \
    tesseract-ocr-vie \
    libgl1-mesa-glx \
    libglib2.0-0 \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN pip install --no-cache-dir uv

WORKDIR /app

# Copy project files
COPY pyproject.toml .python-version ./
COPY src/ src/
COPY config/ config/
COPY data/ data/
COPY scripts/ scripts/

# Sync dependencies
RUN uv sync --no-dev

ENV PYTHONPATH=/app/src

CMD ["uv", "run", "python", "-m", "rag.pipeline.run_step0_4", "--config", "config/config.yaml"]
