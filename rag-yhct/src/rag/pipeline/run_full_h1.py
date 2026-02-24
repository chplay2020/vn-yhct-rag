"""Run full H1 pipeline — one command from scan to Qdrant + report.

Usage:
    cd rag-yhct
    PYTHONPATH=src uv run python -m rag.pipeline.run_full_h1
    PYTHONPATH=src uv run python -m rag.pipeline.run_full_h1 --with-embedding

Flags:
    --with-embedding  After B4 dummy index, run B5 real embedding + upsert

Environment variables:
    SOURCES_YAML — Override sources manifest (default: creates data/sources_full.yaml)
    MAX_WORKERS  — Concurrency for future use (default: 4)
"""

from __future__ import annotations

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any

import yaml  # type: ignore

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)

# ---------------------------------------------------------------------------
# Path constants for FULL run (do not overwrite test outputs)
# ---------------------------------------------------------------------------
CONFIG_PATH = "config/config.yaml"
SOURCES_FULL = "data/sources_full.yaml"
RAW_DIR = "data/raw"
COLLECTION_EMB = "yhct_chunks_v2_full_emb"

RAW_OUT = "data/ingest/raw_passages_full.jsonl"
CLEAN_OUT = "data/clean/clean_passages_v2_full.jsonl"
CHUNKS_V1_OUT = "data/chunks/chunks_v1_full.jsonl"
CHUNKS_V2_OUT = "data/chunks/chunks_v2_full.jsonl"
COLLECTION = "yhct_chunks_v2_full"
REPORT_OUT = "data/reports/h1_full_report.json"


def _load_config() -> dict[str, Any]:
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return yaml.safe_load(f)


def _step_banner(step: str) -> float:
    logger.info("=" * 60)
    logger.info("  %s", step)
    logger.info("=" * 60)
    return time.time()


def _step_done(t0: float, msg: str = "") -> None:
    elapsed = time.time() - t0
    logger.info("  Done in %.1fs %s", elapsed, msg)


def main(with_embedding: bool = False) -> None:
    t_total = time.time()

    # ------------------------------------------------------------------
    # Step 0: Generate sources_full.yaml
    # ------------------------------------------------------------------
    sources_yaml = os.environ.get("SOURCES_YAML", SOURCES_FULL)

    if not Path(sources_yaml).exists() or sources_yaml == SOURCES_FULL:
        t0 = _step_banner("STEP 0: Generate sources_full.yaml")
        from tools.generate_sources_yaml import scan_raw_dir, generate_yaml

        sources = scan_raw_dir(RAW_DIR, base_dir=".")
        if not sources:
            logger.error("No files found in %s — aborting.", RAW_DIR)
            sys.exit(1)
        generate_yaml(sources, SOURCES_FULL)
        sources_yaml = SOURCES_FULL
        os.environ["SOURCES_YAML"] = SOURCES_FULL
        _step_done(t0, f"({len(sources)} files)")
    else:
        logger.info("Using existing sources manifest: %s", sources_yaml)
        os.environ["SOURCES_YAML"] = sources_yaml

    # ------------------------------------------------------------------
    # Step 1: Ingest (B1)
    # ------------------------------------------------------------------
    t0 = _step_banner("STEP 1: Ingest (B1) -> raw_passages_full.jsonl")
    config = _load_config()
    config["ingest"]["output_jsonl"] = RAW_OUT
    # SOURCES_YAML env is read inside run_ingest

    from rag.ingest.ingest_any import run_ingest
    raw_count = run_ingest(config)
    _step_done(t0, f"({raw_count} passages)")

    # ------------------------------------------------------------------
    # Step 2: Clean (B2)
    # ------------------------------------------------------------------
    t0 = _step_banner("STEP 2: Clean (B2) -> clean_passages_v2_full.jsonl")
    config = _load_config()
    config["ingest"]["output_jsonl"] = RAW_OUT  # input for clean
    config["clean"]["output_jsonl"] = CLEAN_OUT

    from rag.clean.clean_normalize import run_clean
    clean_count = run_clean(config)
    _step_done(t0, f"({clean_count} passages)")

    # ------------------------------------------------------------------
    # Step 3: Chunk (B3)
    # ------------------------------------------------------------------
    t0 = _step_banner("STEP 3: Chunk (B3) -> chunks_v1_full.jsonl")
    config = _load_config()
    config["clean"]["output_jsonl"] = CLEAN_OUT  # input for chunk
    config["index"]["input_chunks"] = CHUNKS_V1_OUT

    from rag.chunk.chunk_by_structure import run_chunk
    chunk_v1_count = run_chunk(config)
    _step_done(t0, f"({chunk_v1_count} chunks)")

    # ------------------------------------------------------------------
    # Step 4: Clean v2 (post-chunk normalization)
    # ------------------------------------------------------------------
    t0 = _step_banner("STEP 4: Clean v2 -> chunks_v2_full.jsonl")
    from rag.clean_v2 import process_chunks
    from rag.utils.io import read_jsonl, write_jsonl, ensure_parent_dir

    chunks_v1 = read_jsonl(CHUNKS_V1_OUT)
    chunks_v2 = process_chunks(chunks_v1, debug=False)
    noise = sum(1 for r in chunks_v2 if r.get("is_noise"))
    ensure_parent_dir(CHUNKS_V2_OUT)
    write_jsonl(chunks_v2, CHUNKS_V2_OUT)
    _step_done(t0, f"({len(chunks_v2)} chunks, {noise} noise)")

    # ------------------------------------------------------------------
    # Step 5: Index into Qdrant
    # ------------------------------------------------------------------
    t0 = _step_banner(f"STEP 5: Index Qdrant -> collection '{COLLECTION}'")
    config = _load_config()
    config["index"]["input_chunks"] = CHUNKS_V2_OUT
    config["qdrant"]["collection"] = COLLECTION
    config["qdrant"]["recreate"] = True

    from rag.index.index_qdrant import run_index
    run_index(config)
    _step_done(t0)

    # ------------------------------------------------------------------
    # Step 5b: Real embedding + upsert (B5) — optional
    # ------------------------------------------------------------------
    if with_embedding:
        t0 = _step_banner(f"STEP 5b: Embed real vectors -> collection '{COLLECTION_EMB}'")
        config = _load_config()
        embed_cfg = config.get("embed", {})

        from rag.embed.embed_full import run_embed

        summary = run_embed(
            collection=COLLECTION_EMB,
            chunks_path=CHUNKS_V2_OUT,
            qdrant_url=config["qdrant"]["url"],
            ollama_url=embed_cfg.get("ollama_url", "http://localhost:11434"),
            model=embed_cfg.get("model", "bge-m3"),
            embed_batch=embed_cfg.get("embed_batch", 16),
            upsert_batch=embed_cfg.get("upsert_batch", 64),
            min_len=30,
            skip_noise=True,
            max_retries=3,
            recreate=True,
            vector_size=config["qdrant"].get("vector_size", 1024),
        )
        _step_done(t0, f"({summary['embedded']} points with real vectors)")

    # ------------------------------------------------------------------
    # Step 6: Report & verify
    # ------------------------------------------------------------------
    t0 = _step_banner("STEP 6: Report & Verify")
    from rag.report.h1_report import generate_report, write_report, print_summary

    report = generate_report(
        raw_path=RAW_OUT,
        clean_path=CLEAN_OUT,
        chunks_path=CHUNKS_V2_OUT,
        collection=COLLECTION,
        qdrant_url=config["qdrant"]["url"],
    )
    write_report(report, REPORT_OUT)
    print_summary(report)
    _step_done(t0)

    # ------------------------------------------------------------------
    # Final
    # ------------------------------------------------------------------
    total_elapsed = time.time() - t_total
    logger.info("=" * 60)
    logger.info("  FULL H1 PIPELINE COMPLETE in %.1fs", total_elapsed)
    logger.info("=" * 60)

    if report["status"] != "ok":
        logger.error("Report status: %s — see %s", report["status"], REPORT_OUT)
        sys.exit(1)


if __name__ == "__main__":
    import argparse as _ap
    _parser = _ap.ArgumentParser(description="Full H1 pipeline")
    _parser.add_argument("--with-embedding", "--with_embedding",
                         action="store_true", default=False,
                         help="Run B5 real embedding after B4 dummy index")
    _args = _parser.parse_args()
    main(with_embedding=_args.with_embedding)
