"""Pipeline runner — executes B1 → B4 sequentially.

Usage:
    python -m rag.pipeline.run_step0_4 --config config/config.yaml
"""

from __future__ import annotations

import argparse
import logging
import time
from typing import Any

import yaml

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)


def _run_step(name: str, func: Any, config: dict[str, Any]) -> None:
    """Run a single pipeline step with timing and error handling."""
    logger.info("=" * 60)
    logger.info("START  %s", name)
    logger.info("=" * 60)
    t0 = time.time()
    try:
        func(config)
    except Exception as exc:
        logger.error("FAILED %s: %s", name, exc, exc_info=True)
        raise
    elapsed = time.time() - t0
    logger.info("DONE   %s  (%.1f s)", name, elapsed)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run pipeline B1 → B4")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    logger.info("Pipeline starting with config: %s", args.config)
    t_total = time.time()

    # B1 — Ingest
    from rag.ingest.ingest_any import run_ingest
    _run_step("B1-Ingest", run_ingest, config)

    # B2 — Clean
    from rag.clean.clean_normalize import run_clean
    _run_step("B2-Clean", run_clean, config)

    # B3 — Chunk
    from rag.chunk.chunk_by_structure import run_chunk
    _run_step("B3-Chunk", run_chunk, config)

    # B4 — Index (Qdrant)
    from rag.index.index_qdrant import run_index
    _run_step("B4-Index", run_index, config)

    elapsed_total = time.time() - t_total
    logger.info("=" * 60)
    logger.info("PIPELINE COMPLETE  (total %.1f s)", elapsed_total)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
