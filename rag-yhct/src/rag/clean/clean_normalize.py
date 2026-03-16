"""B2 — Clean & normalize passages → clean_passages.jsonl

Usage:
    python -m rag.clean.clean_normalize --config config/config.yaml
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

import yaml

from rag.utils.io import read_jsonl, write_jsonl, ensure_parent_dir
from rag.utils.text import normalize_vi_en

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")


def clean_record(rec: dict[str, Any], min_ocr_conf: float = 0.55) -> dict[str, Any]:
    """Clean a single passage record in place."""
    text = rec.get("text", "")
    cleaned = normalize_vi_en(text)
    rec["text"] = cleaned
    rec["text_cleaned"] = True

    # Safety check for low OCR confidence
    ocr_conf = rec.get("ocr_confidence")
    if ocr_conf is not None and ocr_conf < min_ocr_conf:
        rec["low_ocr_confidence"] = True

    return rec


def run_clean(config: dict[str, Any]) -> int:
    """Run B2 clean for all records. Returns record count."""
    input_path = config["ingest"]["output_jsonl"]
    output_path = config["clean"]["output_jsonl"]

    min_ocr_conf = config.get("ingest", {}).get("ocr", {}).get("min_confidence", 0.55)

    records = read_jsonl(input_path)
    if not records:
        logger.warning("No records to clean from %s", input_path)
        return 0

    cleaned: list[dict[str, Any]] = []
    for rec in records:
        try:
            cleaned.append(clean_record(rec, min_ocr_conf))
        except Exception as exc:
            logger.error("Error cleaning record source_id=%s: %s", rec.get("source_id"), exc)
            cleaned.append(rec)  # keep original on error

    ensure_parent_dir(output_path)
    write_jsonl(cleaned, output_path)
    logger.info("B2 Clean complete: %d records → %s", len(cleaned), output_path)
    return len(cleaned)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="B2 — Clean & normalize")
    parser.add_argument("--config", required=True, help="Path to config.yaml")
    parser.add_argument("--input", default=None, help="Override input JSONL path")
    parser.add_argument("--output", default=None, help="Override output JSONL path")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if args.input:
        config["ingest"]["output_jsonl"] = args.input
    if args.output:
        config["clean"]["output_jsonl"] = args.output

    run_clean(config)


if __name__ == "__main__":
    main()
