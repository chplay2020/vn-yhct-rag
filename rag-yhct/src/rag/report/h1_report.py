"""H1 Report — Verify pipeline outputs and generate quality report.

Usage:
    python -m rag.report.h1_report \
        --raw data/ingest/raw_passages_full.jsonl \
        --clean data/clean/clean_passages_v2_full.jsonl \
        --chunks data/chunks/chunks_v2_full.jsonl \
        --collection yhct_chunks_v2_full \
        --qdrant-url http://localhost:6333 \
        --out data/reports/h1_full_report.json
"""

from __future__ import annotations

import argparse
import json
import logging
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

from rag.utils.io import read_jsonl

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(name)s | %(message)s")


# ---------------------------------------------------------------------------
# Quality heuristics
# ---------------------------------------------------------------------------

_RE_FFFD = re.compile(r"\ufffd")
_RE_LETTER_NL_LETTER = re.compile(r"[A-Za-z]\n[A-Za-z]")
_RE_SCI_SPLIT = re.compile(r"[a-z]{2,}\n[a-z]{2,}")


def _sample_errors(records: list[dict[str, Any]], max_samples: int = 20) -> dict[str, list[dict[str, str]]]:
    """Find top samples still containing OCR/PDF artifacts."""
    fffd_samples: list[dict[str, str]] = []
    nl_split_samples: list[dict[str, str]] = []
    sci_split_samples: list[dict[str, str]] = []

    for rec in records:
        text = rec.get("text_norm") or rec.get("text", "")
        chunk_id = rec.get("chunk_id", "?")
        source_id = rec.get("source_id", "?")

        if _RE_FFFD.search(text) and len(fffd_samples) < max_samples:
            fffd_samples.append({
                "chunk_id": chunk_id,
                "source_id": source_id,
                "snippet": text[:200],
            })

        if _RE_LETTER_NL_LETTER.search(text) and len(nl_split_samples) < max_samples:
            m = _RE_LETTER_NL_LETTER.search(text)
            start = max(0, m.start() - 20) if m else 0
            end = min(len(text), (m.end() if m else 0) + 20)
            nl_split_samples.append({
                "chunk_id": chunk_id,
                "source_id": source_id,
                "snippet": text[start:end],
            })

        if _RE_SCI_SPLIT.search(text) and len(sci_split_samples) < max_samples:
            m = _RE_SCI_SPLIT.search(text)
            start = max(0, m.start() - 20) if m else 0
            end = min(len(text), (m.end() if m else 0) + 20)
            sci_split_samples.append({
                "chunk_id": chunk_id,
                "source_id": source_id,
                "snippet": text[start:end],
            })

    return {
        "fffd_replacement_char": fffd_samples,
        "letter_newline_letter": nl_split_samples,
        "scientific_name_split": sci_split_samples,
    }


# ---------------------------------------------------------------------------
# Main report generator
# ---------------------------------------------------------------------------

def _counter_dict(records: list[dict[str, Any]], key: str) -> dict[str, int]:
    return dict(Counter(r.get(key, "unknown") for r in records).most_common())


def generate_report(
    raw_path: str,
    clean_path: str,
    chunks_path: str,
    collection: str,
    qdrant_url: str,
) -> dict[str, Any]:
    """Generate H1 quality report."""
    report: dict[str, Any] = {"status": "ok", "errors": []}

    # Load data
    raw = read_jsonl(raw_path)
    clean = read_jsonl(clean_path)
    chunks = read_jsonl(chunks_path)

    # --- Counts ---
    report["raw"] = {
        "count": len(raw),
        "doc_type_counts": _counter_dict(raw, "doc_type"),
        "unique_source_ids": len(set(r.get("source_id", "") for r in raw)),
        "unique_file_paths": len(set(r.get("file_path", "") for r in raw)),
    }
    report["clean"] = {
        "count": len(clean),
        "doc_type_counts": _counter_dict(clean, "doc_type"),
    }
    report["chunks"] = {
        "count": len(chunks),
        "unique_chunk_ids": len(set(r.get("chunk_id", "") for r in chunks)),
        "doc_type_counts": _counter_dict(chunks, "doc_type"),
        "unique_source_ids": len(set(r.get("source_id", "") for r in chunks)),
        "unique_file_paths": len(set(r.get("file_path", "") for r in chunks)),
    }

    # Noise ratio
    noise_count = sum(1 for r in chunks if r.get("is_noise"))
    report["chunks"]["noise_count"] = noise_count
    report["chunks"]["noise_ratio"] = round(noise_count / max(len(chunks), 1), 4)

    # Chunk strategy distribution
    strategy_counts = Counter(r.get("chunk_strategy", "page") for r in chunks if r.get("doc_type") == "pdf")
    report["chunks"]["pdf_strategy_counts"] = dict(strategy_counts)

    # --- Qdrant verify ---
    try:
        from qdrant_client import QdrantClient  # type: ignore
        client = QdrantClient(url=qdrant_url)  # type: ignore
        collection_info = client.get_collection(collection)  # type: ignore
        qdrant_count = client.count(collection, exact=True).count  # type: ignore
        report["qdrant"] = {
            "collection": collection,
            "point_count": qdrant_count,
            "points_count": getattr(collection_info, "points_count", None),  # type: ignore
        }

        # Verify match (against unique chunk_ids, since Qdrant deduplicates)
        unique_chunks = report["chunks"]["unique_chunk_ids"]
        if qdrant_count != unique_chunks:
            report["errors"].append(
                f"MISMATCH: unique_chunks={unique_chunks} != qdrant={qdrant_count}"
            )
            report["status"] = "FAIL"
        else:
            report["qdrant"]["match"] = True
    except Exception as exc:
        report["qdrant"] = {"error": str(exc)}
        report["errors"].append(f"Qdrant verify failed: {exc}")
        report["status"] = "FAIL"

    # --- Quality samples ---
    report["error_samples"] = _sample_errors(chunks)

    # Count totals for each error type
    fffd_total = sum(1 for r in chunks if _RE_FFFD.search(r.get("text_norm") or r.get("text", "")))
    nl_total = sum(1 for r in chunks if _RE_LETTER_NL_LETTER.search(r.get("text_norm") or r.get("text", "")))
    sci_total = sum(1 for r in chunks if _RE_SCI_SPLIT.search(r.get("text_norm") or r.get("text", "")))
    report["error_counts"] = {
        "fffd_chunks": fffd_total,
        "letter_nl_letter_chunks": nl_total,
        "sci_name_split_chunks": sci_total,
    }

    return report


def write_report(report: dict[str, Any], out_path: str) -> None:
    """Write report as JSON."""
    p = Path(out_path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    logger.info("Report written to %s", out_path)


def print_summary(report: dict[str, Any]) -> None:
    """Print human-readable summary to stdout."""
    print("\n" + "=" * 60)
    print("  H1 FULL PIPELINE REPORT")
    print("=" * 60)
    print(f"  Status: {report['status']}")
    print(f"  Raw passages:    {report['raw']['count']}")
    print(f"    doc_types:     {report['raw']['doc_type_counts']}")
    print(f"    unique files:  {report['raw']['unique_file_paths']}")
    print(f"  Clean passages:  {report['clean']['count']}")
    print(f"  Chunks:          {report['chunks']['count']}")
    print(f"    doc_types:     {report['chunks']['doc_type_counts']}")
    print(f"    noise:         {report['chunks']['noise_count']} ({report['chunks']['noise_ratio']:.1%})")
    if report.get("qdrant"):
        q = report["qdrant"]
        if "error" in q:
            print(f"  Qdrant:          ERROR -- {q['error']}")
        else:
            match_str = "MATCH" if q.get("match") else "MISMATCH"
            print(f"  Qdrant points:   {q['point_count']}  {match_str}")
    ecounts = report.get("error_counts", {})
    print(f"  Error chunks:")
    print(f"    U+FFFD:        {ecounts.get('fffd_chunks', 0)}")
    print(f"    letter\\nletter: {ecounts.get('letter_nl_letter_chunks', 0)}")
    print(f"    sci-split:     {ecounts.get('sci_name_split_chunks', 0)}")
    if report.get("errors"):
        print("  ERRORS:")
        for err in report["errors"]:
            print(f"    - {err}")
    print("=" * 60 + "\n")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="H1 pipeline verification report")
    parser.add_argument("--raw", required=True, help="Raw passages JSONL")
    parser.add_argument("--clean", required=True, help="Clean passages JSONL")
    parser.add_argument("--chunks", required=True, help="Chunks JSONL (v2)")
    parser.add_argument("--collection", default="yhct_chunks_v2_full")
    parser.add_argument("--qdrant-url", default="http://localhost:6333")
    parser.add_argument("--out", default="data/reports/h1_full_report.json")
    args = parser.parse_args()

    report = generate_report(
        args.raw, args.clean, args.chunks,
        args.collection, args.qdrant_url,
    )
    write_report(report, args.out)
    print_summary(report)

    if report["status"] != "ok":
        sys.exit(1)


if __name__ == "__main__":
    main()
