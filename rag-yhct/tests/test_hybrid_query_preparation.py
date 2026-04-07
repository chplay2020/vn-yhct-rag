"""Tests for safe accent restoration in hybrid retriever query handling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import rag.retrieve.hybrid_retriever as hr
from pytest import MonkeyPatch


def test_query_with_accents_kept_unchanged() -> None:
    q = "tác dụng của cây ngải cứu"
    info = hr.prepare_hybrid_queries(q, chunks_path="missing.jsonl")

    assert info["query_after_normalize"] == q
    assert info["query_accentless_for_matching"] == "tac dung cua cay ngai cuu"
    assert info["query_after_restore"] == q
    assert info["restore_changed"] is False
    assert info["restore_triggered"] is False
    assert info["restore_method"] == "none"
    assert info["vector_queries_used"] == [q]
    assert info["bm25_queries_used"] == [q]
    assert info["dual_query_enabled"] is False


def test_query_without_accents_restored_for_vector(monkeypatch: MonkeyPatch) -> None:
    q = "tac dung cua cay ngai cuu"

    def _fake_restore(text: str, *, chunks_path: str) -> str:
        assert chunks_path == "dummy.jsonl"
        return "tác dụng của cây ngải cứu"

    monkeypatch.setattr(hr, "restore_query_diacritics_from_corpus", _fake_restore)

    info = hr.prepare_hybrid_queries(q, chunks_path="dummy.jsonl")

    assert info["restore_method"] in {"corpus", "domain_lexicon"}
    assert info["restore_changed"] is True
    assert info["restore_triggered"] is True
    assert info["query_final_restored"] == "tác dụng của cây ngải cứu"
    assert info["query_after_restore"] == info["query_final_restored"]
    assert info["dual_query_enabled"] is True
    assert info["vector_queries_used"] == ["tac dung cua cay ngai cuu", "tác dụng của cây ngải cứu"]
    assert info["bm25_queries_used"][0] == "tac dung cua cay ngai cuu"


def test_mixed_query_triggers_and_uses_accentless_lexicon(monkeypatch: MonkeyPatch) -> None:
    q = "tac dung cua cay sử quan tu"

    def _fake_restore(text: str, *, chunks_path: str) -> str:
        # Lexicon already restored phrase; corpus should not degrade it.
        return text

    monkeypatch.setattr(hr, "restore_query_diacritics_from_corpus", _fake_restore)

    info = hr.prepare_hybrid_queries(q, chunks_path="dummy.jsonl")

    assert info["query_after_normalize"] == "tac dung cua cay sử quan tu"
    assert info["query_accentless_for_matching"] == "tac dung cua cay su quan tu"
    assert info["restore_triggered"] is True
    assert info["query_after_lexicon_restore"] == "tac dung cua cay sử quân tử"
    assert any(m["folded"] == "su quan tu" for m in info["domain_lexicon_matches"])
    assert info["query_final_restored"] == "tac dung cua cay sử quân tử"
    assert info["vector_queries_used"][0] == "tac dung cua cay sử quan tu"
    assert info["vector_queries_used"][-1] == "tac dung cua cay sử quân tử"


def test_ambiguous_restore_falls_back_on_suspicious_output(monkeypatch: MonkeyPatch) -> None:
    q = "cay la ma"

    def _fake_restore(text: str, *, chunks_path: str) -> str:
        # Suspicious because folded text changes semantically, not just diacritics.
        return "la cay tam"

    monkeypatch.setattr(hr, "restore_query_diacritics_from_corpus", _fake_restore)

    info = hr.prepare_hybrid_queries(q, chunks_path="dummy.jsonl")

    assert info["restore_method"] == "none"
    assert info["restore_changed"] is False
    assert info["vector_queries_used"] == [info["query_after_normalize"]]
    assert info["dual_query_enabled"] is False


def test_domain_phrase_lexicon_restoration() -> None:
    q = "kinh nguyet khong deu"
    info = hr.prepare_hybrid_queries(q, chunks_path="missing.jsonl")

    assert info["query_after_lexicon_restore"] == "kinh nguyệt không đều"
    assert info["query_final_restored"] == "kinh nguyệt không đều"
    assert info["query_after_restore"] == "kinh nguyệt không đều"
    assert info["restore_method"] == "domain_lexicon"
    assert info["restore_changed"] is True
    assert info["domain_lexicon_matches"]
    assert info["dual_query_enabled"] is True


def test_low_confidence_restore_keeps_original_primary(monkeypatch: MonkeyPatch) -> None:
    q = "thuoc nam re tien"

    def _fake_restore(text: str, *, chunks_path: str) -> str:
        return "thuoc năm re tien"

    monkeypatch.setattr(hr, "restore_query_diacritics_from_corpus", _fake_restore)
    info = hr.prepare_hybrid_queries(q, chunks_path="dummy.jsonl")

    assert info["restore_method"] == "none"
    assert info["restore_confidence"] == 0.0
    assert info["restore_conf_band"] == "low"
    assert info["query_after_restore"] == info["query_after_normalize"]
    assert info["vector_queries_used"] == [info["query_after_normalize"]]


def test_debug_json_contains_query_variants(tmp_path: Path) -> None:
    query_debug: dict[str, Any] = {
        "query_original": "tac dung cua cay ngai cuu",
        "query_after_normalize": "tac dung cua cay ngai cuu",
        "query_accentless_for_matching": "tac dung cua cay ngai cuu",
        "query_after_lexicon_restore": "tác dụng của cây ngải cứu",
        "query_after_corpus_restore": "tác dụng của cây ngải cứu",
        "query_final_restored": "tác dụng của cây ngải cứu",
        "query_after_restore": "tác dụng của cây ngải cứu",
        "accented_token_count": 0,
        "non_accented_token_count": 6,
        "total_candidate_tokens": 6,
        "non_accented_token_ratio": 1.0,
        "restore_triggered": True,
        "restore_changed": True,
        "restore_confidence": 0.5,
        "restore_method": "corpus",
        "restore_conf_band": "medium",
        "restore_candidates": [],
        "domain_lexicon_matches": [],
        "lexicon_restore_changed": False,
        "lexicon_restore_confidence": 0.0,
        "vector_queries_used": ["tac dung cua cay ngai cuu", "tác dụng của cây ngải cứu"],
        "bm25_queries_used": ["tac dung cua cay ngai cuu"],
        "dual_query_enabled": True,
        "vector_query_used": "tác dụng của cây ngải cứu",
        "bm25_query_used": "tac dung cua cay ngai cuu",
    }

    hr._save_debug_json(  # pyright: ignore[reportPrivateUsage]
        "tac dung cua cay ngai cuu",
        "hybrid_rrf",
        5,
        [],
        str(tmp_path),
        bm25_raw=[],
        vector_raw=[],
        query_debug=query_debug,
    )

    created: list[Path] = list(tmp_path.glob("hybrid_rrf_*.json"))
    assert created

    payload: dict[str, Any] = json.loads(created[0].read_text(encoding="utf-8"))
    assert "query_debug" in payload
    assert payload["query_debug"]["query_original"] == query_debug["query_original"]
    assert payload["query_debug"]["query_after_restore"] == query_debug["query_after_restore"]
    assert payload["query_debug"]["query_accentless_for_matching"] == query_debug["query_accentless_for_matching"]
    assert payload["query_debug"]["query_final_restored"] == query_debug["query_final_restored"]
    assert payload["query_debug"]["dual_query_enabled"] is True
