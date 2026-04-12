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
    assert info["query_after_lexicon_restore"] == q
    assert info["query_after_general_restore"] == q
    assert info["query_after_restore"] == q
    assert info["restore_changed"] is False
    assert info["restore_triggered"] is False
    assert info["restore_method"] == "none"
    assert info["general_restore_triggered"] is False
    assert info["general_restore_changed"] is False
    assert info["general_restore_confidence"] == 0.0
    assert info["general_restore_conf_band"] == "none"
    assert info["final_restore_source"] == "normalized"
    assert info["vector_queries_used"] == [q]
    assert info["bm25_queries_used"] == [q]
    assert info["dual_query_enabled"] is False


def test_query_without_accents_restored_for_vector(monkeypatch: MonkeyPatch) -> None:
    q = "tac dung cua cay ngai cuu"

    def _fake_restore(text: str, *, chunks_path: str) -> str:
        assert chunks_path == "dummy.jsonl"
        return "tác dụng của cây ngải cứu"

    monkeypatch.setattr(hr, "restore_query_diacritics_tokenwise_from_corpus", _fake_restore)

    info = hr.prepare_hybrid_queries(q, chunks_path="dummy.jsonl")

    assert info["restore_method"] in {"general_corpus", "domain_lexicon"}
    assert info["restore_changed"] is True
    assert info["restore_triggered"] is True
    assert info["query_after_lexicon_restore"] == "tac dung cua cay ngải cứu"
    assert info["query_after_general_restore"] == "tác dụng của cây ngải cứu"
    assert info["query_final_restored"] == "tác dụng của cây ngải cứu"
    assert info["query_after_restore"] == info["query_final_restored"]
    assert info["general_restore_triggered"] is True
    assert info["general_restore_confidence"] > 0.0
    assert info["final_restore_source"] == "general"
    assert info["dual_query_enabled"] is True
    assert info["vector_queries_used"] == ["tac dung cua cay ngai cuu", "tác dụng của cây ngải cứu"]
    assert info["bm25_queries_used"][0] == "tac dung cua cay ngai cuu"


def test_mixed_query_triggers_and_uses_accentless_lexicon(monkeypatch: MonkeyPatch) -> None:
    q = "tac dung cua cay sử quan tu"

    def _fake_restore(text: str, *, chunks_path: str) -> str:
        # Lexicon already restored phrase; corpus should not degrade it.
        return text

    monkeypatch.setattr(hr, "restore_query_diacritics_tokenwise_from_corpus", _fake_restore)

    info = hr.prepare_hybrid_queries(q, chunks_path="dummy.jsonl")

    assert info["query_after_normalize"] == "tac dung cua cay sử quan tu"
    assert info["query_accentless_for_matching"] == "tac dung cua cay su quan tu"
    assert info["restore_triggered"] is True
    assert info["query_after_lexicon_restore"] == "tac dung cua cay sử quân tử"
    assert info["query_after_general_restore"] == "tac dung cua cay sử quân tử"
    assert any(m["folded"] == "su quan tu" for m in info["domain_lexicon_matches"])
    assert info["query_final_restored"] == "tac dung cua cay sử quân tử"
    assert info["general_restore_triggered"] is True
    assert info["general_restore_changed"] is False
    assert info["final_restore_source"] == "lexicon"
    assert info["vector_queries_used"][0] == "tac dung cua cay sử quan tu"
    assert info["vector_queries_used"][-1] == "tac dung cua cay sử quân tử"


def test_ambiguous_restore_falls_back_on_suspicious_output(monkeypatch: MonkeyPatch) -> None:
    q = "cay la ma"

    def _fake_restore(text: str, *, chunks_path: str) -> str:
        # Suspicious because folded text changes semantically, not just diacritics.
        return "la cay tam"

    monkeypatch.setattr(hr, "restore_query_diacritics_tokenwise_from_corpus", _fake_restore)

    info = hr.prepare_hybrid_queries(q, chunks_path="dummy.jsonl")

    assert info["general_restore_triggered"] is True
    assert info["general_restore_changed"] is True
    assert info["general_restore_confidence"] == 0.0
    assert info["general_restore_conf_band"] == "none"
    assert info["restore_method"] in {"domain_lexicon", "none"}
    assert info["final_restore_source"] in {"lexicon", "normalized"}
    assert info["query_final_restored"] in {
        info["query_after_normalize"],
        info["query_after_lexicon_restore"],
    }
    assert "la cay tam" not in info["vector_queries_used"]


def test_domain_phrase_lexicon_restoration() -> None:
    q = "kinh nguyet khong deu"
    info = hr.prepare_hybrid_queries(q, chunks_path="missing.jsonl")

    assert info["query_after_lexicon_restore"] == "kinh nguyệt không đều"
    assert info["query_final_restored"] == "kinh nguyệt không đều"
    assert info["query_after_restore"] == "kinh nguyệt không đều"
    assert info["restore_method"] == "domain_lexicon"
    assert info["restore_changed"] is True
    assert info["final_restore_source"] == "lexicon"
    assert info["domain_lexicon_matches"]
    assert info["dual_query_enabled"] is True


def test_low_confidence_restore_keeps_original_primary(monkeypatch: MonkeyPatch) -> None:
    q = "thuoc nam re tien"

    def _fake_restore(text: str, *, chunks_path: str) -> str:
        return "thuoc năm re tien"

    monkeypatch.setattr(hr, "restore_query_diacritics_tokenwise_from_corpus", _fake_restore)
    info = hr.prepare_hybrid_queries(q, chunks_path="dummy.jsonl")

    assert info["general_restore_triggered"] is True
    assert info["general_restore_changed"] is True
    assert info["general_restore_confidence"] > 0.0
    assert info["general_restore_conf_band"] == "medium"
    assert info["query_after_general_restore"] == "thuoc năm re tien"
    assert info["query_after_restore"] == info["query_after_normalize"]
    assert info["final_restore_source"] == "normalized"
    assert info["vector_queries_used"][0] == info["query_after_normalize"]
    assert "thuoc năm re tien" in info["vector_queries_used"]


def test_general_restore_low_confidence_falls_back_to_lexicon(monkeypatch: MonkeyPatch) -> None:
    q = "tac dung cua hoang ky"

    def _fake_general(text: str, *, chunks_path: str) -> str:
        # Suspicious rewrite: folded content changes, must be rejected.
        return "tra loi linh tinh"

    monkeypatch.setattr(hr, "restore_query_diacritics_tokenwise_from_corpus", _fake_general)
    info = hr.prepare_hybrid_queries(q, chunks_path="dummy.jsonl")

    assert info["query_after_lexicon_restore"] == "tac dung cua hoàng kỳ"
    assert info["general_restore_triggered"] is True
    assert info["general_restore_confidence"] == 0.0
    assert info["query_final_restored"] == "tac dung cua hoàng kỳ"
    assert info["final_restore_source"] == "lexicon"
    assert "tra loi linh tinh" not in info["vector_queries_used"]


def test_mixed_accent_query_general_stage_still_updates_common_words() -> None:
    q = "tac dụng cua cay ngai cuu la gi?"
    info = hr.prepare_hybrid_queries(q, chunks_path="data/chunks/chunks_v2_full.jsonl")

    assert info["query_after_lexicon_restore"] == "tac dụng cua cay ngải cứu la gi?"
    assert info["general_restore_triggered"] is True
    # General stage should improve at least one remaining non-accented token.
    assert info["general_restore_changed"] is True
    assert info["query_after_general_restore"] != info["query_after_lexicon_restore"]
    assert "ngải cứu" in info["query_after_general_restore"]


def test_debug_json_contains_query_variants(tmp_path: Path) -> None:
    query_debug: dict[str, Any] = {
        "query_original": "tac dung cua cay ngai cuu",
        "query_after_normalize": "tac dung cua cay ngai cuu",
        "query_accentless_for_matching": "tac dung cua cay ngai cuu",
        "query_after_lexicon_restore": "tac dung cua cay ngải cứu",
        "query_after_general_restore": "tác dụng của cây ngải cứu",
        "query_after_corpus_restore": "tác dụng của cây ngải cứu",
        "query_final_restored": "tác dụng của cây ngải cứu",
        "query_after_restore": "tác dụng của cây ngải cứu",
        "accented_token_count": 0,
        "non_accented_token_count": 6,
        "total_candidate_tokens": 6,
        "non_accented_token_ratio": 1.0,
        "restore_triggered": True,
        "restore_changed": True,
        "restore_confidence": 0.72,
        "restore_method": "general_corpus",
        "restore_conf_band": "medium",
        "restore_candidates": [],
        "domain_lexicon_matches": [],
        "lexicon_restore_changed": True,
        "lexicon_restore_confidence": 0.36,
        "general_restore_triggered": True,
        "general_restore_changed": True,
        "general_restore_confidence": 0.72,
        "general_restore_conf_band": "high",
        "final_restore_source": "general",
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
    assert payload["query_debug"]["query_after_general_restore"] == query_debug["query_after_general_restore"]
    assert payload["query_debug"]["query_final_restored"] == query_debug["query_final_restored"]
    assert payload["query_debug"]["general_restore_triggered"] is True
    assert payload["query_debug"]["general_restore_conf_band"] == "high"
    assert payload["query_debug"]["final_restore_source"] == "general"
    assert payload["query_debug"]["dual_query_enabled"] is True
