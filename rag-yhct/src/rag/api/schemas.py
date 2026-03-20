from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    query: str = Field(min_length=1)
    mode: str = Field(default="hybrid_rrf")
    use_gate: bool = True
    build_context: bool = True
    generate_answer: bool = True


class AskResponse(BaseModel):
    query: str
    mode: str
    answer: str
    key_concepts: list[str]
    limits: str
    safety_note: str
    abstained: bool
    gate_result: dict[str, Any] | None
    evidence: list[dict[str, Any]]
    retrieval_results: list[dict[str, Any]]
    context_debug: dict[str, Any]
