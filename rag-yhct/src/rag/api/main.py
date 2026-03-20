from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from rag.api.schemas import AskRequest, AskResponse
from rag.api.service import run_rag_pipeline


app = FastAPI(title="RAG YHCT API", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/ask", response_model=AskResponse)
def ask(payload: AskRequest) -> AskResponse:
    data = run_rag_pipeline(
        query=payload.query,
        mode=payload.mode,
        use_gate=payload.use_gate,
        build_context=payload.build_context,
        generate_answer=payload.generate_answer,
    )
    return AskResponse(**data)
