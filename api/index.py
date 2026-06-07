from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from medium_rag.config import load_experiment_config
from medium_rag.logging import eval_run_log_path, rag_trace_log_path
from medium_rag.pipeline import RagPipeline

app = FastAPI(redirect_slashes=True)
CONFIG = load_experiment_config()
PIPELINE = RagPipeline(CONFIG)


class PromptRequest(BaseModel):
    question: str = Field(..., min_length=1)
    log_rag_trace: bool = True
    eval_run_id: str | None = None
    expected_answer: str | None = None
    evaluation: dict[str, Any] | None = None


@app.post("/api/prompt")
def prompt(payload: PromptRequest) -> dict:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="question must not be empty")

    try:
        return PIPELINE.answer_question(
            question,
            log_trace=payload.log_rag_trace,
            trace_source="api",
            eval_run_id=payload.eval_run_id,
            expected_answer=payload.expected_answer,
            evaluation=payload.evaluation,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/stats/")
def stats() -> dict:
    return {
        **CONFIG.config_summary(),
        "chat_model": CONFIG.generation.chat_model,
        "embedding_model": CONFIG.embedding.model,
        "rag_trace_log_path": str(rag_trace_log_path()),
        "eval_run_log_path": str(eval_run_log_path()),
    }
