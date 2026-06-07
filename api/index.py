from __future__ import annotations

from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from medium_rag.config import load_experiment_config
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


def api_prompt_contract(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "response": result["response"],
        "context": [
            {
                "article_id": item.get("article_id", ""),
                "title": item.get("title", ""),
                "chunk": item.get("chunk", ""),
                "score": item.get("score", 0.0),
            }
            for item in result.get("context", [])
        ],
        "Augmented_prompt": {
            "System": (result.get("Augmented_prompt") or {}).get("System", ""),
            "User": (result.get("Augmented_prompt") or {}).get("User", ""),
        },
    }


def api_stats_contract() -> dict[str, Any]:
    return {
        "chunk_size": CONFIG.chunking.chunk_size,
        "overlap_ratio": CONFIG.chunking.overlap_ratio,
        "top_k": CONFIG.retrieval.top_k,
    }


@app.post("/api/prompt")
def prompt(payload: PromptRequest) -> dict:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="question must not be empty")

    try:
        result = PIPELINE.answer_question(
            question,
            log_trace=payload.log_rag_trace,
            trace_source="api",
            eval_run_id=payload.eval_run_id,
            expected_answer=payload.expected_answer,
            evaluation=payload.evaluation,
        )
        return api_prompt_contract(result)
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.get("/api/stats")
@app.get("/api/stats/")
def stats() -> dict:
    return api_stats_contract()
