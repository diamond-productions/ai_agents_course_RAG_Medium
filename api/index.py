from fastapi import FastAPI, HTTPException
from typing import Any

from pydantic import BaseModel, Field

from api.config import CHAT_MODEL, CHUNK_SIZE, EMBEDDING_MODEL, OVERLAP_RATIO, TOP_K
from rag_logging import eval_run_log_path, rag_trace_log_path
from rag_utils import answer_question

app = FastAPI(redirect_slashes=True)


# ---------------------------------------------------------------------------
# POST /api/prompt
# ---------------------------------------------------------------------------
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
        return answer_question(
            question,
            log_trace=payload.log_rag_trace,
            trace_source="api",
            eval_run_id=payload.eval_run_id,
            expected_answer=payload.expected_answer,
            evaluation=payload.evaluation,
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ---------------------------------------------------------------------------
# GET /api/stats
# ---------------------------------------------------------------------------
@app.get("/api/stats/")
def stats() -> dict:
    return {
        "chunk_size": CHUNK_SIZE,
        "overlap_ratio": OVERLAP_RATIO,
        "top_k": TOP_K,
        "chat_model": CHAT_MODEL,
        "embedding_model": EMBEDDING_MODEL,
        "rag_trace_log_path": str(rag_trace_log_path()),
        "eval_run_log_path": str(eval_run_log_path()),
    }
