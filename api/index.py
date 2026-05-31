from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from api.config import CHAT_MODEL, CHUNK_SIZE, EMBEDDING_MODEL, OVERLAP_RATIO, TOP_K
from rag import answer_question

app = FastAPI(redirect_slashes=True)


# ---------------------------------------------------------------------------
# POST /api/prompt
# ---------------------------------------------------------------------------
class PromptRequest(BaseModel):
    question: str = Field(..., min_length=1)


@app.post("/api/prompt")
def prompt(payload: PromptRequest) -> dict:
    question = payload.question.strip()
    if not question:
        raise HTTPException(status_code=422, detail="question must not be empty")

    try:
        return answer_question(question)
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
    }
