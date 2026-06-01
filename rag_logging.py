from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from api.config import CHUNK_SIZE, OVERLAP_RATIO, TOP_K

DEFAULT_LOG_DIR = Path(os.getenv("RAG_LOG_DIR", "logs"))
DEFAULT_RAG_TRACE_LOG_PATH = DEFAULT_LOG_DIR / "rag_traces.jsonl"
DEFAULT_EVAL_RUN_LOG_PATH = DEFAULT_LOG_DIR / "eval_runs.jsonl"


def utc_timestamp() -> str:
    return datetime.now(UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def rag_trace_log_path() -> Path:
    return Path(os.getenv("RAG_TRACE_LOG_PATH", str(DEFAULT_RAG_TRACE_LOG_PATH)))


def eval_run_log_path() -> Path:
    return Path(os.getenv("EVAL_RUN_LOG_PATH", str(DEFAULT_EVAL_RUN_LOG_PATH)))


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False, default=str))
        handle.write("\n")


def context_for_log(context: list[dict[str, Any]], preview_chars: int = 240) -> list[dict[str, Any]]:
    logged_context = []
    for item in context:
        chunk = str(item.get("chunk", ""))
        logged_context.append(
            {
                "article_id": item.get("article_id", ""),
                "title": item.get("title", ""),
                "score": item.get("score"),
                "chunk_preview": chunk[:preview_chars],
            }
        )
    return logged_context


def augmented_prompt_for_log(result: dict[str, Any]) -> dict[str, str]:
    augmented_prompt = result.get("Augmented_prompt") or {}
    return {
        "system": str(augmented_prompt.get("System", "")),
        "user": str(augmented_prompt.get("User", "")),
    }


def config_for_log(top_k: int = TOP_K) -> dict[str, Any]:
    return {
        "chunk_size": CHUNK_SIZE,
        "overlap_ratio": OVERLAP_RATIO,
        "top_k": top_k,
    }


def build_rag_trace_record(
    *,
    question: str,
    result: dict[str, Any],
    top_k: int = TOP_K,
    source: str | None = None,
) -> dict[str, Any]:
    record = {
        "timestamp": utc_timestamp(),
        "question": question,
        "response": result.get("response", ""),
        "context": context_for_log(result.get("context") or []),
        "augmented_prompt": augmented_prompt_for_log(result),
        "config": config_for_log(top_k=top_k),
    }
    if source:
        record["source"] = source
    return record


def log_rag_trace(
    *,
    question: str,
    result: dict[str, Any],
    top_k: int = TOP_K,
    source: str | None = None,
) -> Path:
    path = rag_trace_log_path()
    append_jsonl(path, build_rag_trace_record(question=question, result=result, top_k=top_k, source=source))
    return path


def log_eval_run(
    *,
    question: str,
    result: dict[str, Any],
    top_k: int = TOP_K,
    run_id: str | None = None,
    expected_answer: str | None = None,
    evaluation: dict[str, Any] | None = None,
    source: str | None = None,
) -> Path:
    record = build_rag_trace_record(question=question, result=result, top_k=top_k, source=source)
    if run_id:
        record["run_id"] = run_id
    if expected_answer is not None:
        record["expected_answer"] = expected_answer
    if evaluation is not None:
        record["evaluation"] = evaluation
    path = eval_run_log_path()
    append_jsonl(path, record)
    return path
