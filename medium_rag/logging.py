from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from medium_rag.config import RagExperimentConfig, load_experiment_config

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


def read_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            raw_line = line.strip()
            if not raw_line:
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError:
                record = {
                    "timestamp": "",
                    "question": f"Invalid JSONL record on line {line_number}",
                    "response": raw_line,
                    "context": [],
                    "augmented_prompt": {},
                    "config": {},
                    "parse_error": True,
                }
            record["_line_number"] = line_number
            records.append(record)
    records.reverse()
    return records[:limit] if limit is not None else records


def context_for_log(context: list[dict[str, Any]], preview_chars: int = 240) -> list[dict[str, Any]]:
    return [
        {
            "article_id": item.get("article_id", ""),
            "title": item.get("title", ""),
            "score": item.get("score"),
            "chunk_preview": str(item.get("chunk", ""))[:preview_chars],
        }
        for item in context
    ]


def augmented_prompt_for_log(result: dict[str, Any]) -> dict[str, str]:
    augmented_prompt = result.get("Augmented_prompt") or {}
    return {
        "system": str(augmented_prompt.get("System", "")),
        "user": str(augmented_prompt.get("User", "")),
    }


def config_for_log(config: RagExperimentConfig | None = None) -> dict[str, Any]:
    resolved_config = config or load_experiment_config()
    return resolved_config.config_summary()


def build_rag_trace_record(
    *,
    question: str,
    result: dict[str, Any],
    config: RagExperimentConfig | None = None,
    source: str | None = None,
) -> dict[str, Any]:
    record = {
        "timestamp": utc_timestamp(),
        "question": question,
        "response": result.get("response", ""),
        "context": context_for_log(result.get("context") or []),
        "augmented_prompt": augmented_prompt_for_log(result),
        "config": config_for_log(config),
    }
    if source:
        record["source"] = source
    return record


def log_rag_trace(
    *,
    question: str,
    result: dict[str, Any],
    config: RagExperimentConfig | None = None,
    source: str | None = None,
) -> Path:
    path = rag_trace_log_path()
    append_jsonl(path, build_rag_trace_record(question=question, result=result, config=config, source=source))
    return path


def log_eval_run(
    *,
    question: str,
    result: dict[str, Any],
    config: RagExperimentConfig | None = None,
    run_id: str | None = None,
    expected_answer: str | None = None,
    evaluation: dict[str, Any] | None = None,
    source: str | None = None,
) -> Path:
    record = build_rag_trace_record(question=question, result=result, config=config, source=source)
    if run_id:
        record["run_id"] = run_id
    if expected_answer is not None:
        record["expected_answer"] = expected_answer
    if evaluation is not None:
        record["evaluation"] = evaluation
    path = eval_run_log_path()
    append_jsonl(path, record)
    return path
