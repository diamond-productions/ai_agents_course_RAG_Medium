from __future__ import annotations

from medium_rag.logging import eval_run_log_path, rag_trace_log_path


def test_vercel_defaults_logs_to_tmp(monkeypatch) -> None:
    monkeypatch.setenv("VERCEL", "1")
    monkeypatch.delenv("RAG_LOG_DIR", raising=False)
    monkeypatch.delenv("RAG_TRACE_LOG_PATH", raising=False)
    monkeypatch.delenv("EVAL_RUN_LOG_PATH", raising=False)
    assert str(rag_trace_log_path()) == "/tmp/rag_logs/rag_traces.jsonl"
    assert str(eval_run_log_path()) == "/tmp/rag_logs/eval_runs.jsonl"
