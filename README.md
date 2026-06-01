# Medium RAG Assistant

## Development Chat UI

Run the Streamlit development interface:

```sh
uv run streamlit run streamlit_app.py
```

By default the UI calls the local RAG function directly. To test the deployed API shape locally, start FastAPI in a second terminal:

```sh
uv run uvicorn api.index:app --reload
```

Then enable `Call FastAPI endpoint` in the Streamlit sidebar.

The UI shows:

- The assistant answer.
- Referenced articles and retrieved chunks.
- The exact augmented system/user prompt sent to the model.
- Model and retrieval metadata.
- Token usage returned by the model provider.
- Estimated cost using gpt-5-mini defaults: input `$0.25`/1M tokens, output `$2.00`/1M tokens, and text-embedding-3-small query embedding `$0.02`/1M tokens. Override with `CHAT_INPUT_COST_PER_1M_TOKENS`, `CHAT_OUTPUT_COST_PER_1M_TOKENS`, and `EMBEDDING_COST_PER_1M_TOKENS`.
- JSONL trace paths for RAG queries and eval runs.

## JSONL Logging

Every call to `answer_question()` writes a compact RAG trace to `logs/rag_traces.jsonl` by default. Override the location with `RAG_TRACE_LOG_PATH` or set a shared directory with `RAG_LOG_DIR`.

Each line includes:

```json
{
  "timestamp": "2026-06-01T12:00:00Z",
  "question": "List exactly 3 articles about education.",
  "response": "...",
  "context": [
    {
      "article_id": "1234",
      "title": "Example title",
      "score": 0.82,
      "chunk_preview": "..."
    }
  ],
  "augmented_prompt": {
    "system": "...",
    "user": "..."
  },
  "config": {
    "chunk_size": 800,
    "overlap_ratio": 0.15,
    "top_k": 7
  }
}
```

Eval runs are appended to `logs/eval_runs.jsonl` when an API, Streamlit, or CLI caller provides `eval_run_id`, `expected_answer`, or `evaluation`. Override that path with `EVAL_RUN_LOG_PATH`.

FastAPI accepts optional eval fields on `POST /api/prompt`:

```json
{
  "question": "Your question",
  "eval_run_id": "smoke-2026-06-01",
  "expected_answer": "Expected answer text"
}
```

The CLI supports the same logging:

```sh
uv run python rag.py "Your question" --eval-run-id smoke-2026-06-01 --expected-answer "Expected answer text"
```
