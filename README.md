# Medium RAG Assistant

## Development Chat UI

Run the Streamlit development interface:

```sh
uv run rag-web
```

By default the UI calls the local RAG function directly. To test the deployed API shape locally, start FastAPI in a second terminal:

```sh
uv run rag-api
```

Then enable `Call FastAPI endpoint` in the Streamlit sidebar.

To start both the API and Streamlit app from one terminal:

```sh
uv run rag-all
```

Run the standalone Medium dataset explorer:

```sh
uv run rag-explorer
```

You can also use the grouped launcher:

```sh
uv run rag-dev api
uv run rag-dev web
uv run rag-dev all
```

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

## Reindexing

The default Pinecone namespace is `medium-300-chunk800-overlap120-v2`. Rebuild that namespace after changing chunk text, chunk IDs, or metadata:

```sh
uv run python scripts/prepare_embeddings.py
```

Records use deterministic IDs in the form `medium-300:{article_id}:{chunk_index}`. The embedded chunk text includes title, authors, tags, and passage text. Retrieval queries more candidate chunks than the final prompt size, applies Maximal Marginal Relevance, and deduplicates results by article before sending context to the model.

Tune retrieval with:

```sh
RETRIEVAL_CANDIDATE_K=20
MMR_ENABLED=true
MMR_LAMBDA=0.65
```

Higher `MMR_LAMBDA` favors relevance; lower values favor diversity.

## AI Benchmark

Generate grounded benchmark questions from `data/medium-300-sample.csv`:

```sh
uv run python scripts/generate_ai_benchmark.py --articles 30 --cases-per-article 2
```

This writes JSONL cases to `eval/medium_300_ai_benchmark.jsonl`. Each case includes the expected source article, question, expected answer, and supporting evidence quote.

Run the benchmark against the current RAG pipeline:

```sh
uv run python scripts/run_ai_benchmark.py --top-k 7
```

This writes detailed results to `eval/medium_300_ai_benchmark_results.jsonl` and aggregate metrics to `eval/medium_300_ai_benchmark_results.summary.json`, including answer accuracy, grounded rate, mean judge score, and retrieval hit rate.

For a quick smoke test:

```sh
uv run python scripts/generate_ai_benchmark.py --articles 2 --cases-per-article 1
uv run python scripts/run_ai_benchmark.py --limit 2
```
