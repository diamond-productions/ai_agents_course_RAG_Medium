# Development Guide

Local development uses `uv` and Python `>=3.14,<3.15`.

## Setup

Create a local environment and install dependencies:

```sh
uv sync --group dev
```

Create a `.env` file from the example and fill in the required keys:

```sh
cp .env.example .env
```

Required for live RAG calls:

```text
LLMOD_API_KEY
PINECONE_API_KEY
```

The example environment uses `configs/production.yaml`. For local sample experiments, set:

```sh
RAG_EXPERIMENT_CONFIG=configs/experiments/dense_mmr.yaml
```

## Development Chat UI

Run the Streamlit development interface:

```sh
uv run rag-web
```

By default the UI calls the local RAG function directly with `configs/production.yaml` unless `RAG_EXPERIMENT_CONFIG` is already set. To test the deployed API, enable `Call FastAPI endpoint` in the sidebar. The default deployed URL is `https://ozd-ai-agents-course-rag-medium.vercel.app`.

To test a local FastAPI server instead, start FastAPI in a second terminal:

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
- Estimated cost using gpt-5-mini defaults.
- JSONL trace paths for RAG queries and eval runs.

Cost defaults are input `$0.25` per 1M tokens, output `$2.00` per 1M tokens, and `text-embedding-3-small` query embedding `$0.02` per 1M tokens. Override with `CHAT_INPUT_COST_PER_1M_TOKENS`, `CHAT_OUTPUT_COST_PER_1M_TOKENS`, and `EMBEDDING_COST_PER_1M_TOKENS`.

## CLI Queries

Run a direct pipeline query:

```sh
uv run python rag.py "List exactly 3 articles about AI" --config configs/experiments/dense_mmr.yaml
```

Disable trace logging for a query:

```sh
uv run python rag.py "List exactly 3 articles about AI" --config configs/experiments/dense_mmr.yaml --no-log
```

Log eval metadata with a query:

```sh
uv run python rag.py "Your question" \
  --config configs/experiments/dense_mmr.yaml \
  --eval-run-id smoke-2026-06-01 \
  --expected-answer "Expected answer text"
```

## Experiment Configs

RAG behavior is driven by YAML experiment configs in `configs/experiments/`.

Included configs:

- `dense_mmr.yaml`: default dense retrieval with article dedupe and MMR.
- `dense_no_mmr.yaml`: dense retrieval with article dedupe and no MMR.
- `baseline_dense.yaml`: dense top-k retrieval without dedupe or MMR.
- `chunk_500_mmr.yaml`: 500-token chunking with MMR in a separate namespace.
- `dense_mmr_chunk512_overlap015_lambda050.yaml`: 512-token chunking with MMR lambda `0.50`.
- `dense_mmr_chunk512_overlap015_lambda075.yaml`: 512-token chunking with MMR lambda `0.75`.
- `dense_mmr_chunk512_overlap015_lambda085.yaml`: 512-token chunking with MMR lambda `0.85`.
- `full_dense_mmr_chunk512_overlap015_lambda075.yaml`: full 50 MB dataset config.
- `configs/production.yaml`: stable production config used by the deployed API.

Use a config explicitly:

```sh
uv run python rag.py "List exactly 3 articles about AI" --config configs/experiments/dense_mmr.yaml --no-log
```

Or set the default for apps/wrappers:

```sh
RAG_EXPERIMENT_CONFIG=configs/experiments/dense_no_mmr.yaml uv run rag-web
```

## Reindexing

Rebuild the configured Pinecone namespace after changing chunking, embedded text, embedding model, dimensions, namespace, or dataset:

```sh
uv run python scripts/prepare_embeddings.py --config configs/experiments/dense_mmr.yaml --force
```

Preview the full dataset chunk volume without embeddings or Pinecone writes:

```sh
uv run python scripts/prepare_embeddings.py \
  --config configs/production.yaml \
  --dry-run
```

Index a small full-config smoke slice:

```sh
uv run python scripts/prepare_embeddings.py \
  --config configs/production.yaml \
  --limit-articles 25 \
  --force
```

Index the full dataset:

```sh
uv run python scripts/prepare_embeddings.py \
  --config configs/production.yaml \
  --force
```

Pure retrieval-strategy experiments do not require reindexing when the namespace and chunking config are unchanged.

## JSONL Logging

Every call to `answer_question()` writes a compact RAG trace to `logs/rag_traces.jsonl` by default. Override the location with `RAG_TRACE_LOG_PATH` or set a shared directory with `RAG_LOG_DIR`.

Each trace line includes:

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

## Benchmark Suite

Generate an AI-curated requirement-focused benchmark suite from `data/medium-300-sample.csv`:

```sh
uv run python scripts/generate_ai_benchmark.py --config configs/experiments/dense_mmr.yaml --case-count 25
```

This uses `OPENROUTER_API_KEY` and the selected `OPENROUTER_MODEL` to write JSONL cases to `eval/medium_300_ai_benchmark.jsonl` for the assignment query types: precise fact retrieval, multi-result topic listing, key idea summary, recommendation with evidence, and unanswerable guardrail cases.

Each case includes the expected source article when applicable, question, expected answer, supporting evidence quote, `question_type`, and `expected_titles` for multi-result listing retrieval diagnostics.

Run the benchmark against the current RAG pipeline:

```sh
uv run python scripts/run_ai_benchmark.py --config configs/experiments/dense_mmr.yaml --output eval/dense_mmr_results.jsonl
```

This also uses `OPENROUTER_API_KEY` and the selected `OPENROUTER_MODEL` for the benchmark judge. Override the endpoint with `OPENROUTER_API_BASE`. It writes detailed results to the selected `--output` path and aggregate metrics to the matching `.summary.json` path.

Metrics include answer accuracy, context faithfulness/grounded rate, mean judge score, Hit@1/3/5/K, MRR@K, Recall@K, expected-title recall for listing cases, and per-question-type metrics.

For `multi_result_topic_listing`, answer correctness is based on returning exactly 3 distinct retrieved titles that are relevant to the requested topic. The AI-curated `expected_titles` are used only to diagnose retrieval recall, not as the only valid answer set.

Compare retrieval variants:

```sh
uv run python scripts/run_ai_benchmark.py --config configs/experiments/dense_no_mmr.yaml --output eval/dense_no_mmr_results.jsonl
uv run python scripts/run_ai_benchmark.py --config configs/experiments/dense_mmr.yaml --output eval/dense_mmr_results.jsonl
```

For a quick smoke test:

```sh
uv run python scripts/generate_ai_benchmark.py --config configs/experiments/dense_mmr.yaml --articles 10 --case-count 3
uv run python scripts/run_ai_benchmark.py --config configs/experiments/dense_mmr.yaml --limit 2
```

## Tests

Run non-live checks:

```sh
uv run python -m py_compile medium_rag/*.py medium_rag/vectorstores/*.py medium_rag/retrieval/*.py medium_rag/evaluation/*.py rag.py rag_utils.py rag_logging.py scripts/prepare_embeddings.py scripts/generate_ai_benchmark.py scripts/run_ai_benchmark.py api/index.py
uv run pytest
```
