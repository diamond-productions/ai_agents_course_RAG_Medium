# Deployment Guide

The production API is deployed with Vercel and served by `api/index.py`.

## Preconditions

The deployed API should query an already-indexed Pinecone namespace. Do not index during Vercel build or function startup, and do not bundle `data/medium-english-50mb.csv`.

Index the production namespace before deploying:

```sh
uv run python scripts/prepare_embeddings.py \
  --config configs/production.yaml \
  --force
```

For a smaller smoke indexing run:

```sh
uv run python scripts/prepare_embeddings.py \
  --config configs/production.yaml \
  --limit-articles 25 \
  --force
```

## Vercel Environment

Set these environment variables in the linked Vercel project:

```text
LLMOD_API_KEY
LLMOD_API_BASE=https://api.llmod.ai/v1
PINECONE_API_KEY
RAG_EXPERIMENT_CONFIG=configs/production.yaml
PINECONE_INDEX_NAME=medium-articles-full
PINECONE_NAMESPACE=medium-english-50mb-chunk512-overlap077-v1
RAG_LOG_DIR=/tmp/rag_logs
```

Optional overrides:

```text
RAG_DATASET_NAME
RAG_DATASET_PATH
RAG_VECTOR_ID_PREFIX
PINECONE_BATCH_SIZE
PINECONE_MAX_BATCH_BYTES
EMBEDDING_BATCH_SIZE
RAG_TRACE_LOG_PATH
EVAL_RUN_LOG_PATH
CHAT_INPUT_COST_PER_1M_TOKENS
CHAT_OUTPUT_COST_PER_1M_TOKENS
EMBEDDING_COST_PER_1M_TOKENS
```

## Deploy

Deploy through the linked Vercel project.

After deployment, smoke test the preview or production URL:

```sh
curl -s https://<preview-url>/api/stats/
```

```sh
curl -s https://<preview-url>/api/prompt \
  -H 'content-type: application/json' \
  -d '{"question":"List exactly 3 articles about education. Return only the titles."}'
```

## Production Defaults

`configs/production.yaml` uses:

| Parameter | Value |
| --- | --- |
| Experiment | `full_dense_mmr_chunk512_overlap015_lambda075` |
| Dataset | `medium-english-50mb` |
| Chunk size | `512` |
| Overlap ratio | `0.15` |
| Embedding model | `4UHRUIN-text-embedding-3-small` |
| Embedding dimensions | `1536` |
| Pinecone index | `medium-articles-full` |
| Pinecone namespace | `medium-english-50mb-chunk512-overlap077-v1` |
| Retrieval strategy | `dense_mmr` |
| `top_k` | `7` |
| `candidate_k` | `30` |
| MMR lambda | `0.75` |
| Chat model | `4UHRUIN-gpt-5-mini` |
