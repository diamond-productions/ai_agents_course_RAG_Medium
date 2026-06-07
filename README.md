# Medium RAG Assistant

Minimal RAG API for answering questions over indexed Medium articles.

## Setup

```sh
uv sync --group dev
cp .env.example .env
```

Fill in:

```text
LLMOD_API_KEY=
PINECONE_API_KEY=
```

The default environment uses `configs/production.yaml`.

## Run Locally

```sh
uv run rag-api
```

## Deploy

Deploy through the linked Vercel project. The API expects an already-indexed Pinecone namespace.

Required Vercel env vars:

```text
LLMOD_API_KEY
LLMOD_API_BASE=https://api.llmod.ai/v1
PINECONE_API_KEY
RAG_EXPERIMENT_CONFIG=configs/production.yaml
PINECONE_INDEX_NAME=medium-articles-full
PINECONE_NAMESPACE=medium-english-50mb-chunk512-overlap077-v1
RAG_LOG_DIR=/tmp/rag_logs
```

## Call The API

```sh
curl -s http://127.0.0.1:8000/api/stats/
```

```sh
curl -s http://127.0.0.1:8000/api/prompt \
  -H 'content-type: application/json' \
  -d '{"question":"List exactly 3 articles about education. Return only the titles."}'
```

Response shape:

```json
{
  "response": "...",
  "context": [
    {
      "article_id": "1234",
      "title": "Example title",
      "chunk": "Retrieved passage text...",
      "score": 0.8123
    }
  ],
  "Augmented_prompt": {
    "System": "...",
    "User": "..."
  }
}
```

More details:

- [DEVELOPMENT.md](DEVELOPMENT.md)
- [DEPLOYMENT.md](DEPLOYMENT.md)
- [BENCHMARK_METHODOLOGY.md](BENCHMARK_METHODOLOGY.md)
