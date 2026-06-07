from __future__ import annotations

import os

from langchain_openai import OpenAIEmbeddings

from medium_rag.config import EmbeddingConfig


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def build_embeddings(config: EmbeddingConfig) -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=config.model,
        dimensions=config.dimensions,
        api_key=_require_env("LLMOD_API_KEY"),
        base_url=os.getenv("LLMOD_API_BASE") or config.api_base,
    )
