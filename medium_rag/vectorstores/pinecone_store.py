from __future__ import annotations

import os
import time
from typing import Any

from pinecone import Pinecone, ServerlessSpec

from medium_rag.config import PineconeConfig
from medium_rag.types import Chunk, VectorMatch


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


class PineconeVectorStore:
    def __init__(self, config: PineconeConfig, embedding_dimensions: int):
        self.config = config
        self.embedding_dimensions = embedding_dimensions
        self.pc = Pinecone(api_key=_require_env("PINECONE_API_KEY"))
        self._index = None

    @property
    def index(self):
        if self._index is None:
            self.ensure_index()
            self._index = self.pc.Index(self.config.index_name)
        return self._index

    def ensure_index(self) -> None:
        if self.pc.has_index(self.config.index_name):
            return
        self.pc.create_index(
            name=self.config.index_name,
            vector_type="dense",
            dimension=self.embedding_dimensions,
            metric=self.config.metric,
            spec=ServerlessSpec(cloud=self.config.cloud, region=self.config.region),
            deletion_protection="disabled",
            tags={"dataset": "medium-300-sample"},
        )
        while not self.pc.describe_index(self.config.index_name).status["ready"]:
            time.sleep(5)

    def namespace_count(self) -> int:
        stats = self.index.describe_index_stats()
        stats_dict = stats.to_dict() if hasattr(stats, "to_dict") else dict(stats)
        namespaces = stats_dict.get("namespaces") or {}
        if self.config.namespace:
            return int((namespaces.get(self.config.namespace) or {}).get("vector_count", 0))
        if "" in namespaces:
            return int((namespaces.get("") or {}).get("vector_count", 0))
        return int(stats_dict.get("total_vector_count") or 0)

    def upsert_chunks(self, chunks: list[Chunk], vectors: list[list[float]], force: bool = False) -> int:
        if self.namespace_count() and not force:
            return self.namespace_count()

        records = []
        for chunk, vector in zip(chunks, vectors, strict=True):
            records.append(
                {
                    "id": chunk.id,
                    "values": vector,
                    "metadata": {
                        "article_id": chunk.article_id,
                        "chunk_id": chunk.chunk_id,
                        "chunk_index": chunk.chunk_index,
                        "title": chunk.title,
                        "url": chunk.url,
                        "authors": chunk.authors,
                        "timestamp": chunk.timestamp,
                        "tags": chunk.tags,
                        "dataset": "medium-300-sample",
                        "text": chunk.text,
                    },
                }
            )

        total = 0
        for start in range(0, len(records), self.config.batch_size):
            batch = records[start : start + self.config.batch_size]
            kwargs: dict[str, Any] = {"vectors": batch}
            if self.config.namespace:
                kwargs["namespace"] = self.config.namespace
            response = self.index.upsert(**kwargs)
            total += int(getattr(response, "upserted_count", 0) or 0)
        return total

    def query(self, vector: list[float], top_k: int, include_values: bool) -> list[VectorMatch]:
        kwargs: dict[str, Any] = {
            "vector": vector,
            "top_k": top_k,
            "include_metadata": True,
            "include_values": include_values,
        }
        if self.config.namespace:
            kwargs["namespace"] = self.config.namespace
        result = self.index.query(**kwargs)
        matches = result.matches if hasattr(result, "matches") else result.get("matches", [])
        return [self._normalize_match(match) for match in matches]

    def _normalize_match(self, match: Any) -> VectorMatch:
        metadata = match.metadata if hasattr(match, "metadata") else match.get("metadata", {})
        values = match.values if hasattr(match, "values") else match.get("values", [])
        score = match.score if hasattr(match, "score") else match.get("score", 0.0)
        match_id = match.id if hasattr(match, "id") else match.get("id", "")
        return VectorMatch(
            id=str(match_id),
            score=float(score or 0.0),
            values=[float(value) for value in values or []],
            metadata=dict(metadata or {}),
        )
