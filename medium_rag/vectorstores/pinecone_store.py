from __future__ import annotations

import json
import os
import time
from collections.abc import Iterator
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

    def chunk_record(self, chunk: Chunk, vector: list[float], dataset_name: str) -> dict[str, Any]:
        return {
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
                "dataset": dataset_name,
                "text": chunk.text,
            },
        }

    def build_records(
        self,
        chunks: list[Chunk],
        vectors: list[list[float]],
        dataset_name: str = "medium-300-sample",
    ) -> list[dict[str, Any]]:
        return [self.chunk_record(chunk, vector, dataset_name) for chunk, vector in zip(chunks, vectors, strict=True)]

    def estimate_record_bytes(self, record: dict[str, Any]) -> int:
        return len(json.dumps(record, ensure_ascii=False, separators=(",", ":"), default=str).encode("utf-8"))

    def iter_upsert_batches(self, records: list[dict[str, Any]]) -> Iterator[list[dict[str, Any]]]:
        batch: list[dict[str, Any]] = []
        batch_bytes = 0
        for record in records:
            record_bytes = self.estimate_record_bytes(record)
            batch_full_by_count = len(batch) >= self.config.batch_size
            batch_full_by_size = batch and batch_bytes + record_bytes > self.config.max_batch_bytes
            if batch_full_by_count or batch_full_by_size:
                yield batch
                batch = []
                batch_bytes = 0
            batch.append(record)
            batch_bytes += record_bytes
        if batch:
            yield batch

    def _upsert_batch(self, batch: list[dict[str, Any]]) -> int:
        kwargs: dict[str, Any] = {"vectors": batch}
        if self.config.namespace:
            kwargs["namespace"] = self.config.namespace
        response = self.index.upsert(**kwargs)
        if isinstance(response, dict):
            return int(response.get("upserted_count", 0) or 0)
        return int(getattr(response, "upserted_count", 0) or 0)

    def upsert_records(self, records: list[dict[str, Any]]) -> int:
        total = 0
        for batch in self.iter_upsert_batches(records):
            total += self._upsert_batch(batch)
        return total

    def upsert_chunks(
        self,
        chunks: list[Chunk],
        vectors: list[list[float]],
        force: bool = False,
        dataset_name: str = "medium-300-sample",
    ) -> int:
        existing_count = self.namespace_count()
        if existing_count and not force:
            return existing_count
        return self.upsert_records(self.build_records(chunks, vectors, dataset_name))

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
