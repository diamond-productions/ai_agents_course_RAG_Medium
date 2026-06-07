from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Article:
    article_id: str
    title: str
    text: str
    url: str = ""
    authors: str = ""
    timestamp: str = ""
    tags: str = ""


@dataclass(frozen=True)
class Chunk:
    id: str
    article_id: str
    chunk_id: str
    chunk_index: int
    text: str
    title: str
    url: str = ""
    authors: str = ""
    timestamp: str = ""
    tags: str = ""


@dataclass(frozen=True)
class RetrievedContext:
    article_id: str
    title: str
    chunk: str
    score: float
    url: str = ""
    authors: str = ""
    timestamp: str = ""
    tags: str = ""
    chunk_id: str = ""
    chunk_index: int | None = None

    def as_api_dict(self) -> dict[str, Any]:
        return {
            "article_id": self.article_id,
            "title": self.title,
            "chunk": self.chunk,
            "score": self.score,
            "url": self.url,
            "authors": self.authors,
            "timestamp": self.timestamp,
            "tags": self.tags,
            "chunk_id": self.chunk_id,
            "chunk_index": self.chunk_index,
        }


@dataclass(frozen=True)
class VectorMatch:
    id: str
    score: float
    values: list[float]
    metadata: dict[str, Any]


@dataclass(frozen=True)
class GenerationResult:
    response: str
    augmented_prompt: dict[str, str]
    usage: dict[str, int | None]
    cost: dict[str, Any]
    metadata: dict[str, Any]
