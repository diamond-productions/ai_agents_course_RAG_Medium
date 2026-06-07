from __future__ import annotations

from typing import Protocol

from medium_rag.types import RetrievedContext


class Retriever(Protocol):
    def retrieve(self, question: str) -> list[RetrievedContext]:
        ...
