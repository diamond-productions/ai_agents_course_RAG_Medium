from __future__ import annotations

from langchain_openai import OpenAIEmbeddings

from medium_rag.config import RetrievalConfig
from medium_rag.retrieval.postprocess import dedupe_matches_by_article, matches_to_context
from medium_rag.types import RetrievedContext
from medium_rag.vectorstores import PineconeVectorStore


class DenseRetriever:
    def __init__(
        self,
        embeddings: OpenAIEmbeddings,
        store: PineconeVectorStore,
        config: RetrievalConfig,
    ):
        self.embeddings = embeddings
        self.store = store
        self.config = config

    def retrieve(self, question: str) -> list[RetrievedContext]:
        query_vector = [float(value) for value in self.embeddings.embed_query(question)]
        candidate_k = max(self.config.top_k, self.config.candidate_k)
        matches = self.store.query(query_vector, top_k=candidate_k, include_values=False)
        if self.config.dedupe_by_article:
            matches = dedupe_matches_by_article(matches, self.config.top_k)
        else:
            matches = matches[: self.config.top_k]
        return matches_to_context(matches)
