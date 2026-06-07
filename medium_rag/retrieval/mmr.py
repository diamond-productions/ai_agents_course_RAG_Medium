from __future__ import annotations

import math

from langchain_openai import OpenAIEmbeddings

from medium_rag.config import RetrievalConfig
from medium_rag.retrieval.postprocess import article_key, dedupe_matches_by_article, matches_to_context
from medium_rag.types import RetrievedContext, VectorMatch
from medium_rag.vectorstores import PineconeVectorStore


def cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def select_mmr_matches(
    matches: list[VectorMatch],
    query_vector: list[float],
    limit: int,
    lambda_mult: float,
    dedupe_by_article: bool = True,
) -> list[VectorMatch]:
    candidates = [match for match in matches if match.values]
    if not candidates:
        return dedupe_matches_by_article(matches, limit) if dedupe_by_article else matches[:limit]

    lambda_mult = max(0.0, min(float(lambda_mult), 1.0))
    selected: list[VectorMatch] = []
    selected_articles: set[str] = set()
    remaining = candidates.copy()

    while remaining and len(selected) < limit:
        best_match = None
        best_score = float("-inf")
        for match in remaining:
            key = article_key(match)
            if dedupe_by_article and key and key in selected_articles:
                continue
            relevance = cosine_similarity(query_vector, match.values)
            redundancy = max(
                (cosine_similarity(match.values, selected_match.values) for selected_match in selected),
                default=0.0,
            )
            mmr_score = lambda_mult * relevance - (1.0 - lambda_mult) * redundancy
            if mmr_score > best_score:
                best_score = mmr_score
                best_match = match

        if best_match is None:
            break

        remaining.remove(best_match)
        selected.append(best_match)
        key = article_key(best_match)
        if key:
            selected_articles.add(key)

    if len(selected) < limit:
        selected_ids = {match.id for match in selected}
        fallback = dedupe_matches_by_article(matches, limit) if dedupe_by_article else matches[:limit]
        for match in fallback:
            if match.id in selected_ids:
                continue
            selected.append(match)
            if len(selected) >= limit:
                break
    return selected[:limit]


class DenseMmrRetriever:
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
        matches = self.store.query(query_vector, top_k=candidate_k, include_values=True)
        if self.config.mmr_enabled:
            matches = select_mmr_matches(
                matches,
                query_vector,
                limit=self.config.top_k,
                lambda_mult=self.config.mmr_lambda,
                dedupe_by_article=self.config.dedupe_by_article,
            )
        elif self.config.dedupe_by_article:
            matches = dedupe_matches_by_article(matches, self.config.top_k)
        else:
            matches = matches[: self.config.top_k]
        return matches_to_context(matches)
