from __future__ import annotations

from medium_rag.retrieval.mmr import select_mmr_matches
from medium_rag.types import VectorMatch


def test_mmr_selects_relevant_first_and_dedupes_articles() -> None:
    matches = [
        VectorMatch("a1", 0.99, [1.0, 0.0], {"article_id": "a"}),
        VectorMatch("a2", 0.98, [0.99, 0.01], {"article_id": "a"}),
        VectorMatch("b1", 0.75, [0.7, 0.7], {"article_id": "b"}),
        VectorMatch("c1", 0.5, [0.0, 1.0], {"article_id": "c"}),
    ]
    selected = select_mmr_matches(matches, [1.0, 0.0], limit=3, lambda_mult=0.65)
    assert selected[0].id == "a1"
    assert len({match.metadata["article_id"] for match in selected}) == 3


def test_lower_mmr_lambda_can_select_more_diverse_match() -> None:
    matches = [
        VectorMatch("a", 0.99, [1.0, 0.0], {"article_id": "a"}),
        VectorMatch("b", 0.98, [0.95, 0.05], {"article_id": "b"}),
        VectorMatch("c", 0.6, [0.0, 1.0], {"article_id": "c"}),
    ]
    diverse = select_mmr_matches(matches, [1.0, 0.0], limit=2, lambda_mult=0.1, dedupe_by_article=False)
    relevant = select_mmr_matches(matches, [1.0, 0.0], limit=2, lambda_mult=0.9, dedupe_by_article=False)
    assert [match.id for match in diverse] == ["a", "c"]
    assert [match.id for match in relevant] == ["a", "b"]


def test_mmr_falls_back_when_values_missing() -> None:
    matches = [
        VectorMatch("a", 0.9, [], {"article_id": "a"}),
        VectorMatch("a2", 0.8, [], {"article_id": "a"}),
        VectorMatch("b", 0.7, [], {"article_id": "b"}),
    ]
    selected = select_mmr_matches(matches, [1.0, 0.0], limit=3, lambda_mult=0.5)
    assert [match.id for match in selected] == ["a", "b"]
