from __future__ import annotations

from medium_rag.retrieval.postprocess import dedupe_matches_by_article, match_to_context
from medium_rag.types import VectorMatch


def test_dedupe_matches_by_article_keeps_first_match() -> None:
    matches = [
        VectorMatch("1", 0.9, [], {"article_id": "a", "title": "A"}),
        VectorMatch("2", 0.8, [], {"article_id": "a", "title": "A"}),
        VectorMatch("3", 0.7, [], {"article_id": "b", "title": "B"}),
    ]
    deduped = dedupe_matches_by_article(matches, limit=3)
    assert [match.id for match in deduped] == ["1", "3"]


def test_dedupe_matches_by_title_when_article_id_missing() -> None:
    matches = [
        VectorMatch("1", 0.9, [], {"title": "Same"}),
        VectorMatch("2", 0.8, [], {"title": "Same"}),
        VectorMatch("3", 0.7, [], {"title": "Other"}),
    ]
    deduped = dedupe_matches_by_article(matches, limit=3)
    assert [match.id for match in deduped] == ["1", "3"]


def test_match_to_context_preserves_chunk_metadata() -> None:
    context = match_to_context(
        VectorMatch(
            "id",
            0.5,
            [],
            {
                "article_id": "a",
                "title": "Title",
                "text": "Chunk",
                "chunk_id": "a-0000",
                "chunk_index": 0,
            },
        )
    )
    assert context.article_id == "a"
    assert context.chunk == "Chunk"
    assert context.chunk_id == "a-0000"
    assert context.chunk_index == 0
