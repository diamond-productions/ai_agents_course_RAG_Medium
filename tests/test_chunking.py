from __future__ import annotations

from medium_rag.chunking import build_chunk_text, split_articles
from medium_rag.config import ChunkingConfig
from medium_rag.types import Article


def test_build_chunk_text_includes_metadata_and_passage() -> None:
    article = Article(
        article_id="1",
        title="Example",
        authors="Alice",
        tags="AI, RAG",
        text="Body text",
    )
    text = build_chunk_text(article)
    assert "Title: Example" in text
    assert "Authors: Alice" in text
    assert "Tags: AI, RAG" in text
    assert "Passage:\nBody text" in text


def test_split_articles_uses_deterministic_ids_and_resets_per_article() -> None:
    config = ChunkingConfig(
        chunk_size=10,
        overlap_ratio=0.0,
        tokenizer="approx",
        separators=[" "],
    )
    articles = [
        Article(article_id="a", title="A", text="one two three four five six seven eight nine ten eleven twelve"),
        Article(article_id="b", title="B", text="one two three four five six seven eight nine ten eleven twelve"),
    ]
    chunks = split_articles(articles, config)
    assert chunks[0].id == "medium-300:a:0000"
    assert chunks[0].chunk_id == "a-0000"
    assert chunks[0].chunk_index == 0
    assert any(chunk.id == "medium-300:b:0000" for chunk in chunks)
    assert [chunk.id for chunk in chunks] == [chunk.id for chunk in split_articles(articles, config)]
