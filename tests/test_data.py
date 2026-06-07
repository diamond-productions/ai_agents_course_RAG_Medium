from __future__ import annotations

from pathlib import Path

from medium_rag.data import load_medium_articles


def test_load_medium_articles_skips_empty_text_and_uses_row_index(tmp_path: Path) -> None:
    csv_path = tmp_path / "articles.csv"
    csv_path.write_text(
        "title,text,url,authors,timestamp,tags\n"
        "First,hello world,http://one,Author,2020,Tag\n"
        "Empty,,http://empty,Author,2020,Tag\n"
        "Third,more text,http://three,,,Other\n",
        encoding="utf-8",
    )
    articles = load_medium_articles(csv_path)
    assert [article.article_id for article in articles] == ["0", "2"]
    assert articles[0].title == "First"
    assert articles[1].authors == ""
