from __future__ import annotations

from pathlib import Path

from medium_rag.data import iter_medium_articles, load_medium_articles


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


def test_iter_medium_articles_is_lazy_and_matches_loader(tmp_path: Path) -> None:
    csv_path = tmp_path / "articles.csv"
    csv_path.write_text(
        "title,text,url,authors,timestamp,tags\n"
        "First,hello world,http://one,Author,2020,Tag\n"
        "Empty,,http://empty,Author,2020,Tag\n"
        "Third,more text,http://three,,,Other\n",
        encoding="utf-8",
    )
    iterator = iter_medium_articles(csv_path)
    assert iter(iterator) is iterator
    assert [article.article_id for article in iterator] == ["0", "2"]
    assert [article.article_id for article in load_medium_articles(csv_path)] == ["0", "2"]
