from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

from medium_rag.types import Article


def _clean(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def load_medium_articles(path: str | Path) -> list[Article]:
    articles: list[Article] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for article_id, row in enumerate(csv.DictReader(handle)):
            text = _clean(row.get("text"))
            if not text:
                continue
            articles.append(
                Article(
                    article_id=str(article_id),
                    title=_clean(row.get("title")),
                    text=text,
                    url=_clean(row.get("url")),
                    authors=_clean(row.get("authors")),
                    timestamp=_clean(row.get("timestamp")),
                    tags=_clean(row.get("tags")),
                )
            )
    return articles
