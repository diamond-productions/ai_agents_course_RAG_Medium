from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from api.config import SAMPLE_DATA_PATH
from rag_utils import build_llm


DEFAULT_OUTPUT_PATH = Path("eval/medium_300_ai_benchmark.jsonl")


SYSTEM_PROMPT = """You create grounded RAG evaluation questions from Medium articles.
Do not use facts outside the provided article."""


USER_PROMPT = """Create {cases_per_article} evaluation cases for this article.

Rules:
- Questions must be answerable from the article excerpt only.
- Prefer questions that require a specific detail, claim, comparison, reason, or summary.
- Do not ask for information that depends on images or external links.
- Each expected_answer must be concise and fully grounded in the excerpt.
- evidence_quote must be a short exact quote from the excerpt that supports the answer.

Article metadata:
article_id: {article_id}
title: {title}
authors: {authors}
url: {url}
tags: {tags}

Article excerpt:
{article_text}
"""


class GeneratedCase(BaseModel):
    question: str = Field(description="A question answerable from the article excerpt only.")
    expected_answer: str = Field(description="A concise answer grounded in the article excerpt.")
    evidence_quote: str = Field(description="A short exact quote from the excerpt supporting the answer.")
    question_type: str = Field(description="The kind of question, such as detail, claim, reason, or summary.")


class GeneratedBenchmarkCases(BaseModel):
    cases: list[GeneratedCase] = Field(description="Generated benchmark cases.")


def load_articles(path: str) -> list[dict[str, str]]:
    articles: list[dict[str, str]] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for article_id, row in enumerate(csv.DictReader(handle)):
            text = str(row.get("text", "")).strip()
            if not text:
                continue
            articles.append(
                {
                    "article_id": str(article_id),
                    "title": str(row.get("title", "")).strip(),
                    "text": text,
                    "url": str(row.get("url", "")).strip(),
                    "authors": str(row.get("authors", "")).strip(),
                    "tags": str(row.get("tags", "")).strip(),
                }
            )
    return articles


def generate_cases(
    *,
    data_path: str,
    output_path: Path,
    articles: int,
    cases_per_article: int,
    seed: int,
    max_article_chars: int,
) -> None:
    rng = random.Random(seed)
    rows = load_articles(data_path)
    selected = rng.sample(rows, k=min(articles, len(rows)))

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("user", USER_PROMPT),
        ]
    )
    chain = prompt | build_llm().with_structured_output(GeneratedBenchmarkCases)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for article in selected:
            generated = chain.invoke(
                {
                    "cases_per_article": cases_per_article,
                    "article_id": article["article_id"],
                    "title": article["title"],
                    "authors": article["authors"],
                    "url": article["url"],
                    "tags": article["tags"],
                    "article_text": article["text"][:max_article_chars],
                }
            )
            cases = generated.cases if isinstance(generated, GeneratedBenchmarkCases) else []
            for case_index, case in enumerate(cases[:cases_per_article], start=1):
                record = {
                    "case_id": f"medium-300-{article['article_id']}-{case_index}",
                    "article_id": article["article_id"],
                    "title": article["title"],
                    "url": article["url"],
                    "authors": article["authors"],
                    "tags": article["tags"],
                    "question": case.question.strip(),
                    "expected_answer": case.expected_answer.strip(),
                    "evidence_quote": case.evidence_quote.strip(),
                    "question_type": case.question_type.strip() or "grounded",
                    "generator": "ai",
                }
                if record["question"] and record["expected_answer"]:
                    handle.write(json.dumps(record, ensure_ascii=False))
                    handle.write("\n")
                    written += 1
    print(f"Wrote {written} benchmark cases to {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate AI-created RAG benchmark cases for medium-300.")
    parser.add_argument("--data-path", default=SAMPLE_DATA_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--articles", type=int, default=30, help="Number of articles to sample.")
    parser.add_argument("--cases-per-article", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-article-chars", type=int, default=6000)
    args = parser.parse_args()

    generate_cases(
        data_path=args.data_path,
        output_path=args.output,
        articles=args.articles,
        cases_per_article=args.cases_per_article,
        seed=args.seed,
        max_article_chars=args.max_article_chars,
    )


if __name__ == "__main__":
    main()
