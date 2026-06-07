from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Literal

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from medium_rag.config import RagExperimentConfig
from medium_rag.data import load_medium_articles
from medium_rag.evaluation.judge import build_benchmark_llm
from medium_rag.types import Article

DEFAULT_CASE_COUNT = 25
QuestionType = Literal[
    "precise_fact_retrieval",
    "multi_result_topic_listing",
    "key_idea_summary",
    "recommendation_with_evidence",
    "unknown_or_unanswerable",
]

SYSTEM_PROMPT = """You create high-quality RAG benchmark cases from Medium articles.
Use only the provided article dataset excerpts. Do not use outside facts."""

USER_PROMPT = """Create exactly {case_count} benchmark cases for a Medium-article RAG assistant.

The benchmark must cover the assignment requirements with this approximate mix:
- 8 precise_fact_retrieval cases: locate one concrete article from semantic criteria and return requested metadata.
- 5 multi_result_topic_listing cases: ask for exactly 3 distinct article titles matching a topic.
- 6 key_idea_summary cases: locate an article and summarize its central idea.
- 4 recommendation_with_evidence cases: recommend one article for a user need and justify with evidence.
- 2 unknown_or_unanswerable cases: require the assistant to refuse because the answer is not in the dataset.

Rules:
- Every answerable case must be grounded in the provided articles only.
- Prefer realistic, natural user questions over template-like questions.
- For answerable single-article cases, set source_article_ids to the relevant article_id.
- For multi_result_topic_listing, set source_article_ids and expected_titles to exactly 3 distinct relevant articles.
- For unknown_or_unanswerable, leave source_article_ids and expected_titles empty and use the expected answer:
  I don’t know based on the provided Medium articles data.
- evidence_quote must be a short exact quote from the provided excerpt when the case is answerable.
- Do not ask for information that depends on images, external links, live data, or outside knowledge unless the case is unknown_or_unanswerable.

Available articles:
{articles_json}
"""


class GeneratedCase(BaseModel):
    question: str = Field(description="Natural-language benchmark question.")
    expected_answer: str = Field(description="Concise expected answer grounded in the article excerpts.")
    evidence_quote: str = Field(description="Short exact quote supporting the expected answer, or empty if unanswerable.")
    question_type: QuestionType
    source_article_ids: list[str] = Field(default_factory=list)
    expected_titles: list[str] = Field(default_factory=list)


class GeneratedBenchmarkCases(BaseModel):
    cases: list[GeneratedCase] = Field(description="AI-curated benchmark cases.")


def _article_payload(article: Article, max_article_chars: int) -> dict[str, str]:
    return {
        "article_id": article.article_id,
        "title": article.title,
        "authors": article.authors,
        "url": article.url,
        "tags": article.tags,
        "excerpt": article.text[:max_article_chars],
    }


def _select_articles(rows: list[Article], articles: int, seed: int) -> list[Article]:
    rng = random.Random(seed)
    candidates = [row for row in rows if row.title and row.text]
    return rng.sample(candidates, k=min(articles, len(candidates)))


def _case_record(case: GeneratedCase, index: int, articles_by_id: dict[str, Article]) -> dict[str, object]:
    source_article_id = case.source_article_ids[0] if case.source_article_ids else ""
    article = articles_by_id.get(source_article_id)
    record: dict[str, object] = {
        "case_id": f"ai-curated-{index:03d}",
        "article_id": source_article_id,
        "title": article.title if article else "",
        "url": article.url if article else "",
        "authors": article.authors if article else "",
        "tags": article.tags if article else "",
        "source_article_ids": case.source_article_ids,
        "question": case.question.strip(),
        "expected_answer": case.expected_answer.strip(),
        "evidence_quote": case.evidence_quote.strip(),
        "question_type": case.question_type,
        "generator": "ai_curated",
    }
    if case.question_type == "multi_result_topic_listing" and case.expected_titles:
        record["expected_titles"] = [title.strip() for title in case.expected_titles if title.strip()]
    return record


def _write_record(handle, record: dict[str, object]) -> int:
    if str(record.get("question", "")).strip() and str(record.get("expected_answer", "")).strip():
        handle.write(json.dumps(record, ensure_ascii=False))
        handle.write("\n")
        return 1
    return 0


def generate_benchmark_cases(
    config: RagExperimentConfig,
    output_path: Path,
    articles: int,
    case_count: int,
    seed: int,
    max_article_chars: int,
) -> int:
    rows = load_medium_articles(config.dataset.path)
    selected = _select_articles(rows, articles, seed)
    articles_by_id = {article.article_id: article for article in selected}
    articles_json = json.dumps(
        [_article_payload(article, max_article_chars) for article in selected],
        ensure_ascii=False,
        indent=2,
    )

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("user", USER_PROMPT),
        ]
    )
    chain = prompt | build_benchmark_llm(config.generation).with_structured_output(GeneratedBenchmarkCases)
    generated = chain.invoke({"case_count": case_count, "articles_json": articles_json})
    cases = generated.cases if isinstance(generated, GeneratedBenchmarkCases) else []

    output_path.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with output_path.open("w", encoding="utf-8") as handle:
        for index, case in enumerate(cases[:case_count], start=1):
            written += _write_record(handle, _case_record(case, index, articles_by_id))
    return written
