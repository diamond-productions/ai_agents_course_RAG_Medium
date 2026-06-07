from __future__ import annotations

import os
import re
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from medium_rag.config import GenerationConfig

DEFAULT_OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
DEFAULT_OPENROUTER_JUDGE_MODEL = "openai/gpt-4o-mini"

JUDGE_SYSTEM_PROMPT = """You are a strict evaluator for RAG answers.
Judge answer correctness against the expected answer and evidence. Judge faithfulness against the retrieved context only."""

JUDGE_USER_PROMPT = """Question:
{question}

Expected answer:
{expected_answer}

Evidence quote:
{evidence_quote}

Retrieved context:
{retrieved_context}

Model answer:
{actual_answer}
"""

LISTING_JUDGE_SYSTEM_PROMPT = """You are a strict evaluator for RAG multi-result listing answers.
Use only the retrieved context to judge title relevance and answer faithfulness."""

LISTING_JUDGE_USER_PROMPT = """Question:
{question}

Retrieved context:
{retrieved_context}

Model answer:
{actual_answer}

Return the article titles the model intended as its final answer. Then judge whether each returned title is relevant to the question using only the retrieved context. Also judge whether every factual claim in the model answer is supported by retrieved context.
"""


class JudgeResult(BaseModel):
    answer_correct: bool = Field(description="Whether the answer captures the expected answer.")
    faithful: bool = Field(description="Whether every factual claim in the answer is supported by retrieved context.")
    score: float = Field(ge=0.0, le=1.0, description="Overall answer quality from 0 to 1.")
    rationale: str = Field(description="A short explanation of the judgment.")


class TitleRelevance(BaseModel):
    title: str
    relevant: bool
    rationale: str


class ListingJudgeResult(BaseModel):
    returned_titles: list[str] = Field(description="Article titles the model returned as its final answer.")
    title_count_ok: bool = Field(description="Whether the model returned exactly three article titles.")
    titles_relevant: bool = Field(description="Whether every returned title is relevant to the question.")
    faithful: bool = Field(description="Whether every factual claim is supported by retrieved context.")
    score: float = Field(ge=0.0, le=1.0, description="Overall listing answer quality from 0 to 1.")
    title_relevance: list[TitleRelevance] = Field(description="Per-title relevance judgments.")
    rationale: str = Field(description="A short explanation of the judgment.")


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def build_judge_llm(config: GenerationConfig) -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("OPENROUTER_MODEL") or DEFAULT_OPENROUTER_JUDGE_MODEL,
        api_key=_require_env("OPENROUTER_API_KEY"),
        base_url=os.getenv("OPENROUTER_API_BASE") or DEFAULT_OPENROUTER_API_BASE,
    )


def normalize_title(value: str) -> str:
    value = re.sub(r"^\s*[-*•\d.)\]]+\s*", "", str(value))
    value = value.strip().strip("\"'`“”‘’")
    value = re.sub(r"\s+", " ", value)
    return value.rstrip(".,;:").casefold()


def format_retrieved_context(context: list[dict[str, Any]]) -> str:
    blocks = []
    for index, item in enumerate(context, start=1):
        chunk = str(item.get("chunk") or item.get("chunk_preview") or "")
        blocks.append(
            "\n".join(
                [
                    f"[{index}] article_id: {item.get('article_id', '')}",
                    f"title: {item.get('title', '')}",
                    f"authors: {item.get('authors', '')}",
                    f"url: {item.get('url', '')}",
                    f"passage: {chunk[:1600]}",
                ]
            )
        )
    return "\n\n".join(blocks) if blocks else "(no retrieved context)"


def build_judge_chain(config: GenerationConfig) -> Any:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", JUDGE_SYSTEM_PROMPT),
            ("user", JUDGE_USER_PROMPT),
        ]
    )
    return prompt | build_judge_llm(config).with_structured_output(JudgeResult)


def build_listing_judge_chain(config: GenerationConfig) -> Any:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", LISTING_JUDGE_SYSTEM_PROMPT),
            ("user", LISTING_JUDGE_USER_PROMPT),
        ]
    )
    return prompt | build_judge_llm(config).with_structured_output(ListingJudgeResult)


def judge_answer(
    judge_chain: Any,
    case: dict[str, Any],
    actual_answer: str,
    retrieved_context: list[dict[str, Any]],
) -> dict[str, Any]:
    judgment = judge_chain.invoke(
        {
            "question": case["question"],
            "expected_answer": case["expected_answer"],
            "evidence_quote": case.get("evidence_quote", ""),
            "retrieved_context": format_retrieved_context(retrieved_context),
            "actual_answer": actual_answer,
        }
    )
    if not isinstance(judgment, JudgeResult):
        raise TypeError(f"Expected JudgeResult, got {type(judgment).__name__}")
    return {
        "mode": "standard",
        "answer_correct": judgment.answer_correct,
        "faithful": judgment.faithful,
        "grounded": judgment.faithful,
        "score": judgment.score,
        "rationale": judgment.rationale,
    }


def _retrieved_title_map(retrieved_context: list[dict[str, Any]]) -> dict[str, str]:
    title_map: dict[str, str] = {}
    for item in retrieved_context:
        title = str(item.get("title", "")).strip()
        if title:
            title_map.setdefault(normalize_title(title), title)
    return title_map


def judge_listing_answer(
    judge_chain: Any,
    case: dict[str, Any],
    actual_answer: str,
    retrieved_context: list[dict[str, Any]],
) -> dict[str, Any]:
    judgment = judge_chain.invoke(
        {
            "question": case["question"],
            "retrieved_context": format_retrieved_context(retrieved_context),
            "actual_answer": actual_answer,
        }
    )
    if not isinstance(judgment, ListingJudgeResult):
        raise TypeError(f"Expected ListingJudgeResult, got {type(judgment).__name__}")

    title_map = _retrieved_title_map(retrieved_context)
    returned_titles = [title.strip() for title in judgment.returned_titles if title.strip()]
    normalized_titles = [normalize_title(title) for title in returned_titles]
    title_count_ok = len(returned_titles) == 3
    titles_distinct = len(set(normalized_titles)) == len(normalized_titles)
    titles_in_retrieved_context = all(title in title_map for title in normalized_titles)
    titles_relevant = (
        bool(judgment.titles_relevant)
        and len(judgment.title_relevance) == len(returned_titles)
        and all(bool(item.relevant) for item in judgment.title_relevance)
    )
    answer_correct = title_count_ok and titles_distinct and titles_in_retrieved_context and titles_relevant

    return {
        "mode": "listing",
        "answer_correct": answer_correct,
        "faithful": judgment.faithful,
        "grounded": judgment.faithful,
        "score": judgment.score,
        "returned_titles": returned_titles,
        "title_count_ok": title_count_ok,
        "titles_distinct": titles_distinct,
        "titles_in_retrieved_context": titles_in_retrieved_context,
        "titles_relevant": titles_relevant,
        "title_relevance": [
            {"title": item.title, "relevant": item.relevant, "rationale": item.rationale}
            for item in judgment.title_relevance
        ],
        "rationale": judgment.rationale,
    }
