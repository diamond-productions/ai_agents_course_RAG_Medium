from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from statistics import mean
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field

from api.config import TOP_K
from rag_utils import answer_question, build_llm


DEFAULT_CASES_PATH = Path("eval/medium_300_ai_benchmark.jsonl")
DEFAULT_OUTPUT_PATH = Path("eval/medium_300_ai_benchmark_results.jsonl")


JUDGE_SYSTEM_PROMPT = """You are a strict evaluator for RAG answers.
Judge only against the expected answer and evidence. Do not require exact wording."""


JUDGE_USER_PROMPT = """Question:
{question}

Expected answer:
{expected_answer}

Evidence quote:
{evidence_quote}

Model answer:
{actual_answer}
"""


class JudgeResult(BaseModel):
    answer_correct: bool = Field(description="Whether the answer captures the expected answer.")
    grounded: bool = Field(description="Whether the answer is supported by the expected answer and evidence.")
    score: float = Field(ge=0.0, le=1.0, description="Overall answer quality from 0 to 1.")
    rationale: str = Field(description="A short explanation of the judgment.")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def retrieval_hit(case: dict[str, Any], context: list[dict[str, Any]]) -> bool:
    expected_article_id = str(case.get("article_id", ""))
    expected_title = str(case.get("title", "")).strip().casefold()
    for item in context:
        actual_article_id = str(item.get("article_id", ""))
        actual_title = str(item.get("title", "")).strip().casefold()
        if expected_article_id and actual_article_id and expected_article_id == actual_article_id:
            return True
        if expected_title and actual_title and expected_title == actual_title:
            return True
    return False


def judge_answer(judge_chain: Any, case: dict[str, Any], actual_answer: str) -> dict[str, Any]:
    judgment = judge_chain.invoke(
        {
            "question": case["question"],
            "expected_answer": case["expected_answer"],
            "evidence_quote": case.get("evidence_quote", ""),
            "actual_answer": actual_answer,
        }
    )
    if not isinstance(judgment, JudgeResult):
        raise TypeError(f"Expected JudgeResult, got {type(judgment).__name__}")
    return {
        "answer_correct": judgment.answer_correct,
        "grounded": judgment.grounded,
        "score": judgment.score,
        "rationale": judgment.rationale,
    }


def summarize(records: list[dict[str, Any]]) -> dict[str, Any]:
    if not records:
        return {"cases": 0}
    question_types = Counter(str(record.get("question_type", "unknown")) for record in records)
    return {
        "cases": len(records),
        "answer_accuracy": mean(1.0 if record["evaluation"]["answer_correct"] else 0.0 for record in records),
        "grounded_rate": mean(1.0 if record["evaluation"]["grounded"] else 0.0 for record in records),
        "mean_judge_score": mean(float(record["evaluation"]["score"]) for record in records),
        "retrieval_hit_rate": mean(1.0 if record["retrieval"]["hit"] else 0.0 for record in records),
        "question_types": dict(question_types),
    }


def run_benchmark(cases_path: Path, output_path: Path, top_k: int, limit: int | None) -> None:
    cases = read_jsonl(cases_path)
    if limit is not None:
        cases = cases[:limit]

    judge_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", JUDGE_SYSTEM_PROMPT),
            ("user", JUDGE_USER_PROMPT),
        ]
    )
    judge_chain = judge_prompt | build_llm().with_structured_output(JudgeResult)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    with output_path.open("w", encoding="utf-8") as handle:
        for index, case in enumerate(cases, start=1):
            result = answer_question(
                case["question"],
                top_k=top_k,
                log_trace=False,
                eval_run_id=None,
                expected_answer=None,
                evaluation=None,
            )
            evaluation = judge_answer(judge_chain, case, result["response"])
            record = {
                **case,
                "actual_answer": result["response"],
                "retrieval": {
                    "hit": retrieval_hit(case, result["context"]),
                    "top_k": top_k,
                    "titles": [item.get("title", "") for item in result["context"]],
                    "scores": [item.get("score") for item in result["context"]],
                },
                "evaluation": evaluation,
                "usage": result.get("usage", {}),
                "cost": result.get("cost", {}),
                "metadata": result.get("metadata", {}),
            }
            handle.write(json.dumps(record, ensure_ascii=False, default=str))
            handle.write("\n")
            records.append(record)
            print(
                f"[{index}/{len(cases)}] "
                f"hit={record['retrieval']['hit']} "
                f"correct={evaluation['answer_correct']} "
                f"score={evaluation['score']:.2f} "
                f"{case['case_id']}"
            )

    summary_path = output_path.with_suffix(".summary.json")
    summary = summarize(records)
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"Wrote results to {output_path}")
    print(f"Wrote summary to {summary_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AI-created medium-300 RAG benchmark.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--top-k", type=int, default=TOP_K)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    run_benchmark(args.cases, args.output, args.top_k, args.limit)


if __name__ == "__main__":
    main()
