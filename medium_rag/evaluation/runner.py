from __future__ import annotations

import json
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from medium_rag.config import RagExperimentConfig
from medium_rag.evaluation.judge import build_judge_chain, build_listing_judge_chain, judge_answer, judge_listing_answer, normalize_title
from medium_rag.pipeline import RagPipeline


@dataclass(frozen=True)
class BenchmarkSummary:
    experiment_name: str
    retrieval_strategy: str
    top_k: int
    candidate_k: int
    mmr_lambda: float
    cases: int
    answer_accuracy: float | None
    grounded_rate: float | None
    mean_judge_score: float | None
    retrieval_hit_rate: float | None
    faithfulness_rate: float | None
    retrieval_metrics: dict[str, float | None]
    question_types: dict[str, int]
    question_type_metrics: dict[str, dict[str, float | int | None]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "experiment_name": self.experiment_name,
            "retrieval_strategy": self.retrieval_strategy,
            "top_k": self.top_k,
            "candidate_k": self.candidate_k,
            "mmr_lambda": self.mmr_lambda,
            "cases": self.cases,
            "answer_accuracy": self.answer_accuracy,
            "grounded_rate": self.grounded_rate,
            "mean_judge_score": self.mean_judge_score,
            "retrieval_hit_rate": self.retrieval_hit_rate,
            "faithfulness_rate": self.faithfulness_rate,
            "retrieval_metrics": self.retrieval_metrics,
            "question_types": self.question_types,
            "question_type_metrics": self.question_type_metrics,
        }


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def relevant_article_ids(case: dict[str, Any]) -> set[str]:
    ids = {str(value).strip() for value in case.get("source_article_ids", []) if str(value).strip()}
    article_id = str(case.get("article_id", "")).strip()
    if article_id:
        ids.add(article_id)
    return ids


def relevant_titles(case: dict[str, Any]) -> set[str]:
    titles = {normalize_title(value) for value in case.get("expected_titles", []) if str(value).strip()}
    title = str(case.get("title", "")).strip()
    if title and not titles:
        titles.add(normalize_title(title))
    return titles


def context_article_id_at(context: list[dict[str, Any]], rank: int) -> str:
    if rank < 1 or rank > len(context):
        return ""
    return str(context[rank - 1].get("article_id", "")).strip()


def context_title_at(context: list[dict[str, Any]], rank: int) -> str:
    if rank < 1 or rank > len(context):
        return ""
    return normalize_title(str(context[rank - 1].get("title", "")))


def _null_retrieval_metrics() -> dict[str, Any]:
    return {
        "hit": None,
        "hit_at_1": None,
        "hit_at_3": None,
        "hit_at_5": None,
        "hit_at_k": None,
        "mrr_at_k": None,
        "recall_at_k": None,
        "expected_title_recall_at_k": None,
        "expected_titles_hit_at_k": None,
    }


def _rank_matches(case: dict[str, Any], context: list[dict[str, Any]]) -> list[bool]:
    ids = relevant_article_ids(case)
    titles = relevant_titles(case)
    matches = []
    for item in context:
        article_id = str(item.get("article_id", "")).strip()
        title = normalize_title(str(item.get("title", "")))
        matches.append((bool(ids) and article_id in ids) or (bool(titles) and title in titles))
    return matches


def compute_retrieval_metrics(case: dict[str, Any], context: list[dict[str, Any]], top_k: int) -> dict[str, Any]:
    if case.get("question_type") == "unknown_or_unanswerable":
        return _null_retrieval_metrics()

    ranks = _rank_matches(case, context[:top_k])
    ids = relevant_article_ids(case)
    titles = relevant_titles(case)
    expected_count = len(titles) if case.get("expected_titles") else max(len(ids), len(titles), 1)
    found_count = 0
    if case.get("expected_titles"):
        retrieved_titles = {normalize_title(str(item.get("title", ""))) for item in context[:top_k]}
        found_count = len(titles & retrieved_titles)
    else:
        found_ids = {str(item.get("article_id", "")).strip() for item in context[:top_k]} & ids
        found_titles = {normalize_title(str(item.get("title", ""))) for item in context[:top_k]} & titles
        found_count = max(len(found_ids), len(found_titles))

    first_rank = next((index for index, is_match in enumerate(ranks, start=1) if is_match), None)
    expected_title_recall = None
    expected_titles_hit = None
    if case.get("expected_titles"):
        expected_title_recall = found_count / expected_count if expected_count else None
        expected_titles_hit = found_count == expected_count if expected_count else None

    def hit_at(limit: int) -> bool:
        return any(ranks[: min(limit, top_k)])

    return {
        "hit": any(ranks),
        "hit_at_1": hit_at(1),
        "hit_at_3": hit_at(3),
        "hit_at_5": hit_at(5),
        "hit_at_k": any(ranks),
        "mrr_at_k": (1.0 / first_rank) if first_rank else 0.0,
        "recall_at_k": found_count / expected_count if expected_count else None,
        "expected_title_recall_at_k": expected_title_recall,
        "expected_titles_hit_at_k": expected_titles_hit,
    }


def compact_context(context: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "article_id": item.get("article_id", ""),
            "title": item.get("title", ""),
            "authors": item.get("authors", ""),
            "url": item.get("url", ""),
            "score": item.get("score"),
            "chunk_preview": str(item.get("chunk", ""))[:1200],
        }
        for item in context
    ]


def _mean_or_none(values: list[float]) -> float | None:
    return mean(values) if values else None


RETRIEVAL_METRIC_KEYS = [
    "hit_at_1",
    "hit_at_3",
    "hit_at_5",
    "hit_at_k",
    "mrr_at_k",
    "recall_at_k",
    "expected_title_recall_at_k",
    "expected_titles_hit_at_k",
]


def _metric_mean(records: list[dict[str, Any]], metric: str) -> float | None:
    values = []
    for record in records:
        value = record["retrieval"].get(metric)
        if value is None:
            continue
        values.append(1.0 if isinstance(value, bool) and value else 0.0 if isinstance(value, bool) else float(value))
    return _mean_or_none(values)


def _retrieval_metric_summary(records: list[dict[str, Any]]) -> dict[str, float | None]:
    return {metric: _metric_mean(records, metric) for metric in RETRIEVAL_METRIC_KEYS}


def _question_type_metrics(records: list[dict[str, Any]]) -> dict[str, dict[str, float | int | None]]:
    metrics: dict[str, dict[str, float | int | None]] = {}
    question_types = sorted({str(record.get("question_type", "unknown")) for record in records})
    for question_type in question_types:
        matching = [record for record in records if str(record.get("question_type", "unknown")) == question_type]
        retrieval_values = [
            1.0 if record["retrieval"]["hit"] else 0.0
            for record in matching
            if record["retrieval"]["hit"] is not None
        ]
        metric = {
            "cases": len(matching),
            "answer_accuracy": _mean_or_none(
                [1.0 if record["evaluation"]["answer_correct"] else 0.0 for record in matching]
            ),
            "faithfulness_rate": _mean_or_none(
                [1.0 if record["evaluation"].get("faithful") else 0.0 for record in matching]
            ),
            "grounded_rate": _mean_or_none(
                [1.0 if record["evaluation"]["grounded"] else 0.0 for record in matching]
            ),
            "mean_judge_score": _mean_or_none(
                [float(record["evaluation"]["score"]) for record in matching]
            ),
            "retrieval_hit_rate": _mean_or_none(retrieval_values),
        }
        metric.update(_retrieval_metric_summary(matching))
        if question_type == "multi_result_topic_listing":
            metric.update(
                {
                    "listing_valid_rate": _mean_or_none(
                        [1.0 if record["evaluation"]["answer_correct"] else 0.0 for record in matching]
                    ),
                    "listing_count_ok_rate": _mean_or_none(
                        [1.0 if record["evaluation"].get("title_count_ok") else 0.0 for record in matching]
                    ),
                    "listing_distinct_rate": _mean_or_none(
                        [1.0 if record["evaluation"].get("titles_distinct") else 0.0 for record in matching]
                    ),
                    "listing_context_title_rate": _mean_or_none(
                        [1.0 if record["evaluation"].get("titles_in_retrieved_context") else 0.0 for record in matching]
                    ),
                    "listing_relevance_rate": _mean_or_none(
                        [1.0 if record["evaluation"].get("titles_relevant") else 0.0 for record in matching]
                    ),
                }
            )
        metrics[question_type] = metric
    return metrics


def summarize(config: RagExperimentConfig, records: list[dict[str, Any]]) -> BenchmarkSummary:
    question_types = Counter(str(record.get("question_type", "unknown")) for record in records)
    if not records:
        return BenchmarkSummary(
            experiment_name=config.experiment_name,
            retrieval_strategy=config.retrieval.strategy,
            top_k=config.retrieval.top_k,
            candidate_k=config.retrieval.candidate_k,
            mmr_lambda=config.retrieval.mmr_lambda,
            cases=0,
            answer_accuracy=None,
            grounded_rate=None,
            mean_judge_score=None,
            retrieval_hit_rate=None,
            faithfulness_rate=None,
            retrieval_metrics={},
            question_types={},
            question_type_metrics={},
        )
    retrieval_values = [
        1.0 if record["retrieval"]["hit"] else 0.0
        for record in records
        if record["retrieval"]["hit"] is not None
    ]
    faithfulness_rate = mean(1.0 if record["evaluation"].get("faithful") else 0.0 for record in records)
    return BenchmarkSummary(
        experiment_name=config.experiment_name,
        retrieval_strategy=config.retrieval.strategy,
        top_k=config.retrieval.top_k,
        candidate_k=config.retrieval.candidate_k,
        mmr_lambda=config.retrieval.mmr_lambda,
        cases=len(records),
        answer_accuracy=mean(1.0 if record["evaluation"]["answer_correct"] else 0.0 for record in records),
        grounded_rate=mean(1.0 if record["evaluation"]["grounded"] else 0.0 for record in records),
        mean_judge_score=mean(float(record["evaluation"]["score"]) for record in records),
        retrieval_hit_rate=_mean_or_none(retrieval_values),
        faithfulness_rate=faithfulness_rate,
        retrieval_metrics=_retrieval_metric_summary(records),
        question_types=dict(question_types),
        question_type_metrics=_question_type_metrics(records),
    )


def run_benchmark(
    config: RagExperimentConfig,
    cases_path: Path,
    output_path: Path,
    limit: int | None = None,
) -> BenchmarkSummary:
    cases = read_jsonl(cases_path)
    if limit is not None:
        cases = cases[:limit]
    pipeline = RagPipeline(config)
    judge_chain = build_judge_chain(config.generation)
    listing_judge_chain = build_listing_judge_chain(config.generation)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    with output_path.open("w", encoding="utf-8") as handle:
        for index, case in enumerate(cases, start=1):
            result = pipeline.answer_question(case["question"], log_trace=False)
            context = result["context"]
            if case.get("question_type") == "multi_result_topic_listing":
                evaluation = judge_listing_answer(listing_judge_chain, case, result["response"], context)
            else:
                evaluation = judge_answer(judge_chain, case, result["response"], context)
            retrieval = {
                **compute_retrieval_metrics(case, context, config.retrieval.top_k),
                "top_k": config.retrieval.top_k,
                "titles": [item.get("title", "") for item in context],
                "scores": [item.get("score") for item in context],
                "context": compact_context(context),
            }
            record = {
                **case,
                "actual_answer": result["response"],
                "retrieval": retrieval,
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

    summary = summarize(config, records)
    summary_path = output_path.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary.as_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    return summary
