from __future__ import annotations

import pytest

from medium_rag.evaluation.runner import compute_retrieval_metrics, compact_context


def test_single_relevant_article_rank_one_metrics() -> None:
    case = {"article_id": "a1", "title": "Expected"}
    context = [{"article_id": "a1", "title": "Expected"}, {"article_id": "a2", "title": "Other"}]

    metrics = compute_retrieval_metrics(case, context, top_k=7)

    assert metrics["hit_at_1"] is True
    assert metrics["hit_at_3"] is True
    assert metrics["hit_at_k"] is True
    assert metrics["mrr_at_k"] == 1.0
    assert metrics["recall_at_k"] == 1.0


def test_single_relevant_article_rank_three_metrics() -> None:
    case = {"article_id": "a3", "title": "Expected"}
    context = [
        {"article_id": "a1", "title": "First"},
        {"article_id": "a2", "title": "Second"},
        {"article_id": "a3", "title": "Expected"},
    ]

    metrics = compute_retrieval_metrics(case, context, top_k=7)

    assert metrics["hit_at_1"] is False
    assert metrics["hit_at_3"] is True
    assert metrics["mrr_at_k"] == pytest.approx(1 / 3)
    assert metrics["recall_at_k"] == 1.0


def test_single_relevant_article_absent_metrics() -> None:
    case = {"article_id": "missing", "title": "Missing"}
    context = [{"article_id": "a1", "title": "First"}]

    metrics = compute_retrieval_metrics(case, context, top_k=7)

    assert metrics["hit_at_k"] is False
    assert metrics["mrr_at_k"] == 0.0
    assert metrics["recall_at_k"] == 0.0


def test_listing_expected_title_recall_metrics() -> None:
    case = {
        "question_type": "multi_result_topic_listing",
        "expected_titles": ["First", "Second", "Third"],
    }
    context = [
        {"article_id": "a1", "title": "First"},
        {"article_id": "a2", "title": "Other"},
        {"article_id": "a3", "title": "Second"},
    ]

    metrics = compute_retrieval_metrics(case, context, top_k=7)

    assert metrics["hit_at_k"] is True
    assert metrics["mrr_at_k"] == 1.0
    assert metrics["recall_at_k"] == pytest.approx(2 / 3)
    assert metrics["expected_title_recall_at_k"] == pytest.approx(2 / 3)
    assert metrics["expected_titles_hit_at_k"] is False


def test_unanswerable_retrieval_metrics_are_null() -> None:
    case = {"question_type": "unknown_or_unanswerable"}
    context = [{"article_id": "a1", "title": "First"}]

    metrics = compute_retrieval_metrics(case, context, top_k=7)

    assert all(value is None for value in metrics.values())


def test_compact_context_stores_preview_not_full_chunk() -> None:
    context = [
        {
            "article_id": "a1",
            "title": "First",
            "authors": "Ada",
            "url": "https://example.test",
            "score": 0.9,
            "chunk": "x" * 1300,
        }
    ]

    compact = compact_context(context)

    assert compact[0]["article_id"] == "a1"
    assert compact[0]["title"] == "First"
    assert len(compact[0]["chunk_preview"]) == 1200
