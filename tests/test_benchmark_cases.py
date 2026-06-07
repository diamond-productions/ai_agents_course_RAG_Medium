from __future__ import annotations

from medium_rag.evaluation.benchmark_cases import GeneratedCase, _case_record
from medium_rag.types import Article


def test_ai_curated_listing_case_record_preserves_expected_titles() -> None:
    articles = {
        "1": Article(article_id="1", title="First", text="Text", authors="Ada", tags="Education"),
        "2": Article(article_id="2", title="Second", text="Text", authors="Ben", tags="Education"),
        "3": Article(article_id="3", title="Third", text="Text", authors="Cam", tags="Education"),
    }
    case = GeneratedCase(
        question="List exactly 3 distinct articles about education. Return only the titles.",
        expected_answer="First\nSecond\nThird",
        evidence_quote="Education",
        question_type="multi_result_topic_listing",
        source_article_ids=["1", "2", "3"],
        expected_titles=["First", "Second", "Third"],
    )

    record = _case_record(case, 1, articles)

    assert record["case_id"] == "ai-curated-001"
    assert record["article_id"] == "1"
    assert record["question_type"] == "multi_result_topic_listing"
    assert record["expected_titles"] == ["First", "Second", "Third"]
    assert record["generator"] == "ai_curated"
