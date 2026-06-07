from __future__ import annotations

from medium_rag.evaluation.judge import (
    JudgeResult,
    ListingJudgeResult,
    TitleRelevance,
    judge_answer,
    judge_listing_answer,
    normalize_title,
)


class FakeListingChain:
    def __init__(self, result: ListingJudgeResult):
        self.result = result

    def invoke(self, payload):
        return self.result


class FakeStandardChain:
    def __init__(self, result: JudgeResult):
        self.result = result

    def invoke(self, payload):
        return self.result


def test_normalize_title_removes_bullets_quotes_and_case() -> None:
    assert normalize_title("  1. “User Experience Design Process.” ") == "user experience design process"


def test_listing_answer_passes_for_three_distinct_retrieved_relevant_titles() -> None:
    result = ListingJudgeResult(
        returned_titles=["First", "Second", "Third"],
        title_count_ok=True,
        titles_relevant=True,
        faithful=True,
        score=1.0,
        title_relevance=[
            TitleRelevance(title="First", relevant=True, rationale="Relevant."),
            TitleRelevance(title="Second", relevant=True, rationale="Relevant."),
            TitleRelevance(title="Third", relevant=True, rationale="Relevant."),
        ],
        rationale="All titles are relevant.",
    )
    context = [{"title": "First"}, {"title": "Second"}, {"title": "Third"}]

    evaluation = judge_listing_answer(FakeListingChain(result), {"question": "List exactly 3."}, "", context)

    assert evaluation["mode"] == "listing"
    assert evaluation["answer_correct"] is True
    assert evaluation["grounded"] is True
    assert evaluation["faithful"] is True


def test_listing_answer_fails_duplicate_title() -> None:
    result = ListingJudgeResult(
        returned_titles=["First", "First", "Second"],
        title_count_ok=True,
        titles_relevant=True,
        faithful=True,
        score=0.5,
        title_relevance=[
            TitleRelevance(title="First", relevant=True, rationale="Relevant."),
            TitleRelevance(title="First", relevant=True, rationale="Relevant."),
            TitleRelevance(title="Second", relevant=True, rationale="Relevant."),
        ],
        rationale="Duplicates.",
    )
    context = [{"title": "First"}, {"title": "Second"}]

    evaluation = judge_listing_answer(FakeListingChain(result), {"question": "List exactly 3."}, "", context)

    assert evaluation["answer_correct"] is False
    assert evaluation["titles_distinct"] is False


def test_listing_answer_fails_title_not_in_retrieved_context() -> None:
    result = ListingJudgeResult(
        returned_titles=["First", "Second", "Missing"],
        title_count_ok=True,
        titles_relevant=True,
        faithful=True,
        score=0.5,
        title_relevance=[
            TitleRelevance(title="First", relevant=True, rationale="Relevant."),
            TitleRelevance(title="Second", relevant=True, rationale="Relevant."),
            TitleRelevance(title="Missing", relevant=True, rationale="Relevant."),
        ],
        rationale="One title missing.",
    )
    context = [{"title": "First"}, {"title": "Second"}]

    evaluation = judge_listing_answer(FakeListingChain(result), {"question": "List exactly 3."}, "", context)

    assert evaluation["answer_correct"] is False
    assert evaluation["titles_in_retrieved_context"] is False


def test_listing_answer_fails_irrelevant_title() -> None:
    result = ListingJudgeResult(
        returned_titles=["First", "Second", "Third"],
        title_count_ok=True,
        titles_relevant=False,
        faithful=True,
        score=0.5,
        title_relevance=[
            TitleRelevance(title="First", relevant=True, rationale="Relevant."),
            TitleRelevance(title="Second", relevant=False, rationale="Irrelevant."),
            TitleRelevance(title="Third", relevant=True, rationale="Relevant."),
        ],
        rationale="One title irrelevant.",
    )
    context = [{"title": "First"}, {"title": "Second"}, {"title": "Third"}]

    evaluation = judge_listing_answer(FakeListingChain(result), {"question": "List exactly 3."}, "", context)

    assert evaluation["answer_correct"] is False
    assert evaluation["titles_relevant"] is False


def test_standard_judge_uses_faithful_as_grounded_alias() -> None:
    result = JudgeResult(answer_correct=True, faithful=False, score=0.8, rationale="Unsupported extra claim.")

    evaluation = judge_answer(
        FakeStandardChain(result),
        {"question": "Q", "expected_answer": "A", "evidence_quote": "A"},
        "A plus unsupported detail.",
        [{"title": "Title", "chunk": "A"}],
    )

    assert evaluation["mode"] == "standard"
    assert evaluation["answer_correct"] is True
    assert evaluation["faithful"] is False
    assert evaluation["grounded"] is False
