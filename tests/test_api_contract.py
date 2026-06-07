from __future__ import annotations

from typing import Any


def test_prompt_api_contract_filters_extra_fields() -> None:
    from api.index import api_prompt_contract

    result: dict[str, Any] = {
        "response": "Answer",
        "context": [
            {
                "article_id": "1234",
                "title": "Sample article title",
                "chunk": "article chunk retrieved",
                "score": 0.1234,
                "url": "https://example.test",
            }
        ],
        "Augmented_prompt": {"System": "S", "User": "U"},
        "metadata": {"extra": True},
        "usage": {"total_tokens": 1},
        "cost": {"estimated_total_cost": 0.0},
    }

    assert api_prompt_contract(result) == {
        "response": "Answer",
        "context": [
            {
                "article_id": "1234",
                "title": "Sample article title",
                "chunk": "article chunk retrieved",
                "score": 0.1234,
            }
        ],
        "Augmented_prompt": {"System": "S", "User": "U"},
    }
