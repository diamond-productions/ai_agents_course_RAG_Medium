from __future__ import annotations

from medium_rag.config import load_experiment_config
from medium_rag.types import GenerationResult, RetrievedContext


class FakeRetriever:
    def retrieve(self, question: str) -> list[RetrievedContext]:
        return [RetrievedContext(article_id="1", title="Title", chunk="Chunk", score=0.9)]


def test_pipeline_answer_question_contract(monkeypatch) -> None:
    import medium_rag.pipeline as pipeline_module

    def fake_build_retriever(config):
        return FakeRetriever()

    def fake_generate_answer(question, context, config):
        return GenerationResult(
            response="Answer",
            augmented_prompt={"System": "S", "User": "U"},
            usage={"input_tokens": 1, "output_tokens": 1, "total_tokens": 2},
            cost={"estimated_total_cost": 0.0},
            metadata={"experiment_name": config.experiment_name},
        )

    monkeypatch.setattr(pipeline_module, "build_retriever", fake_build_retriever)
    monkeypatch.setattr(pipeline_module, "generate_answer", fake_generate_answer)

    pipeline = pipeline_module.RagPipeline(load_experiment_config("configs/experiments/dense_mmr.yaml"))
    result = pipeline.answer_question("question", log_trace=False)

    assert set(result) == {"response", "context", "Augmented_prompt", "metadata", "usage", "cost"}
    assert result["response"] == "Answer"
    assert result["context"][0]["article_id"] == "1"
