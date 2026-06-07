from __future__ import annotations

from typing import Any

from medium_rag.config import RagExperimentConfig
from medium_rag.generation import generate_answer
from medium_rag.logging import log_eval_run, log_rag_trace
from medium_rag.retrieval import build_retriever


class RagPipeline:
    def __init__(self, config: RagExperimentConfig):
        self.config = config
        self.retriever = build_retriever(config)

    def answer_question(
        self,
        question: str,
        *,
        log_trace: bool = True,
        trace_source: str | None = None,
        eval_run_id: str | None = None,
        expected_answer: str | None = None,
        evaluation: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        context = self.retriever.retrieve(question)
        generation = generate_answer(question, context, self.config)
        result = {
            "response": generation.response,
            "context": [item.as_api_dict() for item in context],
            "Augmented_prompt": generation.augmented_prompt,
            "metadata": generation.metadata,
            "usage": generation.usage,
            "cost": generation.cost,
        }
        if log_trace:
            result["trace_log_path"] = str(
                log_rag_trace(question=question, result=result, config=self.config, source=trace_source)
            )
        if eval_run_id or expected_answer is not None or evaluation is not None:
            result["eval_log_path"] = str(
                log_eval_run(
                    question=question,
                    result=result,
                    config=self.config,
                    run_id=eval_run_id,
                    expected_answer=expected_answer,
                    evaluation=evaluation,
                    source=trace_source,
                )
            )
        return result
