from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

from medium_rag.chunking import approx_token_count
from medium_rag.config import GenerationConfig, RagExperimentConfig
from medium_rag.types import GenerationResult, RetrievedContext

CHAT_INPUT_COST_PER_1M_TOKENS = 0.25
CHAT_OUTPUT_COST_PER_1M_TOKENS = 2.00
EMBEDDING_COST_PER_1M_TOKENS = 0.02


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def build_llm(config: GenerationConfig) -> ChatOpenAI:
    return ChatOpenAI(
        model=config.chat_model,
        api_key=_require_env("LLMOD_API_KEY"),
        base_url=os.getenv("LLMOD_API_BASE") or config.api_base,
    )


def load_system_prompt(path: str | Path) -> str:
    return Path(path).read_text(encoding="utf-8").strip()


def format_context(context: list[RetrievedContext]) -> str:
    blocks = []
    for i, item in enumerate(context, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[{i}] title: {item.title}",
                    f"article_id: {item.article_id}",
                    f"authors: {item.authors}",
                    f"url: {item.url}",
                    f"score: {item.score:.4f}",
                    f"passage: {item.chunk}",
                ]
            )
        )
    return "\n\n".join(blocks)


def build_user_prompt(question: str, context: list[RetrievedContext]) -> str:
    return f"Question:\n{question}\n\nRetrieved context:\n{format_context(context)}"


def message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for item in content:
            if isinstance(item, str):
                parts.append(item)
            elif isinstance(item, dict):
                parts.append(str(item.get("text") or item.get("content") or ""))
            else:
                parts.append(str(item))
        return "\n".join(part for part in parts if part)
    return str(content)


def normalize_usage(message: Any) -> dict[str, int | None]:
    usage = getattr(message, "usage_metadata", None) or {}
    response_metadata = getattr(message, "response_metadata", None) or {}
    token_usage = response_metadata.get("token_usage") or response_metadata.get("usage") or {}
    input_tokens = (
        usage.get("input_tokens")
        or usage.get("prompt_tokens")
        or token_usage.get("input_tokens")
        or token_usage.get("prompt_tokens")
    )
    output_tokens = (
        usage.get("output_tokens")
        or usage.get("completion_tokens")
        or token_usage.get("output_tokens")
        or token_usage.get("completion_tokens")
    )
    total_tokens = usage.get("total_tokens") or token_usage.get("total_tokens")
    if total_tokens is None and input_tokens is not None and output_tokens is not None:
        total_tokens = int(input_tokens) + int(output_tokens)
    return {
        "input_tokens": int(input_tokens) if input_tokens is not None else None,
        "output_tokens": int(output_tokens) if output_tokens is not None else None,
        "total_tokens": int(total_tokens) if total_tokens is not None else None,
    }


def _optional_float_env(name: str, default: float | None = None) -> float | None:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def estimate_cost(usage: dict[str, int | None], question: str) -> dict[str, Any]:
    input_rate = _optional_float_env("CHAT_INPUT_COST_PER_1M_TOKENS", CHAT_INPUT_COST_PER_1M_TOKENS)
    output_rate = _optional_float_env("CHAT_OUTPUT_COST_PER_1M_TOKENS", CHAT_OUTPUT_COST_PER_1M_TOKENS)
    embedding_rate = _optional_float_env("EMBEDDING_COST_PER_1M_TOKENS", EMBEDDING_COST_PER_1M_TOKENS)
    input_cost = None
    output_cost = None
    embedding_cost = None
    if input_rate is not None and usage.get("input_tokens") is not None:
        input_cost = usage["input_tokens"] * input_rate / 1_000_000
    if output_rate is not None and usage.get("output_tokens") is not None:
        output_cost = usage["output_tokens"] * output_rate / 1_000_000
    embedding_tokens = approx_token_count(question)
    if embedding_rate is not None:
        embedding_cost = embedding_tokens * embedding_rate / 1_000_000
    total_cost = None
    if input_cost is not None or output_cost is not None or embedding_cost is not None:
        total_cost = (input_cost or 0.0) + (output_cost or 0.0) + (embedding_cost or 0.0)
    return {
        "currency": "USD",
        "chat_input_cost_per_1m_tokens": input_rate,
        "chat_output_cost_per_1m_tokens": output_rate,
        "embedding_cost_per_1m_tokens": embedding_rate,
        "estimated_embedding_tokens": embedding_tokens,
        "estimated_chat_input_cost": input_cost,
        "estimated_chat_output_cost": output_cost,
        "estimated_embedding_cost": embedding_cost,
        "estimated_total_cost": total_cost,
        "note": "Embedding cost is estimated from the query tokens only; indexing costs are not included.",
    }


def model_metadata(message: Any, config: RagExperimentConfig) -> dict[str, Any]:
    response_metadata = getattr(message, "response_metadata", None) or {}
    return {
        "experiment_name": config.experiment_name,
        "chat_model": config.generation.chat_model,
        "embedding_model": config.embedding.model,
        "embedding_dimensions": config.embedding.dimensions,
        "api_base": os.getenv("LLMOD_API_BASE") or config.generation.api_base,
        "pinecone_index": config.pinecone.index_name,
        "pinecone_namespace": config.pinecone.namespace,
        "chunk_size": config.chunking.chunk_size,
        "overlap_ratio": config.chunking.overlap_ratio,
        "tokenizer": config.chunking.tokenizer,
        "retrieval_strategy": config.retrieval.strategy,
        "retrieval_candidate_k": config.retrieval.candidate_k,
        "top_k": config.retrieval.top_k,
        "mmr_enabled": config.retrieval.mmr_enabled,
        "mmr_lambda": config.retrieval.mmr_lambda,
        "finish_reason": response_metadata.get("finish_reason"),
        "system_fingerprint": response_metadata.get("system_fingerprint"),
    }


def generate_answer(question: str, context: list[RetrievedContext], config: RagExperimentConfig) -> GenerationResult:
    system_prompt = load_system_prompt(config.generation.system_prompt_path)
    user_prompt = build_user_prompt(question, context)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("user", "{user_prompt}"),
        ]
    )
    message = (prompt | build_llm(config.generation)).invoke({"user_prompt": user_prompt})
    usage = normalize_usage(message)
    return GenerationResult(
        response=message_text(message.content),
        augmented_prompt={"System": system_prompt, "User": user_prompt},
        usage=usage,
        cost=estimate_cost(usage, question),
        metadata=model_metadata(message, config),
    )
