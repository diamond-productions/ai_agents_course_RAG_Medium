from __future__ import annotations

import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

from api.config import (
    CHAT_INPUT_COST_PER_1M_TOKENS,
    CHAT_MODEL,
    CHAT_OUTPUT_COST_PER_1M_TOKENS,
    CHUNK_SIZE,
    DEFAULT_BASE_URL,
    EMBEDDING_COST_PER_1M_TOKENS,
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    MMR_ENABLED,
    MMR_LAMBDA,
    OVERLAP_RATIO,
    PINECONE_INDEX_NAME,
    PINECONE_NAMESPACE,
    RETRIEVAL_CANDIDATE_K,
    SAMPLE_DATA_PATH,
    SYSTEM_PROMPT,
    TOP_K,
)
from rag_logging import log_eval_run, log_rag_trace

load_dotenv()

PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
TEXT_KEY = "text"
DATASET_NAME = "medium-300-sample"


@dataclass(frozen=True)
class RetrievedContext:
    article_id: str
    title: str
    chunk: str
    score: float
    url: str = ""
    authors: str = ""
    timestamp: str = ""
    tags: str = ""

    def as_api_dict(self) -> dict[str, Any]:
        return {
            "article_id": self.article_id,
            "title": self.title,
            "chunk": self.chunk,
            "score": self.score,
            "url": self.url,
            "authors": self.authors,
            "timestamp": self.timestamp,
            "tags": self.tags,
        }


def _require_env(name: str) -> str:
    value = os.getenv(name)
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _env_or_default(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value in (None, "") else value


def _bool_env(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    return raw.strip().casefold() in {"1", "true", "yes", "on"}


def _float_env(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw in (None, ""):
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(value, maximum))


def pinecone_namespace() -> str:
    return _env_or_default("PINECONE_NAMESPACE", PINECONE_NAMESPACE)


def build_embeddings() -> OpenAIEmbeddings:
    return OpenAIEmbeddings(
        model=os.getenv("EMBEDDING_MODEL", EMBEDDING_MODEL),
        dimensions=int(os.getenv("EMBEDDING_DIMENSIONS", str(EMBEDDING_DIMENSIONS))),
        api_key=_require_env("LLMOD_API_KEY"),
        base_url=os.getenv("LLMOD_API_BASE", DEFAULT_BASE_URL),
    )


def build_llm() -> ChatOpenAI:
    return ChatOpenAI(
        model=os.getenv("CHAT_MODEL", CHAT_MODEL),
        api_key=_require_env("LLMOD_API_KEY"),
        base_url=os.getenv("LLMOD_API_BASE", DEFAULT_BASE_URL),
    )


def build_pinecone_index(index_name: str | None = None):
    pc = Pinecone(api_key=_require_env("PINECONE_API_KEY"))
    name = index_name or _env_or_default("PINECONE_INDEX_NAME", PINECONE_INDEX_NAME)
    if not pc.has_index(name):
        pc.create_index(
            name=name,
            vector_type="dense",
            dimension=int(os.getenv("EMBEDDING_DIMENSIONS", str(EMBEDDING_DIMENSIONS))),
            metric="cosine",
            spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
            deletion_protection="disabled",
            tags={"dataset": DATASET_NAME},
        )
    return pc.Index(name)


def build_chunk_text(*, title: str, authors: str = "", tags: str = "", text: str) -> str:
    metadata_lines = []
    if title:
        metadata_lines.append(f"Title: {title}")
    if authors:
        metadata_lines.append(f"Authors: {authors}")
    if tags:
        metadata_lines.append(f"Tags: {tags}")

    body = text.strip()
    if metadata_lines:
        return "\n".join(metadata_lines) + f"\n\nPassage:\n{body}"
    return body


def load_sample_documents(path: str = SAMPLE_DATA_PATH) -> list[Document]:
    docs: list[Document] = []
    with Path(path).open(newline="", encoding="utf-8") as handle:
        for article_id, row in enumerate(csv.DictReader(handle)):
            title = row.get("title", "").strip()
            text = row.get("text", "").strip()
            if not text:
                continue

            authors = row.get("authors", "").strip()
            tags = row.get("tags", "").strip()
            page_content = build_chunk_text(title=title, authors=authors, tags=tags, text=text)
            docs.append(
                Document(
                    page_content=page_content,
                    metadata={
                        "article_id": str(article_id),
                        "title": title,
                        "url": row.get("url", ""),
                        "authors": authors,
                        "timestamp": row.get("timestamp", ""),
                        "tags": tags,
                    },
                )
            )
    return docs


def split_documents(docs: list[Document]) -> list[Document]:
    splitter = RecursiveCharacterTextSplitter(
        length_function=approx_token_count,
        chunk_size=CHUNK_SIZE,
        chunk_overlap=int(CHUNK_SIZE * OVERLAP_RATIO),
        separators=["\n\n", "\n", " ", ""],
    )
    splits = splitter.split_documents(docs)
    article_chunk_counts: dict[str, int] = {}
    for doc in splits:
        article_id = str(doc.metadata["article_id"])
        chunk_index = article_chunk_counts.get(article_id, 0)
        article_chunk_counts[article_id] = chunk_index + 1
        doc.metadata["chunk_index"] = chunk_index
        doc.metadata["chunk_id"] = f"{article_id}-{chunk_index:04d}"
    return splits


def approx_token_count(text: str) -> int:
    # Conservative local approximation that avoids downloading tokenizer files.
    return max(1, len(text) // 4)


def index_sample_dataset(force: bool = False) -> int:
    """Index only data/medium-300-sample.csv into Pinecone."""
    index = build_pinecone_index()
    namespace = pinecone_namespace()
    stats = index.describe_index_stats()
    existing_count = _namespace_count(stats, namespace)
    if existing_count and not force:
        return existing_count

    docs = split_documents(load_sample_documents())
    embeddings = build_embeddings()
    vectors = embeddings.embed_documents([doc.page_content for doc in docs])
    records = []
    for doc, vector in zip(docs, vectors, strict=True):
        article_id = doc.metadata["article_id"]
        chunk_index = int(doc.metadata["chunk_index"])
        records.append(
            {
                "id": f"medium-300:{article_id}:{chunk_index:04d}",
                "values": vector,
                "metadata": {
                    **doc.metadata,
                    TEXT_KEY: doc.page_content,
                    "dataset": DATASET_NAME,
                },
            }
        )

    for start in range(0, len(records), 100):
        batch = records[start : start + 100]
        kwargs: dict[str, Any] = {"vectors": batch}
        if namespace:
            kwargs["namespace"] = namespace
        index.upsert(**kwargs)
    return len(records)


def _namespace_count(stats: Any, namespace: str) -> int:
    stats_dict = stats.to_dict() if hasattr(stats, "to_dict") else dict(stats)
    namespaces = stats_dict.get("namespaces") or {}
    if namespace:
        return int((namespaces.get(namespace) or {}).get("vector_count", 0))
    if "" in namespaces:
        return int((namespaces.get("") or {}).get("vector_count", 0))
    return int(stats_dict.get("total_vector_count") or 0)


def _match_metadata(match: Any) -> dict[str, Any]:
    metadata = match.metadata if hasattr(match, "metadata") else match.get("metadata", {})
    return metadata or {}


def _match_values(match: Any) -> list[float]:
    values = match.values if hasattr(match, "values") else match.get("values", [])
    return [float(value) for value in values or []]


def _article_key(match: Any) -> str:
    metadata = _match_metadata(match)
    return str(metadata.get("article_id") or metadata.get("title") or metadata.get("id") or "")


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right, strict=True))
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if not left_norm or not right_norm:
        return 0.0
    return dot / (left_norm * right_norm)


def _dedupe_matches_by_article(matches: list[Any], limit: int) -> list[Any]:
    deduped: list[Any] = []
    seen_articles: set[str] = set()
    for match in matches:
        article_key = _article_key(match)
        if article_key and article_key in seen_articles:
            continue
        if article_key:
            seen_articles.add(article_key)
        deduped.append(match)
        if len(deduped) >= limit:
            break
    return deduped


def _select_mmr_matches(
    *,
    matches: list[Any],
    query_vector: list[float],
    limit: int,
    lambda_mult: float,
) -> list[Any]:
    candidates = [match for match in matches if _match_values(match)]
    if not candidates:
        return _dedupe_matches_by_article(matches, limit)

    selected: list[Any] = []
    selected_vectors: list[list[float]] = []
    selected_articles: set[str] = set()
    remaining = candidates.copy()

    while remaining and len(selected) < limit:
        best_match = None
        best_score = float("-inf")
        for match in remaining:
            article_key = _article_key(match)
            if article_key and article_key in selected_articles:
                continue

            vector = _match_values(match)
            relevance = _cosine_similarity(query_vector, vector)
            redundancy = max(
                (_cosine_similarity(vector, selected_vector) for selected_vector in selected_vectors),
                default=0.0,
            )
            mmr_score = lambda_mult * relevance - (1.0 - lambda_mult) * redundancy
            if mmr_score > best_score:
                best_score = mmr_score
                best_match = match

        if best_match is None:
            break

        remaining.remove(best_match)
        selected.append(best_match)
        selected_vectors.append(_match_values(best_match))
        article_key = _article_key(best_match)
        if article_key:
            selected_articles.add(article_key)

    if len(selected) < limit:
        selected_ids = {id(match) for match in selected}
        for match in _dedupe_matches_by_article(matches, limit):
            if id(match) in selected_ids:
                continue
            selected.append(match)
            if len(selected) >= limit:
                break

    return selected[:limit]


def retrieve_context(question: str, top_k: int = TOP_K) -> list[RetrievedContext]:
    embeddings = build_embeddings()
    index = build_pinecone_index()
    namespace = pinecone_namespace()
    candidate_k = max(top_k, int(os.getenv("RETRIEVAL_CANDIDATE_K", str(RETRIEVAL_CANDIDATE_K))))
    mmr_enabled = _bool_env("MMR_ENABLED", MMR_ENABLED)
    mmr_lambda = _clamp(_float_env("MMR_LAMBDA", MMR_LAMBDA), 0.0, 1.0)
    query_vector = embeddings.embed_query(question)
    kwargs: dict[str, Any] = {
        "vector": query_vector,
        "top_k": candidate_k,
        "include_metadata": True,
        "include_values": mmr_enabled,
    }
    if namespace:
        kwargs["namespace"] = namespace
    result = index.query(**kwargs)
    matches = result.matches if hasattr(result, "matches") else result.get("matches", [])
    matches = list(matches)
    if mmr_enabled:
        matches = _select_mmr_matches(
            matches=matches,
            query_vector=[float(value) for value in query_vector],
            limit=top_k,
            lambda_mult=mmr_lambda,
        )
    else:
        matches = _dedupe_matches_by_article(matches, top_k)

    contexts: list[RetrievedContext] = []
    for match in matches:
        metadata = _match_metadata(match)
        score = match.score if hasattr(match, "score") else match.get("score", 0.0)
        article_id = metadata.get("article_id") or metadata.get("id") or ""
        contexts.append(
            RetrievedContext(
                article_id=str(article_id),
                title=str(metadata.get("title", "")),
                chunk=str(metadata.get(TEXT_KEY, "")),
                score=float(score or 0.0),
                url=str(metadata.get("url", "")),
                authors=str(metadata.get("authors", "")),
                timestamp=str(metadata.get("timestamp", "")),
                tags=str(metadata.get("tags", "")),
            )
        )
    return contexts


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


def _message_text(content: Any) -> str:
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


def _normalize_usage(message: Any) -> dict[str, int | None]:
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


def _estimate_cost(usage: dict[str, int | None], question: str) -> dict[str, Any]:
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


def model_metadata(message: Any, top_k: int) -> dict[str, Any]:
    response_metadata = getattr(message, "response_metadata", None) or {}
    return {
        "chat_model": os.getenv("CHAT_MODEL", CHAT_MODEL),
        "embedding_model": os.getenv("EMBEDDING_MODEL", EMBEDDING_MODEL),
        "embedding_dimensions": int(os.getenv("EMBEDDING_DIMENSIONS", str(EMBEDDING_DIMENSIONS))),
        "api_base": os.getenv("LLMOD_API_BASE", DEFAULT_BASE_URL),
        "pinecone_index": _env_or_default("PINECONE_INDEX_NAME", PINECONE_INDEX_NAME),
        "pinecone_namespace": pinecone_namespace(),
        "chunk_size": CHUNK_SIZE,
        "overlap_ratio": OVERLAP_RATIO,
        "retrieval_candidate_k": int(os.getenv("RETRIEVAL_CANDIDATE_K", str(RETRIEVAL_CANDIDATE_K))),
        "mmr_enabled": _bool_env("MMR_ENABLED", MMR_ENABLED),
        "mmr_lambda": _clamp(_float_env("MMR_LAMBDA", MMR_LAMBDA), 0.0, 1.0),
        "top_k": top_k,
        "finish_reason": response_metadata.get("finish_reason"),
        "system_fingerprint": response_metadata.get("system_fingerprint"),
    }


def answer_question(
    question: str,
    top_k: int = TOP_K,
    *,
    log_trace: bool = True,
    trace_source: str | None = None,
    eval_run_id: str | None = None,
    expected_answer: str | None = None,
    evaluation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    context = retrieve_context(question, top_k=top_k)
    user_prompt = build_user_prompt(question, context)
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("user", "{user_prompt}"),
        ]
    )
    chain = prompt | build_llm()
    message = chain.invoke({"user_prompt": user_prompt})
    usage = _normalize_usage(message)
    result = {
        "response": _message_text(message.content),
        "context": [item.as_api_dict() for item in context],
        "Augmented_prompt": {
            "System": SYSTEM_PROMPT,
            "User": user_prompt,
        },
        "metadata": model_metadata(message, top_k=top_k),
        "usage": usage,
        "cost": _estimate_cost(usage, question),
    }
    if log_trace:
        result["trace_log_path"] = str(log_rag_trace(question=question, result=result, top_k=top_k, source=trace_source))
    if eval_run_id or expected_answer is not None or evaluation is not None:
        result["eval_log_path"] = str(
            log_eval_run(
                question=question,
                result=result,
                top_k=top_k,
                run_id=eval_run_id,
                expected_answer=expected_answer,
                evaluation=evaluation,
                source=trace_source,
            )
        )
    return result
