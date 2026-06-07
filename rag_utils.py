from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.documents import Document

from medium_rag.chunking import approx_token_count, build_chunk_text as _build_article_chunk_text, split_articles
from medium_rag.config import RagExperimentConfig, load_experiment_config
from medium_rag.data import load_medium_articles
from medium_rag.embeddings import build_embeddings as _build_embeddings
from medium_rag.generation import build_llm as _build_llm
from medium_rag.generation import build_user_prompt, format_context, message_text
from medium_rag.indexing import index_dataset
from medium_rag.pipeline import RagPipeline
from medium_rag.types import Article, RetrievedContext
from medium_rag.vectorstores import PineconeVectorStore

_default_pipeline: RagPipeline | None = None
_default_pipeline_config_path: str | None = None


def get_default_config() -> RagExperimentConfig:
    return load_experiment_config()


def get_default_pipeline() -> RagPipeline:
    global _default_pipeline, _default_pipeline_config_path
    config = get_default_config()
    config_key = config.model_dump_json()
    if _default_pipeline is None or _default_pipeline_config_path != config_key:
        _default_pipeline = RagPipeline(config)
        _default_pipeline_config_path = config_key
    return _default_pipeline


def build_pinecone_index(index_name: str | None = None):
    config = get_default_config()
    if index_name:
        data = config.model_dump()
        data["pinecone"]["index_name"] = index_name
        config = RagExperimentConfig.model_validate(data)
    store = PineconeVectorStore(config.pinecone, embedding_dimensions=config.embedding.dimensions)
    return store.index


def pinecone_namespace() -> str:
    return get_default_config().pinecone.namespace


def build_embeddings(config: RagExperimentConfig | None = None):
    return _build_embeddings((config or get_default_config()).embedding)


def build_llm(config: RagExperimentConfig | None = None):
    return _build_llm((config or get_default_config()).generation)


def load_sample_documents(path: str | Path | None = None) -> list[Document]:
    config = get_default_config()
    articles = load_medium_articles(path or config.dataset.path)
    return [
        Document(
            page_content=_build_article_chunk_text(article),
            metadata={
                "article_id": article.article_id,
                "title": article.title,
                "url": article.url,
                "authors": article.authors,
                "timestamp": article.timestamp,
                "tags": article.tags,
            },
        )
        for article in articles
    ]


def build_chunk_text(*, title: str, authors: str = "", tags: str = "", text: str) -> str:
    return _build_article_chunk_text(Article(article_id="", title=title, text=text, authors=authors, tags=tags))


def split_documents(docs: list[Document]) -> list[Document]:
    config = get_default_config()
    articles = [
        Article(
            article_id=str(doc.metadata.get("article_id", index)),
            title=str(doc.metadata.get("title", "")),
            text=doc.page_content,
            url=str(doc.metadata.get("url", "")),
            authors=str(doc.metadata.get("authors", "")),
            timestamp=str(doc.metadata.get("timestamp", "")),
            tags=str(doc.metadata.get("tags", "")),
        )
        for index, doc in enumerate(docs)
    ]
    chunks = split_articles(articles, config.chunking)
    return [
        Document(
            page_content=chunk.text,
            metadata={
                "article_id": chunk.article_id,
                "title": chunk.title,
                "url": chunk.url,
                "authors": chunk.authors,
                "timestamp": chunk.timestamp,
                "tags": chunk.tags,
                "chunk_id": chunk.chunk_id,
                "chunk_index": chunk.chunk_index,
            },
        )
        for chunk in chunks
    ]


def index_sample_dataset(force: bool = False) -> int:
    return index_dataset(get_default_config(), force=force)


def retrieve_context(question: str, top_k: int | None = None) -> list[RetrievedContext]:
    pipeline = get_default_pipeline()
    if top_k is None or top_k == pipeline.config.retrieval.top_k:
        return pipeline.retriever.retrieve(question)
    data = pipeline.config.model_dump()
    data["retrieval"]["top_k"] = top_k
    config = RagExperimentConfig.model_validate(data)
    return RagPipeline(config).retriever.retrieve(question)


def answer_question(
    question: str,
    top_k: int | None = None,
    *,
    log_trace: bool = True,
    trace_source: str | None = None,
    eval_run_id: str | None = None,
    expected_answer: str | None = None,
    evaluation: dict[str, Any] | None = None,
) -> dict[str, Any]:
    pipeline = get_default_pipeline()
    if top_k is not None and top_k != pipeline.config.retrieval.top_k:
        data = pipeline.config.model_dump()
        data["retrieval"]["top_k"] = top_k
        pipeline = RagPipeline(RagExperimentConfig.model_validate(data))
    return pipeline.answer_question(
        question,
        log_trace=log_trace,
        trace_source=trace_source,
        eval_run_id=eval_run_id,
        expected_answer=expected_answer,
        evaluation=evaluation,
    )


__all__ = [
    "RetrievedContext",
    "answer_question",
    "approx_token_count",
    "build_chunk_text",
    "build_embeddings",
    "build_llm",
    "build_pinecone_index",
    "build_user_prompt",
    "format_context",
    "get_default_config",
    "get_default_pipeline",
    "index_sample_dataset",
    "load_sample_documents",
    "message_text",
    "pinecone_namespace",
    "retrieve_context",
    "split_documents",
]
