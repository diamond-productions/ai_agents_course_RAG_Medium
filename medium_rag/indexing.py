from __future__ import annotations

import sys
from dataclasses import dataclass

from medium_rag.chunking import approx_token_count, split_article, vector_id_prefix_for_dataset
from medium_rag.config import RagExperimentConfig
from medium_rag.data import iter_medium_articles
from medium_rag.embeddings import build_embeddings
from medium_rag.types import Chunk
from medium_rag.vectorstores import PineconeVectorStore


@dataclass(frozen=True)
class IndexingResult:
    dataset_name: str
    index_name: str
    namespace: str
    articles_seen: int
    chunks_created: int
    vectors_upserted: int
    namespace_count_before: int
    namespace_count_after: int
    skipped_existing_namespace: bool


@dataclass(frozen=True)
class IndexingEstimate:
    dataset_name: str
    dataset_path: str
    index_name: str
    namespace: str
    articles_seen: int
    chunks_created: int
    estimated_tokens: int


def _vector_id_prefix(config: RagExperimentConfig) -> str:
    return vector_id_prefix_for_dataset(config.dataset.name, config.dataset.vector_id_prefix)


def estimate_dataset(config: RagExperimentConfig, limit_articles: int | None = None) -> IndexingEstimate:
    articles_seen = 0
    chunks_created = 0
    estimated_tokens = 0
    vector_id_prefix = _vector_id_prefix(config)
    for article in iter_medium_articles(config.dataset.path):
        if limit_articles is not None and articles_seen >= limit_articles:
            break
        articles_seen += 1
        chunks = split_article(article, config.chunking, vector_id_prefix)
        chunks_created += len(chunks)
        estimated_tokens += sum(approx_token_count(chunk.text) for chunk in chunks)
    return IndexingEstimate(
        dataset_name=config.dataset.name,
        dataset_path=config.dataset.path,
        index_name=config.pinecone.index_name,
        namespace=config.pinecone.namespace,
        articles_seen=articles_seen,
        chunks_created=chunks_created,
        estimated_tokens=estimated_tokens,
    )


def _print_progress(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def index_dataset(
    config: RagExperimentConfig,
    force: bool = False,
    limit_articles: int | None = None,
    progress_every: int = 500,
) -> IndexingResult:
    vector_id_prefix = _vector_id_prefix(config)
    store = PineconeVectorStore(config.pinecone, embedding_dimensions=config.embedding.dimensions)
    namespace_count_before = store.namespace_count()
    if namespace_count_before and not force:
        return IndexingResult(
            dataset_name=config.dataset.name,
            index_name=config.pinecone.index_name,
            namespace=config.pinecone.namespace,
            articles_seen=0,
            chunks_created=0,
            vectors_upserted=0,
            namespace_count_before=namespace_count_before,
            namespace_count_after=namespace_count_before,
            skipped_existing_namespace=True,
        )

    embeddings = build_embeddings(config.embedding)
    pending_chunks: list[Chunk] = []
    articles_seen = 0
    chunks_created = 0
    vectors_upserted = 0
    last_progress_articles = 0

    def flush_pending() -> None:
        nonlocal vectors_upserted
        if not pending_chunks:
            return
        vectors = embeddings.embed_documents([chunk.text for chunk in pending_chunks])
        records = store.build_records(pending_chunks, vectors, config.dataset.name)
        vectors_upserted += store.upsert_records(records)
        pending_chunks.clear()

    for article in iter_medium_articles(config.dataset.path):
        if limit_articles is not None and articles_seen >= limit_articles:
            break
        articles_seen += 1
        chunks = split_article(article, config.chunking, vector_id_prefix)
        chunks_created += len(chunks)
        pending_chunks.extend(chunks)

        while len(pending_chunks) >= config.embedding.batch_size:
            flush_batch = pending_chunks[: config.embedding.batch_size]
            del pending_chunks[: config.embedding.batch_size]
            vectors = embeddings.embed_documents([chunk.text for chunk in flush_batch])
            records = store.build_records(flush_batch, vectors, config.dataset.name)
            vectors_upserted += store.upsert_records(records)

        if progress_every > 0 and articles_seen % progress_every == 0 and articles_seen != last_progress_articles:
            last_progress_articles = articles_seen
            _print_progress(
                f"Indexed progress: articles={articles_seen} chunks={chunks_created} "
                f"vectors_upserted={vectors_upserted}"
            )

    flush_pending()
    namespace_count_after = store.namespace_count()
    return IndexingResult(
        dataset_name=config.dataset.name,
        index_name=config.pinecone.index_name,
        namespace=config.pinecone.namespace,
        articles_seen=articles_seen,
        chunks_created=chunks_created,
        vectors_upserted=vectors_upserted,
        namespace_count_before=namespace_count_before,
        namespace_count_after=namespace_count_after,
        skipped_existing_namespace=False,
    )
