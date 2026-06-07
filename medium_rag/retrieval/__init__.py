from __future__ import annotations

from medium_rag.config import RagExperimentConfig
from medium_rag.embeddings import build_embeddings
from medium_rag.retrieval.base import Retriever
from medium_rag.retrieval.dense import DenseRetriever
from medium_rag.retrieval.mmr import DenseMmrRetriever
from medium_rag.vectorstores import PineconeVectorStore


def build_retriever(config: RagExperimentConfig) -> Retriever:
    embeddings = build_embeddings(config.embedding)
    store = PineconeVectorStore(config.pinecone, embedding_dimensions=config.embedding.dimensions)
    if config.retrieval.strategy == "dense":
        return DenseRetriever(embeddings, store, config.retrieval)
    if config.retrieval.strategy == "dense_mmr":
        return DenseMmrRetriever(embeddings, store, config.retrieval)
    raise ValueError(f"Unknown retrieval strategy: {config.retrieval.strategy}")


__all__ = ["DenseMmrRetriever", "DenseRetriever", "Retriever", "build_retriever"]
