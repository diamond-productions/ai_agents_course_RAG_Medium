from __future__ import annotations

from medium_rag.chunking import split_articles
from medium_rag.config import RagExperimentConfig
from medium_rag.data import load_medium_articles
from medium_rag.embeddings import build_embeddings
from medium_rag.vectorstores import PineconeVectorStore


def index_dataset(config: RagExperimentConfig, force: bool = False) -> int:
    articles = load_medium_articles(config.dataset.path)
    chunks = split_articles(articles, config.chunking)
    embeddings = build_embeddings(config.embedding)
    vectors = embeddings.embed_documents([chunk.text for chunk in chunks])
    store = PineconeVectorStore(config.pinecone, embedding_dimensions=config.embedding.dimensions)
    return store.upsert_chunks(chunks, vectors, force=force)
