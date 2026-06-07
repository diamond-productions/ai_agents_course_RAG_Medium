from __future__ import annotations

from medium_rag.config import PineconeConfig
from medium_rag.types import Chunk
from medium_rag.vectorstores.pinecone_store import PineconeVectorStore


def _store(batch_size: int = 2, max_batch_bytes: int = 1_800_000) -> PineconeVectorStore:
    store = object.__new__(PineconeVectorStore)
    store.config = PineconeConfig(
        index_name="idx",
        namespace="ns",
        cloud="aws",
        region="us-east-1",
        metric="cosine",
        batch_size=batch_size,
        max_batch_bytes=max_batch_bytes,
    )
    return store


def _record(text: str) -> dict:
    return {"id": text, "values": [0.1, 0.2], "metadata": {"text": text}}


def test_iter_upsert_batches_splits_by_count() -> None:
    batches = list(_store(batch_size=2).iter_upsert_batches([_record("a"), _record("b"), _record("c")]))
    assert [len(batch) for batch in batches] == [2, 1]


def test_iter_upsert_batches_splits_by_byte_limit() -> None:
    records = [_record("a" * 100), _record("b" * 100), _record("c" * 100)]
    batches = list(_store(batch_size=10, max_batch_bytes=180).iter_upsert_batches(records))
    assert [len(batch) for batch in batches] == [1, 1, 1]


def test_build_records_includes_dataset_and_text_metadata() -> None:
    store = _store()
    chunk = Chunk(
        id="medium-full:1:0000",
        article_id="1",
        chunk_id="1-0000",
        chunk_index=0,
        text="chunk text",
        title="Title",
    )
    record = store.build_records([chunk], [[0.1, 0.2]], dataset_name="medium-english-50mb")[0]
    assert record["metadata"]["dataset"] == "medium-english-50mb"
    assert record["metadata"]["text"] == "chunk text"
    assert record["metadata"]["title"] == "Title"
