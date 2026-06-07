from __future__ import annotations

from pathlib import Path
from typing import Any

from medium_rag.config import load_experiment_config
from medium_rag.indexing import estimate_dataset, index_dataset


def _write_csv(path: Path) -> None:
    path.write_text(
        "title,text,url,authors,timestamp,tags\n"
        "First,one two three four five six seven eight nine ten,http://one,Author,2020,Tag\n"
        "Second,alpha beta gamma delta epsilon zeta eta theta iota kappa,http://two,Author,2021,Tag\n",
        encoding="utf-8",
    )


def _test_config(csv_path: Path):
    config = load_experiment_config("configs/experiments/full_dense_mmr_chunk512_overlap015_lambda075.yaml")
    data = config.model_dump()
    data["dataset"]["path"] = str(csv_path)
    data["chunking"]["chunk_size"] = 8
    data["chunking"]["overlap_ratio"] = 0.0
    data["chunking"]["tokenizer"] = "approx"
    data["embedding"]["batch_size"] = 2
    data["pinecone"]["batch_size"] = 2
    return type(config).model_validate(data)


def test_estimate_dataset_does_not_call_live_services(tmp_path: Path) -> None:
    csv_path = tmp_path / "articles.csv"
    _write_csv(csv_path)
    estimate = estimate_dataset(_test_config(csv_path))
    assert estimate.dataset_name == "medium-english-50mb"
    assert estimate.articles_seen == 2
    assert estimate.chunks_created >= 2
    assert estimate.estimated_tokens > 0


def test_index_dataset_skips_existing_namespace(monkeypatch, tmp_path: Path) -> None:
    csv_path = tmp_path / "articles.csv"
    _write_csv(csv_path)

    class FakeStore:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def namespace_count(self) -> int:
            return 10

    def fail_build_embeddings(config):  # pragma: no cover - should not be called
        raise AssertionError("embeddings should not be built when namespace exists")

    import medium_rag.indexing as indexing_module

    monkeypatch.setattr(indexing_module, "PineconeVectorStore", FakeStore)
    monkeypatch.setattr(indexing_module, "build_embeddings", fail_build_embeddings)
    result = index_dataset(_test_config(csv_path), force=False)
    assert result.skipped_existing_namespace is True
    assert result.namespace_count_before == 10
    assert result.vectors_upserted == 0


def test_index_dataset_streams_embedding_batches(monkeypatch, tmp_path: Path) -> None:
    csv_path = tmp_path / "articles.csv"
    _write_csv(csv_path)
    embedded_batches: list[int] = []
    upserted_ids: list[str] = []

    class FakeEmbeddings:
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            embedded_batches.append(len(texts))
            return [[float(i)] for i, _ in enumerate(texts)]

    class FakeStore:
        count = 0

        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

        def namespace_count(self) -> int:
            return self.count

        def build_records(self, chunks, vectors, dataset_name: str):
            return [
                {
                    "id": chunk.id,
                    "values": vector,
                    "metadata": {"dataset": dataset_name, "text": chunk.text},
                }
                for chunk, vector in zip(chunks, vectors, strict=True)
            ]

        def upsert_records(self, records) -> int:
            upserted_ids.extend(record["id"] for record in records)
            self.count += len(records)
            return len(records)

    import medium_rag.indexing as indexing_module

    monkeypatch.setattr(indexing_module, "PineconeVectorStore", FakeStore)
    monkeypatch.setattr(indexing_module, "build_embeddings", lambda config: FakeEmbeddings())
    result = index_dataset(_test_config(csv_path), force=True, progress_every=0)
    assert result.articles_seen == 2
    assert result.vectors_upserted == len(upserted_ids)
    assert embedded_batches
    assert max(embedded_batches) <= 2
    assert all(vector_id.startswith("medium-full:") for vector_id in upserted_ids)
