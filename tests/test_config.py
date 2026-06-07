from __future__ import annotations

import pytest
from pydantic import ValidationError

from medium_rag.config import RagExperimentConfig, load_experiment_config


def test_loads_default_dense_mmr_config() -> None:
    config = load_experiment_config("configs/experiments/dense_mmr.yaml")
    assert config.experiment_name == "dense_mmr_chunk800_overlap015"
    assert config.retrieval.strategy == "dense_mmr"
    assert config.retrieval.mmr_lambda == 0.65
    assert config.embedding.batch_size == 64
    assert config.pinecone.max_batch_bytes == 1_800_000


def test_rejects_missing_required_fields() -> None:
    with pytest.raises(ValidationError):
        RagExperimentConfig.model_validate({"experiment_name": "broken"})


def test_clamps_mmr_lambda() -> None:
    data = load_experiment_config("configs/experiments/dense_mmr.yaml").model_dump()
    data["retrieval"]["mmr_lambda"] = 2.0
    config = RagExperimentConfig.model_validate(data)
    assert config.retrieval.mmr_lambda == 1.0


def test_env_overrides_provider_and_pinecone_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLMOD_API_BASE", "https://example.test/v1")
    monkeypatch.setenv("RAG_DATASET_NAME", "override-dataset")
    monkeypatch.setenv("RAG_DATASET_PATH", "data/override.csv")
    monkeypatch.setenv("RAG_VECTOR_ID_PREFIX", "override-prefix")
    monkeypatch.setenv("EMBEDDING_BATCH_SIZE", "11")
    monkeypatch.setenv("PINECONE_INDEX_NAME", "override-index")
    monkeypatch.setenv("PINECONE_NAMESPACE", "override-namespace")
    monkeypatch.setenv("PINECONE_BATCH_SIZE", "12")
    monkeypatch.setenv("PINECONE_MAX_BATCH_BYTES", "12345")
    config = load_experiment_config("configs/experiments/dense_mmr.yaml")
    assert config.dataset.name == "override-dataset"
    assert config.dataset.path == "data/override.csv"
    assert config.dataset.vector_id_prefix == "override-prefix"
    assert config.embedding.api_base == "https://example.test/v1"
    assert config.embedding.batch_size == 11
    assert config.generation.api_base == "https://example.test/v1"
    assert config.pinecone.index_name == "override-index"
    assert config.pinecone.namespace == "override-namespace"
    assert config.pinecone.batch_size == 12
    assert config.pinecone.max_batch_bytes == 12345


def test_loads_full_dataset_config() -> None:
    config = load_experiment_config("configs/experiments/full_dense_mmr_chunk512_overlap015_lambda075.yaml")
    assert config.dataset.name == "medium-english-50mb"
    assert config.dataset.vector_id_prefix == "medium-full"
    assert config.pinecone.index_name == "medium-articles-full"
    assert config.pinecone.batch_size == 50
    assert config.retrieval.candidate_k == 30
