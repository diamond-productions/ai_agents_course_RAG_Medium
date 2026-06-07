from __future__ import annotations

import pytest
from pydantic import ValidationError

from medium_rag.config import RagExperimentConfig, load_experiment_config


def test_loads_default_dense_mmr_config() -> None:
    config = load_experiment_config("configs/experiments/dense_mmr.yaml")
    assert config.experiment_name == "dense_mmr_chunk800_overlap015"
    assert config.retrieval.strategy == "dense_mmr"
    assert config.retrieval.mmr_lambda == 0.65


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
    monkeypatch.setenv("PINECONE_INDEX_NAME", "override-index")
    monkeypatch.setenv("PINECONE_NAMESPACE", "override-namespace")
    config = load_experiment_config("configs/experiments/dense_mmr.yaml")
    assert config.embedding.api_base == "https://example.test/v1"
    assert config.generation.api_base == "https://example.test/v1"
    assert config.pinecone.index_name == "override-index"
    assert config.pinecone.namespace == "override-namespace"
