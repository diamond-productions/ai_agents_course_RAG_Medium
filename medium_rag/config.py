from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

import yaml
from dotenv import load_dotenv
from pydantic import BaseModel, Field, field_validator

load_dotenv()

DEFAULT_EXPERIMENT_CONFIG_PATH = Path("configs/experiments/dense_mmr.yaml")


class DatasetConfig(BaseModel):
    name: str
    path: str
    vector_id_prefix: str | None = None


class ChunkingConfig(BaseModel):
    chunk_size: int = Field(gt=0)
    overlap_ratio: float = Field(ge=0.0, lt=1.0)
    tokenizer: Literal["approx", "tiktoken"]
    separators: list[str]


class EmbeddingConfig(BaseModel):
    model: str
    dimensions: int = Field(gt=0)
    api_base: str
    batch_size: int = Field(default=64, gt=0)


class PineconeConfig(BaseModel):
    index_name: str
    namespace: str
    cloud: str
    region: str
    metric: Literal["cosine"]
    batch_size: int = Field(gt=0)
    max_batch_bytes: int = Field(default=1_800_000, gt=0)


class RetrievalConfig(BaseModel):
    strategy: Literal["dense", "dense_mmr"]
    top_k: int = Field(gt=0)
    candidate_k: int = Field(gt=0)
    dedupe_by_article: bool
    mmr_enabled: bool
    mmr_lambda: float = 0.65

    @field_validator("mmr_lambda")
    @classmethod
    def clamp_mmr_lambda(cls, value: float) -> float:
        return max(0.0, min(float(value), 1.0))


class GenerationConfig(BaseModel):
    chat_model: str
    api_base: str
    system_prompt_path: str


class RagExperimentConfig(BaseModel):
    experiment_name: str
    dataset: DatasetConfig
    chunking: ChunkingConfig
    embedding: EmbeddingConfig
    pinecone: PineconeConfig
    retrieval: RetrievalConfig
    generation: GenerationConfig

    def config_summary(self) -> dict[str, object]:
        return {
            "experiment_name": self.experiment_name,
            "dataset_name": self.dataset.name,
            "dataset_path": self.dataset.path,
            "vector_id_prefix": self.dataset.vector_id_prefix,
            "chunk_size": self.chunking.chunk_size,
            "overlap_ratio": self.chunking.overlap_ratio,
            "tokenizer": self.chunking.tokenizer,
            "top_k": self.retrieval.top_k,
            "retrieval_candidate_k": self.retrieval.candidate_k,
            "retrieval_strategy": self.retrieval.strategy,
            "mmr_enabled": self.retrieval.mmr_enabled,
            "mmr_lambda": self.retrieval.mmr_lambda,
            "pinecone_namespace": self.pinecone.namespace,
        }


def _env_or_default(name: str, default: str) -> str:
    value = os.getenv(name)
    return default if value in (None, "") else value


def _optional_int_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value in (None, ""):
        return default
    return int(value)


def apply_env_overrides(config: RagExperimentConfig) -> RagExperimentConfig:
    data = config.model_dump()
    data["dataset"]["name"] = _env_or_default("RAG_DATASET_NAME", config.dataset.name)
    data["dataset"]["path"] = _env_or_default("RAG_DATASET_PATH", config.dataset.path)
    vector_id_prefix = os.getenv("RAG_VECTOR_ID_PREFIX")
    if vector_id_prefix not in (None, ""):
        data["dataset"]["vector_id_prefix"] = vector_id_prefix
    data["embedding"]["api_base"] = _env_or_default("LLMOD_API_BASE", config.embedding.api_base)
    data["embedding"]["batch_size"] = _optional_int_env("EMBEDDING_BATCH_SIZE", config.embedding.batch_size)
    data["generation"]["api_base"] = _env_or_default("LLMOD_API_BASE", config.generation.api_base)
    data["pinecone"]["index_name"] = _env_or_default("PINECONE_INDEX_NAME", config.pinecone.index_name)
    data["pinecone"]["namespace"] = _env_or_default("PINECONE_NAMESPACE", config.pinecone.namespace)
    data["pinecone"]["batch_size"] = _optional_int_env("PINECONE_BATCH_SIZE", config.pinecone.batch_size)
    data["pinecone"]["max_batch_bytes"] = _optional_int_env(
        "PINECONE_MAX_BATCH_BYTES",
        config.pinecone.max_batch_bytes,
    )
    return RagExperimentConfig.model_validate(data)


def load_experiment_config(path: str | Path | None = None) -> RagExperimentConfig:
    config_path = Path(path or os.getenv("RAG_EXPERIMENT_CONFIG") or DEFAULT_EXPERIMENT_CONFIG_PATH)
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return apply_env_overrides(RagExperimentConfig.model_validate(raw))
