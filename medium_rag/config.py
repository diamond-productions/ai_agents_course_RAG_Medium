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


class ChunkingConfig(BaseModel):
    chunk_size: int = Field(gt=0)
    overlap_ratio: float = Field(ge=0.0, lt=1.0)
    tokenizer: Literal["approx", "tiktoken"]
    separators: list[str]


class EmbeddingConfig(BaseModel):
    model: str
    dimensions: int = Field(gt=0)
    api_base: str


class PineconeConfig(BaseModel):
    index_name: str
    namespace: str
    cloud: str
    region: str
    metric: Literal["cosine"]
    batch_size: int = Field(gt=0)


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


def apply_env_overrides(config: RagExperimentConfig) -> RagExperimentConfig:
    data = config.model_dump()
    data["embedding"]["api_base"] = _env_or_default("LLMOD_API_BASE", config.embedding.api_base)
    data["generation"]["api_base"] = _env_or_default("LLMOD_API_BASE", config.generation.api_base)
    data["pinecone"]["index_name"] = _env_or_default("PINECONE_INDEX_NAME", config.pinecone.index_name)
    data["pinecone"]["namespace"] = _env_or_default("PINECONE_NAMESPACE", config.pinecone.namespace)
    return RagExperimentConfig.model_validate(data)


def load_experiment_config(path: str | Path | None = None) -> RagExperimentConfig:
    config_path = Path(path or os.getenv("RAG_EXPERIMENT_CONFIG") or DEFAULT_EXPERIMENT_CONFIG_PATH)
    with config_path.open(encoding="utf-8") as handle:
        raw = yaml.safe_load(handle) or {}
    return apply_env_overrides(RagExperimentConfig.model_validate(raw))
