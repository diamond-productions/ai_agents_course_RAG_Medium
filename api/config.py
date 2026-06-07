from __future__ import annotations

from medium_rag.config import load_experiment_config
from medium_rag.generation import (
    CHAT_INPUT_COST_PER_1M_TOKENS,
    CHAT_OUTPUT_COST_PER_1M_TOKENS,
    EMBEDDING_COST_PER_1M_TOKENS,
    load_system_prompt,
)

_CONFIG = load_experiment_config()

CHUNK_SIZE = _CONFIG.chunking.chunk_size
OVERLAP_RATIO = _CONFIG.chunking.overlap_ratio
TOP_K = _CONFIG.retrieval.top_k
RETRIEVAL_CANDIDATE_K = _CONFIG.retrieval.candidate_k
MMR_ENABLED = _CONFIG.retrieval.mmr_enabled
MMR_LAMBDA = _CONFIG.retrieval.mmr_lambda
EMBEDDING_DIMENSIONS = _CONFIG.embedding.dimensions
EMBEDDING_MODEL = _CONFIG.embedding.model
CHAT_MODEL = _CONFIG.generation.chat_model
PINECONE_INDEX_NAME = _CONFIG.pinecone.index_name
PINECONE_NAMESPACE = _CONFIG.pinecone.namespace
SAMPLE_DATA_PATH = _CONFIG.dataset.path
DEFAULT_BASE_URL = _CONFIG.generation.api_base
SYSTEM_PROMPT = load_system_prompt(_CONFIG.generation.system_prompt_path)

DUMMY_CONTEXT = [
    {
        "article_id": "dummy-1",
        "title": "Dummy Medium Article",
        "chunk": "This is placeholder retrieved context for local API wiring. Replace it with Pinecone retrieval results.",
        "score": 0.0,
    }
]
