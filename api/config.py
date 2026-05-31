CHUNK_SIZE = 800
OVERLAP_RATIO = 0.15
TOP_K = 7
EMBEDDING_DIMENSIONS = 1536
EMBEDDING_MODEL = "4UHRUIN-text-embedding-3-small"
CHAT_MODEL = "4UHRUIN-gpt-5-mini"
PINECONE_INDEX_NAME = "medium-articles"
PINECONE_NAMESPACE = ""
SAMPLE_DATA_PATH = "data/medium-300-sample.csv"
DEFAULT_BASE_URL = "https://api.llmod.ai/v1"

# Default OpenAI gpt-5-mini and text-embedding-3-small prices in USD per 1M tokens.
CHAT_INPUT_COST_PER_1M_TOKENS = 0.25
CHAT_OUTPUT_COST_PER_1M_TOKENS = 2.00
EMBEDDING_COST_PER_1M_TOKENS = 0.02

SYSTEM_PROMPT = """You are a Medium-article assistant that answers questions strictly and only based on the Medium articles dataset context provided to you (metadata and article passages). You must not use any external knowledge, the open internet, or information that is not explicitly contained in the retrieved context. If the answer cannot be determined from the provided context, respond: “I don’t know based on the provided Medium articles data.” Always explain your answer using the given context, quoting or paraphrasing the relevant article passage or metadata when helpful."""

DUMMY_CONTEXT = [
    {
        "article_id": "dummy-1",
        "title": "Dummy Medium Article",
        "chunk": "This is placeholder retrieved context for local API wiring. Replace it with Pinecone retrieval results.",
        "score": 0.0,
    }
]
