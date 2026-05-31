"""
Prepare embeddings from data/medium-300-sample.csv and upsert into Pinecone.

Pipeline:
  1. Load CSV into LangChain Documents (title + text, with metadata)
  2. Split with RecursiveCharacterTextSplitter using tiktoken token counts
  3. Embed chunks with text-embedding-3-small via OpenAI
  4. Create Pinecone index (if needed) and upsert vectors in batches
"""

import itertools
import os
import time

import pandas as pd
import tiktoken
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone, ServerlessSpec

load_dotenv()

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "4UHRUIN-text-embedding-3-small")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME", "medium-articles")
PINECONE_CLOUD = os.getenv("PINECONE_CLOUD", "aws")
PINECONE_REGION = os.getenv("PINECONE_REGION", "us-east-1")
EMBEDDING_DIMENSIONS = 1536

CHUNK_SIZE = 800  # tokens  (≤ 1024 max per instructions)
OVERLAP_RATIO = 0.15
CHUNK_OVERLAP = int(CHUNK_SIZE * OVERLAP_RATIO)  # 120 tokens
UPSERT_BATCH_SIZE = 100
DATA_PATH = os.getenv("DATA_PATH", "data/medium-300-sample.csv")

# ---------------------------------------------------------------------------
# 1. Load CSV
# ---------------------------------------------------------------------------
print(f"Loading {DATA_PATH} ...")
df = pd.read_csv(DATA_PATH)
df = df.where(pd.notna(df), "")  # replace NaN with empty string

docs: list[Document] = []
for idx, row in df.iterrows():
    title = str(row.get("title", "")).strip()
    text = str(row.get("text", "")).strip()
    if not text:
        continue
    # Prepend title so it is always present in the first chunk
    page_content = f"{title}\n\n{text}" if title else text
    metadata = {
        "title": title,
        "url": str(row.get("url", "")),
        "authors": str(row.get("authors", "")),
        "timestamp": str(row.get("timestamp", "")),
        "tags": str(row.get("tags", "")),
    }
    docs.append(Document(page_content=page_content, metadata=metadata))

print(f"Loaded {len(docs)} articles.")

# ---------------------------------------------------------------------------
# 2. Split with tiktoken-based length function
# ---------------------------------------------------------------------------
# cl100k_base is the encoding used by text-embedding-3-small
enc = tiktoken.get_encoding("cl100k_base")


def tiktoken_len(text: str) -> int:
    return len(enc.encode(text))


splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=tiktoken_len,
    separators=["\n\n", "\n", " ", ""],
)

splits = splitter.split_documents(docs)
print(
    f"Split into {len(splits)} chunks  (chunk_size={CHUNK_SIZE} tokens, overlap={CHUNK_OVERLAP})."
)

# ---------------------------------------------------------------------------
# 3. Embed chunks
# ---------------------------------------------------------------------------
embeddings_model = OpenAIEmbeddings(
    model=EMBEDDING_MODEL,
    openai_api_key=os.environ["LLMOD_API_KEY"],
    openai_api_base=os.environ.get("LLMOD_API_BASE", "https://api.llmod.ai/v1"),
    dimensions=EMBEDDING_DIMENSIONS,
)

print(f"Generating embeddings with '{EMBEDDING_MODEL}' ...")
texts = [doc.page_content for doc in splits]
vectors = embeddings_model.embed_documents(texts)
print(f"Generated {len(vectors)} embeddings (dim={len(vectors[0])}).")

# ---------------------------------------------------------------------------
# 4. Upsert into Pinecone
# ---------------------------------------------------------------------------
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

if not pc.has_index(PINECONE_INDEX_NAME):
    print(f"Creating Pinecone index '{PINECONE_INDEX_NAME}' ...")
    pc.create_index(
        name=PINECONE_INDEX_NAME,
        vector_type="dense",
        dimension=EMBEDDING_DIMENSIONS,
        metric="cosine",
        spec=ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION),
        deletion_protection="disabled",
        tags={"dataset": "medium-300-sample"},
    )
    # Wait until the index is ready
    while not pc.describe_index(PINECONE_INDEX_NAME).status["ready"]:
        print("  Waiting for index to be ready ...")
        time.sleep(5)
    print("Index ready.")
else:
    print(f"Using existing Pinecone index '{PINECONE_INDEX_NAME}'.")

index = pc.Index(PINECONE_INDEX_NAME)


def batched(iterable, n):
    """Yield successive n-sized chunks from iterable."""
    it = iter(iterable)
    while chunk := tuple(itertools.islice(it, n)):
        yield chunk


records = [
    {
        "id": f"chunk-{i}",
        "values": vectors[i],
        "metadata": {
            **splits[i].metadata,
            "text": splits[i].page_content,  # store chunk text for retrieval
        },
    }
    for i in range(len(splits))
]

total_upserted = 0
for batch in batched(records, UPSERT_BATCH_SIZE):
    response = index.upsert(vectors=list(batch))
    total_upserted += response.upserted_count
    print(f"  Upserted {total_upserted}/{len(records)} vectors ...")

print(f"\nDone. {total_upserted} vectors upserted into '{PINECONE_INDEX_NAME}'.")
stats = index.describe_index_stats()
print(f"Index stats: {stats}")
