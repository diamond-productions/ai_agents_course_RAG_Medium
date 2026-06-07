from __future__ import annotations

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from medium_rag.config import ChunkingConfig
from medium_rag.types import Article, Chunk


def approx_token_count(text: str) -> int:
    return max(1, len(text) // 4)


def _tiktoken_len(text: str) -> int:
    import tiktoken

    return len(tiktoken.get_encoding("cl100k_base").encode(text))


def build_chunk_text(article: Article) -> str:
    metadata_lines = []
    if article.title:
        metadata_lines.append(f"Title: {article.title}")
    if article.authors:
        metadata_lines.append(f"Authors: {article.authors}")
    if article.tags:
        metadata_lines.append(f"Tags: {article.tags}")

    if metadata_lines:
        return "\n".join(metadata_lines) + f"\n\nPassage:\n{article.text.strip()}"
    return article.text.strip()


def split_articles(articles: list[Article], config: ChunkingConfig) -> list[Chunk]:
    length_function = _tiktoken_len if config.tokenizer == "tiktoken" else approx_token_count
    splitter = RecursiveCharacterTextSplitter(
        length_function=length_function,
        chunk_size=config.chunk_size,
        chunk_overlap=int(config.chunk_size * config.overlap_ratio),
        separators=config.separators,
    )

    docs = [
        Document(
            page_content=build_chunk_text(article),
            metadata={
                "article_id": article.article_id,
                "title": article.title,
                "url": article.url,
                "authors": article.authors,
                "timestamp": article.timestamp,
                "tags": article.tags,
            },
        )
        for article in articles
    ]
    splits = splitter.split_documents(docs)

    chunks: list[Chunk] = []
    article_chunk_counts: dict[str, int] = {}
    for split in splits:
        article_id = str(split.metadata["article_id"])
        chunk_index = article_chunk_counts.get(article_id, 0)
        article_chunk_counts[article_id] = chunk_index + 1
        chunk_id = f"{article_id}-{chunk_index:04d}"
        chunks.append(
            Chunk(
                id=f"medium-300:{article_id}:{chunk_index:04d}",
                article_id=article_id,
                chunk_id=chunk_id,
                chunk_index=chunk_index,
                text=split.page_content,
                title=str(split.metadata.get("title", "")),
                url=str(split.metadata.get("url", "")),
                authors=str(split.metadata.get("authors", "")),
                timestamp=str(split.metadata.get("timestamp", "")),
                tags=str(split.metadata.get("tags", "")),
            )
        )
    return chunks
