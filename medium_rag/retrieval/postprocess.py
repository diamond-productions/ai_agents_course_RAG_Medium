from __future__ import annotations

from medium_rag.types import RetrievedContext, VectorMatch


def article_key(match: VectorMatch) -> str:
    return str(match.metadata.get("article_id") or match.metadata.get("title") or match.id)


def dedupe_matches_by_article(matches: list[VectorMatch], limit: int) -> list[VectorMatch]:
    deduped: list[VectorMatch] = []
    seen: set[str] = set()
    for match in matches:
        key = article_key(match)
        if key and key in seen:
            continue
        if key:
            seen.add(key)
        deduped.append(match)
        if len(deduped) >= limit:
            break
    return deduped


def match_to_context(match: VectorMatch) -> RetrievedContext:
    metadata = match.metadata
    chunk_index = metadata.get("chunk_index")
    return RetrievedContext(
        article_id=str(metadata.get("article_id") or metadata.get("id") or ""),
        title=str(metadata.get("title", "")),
        chunk=str(metadata.get("text", "")),
        score=match.score,
        url=str(metadata.get("url", "")),
        authors=str(metadata.get("authors", "")),
        timestamp=str(metadata.get("timestamp", "")),
        tags=str(metadata.get("tags", "")),
        chunk_id=str(metadata.get("chunk_id", "")),
        chunk_index=int(chunk_index) if chunk_index not in (None, "") else None,
    )


def matches_to_context(matches: list[VectorMatch]) -> list[RetrievedContext]:
    return [match_to_context(match) for match in matches]
