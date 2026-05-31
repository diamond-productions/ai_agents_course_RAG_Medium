from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import streamlit as st

from api.config import CHAT_MODEL, CHUNK_SIZE, EMBEDDING_MODEL, OVERLAP_RATIO, TOP_K
from rag import answer_question


st.set_page_config(page_title="Medium RAG Chat", page_icon="M", layout="wide")


def _post_prompt(api_base_url: str, question: str) -> dict[str, Any]:
    url = f"{api_base_url.rstrip('/')}/api/prompt"
    payload = json.dumps({"question": question}).encode("utf-8")
    request = Request(
        url,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=120) as response:
        return json.loads(response.read().decode("utf-8"))


def _format_number(value: Any) -> str:
    if value is None:
        return "n/a"
    if isinstance(value, float):
        return f"{value:,.6f}"
    if isinstance(value, int):
        return f"{value:,}"
    return str(value)


def _format_money(value: Any) -> str:
    if value is None:
        return "n/a"
    return f"${float(value):,.6f}"


def _unique_articles(context: list[dict[str, Any]]) -> list[dict[str, Any]]:
    articles: dict[str, dict[str, Any]] = {}
    for item in context:
        key = item.get("article_id") or item.get("title") or str(len(articles))
        if key not in articles:
            articles[key] = {
                "article_id": item.get("article_id", ""),
                "title": item.get("title", "Untitled"),
                "url": item.get("url", ""),
                "authors": item.get("authors", ""),
                "timestamp": item.get("timestamp", ""),
                "tags": item.get("tags", ""),
                "best_score": item.get("score"),
                "chunks": 0,
            }
        articles[key]["chunks"] += 1
        score = item.get("score")
        best = articles[key]["best_score"]
        if score is not None and (best is None or score > best):
            articles[key]["best_score"] = score
    return list(articles.values())


def _show_usage(result: dict[str, Any]) -> None:
    usage = result.get("usage") or {}
    cost = result.get("cost") or {}
    cols = st.columns(4)
    cols[0].metric("Input tokens", _format_number(usage.get("input_tokens")))
    cols[1].metric("Output tokens", _format_number(usage.get("output_tokens")))
    cols[2].metric("Total tokens", _format_number(usage.get("total_tokens")))
    cols[3].metric("Estimated cost", _format_money(cost.get("estimated_total_cost")))
    if cost.get("note"):
        st.caption(cost["note"])


def _show_context(context: list[dict[str, Any]]) -> None:
    articles = _unique_articles(context)
    st.subheader("Articles Referenced")
    if not articles:
        st.info("No retrieved articles returned.")
        return

    for article in articles:
        title = article.get("title") or "Untitled"
        score = _format_number(article.get("best_score"))
        with st.expander(f"{title}  |  score {score}", expanded=False):
            st.write(
                {
                    "article_id": article.get("article_id"),
                    "authors": article.get("authors"),
                    "timestamp": article.get("timestamp"),
                    "tags": article.get("tags"),
                    "chunks": article.get("chunks"),
                    "url": article.get("url"),
                }
            )

    st.subheader("Retrieved Chunks")
    for index, item in enumerate(context, start=1):
        title = item.get("title") or "Untitled"
        score = _format_number(item.get("score"))
        with st.expander(f"Chunk {index}: {title}  |  score {score}", expanded=index == 1):
            st.write(
                {
                    "article_id": item.get("article_id"),
                    "authors": item.get("authors"),
                    "timestamp": item.get("timestamp"),
                    "tags": item.get("tags"),
                    "url": item.get("url"),
                }
            )
            st.text_area("Chunk text", item.get("chunk", ""), height=220, key=f"chunk-{id(item)}-{index}")


def _show_prompt(result: dict[str, Any]) -> None:
    augmented_prompt = result.get("Augmented_prompt") or {}
    st.subheader("Augmented Prompt")
    st.text_area("System", augmented_prompt.get("System", ""), height=180)
    st.text_area("User", augmented_prompt.get("User", ""), height=300)


def _show_metadata(result: dict[str, Any]) -> None:
    st.subheader("Model Metadata")
    st.json(result.get("metadata") or {})
    st.subheader("Raw Usage And Cost")
    st.json({"usage": result.get("usage") or {}, "cost": result.get("cost") or {}})


with st.sidebar:
    st.header("RAG Settings")
    use_http_api = st.toggle("Call FastAPI endpoint", value=False)
    api_base_url = st.text_input("API base URL", value="http://127.0.0.1:8000", disabled=not use_http_api)
    st.divider()
    st.write(
        {
            "chat_model": CHAT_MODEL,
            "embedding_model": EMBEDDING_MODEL,
            "chunk_size": CHUNK_SIZE,
            "overlap_ratio": OVERLAP_RATIO,
            "top_k": TOP_K,
        }
    )
    st.caption("Default costs: gpt-5-mini input $0.25/M, output $2.00/M; embeddings $0.02/M.")


st.title("Medium RAG Chat")

if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

question = st.chat_input("Ask a question about the Medium article dataset")
if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving chunks and generating an answer..."):
            try:
                result = _post_prompt(api_base_url, question) if use_http_api else answer_question(question)
            except (HTTPError, URLError, TimeoutError, RuntimeError) as exc:
                st.error(str(exc))
                st.stop()
        answer = result.get("response", "")
        st.markdown(answer)
        _show_usage(result)

    st.session_state.last_result = result
    st.session_state.messages.append({"role": "assistant", "content": answer})

if st.session_state.last_result:
    result = st.session_state.last_result
    tab_context, tab_prompt, tab_metadata, tab_raw = st.tabs(["Sources", "Prompt", "Metadata", "Raw JSON"])
    with tab_context:
        _show_context(result.get("context") or [])
    with tab_prompt:
        _show_prompt(result)
    with tab_metadata:
        _show_metadata(result)
    with tab_raw:
        st.json(result)
