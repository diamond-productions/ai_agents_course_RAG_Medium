from __future__ import annotations

import json
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import streamlit as st

from rag_logging import eval_run_log_path, rag_trace_log_path, read_jsonl
from rag_utils import answer_question, get_default_config


st.set_page_config(page_title="Medium RAG Chat", page_icon="M", layout="wide")


def _post_prompt(
    api_base_url: str,
    question: str,
    *,
    eval_run_id: str | None = None,
    expected_answer: str | None = None,
) -> dict[str, Any]:
    url = f"{api_base_url.rstrip('/')}/api/prompt"
    payload_data: dict[str, Any] = {"question": question, "log_rag_trace": True}
    if eval_run_id:
        payload_data["eval_run_id"] = eval_run_id
    if expected_answer:
        payload_data["expected_answer"] = expected_answer
    payload = json.dumps(payload_data).encode("utf-8")
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
    st.subheader("Log Files")
    st.json(
        {
            "rag_trace_log_path": result.get("trace_log_path") or str(rag_trace_log_path()),
            "eval_run_log_path": result.get("eval_log_path") or str(eval_run_log_path()),
        }
    )
    st.subheader("Raw Usage And Cost")
    st.json({"usage": result.get("usage") or {}, "cost": result.get("cost") or {}})


def _record_matches_search(record: dict[str, Any], query: str) -> bool:
    if not query:
        return True
    needle = query.casefold()
    searchable = [
        record.get("timestamp", ""),
        record.get("question", ""),
        record.get("response", ""),
        record.get("run_id", ""),
        record.get("source", ""),
        record.get("expected_answer", ""),
    ]
    searchable.extend(str(item.get("title", "")) for item in record.get("context") or [])
    return needle in "\n".join(str(value) for value in searchable).casefold()


def _show_logged_context(context: list[dict[str, Any]], key_prefix: str) -> None:
    if not context:
        st.info("No context recorded.")
        return

    for index, item in enumerate(context, start=1):
        title = item.get("title") or "Untitled"
        score = _format_number(item.get("score"))
        with st.expander(f"{index}. {title}  |  score {score}", expanded=index == 1):
            st.write({"article_id": item.get("article_id"), "score": item.get("score")})
            st.text_area(
                "Chunk preview",
                item.get("chunk_preview", ""),
                height=140,
                key=f"{key_prefix}-context-{index}",
            )


def _show_log_record(record: dict[str, Any], key_prefix: str) -> None:
    cols = st.columns(4)
    cols[0].caption(record.get("timestamp") or "No timestamp")
    cols[1].caption(f"source: {record.get('source', 'n/a')}")
    cols[2].caption(f"line: {record.get('_line_number', 'n/a')}")
    cols[3].caption(f"run: {record.get('run_id', 'n/a')}")

    st.markdown("**Question**")
    st.write(record.get("question", ""))
    st.markdown("**Response**")
    st.write(record.get("response", ""))

    if record.get("expected_answer") is not None:
        st.markdown("**Expected Answer**")
        st.write(record.get("expected_answer", ""))
    if record.get("evaluation") is not None:
        st.markdown("**Evaluation**")
        st.json(record.get("evaluation"))

    detail_tabs = st.tabs(["Context", "Prompt", "Config", "Raw"])
    with detail_tabs[0]:
        _show_logged_context(record.get("context") or [], key_prefix)
    with detail_tabs[1]:
        prompt = record.get("augmented_prompt") or {}
        st.text_area("System", prompt.get("system", ""), height=160, key=f"{key_prefix}-system")
        st.text_area("User", prompt.get("user", ""), height=240, key=f"{key_prefix}-user")
    with detail_tabs[2]:
        st.json(record.get("config") or {})
    with detail_tabs[3]:
        st.json(record)


def _show_log_browser() -> None:
    st.header("Conversation Logs")
    log_source = st.radio(
        "Log source",
        ["RAG traces", "Eval runs"],
        horizontal=True,
        label_visibility="collapsed",
    )
    path = rag_trace_log_path() if log_source == "RAG traces" else eval_run_log_path()

    cols = st.columns([2, 1])
    search = cols[0].text_input("Search logs", placeholder="Question, answer, title, source, run ID")
    limit = cols[1].number_input("Recent records", min_value=1, max_value=500, value=50, step=10)

    st.caption(str(path))
    records = [record for record in read_jsonl(path, limit=int(limit)) if _record_matches_search(record, search)]
    if not records:
        st.info("No matching log records found.")
        return

    st.caption(f"Showing {len(records)} record(s), newest first.")
    key_slug = "rag" if log_source == "RAG traces" else "eval"
    for index, record in enumerate(records):
        timestamp = record.get("timestamp") or "No timestamp"
        question = record.get("question") or "Untitled record"
        label = f"{timestamp} | {question[:100]}"
        with st.expander(label, expanded=index == 0):
            _show_log_record(record, key_prefix=f"log-{key_slug}-{record.get('_line_number', index)}")


with st.sidebar:
    current_config = get_default_config()
    st.header("RAG Settings")
    use_http_api = st.toggle("Call FastAPI endpoint", value=False)
    api_base_url = st.text_input("API base URL", value="http://127.0.0.1:8000", disabled=not use_http_api)
    log_eval_run = st.toggle("Log as eval run", value=False)
    eval_run_id = st.text_input("Eval run ID", value="", disabled=not log_eval_run)
    expected_answer = st.text_area("Expected answer", value="", disabled=not log_eval_run)
    st.divider()
    st.write(
        {
            **current_config.config_summary(),
            "chat_model": current_config.generation.chat_model,
            "embedding_model": current_config.embedding.model,
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
                eval_id = eval_run_id.strip() if log_eval_run else None
                expected = expected_answer.strip() if log_eval_run else None
                if use_http_api:
                    result = _post_prompt(
                        api_base_url,
                        question,
                        eval_run_id=eval_id,
                        expected_answer=expected,
                    )
                else:
                    result = answer_question(
                        question,
                        trace_source="streamlit",
                        eval_run_id=eval_id,
                        expected_answer=expected,
                    )
            except (HTTPError, URLError, TimeoutError, RuntimeError) as exc:
                st.error(str(exc))
                st.stop()
        answer = result.get("response", "")
        st.markdown(answer)
        _show_usage(result)

    st.session_state.last_result = result
    st.session_state.messages.append({"role": "assistant", "content": answer})

current_tab, history_tab = st.tabs(["Current Chat", "Log Review"])

with current_tab:
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
    else:
        st.info("Ask a question to inspect the current answer details.")

with history_tab:
    _show_log_browser()
