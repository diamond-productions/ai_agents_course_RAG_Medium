# Medium RAG Assistant

## Development Chat UI

Run the Streamlit development interface:

```sh
uv run streamlit run streamlit_app.py
```

By default the UI calls the local RAG function directly. To test the deployed API shape locally, start FastAPI in a second terminal:

```sh
uv run uvicorn api.index:app --reload
```

Then enable `Call FastAPI endpoint` in the Streamlit sidebar.

The UI shows:

- The assistant answer.
- Referenced articles and retrieved chunks.
- The exact augmented system/user prompt sent to the model.
- Model and retrieval metadata.
- Token usage returned by the model provider.
- Estimated cost using gpt-5-mini defaults: input `$0.25`/1M tokens, output `$2.00`/1M tokens, and text-embedding-3-small query embedding `$0.02`/1M tokens. Override with `CHAT_INPUT_COST_PER_1M_TOKENS`, `CHAT_OUTPUT_COST_PER_1M_TOKENS`, and `EMBEDDING_COST_PER_1M_TOKENS`.
