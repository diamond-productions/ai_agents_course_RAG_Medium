# Medium Dataset Explorer Web App

This standalone web app lives separately from the RAG project code. It reads the CSV files in `../data/` from the browser and provides search, filtering, sorting, tag statistics, pagination, and article detail views.

Run the app-specific server from the project root. It only serves this `web_app/` directory and the two Medium CSV files under `data/`:

```sh
uv run rag-explorer
```

Then open:

```text
http://127.0.0.1:8080/
```

You can also use the `Open CSV` control to load another CSV with the same columns: `title`, `text`, `url`, `authors`, `timestamp`, and `tags`.
