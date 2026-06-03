from __future__ import annotations

from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlparse


ROOT = Path(__file__).resolve().parents[1]
APP_DIR = ROOT / "web_app"
DATA_DIR = ROOT / "data"
ALLOWED_DATA = {
    "medium-300-sample.csv",
    "medium-english-50mb.csv",
}


class DatasetExplorerHandler(SimpleHTTPRequestHandler):
    def translate_path(self, path: str) -> str:
        parsed_path = unquote(urlparse(path).path)

        if parsed_path.startswith("/data/"):
            filename = Path(parsed_path).name
            if filename in ALLOWED_DATA:
                return str(DATA_DIR / filename)
            return str(APP_DIR / "__not_found__")

        app_path = parsed_path.removeprefix("/web_app/")
        if parsed_path in {"", "/"}:
            app_path = "index.html"
        resolved_path = (APP_DIR / app_path.lstrip("/")).resolve()
        if APP_DIR not in resolved_path.parents and resolved_path != APP_DIR:
            return str(APP_DIR / "__not_found__")
        return str(resolved_path)

    def end_headers(self) -> None:
        self.send_header("Cache-Control", "no-store")
        super().end_headers()


def main() -> None:
    server = ThreadingHTTPServer(("127.0.0.1", 8080), DatasetExplorerHandler)
    print("Serving Medium Dataset Explorer at http://127.0.0.1:8080/")
    server.serve_forever()


if __name__ == "__main__":
    main()
