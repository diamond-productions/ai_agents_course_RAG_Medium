from __future__ import annotations

import argparse
import subprocess
import sys
from collections.abc import Sequence


def _run(command: Sequence[str]) -> int:
    return subprocess.run(command, check=False).returncode


def start_api() -> int:
    return _run(["uvicorn", "api.index:app", "--reload"])


def start_web() -> int:
    return _run(["streamlit", "run", "streamlit_app.py"])


def start_all() -> int:
    api = subprocess.Popen(["uvicorn", "api.index:app", "--reload"])
    try:
        return _run(["streamlit", "run", "streamlit_app.py"])
    finally:
        api.terminate()
        try:
            api.wait(timeout=10)
        except subprocess.TimeoutExpired:
            api.kill()
            api.wait()


def main() -> int:
    parser = argparse.ArgumentParser(description="Start development services.")
    parser.add_argument("service", choices=["api", "web", "all"], help="Service to start.")
    args = parser.parse_args()

    if args.service == "api":
        return start_api()
    if args.service == "web":
        return start_web()
    return start_all()


if __name__ == "__main__":
    raise SystemExit(main())
