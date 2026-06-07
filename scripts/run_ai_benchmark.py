from __future__ import annotations

import argparse
import json
from pathlib import Path

from medium_rag.config import DEFAULT_EXPERIMENT_CONFIG_PATH, load_experiment_config
from medium_rag.evaluation import run_benchmark

DEFAULT_CASES_PATH = Path("eval/medium_300_ai_benchmark.jsonl")
DEFAULT_OUTPUT_PATH = Path("eval/medium_300_ai_benchmark_results.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the AI-created medium-300 RAG benchmark.")
    parser.add_argument("--config", default=str(DEFAULT_EXPERIMENT_CONFIG_PATH), help="Path to experiment YAML config.")
    parser.add_argument("--cases", type=Path, default=DEFAULT_CASES_PATH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    summary = run_benchmark(config, args.cases, args.output, args.limit)
    print(json.dumps(summary.as_dict(), indent=2, ensure_ascii=False))
    print(f"Wrote results to {args.output}")
    print(f"Wrote summary to {args.output.with_suffix('.summary.json')}")


if __name__ == "__main__":
    main()
