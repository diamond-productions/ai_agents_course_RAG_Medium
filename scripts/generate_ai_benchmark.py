from __future__ import annotations

import argparse
from pathlib import Path

from medium_rag.config import DEFAULT_EXPERIMENT_CONFIG_PATH, load_experiment_config
from medium_rag.evaluation.benchmark_cases import DEFAULT_CASE_COUNT, generate_benchmark_cases

DEFAULT_OUTPUT_PATH = Path("eval/medium_300_ai_benchmark.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate RAG benchmark cases for the configured Medium dataset.")
    parser.add_argument("--config", default=str(DEFAULT_EXPERIMENT_CONFIG_PATH), help="Path to experiment YAML config.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT_PATH)
    parser.add_argument("--articles", type=int, default=30, help="Number of articles to sample.")
    parser.add_argument(
        "--case-count",
        type=int,
        default=DEFAULT_CASE_COUNT,
        help="Number of AI-curated benchmark cases to write.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-article-chars", type=int, default=6000)
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    written = generate_benchmark_cases(
        config,
        output_path=args.output,
        articles=args.articles,
        case_count=args.case_count,
        seed=args.seed,
        max_article_chars=args.max_article_chars,
    )
    print(f"Wrote {written} benchmark cases to {args.output}")


if __name__ == "__main__":
    main()
