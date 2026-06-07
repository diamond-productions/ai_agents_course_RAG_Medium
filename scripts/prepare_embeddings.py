from __future__ import annotations

import argparse
import json

from medium_rag.config import DEFAULT_EXPERIMENT_CONFIG_PATH, load_experiment_config
from medium_rag.indexing import estimate_dataset, index_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare embeddings and upsert chunks into Pinecone.")
    parser.add_argument("--config", default=str(DEFAULT_EXPERIMENT_CONFIG_PATH), help="Path to experiment YAML config.")
    parser.add_argument("--force", action="store_true", help="Upsert even if namespace already has vectors.")
    parser.add_argument("--dry-run", action="store_true", help="Estimate chunks and tokens without embedding or upserting.")
    parser.add_argument("--limit-articles", type=int, help="Only process the first N non-empty articles.")
    parser.add_argument("--progress-every", type=int, default=500, help="Print progress every N articles.")
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    if args.dry_run:
        estimate = estimate_dataset(config, limit_articles=args.limit_articles)
        print(json.dumps(estimate.__dict__, indent=2, ensure_ascii=False))
        return

    result = index_dataset(
        config,
        force=args.force,
        limit_articles=args.limit_articles,
        progress_every=args.progress_every,
    )
    print(json.dumps(result.__dict__, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
