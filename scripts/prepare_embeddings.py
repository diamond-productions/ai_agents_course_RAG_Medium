from __future__ import annotations

import argparse

from medium_rag.config import DEFAULT_EXPERIMENT_CONFIG_PATH, load_experiment_config
from medium_rag.indexing import index_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare embeddings and upsert chunks into Pinecone.")
    parser.add_argument("--config", default=str(DEFAULT_EXPERIMENT_CONFIG_PATH), help="Path to experiment YAML config.")
    parser.add_argument("--force", action="store_true", help="Upsert even if namespace already has vectors.")
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    count = index_dataset(config, force=args.force)
    print(
        f"Done. {count} vectors available in '{config.pinecone.index_name}' "
        f"namespace '{config.pinecone.namespace}'."
    )


if __name__ == "__main__":
    main()
