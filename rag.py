from __future__ import annotations

import argparse

from medium_rag.config import DEFAULT_EXPERIMENT_CONFIG_PATH, load_experiment_config
from medium_rag.indexing import index_dataset
from medium_rag.pipeline import RagPipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the Medium sample RAG pipeline.")
    parser.add_argument("question", nargs="?", default="List exactly 3 articles about education. Return only the titles.")
    parser.add_argument("--config", default=str(DEFAULT_EXPERIMENT_CONFIG_PATH), help="Path to experiment YAML config.")
    parser.add_argument("--index-sample", action="store_true", help="Index the configured dataset before querying.")
    parser.add_argument("--force-reindex", action="store_true", help="Upsert chunks even if vectors already exist.")
    parser.add_argument("--no-log", action="store_true", help="Disable RAG trace logging for this query.")
    parser.add_argument("--eval-run-id", help="Also append this query result to the eval runs JSONL log.")
    parser.add_argument("--expected-answer", help="Expected answer text to include in the eval run log.")
    args = parser.parse_args()

    config = load_experiment_config(args.config)
    if args.index_sample:
        count = index_dataset(config, force=args.force_reindex)
        print(f"Sample index vector count: {count}")

    result = RagPipeline(config).answer_question(
        args.question,
        log_trace=not args.no_log,
        trace_source="cli",
        eval_run_id=args.eval_run_id,
        expected_answer=args.expected_answer,
    )
    print(result["response"])
    if result.get("trace_log_path"):
        print(f"\nRAG trace log: {result['trace_log_path']}")
    if result.get("eval_log_path"):
        print(f"Eval run log: {result['eval_log_path']}")
    print("\nRetrieved context:")
    for item in result["context"]:
        print(f"- {item['title']} ({item['score']:.4f})")


if __name__ == "__main__":
    main()
