from __future__ import annotations

import argparse

from rag_utils import answer_question, index_sample_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Query the Medium sample RAG pipeline.")
    parser.add_argument("question", nargs="?", default="List exactly 3 articles about education. Return only the titles.")
    parser.add_argument("--index-sample", action="store_true", help="Index only data/medium-300-sample.csv before querying.")
    parser.add_argument("--force-reindex", action="store_true", help="Upsert the sample chunks even if vectors already exist.")
    parser.add_argument("--no-log", action="store_true", help="Disable RAG trace logging for this query.")
    parser.add_argument("--eval-run-id", help="Also append this query result to the eval runs JSONL log.")
    parser.add_argument("--expected-answer", help="Expected answer text to include in the eval run log.")
    args = parser.parse_args()

    if args.index_sample:
        count = index_sample_dataset(force=args.force_reindex)
        print(f"Sample index vector count: {count}")

    result = answer_question(
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
