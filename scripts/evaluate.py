#!/usr/bin/env python
"""
Runs the golden evaluation dataset against the live pipeline and prints the
metrics report (spec section 27).

Usage:
    python scripts/evaluate.py [--limit N] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.evaluation.evaluator import run_evaluation  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the KrishiMitra AI evaluation harness.")
    parser.add_argument("--limit", type=int, default=None, help="Only run the first N cases.")
    parser.add_argument("--json", type=str, default=None, help="Optional path to save full results as JSON.")
    parser.add_argument(
        "--deepeval", action="store_true",
        help="Also run optional DeepEval LLM-as-judge metrics (requires DEEPEVAL_ENABLED=true + OPENAI_API_KEY).",
    )
    args = parser.parse_args()

    print(f"Running evaluation over {'all' if not args.limit else args.limit} cases...\n")
    report = run_evaluation(limit=args.limit, run_deepeval=args.deepeval)
    s = report["summary"]

    print("==================================")
    print("KRISHIMITRA AI EVALUATION")
    print("==================================")
    print(f"Total cases:            {s['total_cases']}")
    print(f"Retrieval Precision:    {s['retrieval_precision']}%")
    print(f"Retrieval Recall:       {s['retrieval_recall']}%")
    print(f"Groundedness:           {s['groundedness']}%")
    print(f"Citation Accuracy:      {s['citation_accuracy']}%")
    print(f"Relevance Accuracy:     {s['relevance_accuracy']}%")
    print(f"Hallucination Rate:     {s['hallucination_rate']}%")
    print(f"Refusal Accuracy:       {s['refusal_accuracy']}%")
    print(f"Eligibility Accuracy:   {s['eligibility_accuracy']}%")
    print(f"Avg latency:            {s['avg_latency_ms']} ms")
    print("==================================")

    if args.deepeval:
        de = report.get("deepeval", {})
        print("\n-- DeepEval (optional, LLM-as-judge) --")
        if not de.get("enabled"):
            print(f"  skipped: {de.get('reason')}")
        else:
            print(f"  Cases evaluated:        {de['cases_evaluated']} (skipped {de['cases_skipped_refused_or_no_evidence']} refused/no-evidence)")
            print(f"  Avg Faithfulness:       {de['avg_faithfulness']}")
            print(f"  Avg Answer Relevancy:   {de['avg_answer_relevancy']}")

    by_category: dict[str, list[dict]] = {}
    for r in report["results"]:
        by_category.setdefault(r["category"], []).append(r)
    print("\nPer-category pass summary:")
    for cat, rows in sorted(by_category.items()):
        refuse_rows = [r for r in rows if r["must_refuse"]]
        ok = sum(1 for r in refuse_rows if r["response_refused"]) + sum(
            1 for r in rows if not r["must_refuse"] and not r["response_refused"]
        )
        print(f"  {cat:22s} {ok}/{len(rows)} behaved as expected")

    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nFull results saved to {args.json}")


if __name__ == "__main__":
    main()
