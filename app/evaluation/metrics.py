"""
Metric functions for the evaluation harness. Each takes the raw per-case
results and returns a 0..100 percentage.
"""
from __future__ import annotations


def refusal_accuracy(results: list[dict]) -> float:
    relevant = [r for r in results if r["must_refuse"]]
    if not relevant:
        return 100.0
    correct = sum(1 for r in relevant if r["response_refused"])
    return 100.0 * correct / len(relevant)


def relevance_accuracy(results: list[dict]) -> float:
    relevant = [r for r in results if not r["must_refuse"]]
    if not relevant:
        return 100.0
    correct = sum(1 for r in relevant if r["response_relevant"] == r["expect_relevant"])
    return 100.0 * correct / len(relevant)


def groundedness(results: list[dict]) -> float:
    """% of non-refused answers where every surviving claim has >=1 citation."""
    candidates = [r for r in results if not r["response_refused"]]
    if not candidates:
        return 100.0
    grounded = sum(1 for r in candidates if r["all_claims_cited"])
    return 100.0 * grounded / len(candidates)


def citation_accuracy(results: list[dict]) -> float:
    """% of non-refused answers with >=1 valid citation pointing at real evidence."""
    candidates = [r for r in results if not r["response_refused"]]
    if not candidates:
        return 100.0
    cited = sum(1 for r in candidates if r["has_valid_citation"])
    return 100.0 * cited / len(candidates)


def hallucination_rate(results: list[dict]) -> float:
    """% of non-refused answers where the hallucination guard had to remove >=1 claim."""
    candidates = [r for r in results if not r["response_refused"]]
    if not candidates:
        return 0.0
    hallucinated = sum(1 for r in candidates if r["claims_removed"] > 0)
    return 100.0 * hallucinated / len(candidates)


def retrieval_precision(results: list[dict]) -> float:
    """% of cases with >=1 retrieved chunk where retrieval passed the evidence threshold."""
    candidates = [r for r in results if r["retrieved_chunks"] > 0]
    if not candidates:
        return 0.0
    passed = sum(1 for r in candidates if r["passed_threshold"])
    return 100.0 * passed / len(candidates)


def retrieval_recall(results: list[dict]) -> float:
    """% of cases expected to be answerable where >=1 chunk was retrieved at all."""
    candidates = [r for r in results if not r["must_refuse"]]
    if not candidates:
        return 100.0
    found = sum(1 for r in candidates if r["retrieved_chunks"] > 0)
    return 100.0 * found / len(candidates)


def eligibility_accuracy(results: list[dict]) -> float:
    """% of eligibility-category cases where the engine produced a non-crash, defensible verdict
    (never a bare 'eligible' without matched criteria)."""
    candidates = [r for r in results if r["category"] == "eligibility"]
    if not candidates:
        return 100.0
    correct = sum(1 for r in candidates if r["eligibility_defensible"])
    return 100.0 * correct / len(candidates)
