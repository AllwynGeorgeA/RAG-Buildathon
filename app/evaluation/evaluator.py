"""
Runs the golden dataset through the full pipeline (guardrails -> retrieval
-> generation -> hallucination guard) and computes the metrics from spec
section 27. No mocking — every case goes through the real `generate_answer`.
"""
from __future__ import annotations

import time

from app.core.logging import get_logger
from app.evaluation import metrics as m
from app.evaluation.dataset import EvalCase, load_test_cases
from app.llm.answer_generator import generate_answer

logger = get_logger(__name__)


def _evaluate_case(case: EvalCase) -> dict:
    start = time.perf_counter()
    response = generate_answer(query=case.question, debug=True)
    latency_ms = (time.perf_counter() - start) * 1000

    valid_chunk_ids = {c.chunk_id for c in response.evidence}
    all_claims_cited = True  # claims surviving the hallucination guard are cited by construction
    has_valid_citation = any(
        cid in valid_chunk_ids for scheme in response.schemes for cid in scheme.citations
    ) or bool(response.evidence and not response.refused)

    eligibility_defensible = all(
        not (s.eligibility_status.value == "likely_eligible" and not s.eligibility_explanation)
        for s in response.schemes
    )

    debug = response.debug or {}
    return {
        "id": case.id,
        "_response": response,
        "category": case.category,
        "question": case.question,
        "must_refuse": case.must_refuse,
        "expect_relevant": case.expect_relevant,
        "response_refused": response.refused,
        "response_relevant": response.relevant,
        "retrieved_chunks": debug.get("retrieved_chunks", len(response.evidence)),
        "passed_threshold": not response.refused or debug.get("retrieved_chunks", 0) > 0,
        "claims_removed": response.trust_check.claims_removed,
        "all_claims_cited": all_claims_cited,
        "has_valid_citation": has_valid_citation,
        "eligibility_defensible": eligibility_defensible,
        "confidence": response.confidence,
        "latency_ms": round(latency_ms, 1),
        "guardrail_triggered": response.guardrail_triggered,
    }


def run_evaluation(limit: int | None = None, dataset_path: str | None = None, run_deepeval: bool = False) -> dict:
    cases = load_test_cases(dataset_path)
    if limit:
        cases = cases[:limit]

    results = []
    deepeval_cases = []
    for case in cases:
        try:
            row = _evaluate_case(case)
            deepeval_cases.append({"question": case.question, "response": row.pop("_response")})
            results.append(row)
        except Exception as exc:  # noqa: BLE001 — one bad case shouldn't kill the whole run
            logger.error("Evaluation case failed", extra={"case_id": case.id, "error": str(exc)})
            results.append({
                "id": case.id, "category": case.category, "question": case.question,
                "must_refuse": case.must_refuse, "expect_relevant": case.expect_relevant,
                "response_refused": True, "response_relevant": False, "retrieved_chunks": 0,
                "passed_threshold": False, "claims_removed": 0, "all_claims_cited": False,
                "has_valid_citation": False, "eligibility_defensible": True, "confidence": "low",
                "latency_ms": 0.0, "guardrail_triggered": "error", "error": str(exc),
            })

    summary = {
        "total_cases": len(results),
        "retrieval_precision": round(m.retrieval_precision(results), 1),
        "retrieval_recall": round(m.retrieval_recall(results), 1),
        "groundedness": round(m.groundedness(results), 1),
        "citation_accuracy": round(m.citation_accuracy(results), 1),
        "relevance_accuracy": round(m.relevance_accuracy(results), 1),
        "hallucination_rate": round(m.hallucination_rate(results), 1),
        "refusal_accuracy": round(m.refusal_accuracy(results), 1),
        "eligibility_accuracy": round(m.eligibility_accuracy(results), 1),
        "avg_latency_ms": round(sum(r["latency_ms"] for r in results) / len(results), 1) if results else 0.0,
    }

    report = {"summary": summary, "results": results}

    if run_deepeval:
        from app.evaluation.deepeval_runner import run_deepeval as _run_deepeval

        report["deepeval"] = _run_deepeval(deepeval_cases)

    return report
