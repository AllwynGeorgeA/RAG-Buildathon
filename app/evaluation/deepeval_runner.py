"""
Optional DeepEval integration — an LLM-as-judge evaluation layer that
supplements (never replaces) the deterministic evaluation harness in
`evaluator.py` / `metrics.py`.

The deterministic harness is the one CI/the demo relies on: it needs no API
key, costs nothing, and its pass/fail criteria are exact (was a claim
actually cited, did the guardrail actually refuse). DeepEval's
Faithfulness/Answer-Relevancy metrics are useful qualitative signal on top
of that, but they cost an LLM call per case and are judgment calls rather
than exact checks — hence "optional, additive" (spec section 27 asks for a
lightweight local harness with "optional RAGAS integration if available";
this project uses DeepEval in that same optional role).

Disabled by default (DEEPEVAL_ENABLED=false). Any failure degrades to an
empty/partial report rather than crashing the run.
"""
from __future__ import annotations

from app.core.config import get_settings
from app.core.logging import get_logger
from app.llm import openai_client
from app.llm.schemas import ChatResponse

logger = get_logger(__name__)


def is_available() -> bool:
    settings = get_settings()
    return settings.deepeval_enabled and openai_client.is_available()


def run_deepeval(cases: list[dict]) -> dict:
    """
    `cases`: list of {"question": str, "response": ChatResponse}.
    Returns a summary dict, or {"enabled": False} if DeepEval isn't configured.
    """
    if not is_available():
        return {"enabled": False, "reason": "DEEPEVAL_ENABLED is false or OPENAI_API_KEY is not set"}

    try:
        from deepeval.metrics import AnswerRelevancyMetric, FaithfulnessMetric
        from deepeval.test_case import LLMTestCase
    except ImportError as exc:
        logger.warning("deepeval not installed", extra={"error": str(exc)})
        return {"enabled": False, "reason": "deepeval package not installed"}

    settings = get_settings()
    faithfulness = FaithfulnessMetric(threshold=0.5, model=settings.openai_chat_model, include_reason=False)
    relevancy = AnswerRelevancyMetric(threshold=0.5, model=settings.openai_chat_model, include_reason=False)

    faithfulness_scores, relevancy_scores, evaluated, skipped = [], [], 0, 0

    for case in cases:
        response: ChatResponse = case["response"]
        if response.refused or not response.evidence:
            skipped += 1
            continue
        test_case = LLMTestCase(
            input=case["question"],
            actual_output=response.answer,
            retrieval_context=[c.text for c in response.evidence],
        )
        try:
            faithfulness.measure(test_case)
            faithfulness_scores.append(faithfulness.score)
        except Exception as exc:  # noqa: BLE001
            logger.warning("DeepEval faithfulness metric failed", extra={"error": str(exc)})
        try:
            relevancy.measure(test_case)
            relevancy_scores.append(relevancy.score)
        except Exception as exc:  # noqa: BLE001
            logger.warning("DeepEval relevancy metric failed", extra={"error": str(exc)})
        evaluated += 1

    def _avg(values: list[float]) -> float | None:
        return round(sum(values) / len(values), 3) if values else None

    return {
        "enabled": True,
        "cases_evaluated": evaluated,
        "cases_skipped_refused_or_no_evidence": skipped,
        "avg_faithfulness": _avg(faithfulness_scores),
        "avg_answer_relevancy": _avg(relevancy_scores),
    }
