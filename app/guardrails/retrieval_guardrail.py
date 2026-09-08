"""
Retrieval guardrail — the "NO EVIDENCE = NO CLAIM" gate.

Runs after hybrid retrieval, before the LLM is called. If evidence is too
weak, we refuse rather than let the LLM improvise.
"""
from __future__ import annotations

from app.core.logging import get_logger
from app.llm.schemas import RetrievalResult

logger = get_logger(__name__)

REFUSAL_NO_EVIDENCE = (
    "I couldn't find enough evidence in the approved knowledge base to answer that "
    "reliably. You can:\n"
    "• Rephrase your question with more detail (crop, state, farmer type)\n"
    "• Upload a related document (PDF/image/Excel) for me to check\n"
    "• Browse the indexed schemes list"
)


def check_retrieval(result: RetrievalResult) -> tuple[bool, str | None]:
    """Return (passed, refusal_message_if_failed)."""
    if not result.chunks or not result.passed_threshold:
        logger.info(
            "Retrieval guardrail refused — insufficient evidence",
            extra={"top_score": result.top_score, "threshold": result.threshold, "chunks": len(result.chunks)},
        )
        return False, REFUSAL_NO_EVIDENCE
    return True, None
