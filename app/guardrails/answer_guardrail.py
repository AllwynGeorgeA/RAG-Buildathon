"""
Answer guardrail — the final gate before a response reaches the user.

Runs the hallucination verifier over the LLM's structured answer and decides
whether the (possibly claim-pruned) answer is still safe to show, or whether
everything was unsupported and the system must refuse outright.
"""
from __future__ import annotations

from app.core.logging import get_logger
from app.guardrails.hallucination import verify_answer
from app.llm.schemas import EvidenceChunk, LLMAnswer, SchemeMatch

logger = get_logger(__name__)

REFUSAL_ALL_CLAIMS_UNSUPPORTED = (
    "I found some related material, but I could not verify a reliable, evidence-backed "
    "answer to your question against the approved knowledge base. Please try rephrasing, "
    "or ask about a specific scheme name."
)


def _filter_scheme_citations(schemes: list[SchemeMatch], valid_chunk_ids: set[str]) -> list[SchemeMatch]:
    filtered = []
    for scheme in schemes:
        citations = [c for c in scheme.citations if c in valid_chunk_ids]
        if not citations:
            # A scheme card with zero verifiable citations must not be shown as fact.
            logger.info("Dropping scheme card with no verifiable citation", extra={"scheme": scheme.scheme_name})
            continue
        filtered.append(scheme.model_copy(update={"citations": citations}))
    return filtered


def check_answer(answer: LLMAnswer, evidence: list[EvidenceChunk]) -> tuple[LLMAnswer, bool, str | None, int]:
    """
    Returns (verified_answer, passed, refusal_message_if_failed, claims_removed_count).
    """
    valid_chunk_ids = {c.chunk_id for c in evidence}
    verified_answer, removed = verify_answer(answer, evidence)
    verified_answer = verified_answer.model_copy(
        update={"schemes": _filter_scheme_citations(verified_answer.schemes, valid_chunk_ids)}
    )

    had_any_content = len(answer.claims) > 0 or len(answer.schemes) > 0
    nothing_survived = len(verified_answer.claims) == 0 and len(verified_answer.schemes) == 0

    if had_any_content and nothing_survived:
        return verified_answer, False, REFUSAL_ALL_CLAIMS_UNSUPPORTED, removed

    return verified_answer, True, None, removed
