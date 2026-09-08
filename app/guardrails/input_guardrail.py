"""
Input guardrail — the first gate every user query passes through, before any
retrieval or LLM call happens.
"""
from __future__ import annotations

from app.core.exceptions import GuardrailRejection
from app.core.logging import get_logger
from app.guardrails import nemo_adapter
from app.guardrails.prompt_injection import detect_prompt_injection
from app.guardrails.relevance import looks_clearly_off_topic

logger = get_logger(__name__)

MAX_QUERY_LENGTH = 2000

REFUSAL_PROMPT_INJECTION = (
    "I can help with farmer and agriculture schemes using information in my approved "
    "knowledge base. I can't follow instructions embedded in a message, reveal internal "
    "system instructions, or answer using knowledge outside that base."
)

REFUSAL_OFF_TOPIC = (
    "I'm designed to help with farmer and agriculture schemes — eligibility, benefits, "
    "and application information from the approved knowledge base. I don't have evidence "
    "in my knowledge base for that topic."
)

REFUSAL_EMPTY = "Please type a question about a farmer scheme, eligibility, or agriculture support."


def check_input(query: str) -> None:
    """Raises GuardrailRejection if the query should be blocked. Returns None otherwise."""
    if not query or not query.strip():
        raise GuardrailRejection(REFUSAL_EMPTY, guardrail="empty_input")

    trimmed = query.strip()
    if len(trimmed) > MAX_QUERY_LENGTH:
        trimmed = trimmed[:MAX_QUERY_LENGTH]

    is_injection, pattern = detect_prompt_injection(trimmed)
    if is_injection:
        logger.info("Prompt injection blocked", extra={"pattern": pattern})
        raise GuardrailRejection(REFUSAL_PROMPT_INJECTION, guardrail="prompt_injection")

    if looks_clearly_off_topic(trimmed):
        logger.info("Off-topic query blocked", extra={"query_preview": trimmed[:80]})
        raise GuardrailRejection(REFUSAL_OFF_TOPIC, guardrail="relevance")

    # Optional supplementary layer — disabled by default, fails open (see nemo_adapter.py).
    nemo_allowed, nemo_reason = nemo_adapter.nemo_check_input(trimmed)
    if not nemo_allowed:
        logger.info("Blocked by NeMo Guardrails self-check-input layer")
        raise GuardrailRejection(nemo_reason or REFUSAL_PROMPT_INJECTION, guardrail="nemo_guardrails")
