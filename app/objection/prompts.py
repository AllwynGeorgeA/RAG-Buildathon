"""
Objection-specific response templates for the deterministic (no-LLM)
fallback path, and the addendum used when the LLM path is available (the
LLM path also receives `objection_mode=True` via app.llm.prompts).
"""
from __future__ import annotations

from app.llm.schemas import EvidenceChunk

NO_EVIDENCE_FOR_OBJECTION = (
    "The indexed source does not provide enough information to confirm or deny that claim."
)


def fallback_objection_response(evidence: list[EvidenceChunk]) -> str:
    """Deterministic objection response used when no LLM is configured."""
    if not evidence:
        return (
            "I couldn't find evidence in the indexed knowledge base that addresses this claim. "
            f"{NO_EVIDENCE_FOR_OBJECTION}"
        )
    lead = "I couldn't find evidence in the indexed source that confirms that statement. Here is what the approved knowledge base actually says:"
    excerpts = "\n\n".join(f"— {c.text.strip()}" for c in evidence[:3])
    return f"{lead}\n\n{excerpts}"
