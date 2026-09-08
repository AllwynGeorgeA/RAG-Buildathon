"""
Hallucination guard — the claim verifier.

Every claim the LLM makes must cite chunk_ids that (a) actually exist in the
retrieved evidence set and (b) lexically overlap with the claim text enough
to be plausible support (catches an LLM citing a real chunk_id for a claim
the chunk doesn't actually say). Claims failing either check are dropped.
If ALL claims fail, the whole answer is rejected upstream.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

from app.core.logging import get_logger
from app.llm.schemas import Claim, EvidenceChunk, LLMAnswer

logger = get_logger(__name__)

_WORD_RE = re.compile(r"[a-zA-Z0-9ऀ-෿]{4,}")
MIN_LEXICAL_OVERLAP = 0.12  # fraction of claim's significant words found in the cited chunk(s)

_STOPWORDS = {
    "this", "that", "with", "from", "have", "will", "your", "their", "there",
    "which", "about", "under", "these", "those", "into", "such", "also",
    "scheme", "schemes", "farmer", "farmers", "government",
}


def _significant_words(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text) if w.lower() not in _STOPWORDS}


@dataclass
class VerifiedClaim:
    claim: str
    supported_by: list[str]
    grounded: bool
    overlap_ratio: float


def verify_claim(claim: Claim, evidence_by_id: dict[str, EvidenceChunk]) -> VerifiedClaim:
    valid_ids = [cid for cid in claim.supported_by if cid in evidence_by_id]
    if not valid_ids:
        return VerifiedClaim(claim=claim.claim, supported_by=[], grounded=False, overlap_ratio=0.0)

    claim_words = _significant_words(claim.claim)
    if not claim_words:
        return VerifiedClaim(claim=claim.claim, supported_by=valid_ids, grounded=True, overlap_ratio=1.0)

    combined_chunk_words: set[str] = set()
    for cid in valid_ids:
        combined_chunk_words |= _significant_words(evidence_by_id[cid].text)

    overlap = len(claim_words & combined_chunk_words) / len(claim_words)
    grounded = overlap >= MIN_LEXICAL_OVERLAP
    return VerifiedClaim(claim=claim.claim, supported_by=valid_ids, grounded=grounded, overlap_ratio=round(overlap, 3))


def verify_answer(answer: LLMAnswer, evidence: list[EvidenceChunk]) -> tuple[LLMAnswer, int]:
    """
    Verify every claim in `answer` against `evidence`. Returns a new LLMAnswer
    with unsupported/ungrounded claims removed, plus a count of removed claims.
    """
    evidence_by_id = {c.chunk_id: c for c in evidence}
    kept_claims: list[Claim] = []
    removed = 0

    for claim in answer.claims:
        verified = verify_claim(claim, evidence_by_id)
        if verified.grounded:
            kept_claims.append(Claim(claim=claim.claim, supported_by=verified.supported_by))
        else:
            removed += 1
            logger.info(
                "Claim rejected by hallucination guard",
                extra={"claim_preview": claim.claim[:100], "overlap": verified.overlap_ratio},
            )

    valid_chunk_ids = set(evidence_by_id.keys())
    verified_citations = [c for c in answer.citations if c in valid_chunk_ids]

    verified_answer = answer.model_copy(update={"claims": kept_claims, "citations": verified_citations})
    return verified_answer, removed
