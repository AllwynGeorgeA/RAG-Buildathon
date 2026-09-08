"""
Structured contracts for retrieval, answer generation, and verification.

These Pydantic models are the backbone of the anti-hallucination design:
the LLM is forced to return `LLMAnswer`, every claim must cite a chunk_id,
and `app.guardrails.hallucination` verifies each claim against the actual
retrieved evidence before anything reaches the user.
"""
from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field


class SourceType(str, Enum):
    VIKASPEDIA = "vikaspedia"
    TRUSTED_DOCUMENT = "trusted_document"
    USER_UPLOAD = "user_upload"


class EvidenceChunk(BaseModel):
    """One retrieved chunk with full provenance — never sent to the LLM without this."""

    chunk_id: str
    document_id: str
    title: str
    section: str = ""
    source_url: str = ""
    source_type: SourceType
    language: str = "en"
    crawl_timestamp: str | None = None
    text: str
    vector_score: float = 0.0
    bm25_score: float = 0.0
    graph_score: float = 0.0
    final_score: float = 0.0
    rank_sources: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    query: str
    chunks: list[EvidenceChunk] = Field(default_factory=list)
    graph_entities: dict = Field(default_factory=dict)
    top_score: float = 0.0
    passed_threshold: bool = False
    threshold: float = 0.0
    correction: dict = Field(default_factory=dict)  # Corrective-RAG trace: verdict, rewritten_query, attempts


class Claim(BaseModel):
    claim: str
    supported_by: list[str] = Field(default_factory=list)  # chunk_ids


class SchemeEligibilityStatus(str, Enum):
    LIKELY_ELIGIBLE = "likely_eligible"
    POSSIBLY_ELIGIBLE = "possibly_eligible"
    INSUFFICIENT_INFORMATION = "insufficient_information"
    LIKELY_NOT_ELIGIBLE = "likely_not_eligible"


class SchemeMatch(BaseModel):
    scheme_name: str
    match_strength: Literal["strong", "moderate", "weak"] = "moderate"
    why_it_matches: str = ""
    benefits: list[str] = Field(default_factory=list)
    eligibility_points: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    eligibility_status: SchemeEligibilityStatus = SchemeEligibilityStatus.INSUFFICIENT_INFORMATION
    eligibility_explanation: str = ""
    next_step: str = ""
    citations: list[str] = Field(default_factory=list)  # chunk_ids
    source_type: SourceType = SourceType.VIKASPEDIA


class LLMAnswer(BaseModel):
    """The strict JSON contract the LLM must return. Never trust it unverified."""

    answer: str
    relevant: bool
    confidence: Literal["high", "medium", "low"] = "low"
    claims: list[Claim] = Field(default_factory=list)
    schemes: list[SchemeMatch] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    objections_addressed: list[str] = Field(default_factory=list)
    citations: list[str] = Field(default_factory=list)


class TrustCheck(BaseModel):
    evidence_found: bool
    source_verified: bool
    hallucination_checked: bool
    claims_removed: int = 0
    eligibility_complete: bool
    confidence: Literal["high", "medium", "low"]


class FarmerProfile(BaseModel):
    state: str | None = None
    district: str | None = None
    crop: str | None = None
    land_size_hectares: float | None = None
    farmer_type: str | None = None  # small/marginal/large
    irrigation: str | None = None
    category: str | None = None  # SC/ST/general/women etc, only if user provides


class ChatResponse(BaseModel):
    answer: str
    relevant: bool
    confidence: Literal["high", "medium", "low"]
    schemes: list[SchemeMatch] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    follow_up_questions: list[str] = Field(default_factory=list)
    evidence: list[EvidenceChunk] = Field(default_factory=list)
    trust_check: TrustCheck
    refused: bool = False
    refusal_reason: str | None = None
    guardrail_triggered: str | None = None
    generated_at: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    debug: dict | None = None
