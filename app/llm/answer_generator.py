"""
The orchestrator: query -> guardrails -> hybrid retrieval -> answer
generation (LLM or deterministic fallback) -> hallucination guard ->
deterministic eligibility engine -> trust panel -> ChatResponse.

This is the single function both the FastAPI `/chat` route and the
Streamlit UI call, so there is exactly one behaviour to reason about, test,
and demo.
"""
from __future__ import annotations

import json
import time

from app.core.config import get_settings
from app.core.exceptions import GuardrailRejection, LLMError
from app.core.logging import get_logger
from app.eligibility.explanation import explain
from app.eligibility.extractor import extract_profile_from_text
from app.eligibility.matcher import assess_eligibility
from app.graph.graph_builder import slug
from app.guardrails.answer_guardrail import check_answer
from app.guardrails.input_guardrail import check_input
from app.guardrails.retrieval_guardrail import check_retrieval
from app.llm import openai_client
from app.llm.prompts import SYSTEM_PROMPT, build_user_prompt
from app.llm.schemas import (
    ChatResponse, Claim, EvidenceChunk, FarmerProfile, LLMAnswer, RetrievalResult,
    SchemeEligibilityStatus, SchemeMatch, SourceType, TrustCheck,
)
from app.objection.handler import detect_objection
from app.objection.prompts import fallback_objection_response
from app.rag.corrective_rag import hybrid_retrieve_with_correction

logger = get_logger(__name__)

MAX_FOLLOWUPS = 2


def _empty_trust_check(evidence_found: bool, confidence: str = "low") -> TrustCheck:
    return TrustCheck(
        evidence_found=evidence_found,
        source_verified=evidence_found,
        hallucination_checked=True,
        claims_removed=0,
        eligibility_complete=False,
        confidence=confidence,  # type: ignore[arg-type]
    )


def _refused_response(reason: str, guardrail: str, evidence: list[EvidenceChunk] | None = None) -> ChatResponse:
    evidence = evidence or []
    return ChatResponse(
        answer=reason,
        relevant=False,
        confidence="low",
        schemes=[],
        missing_information=[],
        follow_up_questions=[],
        evidence=evidence,
        trust_check=_empty_trust_check(evidence_found=bool(evidence)),
        refused=True,
        refusal_reason=reason,
        guardrail_triggered=guardrail,
    )


def _augment_query_with_profile(query: str, profile: FarmerProfile) -> str:
    """Appends known profile attributes not already mentioned in the query text,
    so graph/BM25 matching can use them (see call site for why)."""
    lower = query.lower()
    additions = []
    for value in (profile.crop, profile.state):
        if value and value.lower() not in lower:
            additions.append(value)
    if not additions:
        return query
    return f"{query} {' '.join(additions)}"


def _suggest_follow_ups(profile: FarmerProfile, retrieval: RetrievalResult) -> list[str]:
    questions = []
    if not profile.state:
        questions.append("Which state are you farming in?")
    if not profile.crop:
        questions.append("What crop do you grow?")
    if not profile.land_size_hectares and len(questions) < MAX_FOLLOWUPS:
        questions.append("Roughly how much land do you farm (in acres or hectares)?")
    return questions[:MAX_FOLLOWUPS]


def _scheme_profile_for(scheme_name: str, graph_profiles: dict) -> dict | None:
    target_slug = slug(scheme_name)
    for node_id, profile in graph_profiles.items():
        if target_slug in node_id or target_slug == slug(node_id.split(":", 1)[-1]):
            return profile
    return None


def _apply_eligibility_engine(
    schemes: list[SchemeMatch], farmer_profile: FarmerProfile, graph_profiles: dict
) -> list[SchemeMatch]:
    """
    Overrides every scheme's eligibility_status with the DETERMINISTIC engine's
    verdict — the LLM (if used) never gets the final word on eligibility.
    """
    updated = []
    for scheme in schemes:
        graph_profile = _scheme_profile_for(scheme.scheme_name, graph_profiles)
        if graph_profile and graph_profile.get("eligibility"):
            eligibility_texts = [item["text"] for item in graph_profile["eligibility"]]
        else:
            eligibility_texts = scheme.eligibility_points

        assessment = assess_eligibility(farmer_profile, eligibility_texts)
        merged_missing = list(dict.fromkeys(scheme.missing_information + assessment.missing_information))
        updated.append(
            scheme.model_copy(
                update={
                    "eligibility_status": assessment.status,
                    "eligibility_explanation": explain(assessment),
                    "missing_information": merged_missing,
                }
            )
        )
    return updated


def _confidence_from_score(top_score: float, threshold: float) -> str:
    """Confidence must track actual retrieval strength, not a fixed guess —
    otherwise the Trust Panel's confidence badge would be meaningless."""
    if top_score >= max(threshold * 1.7, 0.55):
        return "high"
    if top_score >= threshold:
        return "medium"
    return "low"


def _merge_duplicate_schemes(schemes: list[SchemeMatch], evidence: list[EvidenceChunk]) -> list[SchemeMatch]:
    """
    The rule-based scheme-name extractor can register the same real-world
    scheme under two names found on the same page (e.g. "PM-KISAN" and
    "Pradhan Mantri Kisan Samman Nidhi"), producing two separate scheme
    cards for one scheme. Detect this by citation-document overlap (both
    cards citing evidence from the same source page) rather than by name
    matching, and merge them into one — keeping the more descriptive name,
    the union of benefits/eligibility points, and whichever eligibility
    verdict is more informative.
    """
    if len(schemes) <= 1:
        return schemes

    doc_id_by_chunk = {c.chunk_id: c.document_id for c in evidence}

    def doc_ids(scheme: SchemeMatch) -> set[str]:
        return {doc_id_by_chunk[c] for c in scheme.citations if c in doc_id_by_chunk}

    merged: list[SchemeMatch] = []
    consumed: set[int] = set()
    for i, scheme in enumerate(schemes):
        if i in consumed:
            continue
        group = [scheme]
        group_docs = doc_ids(scheme)
        for j in range(i + 1, len(schemes)):
            if j in consumed:
                continue
            other_docs = doc_ids(schemes[j])
            if group_docs and other_docs and group_docs & other_docs:
                group.append(schemes[j])
                consumed.add(j)
                group_docs |= other_docs

        if len(group) == 1:
            merged.append(scheme)
            continue

        canonical = max(group, key=lambda s: len(s.scheme_name))
        informative = [s for s in group if s.eligibility_status != SchemeEligibilityStatus.INSUFFICIENT_INFORMATION]
        status_source = informative[0] if informative else canonical
        merged.append(
            canonical.model_copy(
                update={
                    "benefits": list(dict.fromkeys(b for s in group for b in s.benefits))[:6],
                    "eligibility_points": list(dict.fromkeys(e for s in group for e in s.eligibility_points))[:6],
                    "citations": list(dict.fromkeys(c for s in group for c in s.citations))[:8],
                    "eligibility_status": status_source.eligibility_status,
                    "eligibility_explanation": status_source.eligibility_explanation,
                    "match_strength": max((s.match_strength for s in group), key=lambda m: {"weak": 0, "moderate": 1, "strong": 2}[m]),
                }
            )
        )
    return merged


def _build_extractive_answer(
    query: str, retrieval: RetrievalResult, is_objection: bool
) -> LLMAnswer:
    """Deterministic fallback used when OPENAI_API_KEY is not configured."""
    evidence = retrieval.chunks
    graph_matches = retrieval.graph_entities.get("matches", [])
    graph_profiles = retrieval.graph_entities.get("profiles", {})
    confidence = _confidence_from_score(retrieval.top_score, retrieval.threshold)

    if is_objection:
        answer_text = fallback_objection_response(evidence)
        claims = [Claim(claim=c.text[:280], supported_by=[c.chunk_id]) for c in evidence[:3]]
        return LLMAnswer(
            answer=answer_text,
            relevant=bool(evidence),
            confidence=confidence if evidence else "low",
            claims=claims,
            schemes=[],
            missing_information=[],
            objections_addressed=[query[:120]],
            citations=[c.chunk_id for c in evidence[:3]],
        )

    if not evidence:
        return LLMAnswer(answer="", relevant=False, confidence="low")

    lead = "Based on the indexed knowledge base:"
    excerpts = "\n\n".join(f"• {c.text.strip()}" for c in evidence[:3])
    answer_text = f"{lead}\n\n{excerpts}"

    claims = [Claim(claim=c.text[:280], supported_by=[c.chunk_id]) for c in evidence[:3]]

    schemes: list[SchemeMatch] = []
    evidence_by_doc: dict[str, list[str]] = {}
    for c in evidence:
        evidence_by_doc.setdefault(c.document_id, []).append(c.chunk_id)

    for match in graph_matches[:3]:
        profile = graph_profiles.get(match.get("scheme_node", ""), {})
        citations = []
        for doc_id in match.get("document_ids", []):
            citations.extend(evidence_by_doc.get(doc_id, []))
        if not citations:
            continue  # never show a scheme card with zero verifiable evidence
        benefits = [b["text"] for b in profile.get("benefits", [])][:5]
        eligibility_points = [e["text"] for e in profile.get("eligibility", [])][:5]
        why_it_matches = (
            "Found in the knowledge base as directly relevant to your question."
            if match["matched_via"] == ["retrieved_evidence"]
            else f"Matched via: {', '.join(match['matched_via'])}"
        )
        schemes.append(
            SchemeMatch(
                scheme_name=match["scheme_name"],
                match_strength="strong" if match["graph_score"] >= 0.66 else "moderate",
                why_it_matches=why_it_matches,
                benefits=benefits,
                eligibility_points=eligibility_points,
                citations=list(dict.fromkeys(citations))[:5],
                source_type=SourceType.VIKASPEDIA,
            )
        )

    return LLMAnswer(
        answer=answer_text,
        relevant=True,
        confidence=confidence,
        claims=claims,
        schemes=schemes,
        missing_information=[],
        citations=[c.chunk_id for c in evidence[:5]],
    )


def _call_llm(
    query: str, retrieval: RetrievalResult, profile: FarmerProfile, is_objection: bool
) -> LLMAnswer:
    user_prompt = build_user_prompt(query, retrieval.chunks, profile=profile, objection_mode=is_objection)
    raw = openai_client.chat_completion_json(SYSTEM_PROMPT, user_prompt)
    try:
        data = json.loads(raw)
        return LLMAnswer.model_validate(data)
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to parse LLM JSON answer — falling back to extractive", extra={"error": str(exc)})
        raise LLMError(f"Could not parse LLM response: {exc}") from exc


def generate_answer(
    query: str,
    profile: FarmerProfile | None = None,
    chat_history: list[dict] | None = None,
    debug: bool = False,
) -> ChatResponse:
    start = time.perf_counter()
    settings = get_settings()
    profile = profile or FarmerProfile()

    try:
        check_input(query)
    except GuardrailRejection as exc:
        return _refused_response(exc.reason, exc.guardrail)

    profile = extract_profile_from_text(query, base=profile)
    objection = detect_objection(query)

    # Graph/BM25 entity matching only scans the query text — if the farmer's
    # crop/state/type came from the profile panel rather than being typed in
    # this message, surface those terms too so retrieval actually finds them.
    retrieval_query = _augment_query_with_profile(query, profile)

    retrieval_start = time.perf_counter()
    corrective_result = hybrid_retrieve_with_correction(retrieval_query)
    retrieval = corrective_result.retrieval
    retrieval_ms = (time.perf_counter() - retrieval_start) * 1000

    passed, refusal = check_retrieval(retrieval)
    if not passed:
        response = _refused_response(refusal, "retrieval_evidence_threshold", evidence=retrieval.chunks)
        if debug:
            response.debug = _debug_payload(query, retrieval, retrieval_ms, 0.0, None)
        return response

    llm_ms = 0.0
    used_llm = False
    llm_error: str | None = None
    if openai_client.is_available():
        llm_start = time.perf_counter()
        try:
            raw_answer = _call_llm(query, retrieval, profile, objection.is_objection)
            used_llm = True
        except LLMError as exc:
            logger.warning("LLM generation failed — using deterministic fallback", extra={"error": str(exc)})
            llm_error = str(exc)
            raw_answer = _build_extractive_answer(query, retrieval, objection.is_objection)
        llm_ms = (time.perf_counter() - llm_start) * 1000
    else:
        raw_answer = _build_extractive_answer(query, retrieval, objection.is_objection)

    verified_answer, answer_passed, answer_refusal, claims_removed = check_answer(raw_answer, retrieval.chunks)
    if not answer_passed:
        response = _refused_response(answer_refusal, "hallucination_guard", evidence=retrieval.chunks)
        if debug:
            response.debug = _debug_payload(query, retrieval, retrieval_ms, llm_ms, llm_error, used_llm)
        return response

    schemes_with_eligibility = _apply_eligibility_engine(
        verified_answer.schemes, profile, retrieval.graph_entities.get("profiles", {})
    )
    schemes_with_eligibility = _merge_duplicate_schemes(schemes_with_eligibility, retrieval.chunks)
    eligibility_complete = all(
        s.eligibility_status != SchemeEligibilityStatus.INSUFFICIENT_INFORMATION for s in schemes_with_eligibility
    ) if schemes_with_eligibility else False

    follow_ups = [] if objection.is_objection else _suggest_follow_ups(profile, retrieval)
    missing_info = list(dict.fromkeys(
        verified_answer.missing_information + [q for s in schemes_with_eligibility for q in s.missing_information]
    ))

    trust_check = TrustCheck(
        evidence_found=bool(retrieval.chunks),
        source_verified=all(bool(c.source_url) or c.source_type != SourceType.VIKASPEDIA for c in retrieval.chunks) if retrieval.chunks else False,
        hallucination_checked=True,
        claims_removed=claims_removed,
        eligibility_complete=eligibility_complete,
        confidence=verified_answer.confidence,
    )

    response = ChatResponse(
        answer=verified_answer.answer,
        relevant=verified_answer.relevant,
        confidence=verified_answer.confidence,
        schemes=schemes_with_eligibility,
        missing_information=missing_info,
        follow_up_questions=follow_ups,
        evidence=retrieval.chunks,
        trust_check=trust_check,
        refused=False,
    )

    total_ms = (time.perf_counter() - start) * 1000
    if debug or settings.debug_mode:
        response.debug = _debug_payload(query, retrieval, retrieval_ms, llm_ms, llm_error, used_llm, total_ms, profile, objection)

    logger.info(
        "Answer generated",
        extra={"used_llm": used_llm, "claims_removed": claims_removed, "total_ms": round(total_ms, 1)},
    )
    return response


def _debug_payload(query, retrieval, retrieval_ms, llm_ms, llm_error, used_llm=False, total_ms=0.0, profile=None, objection=None):
    return {
        "query": query,
        "used_llm": used_llm,
        "llm_error": llm_error,
        "retrieved_chunks": len(retrieval.chunks),
        "top_score": retrieval.top_score,
        "threshold": retrieval.threshold,
        "graph_matches": [m["scheme_name"] for m in retrieval.graph_entities.get("matches", [])],
        "crag": retrieval.correction,
        "retrieval_latency_ms": round(retrieval_ms, 1),
        "llm_latency_ms": round(llm_ms, 1),
        "total_latency_ms": round(total_ms, 1),
        "farmer_profile": profile.model_dump() if profile else None,
        "objection_detected": objection.objection_type if objection else None,
    }
