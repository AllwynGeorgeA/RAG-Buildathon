"""
Corrective RAG (CRAG).

Classic RAG hands whatever it retrieves straight to the LLM. CRAG adds a
retrieval-evaluator step that grades the retrieved evidence and takes a
corrective action BEFORE generation:

  CORRECT   — top evidence is strongly relevant -> use it as-is.
  AMBIGUOUS — evidence is only partially relevant -> "knowledge refinement":
              strip weak chunks (decompose/recompose) so noise doesn't
              dilute the LLM's context, keeping only the strips that are
              actually relevant.
  INCORRECT — nothing retrieved is relevant enough -> rewrite the query
              (using extracted crop/state/farmer-type/scheme-vocabulary
              signals) and retry retrieval once.

Unlike the original CRAG paper, there is no live web-search fallback here —
spec section 6 forbids any real-time scraping during a chat turn. If a
rewritten query still comes back INCORRECT, the retrieval guardrail's
existing evidence threshold takes over and the system refuses rather than
guessing.
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from app.core.config import get_settings
from app.core.logging import get_logger
from app.graph.graph_builder import CROPS, INDIAN_STATES
from app.llm.schemas import RetrievalResult
from app.rag.hybrid_retriever import hybrid_retrieve

logger = get_logger(__name__)

# Deliberately NOT a list of generic domain filler words ("eligibility",
# "benefit", "subsidy", ...): stacking those onto a rewritten query inflates
# embedding/BM25 similarity against ANY agriculture-scheme chunk regardless
# of actual topical match, which would defeat the evidence-confidence gate.
# The rewrite only ever adds CONCRETE signals actually extracted from the
# query (a graph-matched scheme name, a named crop, a named state) — if none
# are found, there is nothing safe to add and the original query stands.


class CRAGVerdict(str, Enum):
    CORRECT = "correct"
    AMBIGUOUS = "ambiguous"
    INCORRECT = "incorrect"


@dataclass
class CorrectiveResult:
    retrieval: RetrievalResult
    verdict: CRAGVerdict
    attempts: int
    rewritten_query: str | None
    refined: bool


def evaluate_retrieval(retrieval: RetrievalResult, correct_threshold: float | None = None) -> CRAGVerdict:
    settings = get_settings()
    correct_threshold = correct_threshold if correct_threshold is not None else max(settings.retrieval_score_threshold * 1.6, 0.5)
    if not retrieval.chunks:
        return CRAGVerdict.INCORRECT
    if retrieval.top_score >= correct_threshold:
        return CRAGVerdict.CORRECT
    if retrieval.top_score >= retrieval.threshold:
        return CRAGVerdict.AMBIGUOUS
    return CRAGVerdict.INCORRECT


def refine_evidence(retrieval: RetrievalResult, keep_ratio: float = 0.55) -> RetrievalResult:
    """Knowledge refinement: drop chunks far weaker than the best one, so an
    ambiguous retrieval doesn't dilute the LLM's context with noise."""
    if not retrieval.chunks:
        return retrieval
    top = max(c.vector_score for c in retrieval.chunks)
    keep = [c for c in retrieval.chunks if c.vector_score >= keep_ratio * top] or retrieval.chunks[:1]
    if len(keep) == len(retrieval.chunks):
        return retrieval
    logger.info("CRAG refinement dropped weak chunks", extra={"kept": len(keep), "original": len(retrieval.chunks)})
    return retrieval.model_copy(update={"chunks": keep})


def rewrite_query(original_query: str, graph_entities: dict) -> str:
    """Deterministic query expansion: surface the vocabulary a scheme page
    actually uses, even if the farmer phrased the question colloquially."""
    lower = original_query.lower()
    hints: list[str] = []

    matches = graph_entities.get("matches", [])
    for match in matches[:2]:
        hints.append(match["scheme_name"])

    if not hints:
        for crop in CROPS:
            if crop in lower:
                hints.append(crop)
                break
        for state in INDIAN_STATES:
            if state.lower() in lower:
                hints.append(state)
                break

    if not hints:
        return original_query  # nothing concrete to add — rewriting would only add noise

    return " ".join([original_query] + hints).strip()


def hybrid_retrieve_with_correction(query: str, max_attempts: int = 2, **kwargs) -> CorrectiveResult:
    """The corrective wrapper the answer generator should call."""
    retrieval = hybrid_retrieve(query, **kwargs)
    verdict = evaluate_retrieval(retrieval)
    attempts = 1
    rewritten_query: str | None = None
    refined = False

    if verdict == CRAGVerdict.INCORRECT and attempts < max_attempts:
        rewritten_query = rewrite_query(query, retrieval.graph_entities)
        if rewritten_query and rewritten_query.strip().lower() != query.strip().lower():
            retry = hybrid_retrieve(rewritten_query, **kwargs)
            attempts += 1
            if retry.top_score > retrieval.top_score:
                logger.info(
                    "CRAG query rewrite improved retrieval",
                    extra={"original": query, "rewritten": rewritten_query, "before": retrieval.top_score, "after": retry.top_score},
                )
                retrieval = retry
                verdict = evaluate_retrieval(retrieval)

    if verdict == CRAGVerdict.AMBIGUOUS:
        refined_retrieval = refine_evidence(retrieval)
        refined = refined_retrieval is not retrieval
        retrieval = refined_retrieval

    retrieval = retrieval.model_copy(
        update={
            "correction": {
                "verdict": verdict.value,
                "attempts": attempts,
                "rewritten_query": rewritten_query,
                "refined": refined,
            }
        }
    )
    return CorrectiveResult(retrieval=retrieval, verdict=verdict, attempts=attempts, rewritten_query=rewritten_query, refined=refined)
