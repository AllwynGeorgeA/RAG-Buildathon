"""
Hybrid retrieval pipeline: dense vector search + BM25 lexical search +
metadata filtering + knowledge graph traversal + Reciprocal-Rank-Fusion
reranking + evidence threshold gate.

This is the base, single-pass retrieval. `app/rag/corrective_rag.py` wraps
this with a Corrective-RAG loop (evaluate -> refine/rewrite -> retry) for
the answer generator to call. Being a separate, single-pass function keeps
this unit-testable without any correction logic in the way.

Note on scoring: ranking order is decided by RRF (scale-free, fair to both
retrievers), but the evidence CONFIDENCE gate (`passed_threshold`) is based
on the top chunk's raw vector cosine-similarity — RRF scores aren't on a
0..1 scale that a fixed threshold could sensibly gate on.
"""
from __future__ import annotations

from app.core.config import get_settings
from app.core.logging import get_logger
from app.graph.graph_queries import GraphMatch, find_schemes_by_document_ids, get_scheme_profile
from app.graph.knowledge_graph import get_knowledge_graph
from app.llm.schemas import RetrievalResult
from app.rag.bm25_index import get_bm25_index
from app.rag.citations import to_evidence_chunks
from app.rag.graph_retriever import GraphRetrievalResult, retrieve_from_graph
from app.rag.reranker import rerank
from app.rag.retriever import vector_search

logger = get_logger(__name__)


def _merge_matches(entity_matches: list[GraphMatch], evidence_matches: list[GraphMatch]) -> list[GraphMatch]:
    """
    Combines crop/state/farmer-type graph matches with evidence-grounded
    matches (schemes whose own pages the retrieved evidence actually came
    from). A scheme found both ways keeps its entity-match reasoning (more
    specific) while gaining any additional supporting documents.
    """
    merged: dict[str, GraphMatch] = {m.scheme_node: m for m in entity_matches}
    for match in evidence_matches:
        if match.scheme_node in merged:
            existing = merged[match.scheme_node]
            existing.supporting_document_ids |= match.supporting_document_ids
        else:
            merged[match.scheme_node] = match
    return sorted(merged.values(), key=lambda m: m.graph_score, reverse=True)


def hybrid_retrieve(
    query: str,
    top_k: int | None = None,
    source_type: str | None = None,
    language: str | None = None,
) -> RetrievalResult:
    settings = get_settings()
    top_k = top_k or settings.rerank_top_k

    vector_hits = vector_search(query, top_k=settings.retrieval_top_k, source_type=source_type, language=language)
    bm25_hits = get_bm25_index().search(query, top_k=settings.retrieval_top_k)
    graph_result: GraphRetrievalResult = retrieve_from_graph(query)

    # Build a document-level graph score boost: documents backing a matched scheme
    # get a small ranking boost, so graph-confirmed relationships surface higher.
    graph_score_by_doc: dict[str, float] = {}
    for match in graph_result.matches:
        for doc_id in match.supporting_document_ids:
            graph_score_by_doc[doc_id] = max(graph_score_by_doc.get(doc_id, 0.0), match.graph_score)

    ranked = rerank(query, vector_hits, bm25_hits, graph_boost_by_document_id=graph_score_by_doc, top_k=top_k)
    evidence_chunks = to_evidence_chunks(ranked, graph_score_by_doc)

    top_vector_score = max((c.vector_score for c in evidence_chunks), default=0.0)
    passed = top_vector_score >= settings.retrieval_score_threshold

    # Complement crop/state/farmer-type matches with evidence-grounded ones:
    # a scheme with broad, non-crop-specific eligibility (e.g. PM-KISAN) has
    # no FOR_CROP/AVAILABLE_IN edges to match against — correctly so — but
    # should still get a scheme card when retrieval clearly found evidence
    # on its own page(s).
    kg = get_knowledge_graph()
    evidence_doc_ids = [c.document_id for c in evidence_chunks if c.document_id]
    evidence_matches = find_schemes_by_document_ids(kg, evidence_doc_ids)
    all_matches = _merge_matches(graph_result.matches, evidence_matches)
    profiles = dict(graph_result.profiles)
    for match in evidence_matches:
        profiles.setdefault(match.scheme_node, get_scheme_profile(kg, match.scheme_node))

    graph_entities_summary = {
        "matches": [
            {
                "scheme_name": m.scheme_name,
                "scheme_node": m.scheme_node,
                "matched_via": m.matched_via,
                "graph_score": m.graph_score,
                "document_ids": sorted(m.supporting_document_ids),
            }
            for m in all_matches
        ],
        "profiles": profiles,
    }

    result = RetrievalResult(
        query=query,
        chunks=evidence_chunks,
        graph_entities=graph_entities_summary,
        top_score=top_vector_score,
        passed_threshold=passed,
        threshold=settings.retrieval_score_threshold,
    )
    logger.info(
        "Hybrid retrieval complete",
        extra={
            "vector_hits": len(vector_hits),
            "bm25_hits": len(bm25_hits),
            "graph_matches": len(graph_result.matches),
            "top_vector_score": round(top_vector_score, 3),
            "passed_threshold": passed,
        },
    )
    return result
