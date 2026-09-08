"""
Hybrid reranker: fuses dense vector search + BM25 lexical search via
Reciprocal Rank Fusion (RRF), then applies a knowledge-graph confirmation
boost. RRF is used (rather than blending raw scores) because vector
cosine-similarity and BM25 scores live on incomparable scales — RRF only
needs each list's *rank order*, which makes the fusion robust without
manual score normalization.

    RRF(chunk) = sum_over_lists( 1 / (k + rank_in_list) )   [k=60, standard default]

Every chunk keeps its raw vector_score and bm25_score too (surfaced in the
Trust Panel / debug view) — RRF is only used to decide ranking order.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.rag.vector_store import VectorHit

RRF_K = 60


@dataclass
class RankedChunk:
    hit: VectorHit
    vector_score: float
    bm25_score: float
    graph_boost: float
    final_score: float
    rank_sources: list[str] = field(default_factory=list)  # which retrievers surfaced this chunk


def _rrf_ranks(hits: list[VectorHit]) -> dict[str, int]:
    return {h.chunk_id: rank for rank, h in enumerate(hits)}


def rerank(
    query: str,
    vector_hits: list[VectorHit],
    bm25_hits: list[VectorHit] | None = None,
    graph_boost_by_document_id: dict[str, float] | None = None,
    top_k: int = 5,
    graph_weight: float = 0.15,
) -> list[RankedChunk]:
    bm25_hits = bm25_hits or []
    graph_boost_by_document_id = graph_boost_by_document_id or {}

    vector_ranks = _rrf_ranks(vector_hits)
    bm25_ranks = _rrf_ranks(bm25_hits)

    vector_by_id = {h.chunk_id: h for h in vector_hits}
    bm25_by_id = {h.chunk_id: h for h in bm25_hits}
    all_ids = set(vector_by_id) | set(bm25_by_id)

    ranked: list[RankedChunk] = []
    for chunk_id in all_ids:
        hit = vector_by_id.get(chunk_id) or bm25_by_id.get(chunk_id)
        sources = []
        rrf_score = 0.0
        vector_score = 0.0
        bm25_score = 0.0

        if chunk_id in vector_ranks:
            rrf_score += 1.0 / (RRF_K + vector_ranks[chunk_id] + 1)
            vector_score = max(vector_by_id[chunk_id].score, 0.0)
            sources.append("vector")
        if chunk_id in bm25_ranks:
            rrf_score += 1.0 / (RRF_K + bm25_ranks[chunk_id] + 1)
            bm25_score = bm25_by_id[chunk_id].score
            sources.append("bm25")

        doc_id = hit.metadata.get("document_id", "")
        graph_boost = graph_boost_by_document_id.get(doc_id, 0.0)
        final_score = rrf_score + graph_weight * graph_boost

        ranked.append(
            RankedChunk(
                hit=hit,
                vector_score=vector_score,
                bm25_score=bm25_score,
                graph_boost=graph_boost,
                final_score=final_score,
                rank_sources=sources,
            )
        )

    ranked.sort(key=lambda r: r.final_score, reverse=True)
    return ranked[:top_k]
