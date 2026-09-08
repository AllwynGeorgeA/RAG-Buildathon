"""
Converts internal retrieval results into `EvidenceChunk` objects — the only
form of evidence the LLM and the UI are allowed to see. Also formats
human-readable citation strings ("Source: Vikaspedia — Section: ...").
"""
from __future__ import annotations

from app.llm.schemas import EvidenceChunk, SourceType
from app.rag.reranker import RankedChunk


def _source_type(value: str) -> SourceType:
    try:
        return SourceType(value)
    except ValueError:
        return SourceType.USER_UPLOAD


def to_evidence_chunks(ranked: list[RankedChunk], graph_score_by_document_id: dict[str, float] | None = None) -> list[EvidenceChunk]:
    graph_score_by_document_id = graph_score_by_document_id or {}
    chunks: list[EvidenceChunk] = []
    for r in ranked:
        meta = r.hit.metadata
        doc_id = meta.get("document_id", "")
        chunks.append(
            EvidenceChunk(
                chunk_id=r.hit.chunk_id,
                document_id=doc_id,
                title=meta.get("title", ""),
                section=meta.get("section", ""),
                source_url=meta.get("source_url", ""),
                source_type=_source_type(meta.get("source_type", "user_upload")),
                language=meta.get("language", "en"),
                crawl_timestamp=meta.get("crawl_timestamp"),
                text=r.hit.text,
                vector_score=r.vector_score,
                bm25_score=r.bm25_score,
                graph_score=graph_score_by_document_id.get(doc_id, r.graph_boost),
                final_score=r.final_score,
                rank_sources=r.rank_sources,
                metadata=meta,
            )
        )
    return chunks


def format_citation(chunk: EvidenceChunk) -> str:
    label = {
        SourceType.VIKASPEDIA: "Vikaspedia",
        SourceType.TRUSTED_DOCUMENT: "Trusted document",
        SourceType.USER_UPLOAD: "User-provided document",
    }.get(chunk.source_type, "Unknown source")

    parts = [f"📚 {label}"]
    if chunk.title:
        parts.append(f"Title: {chunk.title}")
    if chunk.section:
        parts.append(f"Section: {chunk.section}")
    if chunk.source_url:
        parts.append(f"Source: {chunk.source_url}")
    if chunk.crawl_timestamp:
        parts.append(f"Indexed: {chunk.crawl_timestamp[:10]}")
    return "\n".join(parts)


def citation_block(chunks: list[EvidenceChunk]) -> str:
    if not chunks:
        return "No supporting evidence found in the approved knowledge base."
    return "\n\n".join(format_citation(c) for c in chunks)
