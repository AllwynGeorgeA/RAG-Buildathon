from __future__ import annotations

from app.rag.citations import citation_block, format_citation, to_evidence_chunks
from app.rag.reranker import RankedChunk
from app.rag.vector_store import VectorHit


def _ranked(chunk_id, text, source_url="https://vikaspedia.in/agriculture/pm-kisan", source_type="vikaspedia"):
    hit = VectorHit(
        chunk_id=chunk_id, text=text,
        metadata={
            "document_id": "doc1", "title": "PM-KISAN", "section": "Benefits",
            "source_url": source_url, "source_type": source_type, "language": "en",
            "crawl_timestamp": "2026-01-01T00:00:00",
        },
        score=0.8,
    )
    return RankedChunk(hit=hit, vector_score=0.8, bm25_score=1.2, graph_boost=0.5, final_score=0.9, rank_sources=["vector", "bm25"])


class TestCitations:
    def test_to_evidence_chunks_preserves_provenance(self):
        chunks = to_evidence_chunks([_ranked("c1", "PM-KISAN gives Rs 6000/year.")])
        assert len(chunks) == 1
        chunk = chunks[0]
        assert chunk.chunk_id == "c1"
        assert chunk.source_url == "https://vikaspedia.in/agriculture/pm-kisan"
        assert chunk.source_type.value == "vikaspedia"
        assert chunk.section == "Benefits"

    def test_format_citation_includes_source_and_section(self):
        chunk = to_evidence_chunks([_ranked("c1", "text")])[0]
        text = format_citation(chunk)
        assert "Vikaspedia" in text
        assert "Benefits" in text
        assert "https://vikaspedia.in/agriculture/pm-kisan" in text

    def test_citation_block_empty_evidence(self):
        assert "No supporting evidence" in citation_block([])

    def test_user_upload_labeled_distinctly(self):
        chunk = to_evidence_chunks([_ranked("c1", "text", source_url="", source_type="user_upload")])[0]
        text = format_citation(chunk)
        assert "User-provided document" in text
        assert "Vikaspedia" not in text
