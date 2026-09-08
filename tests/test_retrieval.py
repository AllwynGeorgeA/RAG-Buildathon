from __future__ import annotations

from app.rag.chunker import chunk_document, chunk_text
from app.rag.metadata import RawDocument
from app.rag.reranker import rerank
from app.rag.vector_store import VectorHit


class TestChunker:
    def test_chunk_text_respects_max_chars(self):
        text = "This is a sentence. " * 100
        chunks = chunk_text(text, max_chars=100, overlap_chars=20)
        assert len(chunks) > 1
        assert all(len(c) <= 120 for c in chunks)  # small slack for overlap boundary

    def test_chunk_text_empty(self):
        assert chunk_text("") == []

    def test_chunk_document_preserves_provenance(self):
        doc = RawDocument(
            document_id="d1", title="PM-KISAN", content="Sentence one. Sentence two. Sentence three.",
            source_type="vikaspedia", source_url="https://vikaspedia.in/x", section="Benefits",
        )
        chunks = chunk_document(doc, max_chars=1000)
        assert len(chunks) == 1
        assert chunks[0].document_id == "d1"
        assert chunks[0].source_url == "https://vikaspedia.in/x"
        assert chunks[0].section == "Benefits"


class TestReranker:
    def _hit(self, chunk_id, score, doc_id="d1"):
        return VectorHit(chunk_id=chunk_id, text=f"text {chunk_id}", metadata={"document_id": doc_id}, score=score)

    def test_rrf_prefers_chunk_ranked_high_in_both_lists(self):
        vector_hits = [self._hit("a", 0.9), self._hit("b", 0.5), self._hit("c", 0.3)]
        bm25_hits = [self._hit("a", 5.0), self._hit("c", 2.0)]
        ranked = rerank("query", vector_hits, bm25_hits, top_k=3)
        assert ranked[0].hit.chunk_id == "a"  # top of both lists
        assert set(r.hit.chunk_id for r in ranked) == {"a", "b", "c"}

    def test_graph_boost_affects_ranking(self):
        vector_hits = [self._hit("a", 0.5, doc_id="d1"), self._hit("b", 0.5, doc_id="d2")]
        ranked = rerank("query", vector_hits, [], graph_boost_by_document_id={"d2": 1.0}, top_k=2)
        assert ranked[0].hit.chunk_id == "b"  # graph-confirmed document should rank higher

    def test_empty_hits(self):
        assert rerank("query", [], []) == []
