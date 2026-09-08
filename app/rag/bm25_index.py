"""
BM25 lexical index — the other half of hybrid search alongside dense vector
similarity. Exact keyword/acronym matches ("PM-KISAN", "PMFBY", specific
₹ amounts) are exactly what BM25 is good at and dense embeddings sometimes
miss, so both run on every query and are fused in `reranker.py`.

Rebuilt from the vector store's own documents (single source of truth for
chunk text/metadata — no separate corpus to keep in sync). Cheap to rebuild
at hackathon scale (hundreds–low thousands of chunks), so we do it lazily
on first use per process and invalidate after ingestion.
"""
from __future__ import annotations

import re
import threading
from dataclasses import dataclass

from app.core.logging import get_logger
from app.rag.vector_store import VectorHit, get_vector_store

logger = get_logger(__name__)

_WORD_RE = re.compile(r"[a-zA-Z0-9ऀ-෿\-]+")


def _tokenize(text: str) -> list[str]:
    return [w.lower() for w in _WORD_RE.findall(text)]


@dataclass
class _Corpus:
    chunk_ids: list[str]
    texts: list[str]
    metadatas: list[dict]


class BM25Index:
    def __init__(self):
        self._bm25 = None
        self._corpus: _Corpus | None = None

    def _build(self) -> None:
        from rank_bm25 import BM25Okapi

        store = get_vector_store()
        ids, docs, metas = store.get_all()
        self._corpus = _Corpus(chunk_ids=ids, texts=docs, metadatas=metas)
        tokenized = [_tokenize(t) for t in docs] or [[]]
        self._bm25 = BM25Okapi(tokenized) if docs else None
        logger.info("BM25 index built", extra={"documents": len(docs)})

    def search(self, query: str, top_k: int = 8) -> list[VectorHit]:
        if self._bm25 is None or self._corpus is None:
            self._build()
        if self._bm25 is None or not self._corpus.chunk_ids:
            return []

        scores = self._bm25.get_scores(_tokenize(query))
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        hits = []
        for i in ranked_idx:
            if scores[i] <= 0:
                continue
            hits.append(
                VectorHit(
                    chunk_id=self._corpus.chunk_ids[i],
                    text=self._corpus.texts[i],
                    metadata=self._corpus.metadatas[i] or {},
                    score=float(scores[i]),
                )
            )
        return hits


_lock = threading.Lock()
_index: BM25Index | None = None


def get_bm25_index() -> BM25Index:
    global _index
    if _index is None:
        with _lock:
            if _index is None:
                _index = BM25Index()
    return _index


def reset_bm25_index_singleton() -> None:
    """Called after ingestion so the next query rebuilds against fresh content."""
    global _index
    _index = None
