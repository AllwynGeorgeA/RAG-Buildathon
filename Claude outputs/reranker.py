"""
Cross-encoder reranker — scores (query, chunk) pairs and returns top-k.
Uses a lightweight cross-encoder model that runs on CPU.
"""
from __future__ import annotations
from sentence_transformers import CrossEncoder

_model: CrossEncoder | None = None
_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(_MODEL_NAME)
    return _model


def rerank(query: str, chunks: list[str], top_k: int = 5) -> list[str]:
    """Score all chunks against the query and return top_k by score."""
    if not chunks:
        return []
    model = _get_model()
    pairs = [(query, c) for c in chunks]
    scores = model.predict(pairs)
    ranked = sorted(zip(scores, chunks), key=lambda x: x[0], reverse=True)
    return [c for _, c in ranked[: min(top_k, len(ranked))]]
