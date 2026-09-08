"""
Vector-only retrieval wrapper (metadata filtering + vector similarity).
Used as the first stage of the hybrid retriever.
"""
from __future__ import annotations

from app.core.config import get_settings
from app.rag.vector_store import VectorHit, get_vector_store


def vector_search(
    query: str,
    top_k: int | None = None,
    source_type: str | None = None,
    language: str | None = None,
) -> list[VectorHit]:
    settings = get_settings()
    where: dict = {}
    conditions = []
    if source_type:
        conditions.append({"source_type": source_type})
    if language:
        conditions.append({"language": language})
    if len(conditions) == 1:
        where = conditions[0]
    elif len(conditions) > 1:
        where = {"$and": conditions}

    store = get_vector_store()
    return store.search(query, top_k=top_k or settings.retrieval_top_k, where=where or None)
