"""Content-hash based deduplication across ingestion runs."""
from __future__ import annotations

from app.rag.metadata import RawDocument


def deduplicate(documents: list[RawDocument]) -> list[RawDocument]:
    """Drop documents whose content_hash we've already seen in this batch."""
    seen: set[str] = set()
    unique: list[RawDocument] = []
    for doc in documents:
        if doc.content_hash_value in seen:
            continue
        seen.add(doc.content_hash_value)
        unique.append(doc)
    return unique
