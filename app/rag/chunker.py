"""
Text chunking for RAG.

Simple, dependency-free sliding-window chunker over paragraphs/sentences.
Kept deterministic (no LLM calls) so ingestion is fast, reproducible, and
free of hallucination risk at the chunking stage.
"""
from __future__ import annotations

import re

from app.rag.metadata import Chunk, RawDocument, new_id

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?।])\s+")


def _split_sentences(text: str) -> list[str]:
    text = text.strip()
    if not text:
        return []
    return [s.strip() for s in _SENTENCE_SPLIT.split(text) if s.strip()]


def chunk_text(
    text: str,
    max_chars: int = 900,
    overlap_chars: int = 150,
) -> list[str]:
    """Greedy sentence-packing chunker with character-based overlap."""
    sentences = _split_sentences(text)
    if not sentences:
        return []

    chunks: list[str] = []
    current = ""
    for sentence in sentences:
        candidate = (current + " " + sentence).strip() if current else sentence
        if len(candidate) <= max_chars:
            current = candidate
            continue
        if current:
            chunks.append(current)
        # start new chunk, carrying overlap from the tail of the previous one
        overlap = current[-overlap_chars:] if current else ""
        current = (overlap + " " + sentence).strip() if overlap else sentence
        # guard against a single sentence longer than max_chars
        while len(current) > max_chars:
            chunks.append(current[:max_chars])
            current = current[max_chars - overlap_chars:]
    if current:
        chunks.append(current)
    return chunks


def chunk_document(
    doc: RawDocument,
    max_chars: int = 900,
    overlap_chars: int = 150,
) -> list[Chunk]:
    """Chunk a RawDocument into retrievable Chunk objects, preserving provenance."""
    pieces = chunk_text(doc.content, max_chars=max_chars, overlap_chars=overlap_chars)
    chunks: list[Chunk] = []
    for idx, piece in enumerate(pieces):
        chunks.append(
            Chunk(
                chunk_id=new_id("chunk"),
                document_id=doc.document_id,
                title=doc.title,
                text=piece,
                source_type=doc.source_type,
                source_url=doc.source_url,
                section=doc.section,
                language=doc.language,
                category=doc.category,
                crawl_timestamp=doc.crawl_timestamp,
                file_type=doc.file_type,
                chunk_index=idx,
                extra_metadata=dict(doc.extra_metadata),
            )
        )
    return chunks
