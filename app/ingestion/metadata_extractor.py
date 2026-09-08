"""
Best-effort metadata enrichment for ingested documents that didn't come from
the crawler (uploads don't have a URL/section — we infer what we can, but
never invent a source_url).
"""
from __future__ import annotations

from app.graph.graph_builder import CROPS, INDIAN_STATES
from app.rag.metadata import RawDocument


def enrich_metadata(doc: RawDocument) -> RawDocument:
    text_lower = f"{doc.title}\n{doc.content}".lower()
    if not doc.category:
        if "scheme" in text_lower or "yojana" in text_lower:
            doc.category = "user-provided-scheme-document"
        else:
            doc.category = "user-provided-document"

    mentioned_states = [s for s in INDIAN_STATES if s.lower() in text_lower]
    mentioned_crops = [c for c in CROPS if c in text_lower]
    if mentioned_states or mentioned_crops:
        doc.extra_metadata.setdefault("mentioned_states", mentioned_states)
        doc.extra_metadata.setdefault("mentioned_crops", mentioned_crops)
    return doc
