"""'Show evidence' panel — exact source, section, and a 'View source' link (spec section 45)."""
from __future__ import annotations

import streamlit as st

from app.llm.schemas import EvidenceChunk, SourceType

_SOURCE_LABEL = {
    SourceType.VIKASPEDIA: "📚 Vikaspedia",
    SourceType.TRUSTED_DOCUMENT: "📗 Trusted document",
    SourceType.USER_UPLOAD: "📄 User-provided document",
}


def render_evidence_panel(chunks: list[EvidenceChunk], key_prefix: str = "") -> None:
    if not chunks:
        st.info("No supporting evidence found in the approved knowledge base.")
        return

    with st.expander(f"🔍 Show evidence ({len(chunks)} source excerpt{'s' if len(chunks) != 1 else ''})", expanded=False):
        for i, chunk in enumerate(chunks):
            label = _SOURCE_LABEL.get(chunk.source_type, "Unknown source")
            st.markdown(f"**{label}**" + (f" — {chunk.title}" if chunk.title else ""))
            meta_bits = []
            if chunk.section:
                meta_bits.append(f"Section: *{chunk.section}*")
            if chunk.crawl_timestamp:
                meta_bits.append(f"Indexed: {chunk.crawl_timestamp[:10]}")
            meta_bits.append(f"Relevance: vector={chunk.vector_score:.2f} · bm25={chunk.bm25_score:.2f} · graph={chunk.graph_score:.2f}")
            st.caption(" · ".join(meta_bits))
            st.markdown(f'<div class="km-evidence-item">{chunk.text}</div>', unsafe_allow_html=True)
            if chunk.source_url:
                st.markdown(f"[View source ↗]({chunk.source_url})")
            if i < len(chunks) - 1:
                st.divider()
