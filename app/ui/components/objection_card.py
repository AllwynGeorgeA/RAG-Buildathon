"""Renders the objection/misconception-handling response distinctly (spec section 18/42)."""
from __future__ import annotations

import streamlit as st

from app.llm.schemas import ChatResponse
from app.ui.components.evidence_panel import render_evidence_panel


def render_objection_response(response: ChatResponse) -> None:
    st.markdown('<div class="km-card">', unsafe_allow_html=True)
    st.markdown("**🗣️ Addressing your point**")
    st.write(response.answer)
    render_evidence_panel(response.evidence, key_prefix="objection")
    st.markdown("</div>", unsafe_allow_html=True)
