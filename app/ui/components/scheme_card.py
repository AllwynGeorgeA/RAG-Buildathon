"""The core differentiator card: Scheme -> Why it matches -> Benefits ->
Eligibility -> Missing info -> Next step -> Evidence -> Confidence (spec section 19)."""
from __future__ import annotations

import streamlit as st

from app.llm.schemas import SchemeMatch
from app.ui.components.eligibility_card import render_eligibility
from app.ui.components.evidence_panel import render_evidence_panel


def render_scheme_card(scheme: SchemeMatch, evidence_by_id: dict, index: int = 0) -> None:
    st.markdown(f'<div class="km-scheme-card {scheme.match_strength}">', unsafe_allow_html=True)

    strength_icon = {"strong": "🟢 Strong match", "moderate": "🟡 Moderate match", "weak": "⚪ Weak match"}
    st.markdown(f'<div class="km-scheme-title">🌾 {scheme.scheme_name}</div>', unsafe_allow_html=True)
    st.markdown(f'<span class="km-pill {scheme.match_strength}">{strength_icon.get(scheme.match_strength, scheme.match_strength)}</span>', unsafe_allow_html=True)

    if scheme.why_it_matches:
        st.markdown('<div class="km-section-label">Why it matches</div>', unsafe_allow_html=True)
        st.write(scheme.why_it_matches)

    if scheme.benefits:
        st.markdown('<div class="km-section-label">Benefits</div>', unsafe_allow_html=True)
        for b in scheme.benefits:
            st.markdown(f"- {b}")

    if scheme.eligibility_points:
        st.markdown('<div class="km-section-label">Eligibility (as documented)</div>', unsafe_allow_html=True)
        for e in scheme.eligibility_points:
            st.markdown(f"- {e}")

    st.markdown('<div class="km-section-label">Eligibility assessment</div>', unsafe_allow_html=True)
    render_eligibility(scheme)

    if scheme.next_step:
        st.markdown('<div class="km-section-label">Next step</div>', unsafe_allow_html=True)
        st.write(scheme.next_step)

    cited_chunks = [evidence_by_id[c] for c in scheme.citations if c in evidence_by_id]
    render_evidence_panel(cited_chunks, key_prefix=f"scheme_{index}")

    st.markdown("</div>", unsafe_allow_html=True)
