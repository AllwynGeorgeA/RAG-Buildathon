"""Renders a scheme's deterministic eligibility verdict (never a bare 'yes')."""
from __future__ import annotations

import streamlit as st

from app.eligibility.explanation import status_label
from app.llm.schemas import SchemeMatch


def render_eligibility(scheme: SchemeMatch) -> None:
    st.markdown(f'<span class="km-pill {scheme.eligibility_status.value}">{status_label(scheme.eligibility_status)}</span>', unsafe_allow_html=True)
    if scheme.eligibility_explanation:
        st.caption(scheme.eligibility_explanation)
    if scheme.missing_information:
        st.markdown("**What is still missing:**")
        for item in scheme.missing_information:
            st.markdown(f"- {item}")
