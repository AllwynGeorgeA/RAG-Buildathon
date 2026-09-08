"""Small reusable confidence badge."""
from __future__ import annotations

import streamlit as st

_LABELS = {"high": "🟢 High confidence", "medium": "🟡 Medium confidence", "low": "🔴 Low confidence"}


def render_confidence_badge(confidence: str) -> None:
    css_class = f"km-confidence-{confidence}"
    st.markdown(f'<span class="{css_class}">{_LABELS.get(confidence, confidence)}</span>', unsafe_allow_html=True)
