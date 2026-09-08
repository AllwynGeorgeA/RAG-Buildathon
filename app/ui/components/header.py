"""Top banner: product identity + knowledge-base freshness label (spec section 6/20)."""
from __future__ import annotations

import streamlit as st

from app.core.config import get_settings


def render_header(kb_last_updated: str | None, chunks_indexed: int, schemes_indexed: int) -> None:
    settings = get_settings()
    demo_badge = "🧪 Demo knowledge base" if settings.demo_mode else "📚 Live indexed knowledge base"
    freshness = kb_last_updated[:19].replace("T", " ") if kb_last_updated else "not yet ingested"

    st.markdown(
        f"""
        <div class="km-header">
          <div>
            <h1>🌾 KrishiMitra AI</h1>
            <p>Evidence-first AI for discovering and understanding government schemes for farmers</p>
          </div>
          <div style="text-align:right;">
            <div class="km-badge">{demo_badge}</div>
            <div style="font-size:0.78rem; color:#d7ecdd; margin-top:6px;">
              Knowledge base last updated: {freshness} &nbsp;·&nbsp; {chunks_indexed} chunks &nbsp;·&nbsp; {schemes_indexed} schemes
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
