"""Sidebar: Farmer Profile panel (optional, session-only) + Knowledge Base stats + debug toggle."""
from __future__ import annotations

import streamlit as st

from app.core.config import get_settings
from app.graph.graph_builder import CROPS, INDIAN_STATES
from app.llm.schemas import FarmerProfile


def render_sidebar(chunks_indexed: int, schemes_indexed: int, graph_nodes: int) -> FarmerProfile:
    settings = get_settings()
    with st.sidebar:
        st.markdown("### 👤 Farmer Profile")
        st.caption("Optional — helps narrow eligibility. Stored only in this session.")

        state = st.selectbox("State", [""] + INDIAN_STATES, index=0, key="profile_state")
        crop = st.selectbox("Crop", [""] + CROPS, index=0, key="profile_crop")
        land = st.number_input("Landholding (hectares)", min_value=0.0, max_value=1000.0, value=0.0, step=0.1, key="profile_land")
        farmer_type = st.selectbox(
            "Farmer category",
            ["", "small_marginal", "small", "marginal", "large", "landless", "tenant", "women"],
            key="profile_farmer_type",
        )
        irrigation = st.selectbox("Irrigation", ["", "irrigated", "rainfed", "partially irrigated"], key="profile_irrigation")

        st.divider()
        st.markdown("### 📊 Knowledge Base")
        st.metric("Chunks indexed", chunks_indexed)
        st.metric("Schemes in graph", schemes_indexed)
        st.metric("Graph nodes", graph_nodes)
        if settings.demo_mode:
            st.caption("🧪 Running in DEMO_MODE with a seeded demo knowledge base.")

        st.divider()
        debug = st.checkbox("🛠 Debug mode", value=settings.debug_mode, key="debug_mode_toggle")
        st.session_state["debug_mode"] = debug

        st.divider()
        green_ui_url = f"http://localhost:{settings.api_port}/ui/chatbot-ui-green.html"
        st.link_button("🟢 Open modern Green UI", green_ui_url, use_container_width=True)
        st.caption(f"Served by the FastAPI app — requires `uvicorn app.api.main:app --port {settings.api_port}` running.")

        st.divider()
        st.caption("KrishiMitra AI answers only from the indexed knowledge base — never live web search during chat.")

    return FarmerProfile(
        state=state or None,
        crop=crop or None,
        land_size_hectares=land or None,
        farmer_type=farmer_type or None,
        irrigation=irrigation or None,
    )
