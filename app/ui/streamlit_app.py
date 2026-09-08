"""
KrishiMitra AI — Streamlit front-end.

Calls `app.llm.answer_generator.generate_answer` in-process (no HTTP hop
needed for the demo — the FastAPI service in app/api wraps the exact same
function for programmatic/service access).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import streamlit as st

from app.core.config import get_settings
from app.core.exceptions import (
    FileTooLargeError, IngestionError, OCRLowConfidenceError, TranscriptionError, UnsupportedFileTypeError,
)
from app.core.security import safe_temp_path, validate_upload
from app.graph.knowledge_graph import get_knowledge_graph
from app.ingestion.audio_loader import transcribe
from app.ingestion.document_loader import load_document
from app.ingestion.pipeline import ingest_documents
from app.llm.answer_generator import generate_answer
from app.llm.schemas import ChatResponse, FarmerProfile
from app.objection.handler import detect_objection
from app.rag.vector_store import get_vector_store
from app.ui.components.chat import render_chat_response
from app.ui.components.header import render_header
from app.ui.components.sidebar import render_sidebar

st.set_page_config(page_title="KrishiMitra AI", page_icon="🌾", layout="wide")


@st.cache_resource(show_spinner=False)
def _warm_up():
    """Loads the embedding model / vector store / graph once per process (perf requirement)."""
    from app.rag.vector_store import get_vector_store as _gvs
    from app.graph.knowledge_graph import get_knowledge_graph as _gkg

    _gvs()
    _gkg()
    return True


def _load_css() -> None:
    css_path = Path(__file__).parent / "styles" / "theme.css"
    st.markdown(f"<style>{css_path.read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)


def _kb_stats() -> tuple[int, int, int, str | None]:
    store = get_vector_store()
    kg = get_knowledge_graph()
    chunks = store.count()
    schemes = len(kg.nodes_of_type("Scheme"))
    nodes = kg.node_count()
    last_updated = None
    docs = kg.nodes_of_type("Document")
    return chunks, schemes, nodes, last_updated


def _handle_upload(uploaded_file) -> None:
    try:
        ext = validate_upload(uploaded_file.name, uploaded_file.size, declared_mime=uploaded_file.type)
    except (UnsupportedFileTypeError, FileTooLargeError) as exc:
        st.error(str(exc))
        return

    temp_path = safe_temp_path(ext)
    temp_path.write_bytes(uploaded_file.getvalue())
    try:
        with st.spinner("Processing document — extracting, chunking, embedding, indexing..."):
            documents = load_document(temp_path, source_type="user_upload", title=uploaded_file.name)
            stats = ingest_documents(documents)
        st.success(
            f"✅ Document processed: **{uploaded_file.name}** — "
            f"{stats['documents']} section(s), {stats['chunks']} chunk(s) indexed. "
            f"Labeled as **User-provided document** (not Vikaspedia)."
        )
    except (IngestionError, OCRLowConfidenceError) as exc:
        st.error(str(exc))
    finally:
        temp_path.unlink(missing_ok=True)


def _handle_voice(uploaded_audio) -> str | None:
    ext = Path(uploaded_audio.name).suffix.lower() or ".wav"
    temp_path = safe_temp_path(ext)
    temp_path.write_bytes(uploaded_audio.getvalue())
    try:
        with st.spinner("Transcribing voice input..."):
            text = transcribe(temp_path)
        st.info(f"🎙️ Transcribed query: *\"{text}\"*")
        return text
    except TranscriptionError as exc:
        st.error(str(exc))
        return None
    finally:
        temp_path.unlink(missing_ok=True)


def main() -> None:
    _warm_up()
    _load_css()
    settings = get_settings()

    if "messages" not in st.session_state:
        st.session_state.messages = []  # list[{"role": ..., "content": str, "response": ChatResponse|None}]

    chunks_indexed, schemes_indexed, graph_nodes, last_updated = _kb_stats()
    render_header(last_updated, chunks_indexed, schemes_indexed)
    profile = render_sidebar(chunks_indexed, schemes_indexed, graph_nodes)

    tab_chat, tab_uploads, tab_compare, tab_schemes = st.tabs(
        ["💬 Ask KrishiMitra", "📎 Upload documents", "⚖️ Compare schemes", "📋 Indexed schemes"]
    )

    with tab_uploads:
        st.markdown("#### Upload a document")
        st.caption("PDF, image (photo of a scheme notice), or Excel/CSV. Clearly labeled as a **User-provided document**, never mixed silently with Vikaspedia content.")
        col1, col2 = st.columns(2)
        with col1:
            uploaded_file = st.file_uploader("PDF / Image / Excel / CSV", type=["pdf", "png", "jpg", "jpeg", "xlsx", "xls", "csv"])
            if uploaded_file is not None and st.button("Process document"):
                _handle_upload(uploaded_file)
        with col2:
            uploaded_audio = st.file_uploader("🎙️ Voice question (wav/mp3/m4a)", type=["wav", "mp3", "m4a"])
            if uploaded_audio is not None and st.button("Transcribe & ask"):
                text = _handle_voice(uploaded_audio)
                if text:
                    st.session_state["pending_query"] = text

    with tab_schemes:
        st.markdown("#### Schemes currently in the knowledge graph")
        kg = get_knowledge_graph()
        scheme_nodes = kg.nodes_of_type("Scheme")
        if not scheme_nodes:
            st.info("No schemes indexed yet. Run the ingestion pipeline (see README) or enable DEMO_MODE.")
        else:
            for node in scheme_nodes:
                st.markdown(f"- **{kg.graph.nodes[node].get('name', node)}**")

    with tab_compare:
        st.markdown("#### Compare schemes")
        kg = get_knowledge_graph()
        scheme_names = [kg.graph.nodes[n].get("name", n) for n in kg.nodes_of_type("Scheme")]
        selected = st.multiselect("Pick 2–4 schemes to compare", scheme_names, max_selections=4)
        if len(selected) >= 2 and st.button("Compare"):
            cols = st.columns(len(selected))
            for col, name in zip(cols, selected):
                with col:
                    st.markdown(f"**{name}**")
                    with st.spinner("Retrieving evidence..."):
                        resp = generate_answer(query=f"Tell me about {name}: benefits and eligibility", profile=profile)
                    if resp.refused:
                        st.warning(resp.answer)
                    else:
                        for scheme in resp.schemes:
                            if scheme.scheme_name == name or name.lower() in scheme.scheme_name.lower():
                                st.write("**Benefits:**")
                                for b in scheme.benefits[:4]:
                                    st.markdown(f"- {b}")
                                st.write("**Eligibility:**")
                                for e in scheme.eligibility_points[:4]:
                                    st.markdown(f"- {e}")
                                from app.eligibility.explanation import status_label
                                st.markdown(f"**Assessment:** {status_label(scheme.eligibility_status)}")
                                break
                        else:
                            st.caption("No verifiable evidence-backed card for this scheme yet.")

    with tab_chat:
        st.markdown("###### Try asking:")
        suggestion_cols = st.columns(3)
        suggestions = [
            "I am a farmer growing rice. What schemes can help me?",
            "Am I eligible for any farmer assistance?",
            "Show schemes related to soil health",
        ]
        for col, suggestion in zip(suggestion_cols, suggestions):
            if col.button(suggestion, use_container_width=True):
                st.session_state["pending_query"] = suggestion

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                if msg["role"] == "user":
                    st.write(msg["content"])
                else:
                    render_chat_response(msg["response"], is_objection=msg.get("is_objection", False))

        pending = st.session_state.pop("pending_query", None)
        user_query = st.chat_input("Ask about a farmer scheme, eligibility, or benefit...") or pending

        if user_query:
            st.session_state.messages.append({"role": "user", "content": user_query})
            with st.chat_message("user"):
                st.write(user_query)

            objection = detect_objection(user_query)
            debug = st.session_state.get("debug_mode", settings.debug_mode)

            with st.chat_message("assistant"):
                with st.spinner("Searching the approved knowledge base..."):
                    response: ChatResponse = generate_answer(query=user_query, profile=profile, debug=debug)
                render_chat_response(response, is_objection=objection.is_objection)

            st.session_state.messages.append(
                {"role": "assistant", "content": response.answer, "response": response, "is_objection": objection.is_objection}
            )

    st.markdown(
        '<div class="km-footer-note">KrishiMitra AI answers only from the indexed knowledge base — '
        "never live web search during chat. Every claim is traceable to a cited source.</div>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
