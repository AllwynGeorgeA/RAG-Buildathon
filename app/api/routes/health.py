from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter

from app.core.config import get_settings
from app.graph.knowledge_graph import get_knowledge_graph
from app.llm import openai_client
from app.rag.vector_store import get_vector_store

router = APIRouter()


@router.get("/health")
def health() -> dict:
    settings = get_settings()
    try:
        chunk_count = get_vector_store().count()
        vector_ok = True
    except Exception:  # noqa: BLE001
        chunk_count = 0
        vector_ok = False

    try:
        kg = get_knowledge_graph()
        graph_ok = True
        graph_nodes = kg.node_count()
    except Exception:  # noqa: BLE001
        graph_ok = False
        graph_nodes = 0

    return {
        "status": "ok" if vector_ok and graph_ok else "degraded",
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "demo_mode": settings.demo_mode,
        "vector_store": {"ok": vector_ok, "chunks_indexed": chunk_count},
        "knowledge_graph": {"ok": graph_ok, "nodes": graph_nodes},
        "llm_configured": openai_client.is_available(),
    }
