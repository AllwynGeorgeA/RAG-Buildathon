"""
End-to-end ingestion pipeline:

  RawDocument(s) -> dedupe -> chunk -> embed -> vector store upsert
                 -> knowledge graph merge -> persist raw JSON (audit trail)

Used by:
  - scripts/build_index.py / build_graph.py (Vikaspedia crawl output)
  - app/api/routes/documents.py (user uploads)
  - scripts/demo_seed.py (demo mode)
"""
from __future__ import annotations

import json
from pathlib import Path

from app.core.config import get_settings
from app.core.logging import get_logger
from app.graph.graph_builder import build_graph_from_documents
from app.graph.knowledge_graph import KnowledgeGraph, get_knowledge_graph, reset_knowledge_graph_singleton
from app.ingestion.deduplicator import deduplicate
from app.rag.bm25_index import reset_bm25_index_singleton
from app.rag.chunker import chunk_document
from app.rag.metadata import RawDocument
from app.rag.vector_store import get_vector_store

logger = get_logger(__name__)


def _persist_raw(documents: list[RawDocument], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for doc in documents:
        path = out_dir / f"{doc.document_id}.json"
        path.write_text(json.dumps(vars(doc), indent=2, ensure_ascii=False), encoding="utf-8")


def ingest_documents(
    documents: list[RawDocument],
    update_graph: bool = True,
    persist_raw_dir: str | None = "data/processed",
) -> dict:
    """
    Runs the full pipeline over a batch of RawDocuments and returns stats.
    Idempotent-ish: re-ingesting identical content just re-upserts (same IDs
    won't collide because RawDocument/Chunk IDs are freshly generated per
    call — callers doing incremental crawls should rely on the crawler's own
    manifest to avoid re-submitting unchanged pages).
    """
    settings = get_settings()
    documents = deduplicate(documents)
    if not documents:
        return {"documents": 0, "chunks": 0, "graph_nodes": 0, "graph_edges": 0}

    all_chunks = []
    for doc in documents:
        all_chunks.extend(chunk_document(doc))

    store = get_vector_store()
    store.upsert(all_chunks)
    reset_bm25_index_singleton()  # force rebuild against the newly-upserted content

    graph_stats = {"nodes": 0, "edges": 0}
    if update_graph:
        kg = get_knowledge_graph()
        kg = build_graph_from_documents(documents, kg=kg)
        kg.save()
        reset_knowledge_graph_singleton()  # force reload so the running process picks up changes
        graph_stats = {"nodes": kg.node_count(), "edges": kg.edge_count()}

    if persist_raw_dir:
        _persist_raw(documents, settings.resolve_path(persist_raw_dir))

    logger.info(
        "Ingestion complete",
        extra={"documents": len(documents), "chunks": len(all_chunks), **graph_stats},
    )
    return {
        "documents": len(documents),
        "chunks": len(all_chunks),
        "graph_nodes": graph_stats["nodes"],
        "graph_edges": graph_stats["edges"],
    }
