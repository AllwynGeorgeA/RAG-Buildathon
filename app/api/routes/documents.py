from __future__ import annotations

import shutil

from fastapi import APIRouter, File, HTTPException, UploadFile

from app.core.exceptions import FileTooLargeError, IngestionError, OCRLowConfidenceError, UnsupportedFileTypeError
from app.core.logging import get_logger
from app.core.security import safe_temp_path, sanitize_for_log, validate_upload
from app.graph.graph_builder import CROPS
from app.graph.knowledge_graph import get_knowledge_graph
from app.ingestion.document_loader import load_document
from app.ingestion.pipeline import ingest_documents
from app.rag.vector_store import get_vector_store

logger = get_logger(__name__)
router = APIRouter()


@router.post("/upload")
async def upload_document(file: UploadFile = File(...)) -> dict:
    contents = await file.read()
    try:
        ext = validate_upload(file.filename or "upload", len(contents), declared_mime=file.content_type)
    except (UnsupportedFileTypeError, FileTooLargeError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    temp_path = safe_temp_path(ext)
    try:
        temp_path.write_bytes(contents)
        documents = load_document(temp_path, source_type="user_upload", title=file.filename)
        stats = ingest_documents(documents)
    except (IngestionError, OCRLowConfidenceError) as exc:
        logger.warning("Upload processing failed", extra={"filename": sanitize_for_log(file.filename or "")})
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    finally:
        shutil.rmtree(temp_path, ignore_errors=True) if temp_path.is_dir() else temp_path.unlink(missing_ok=True)

    return {
        "filename": file.filename,
        "source_type": "user_upload",
        "documents_created": stats["documents"],
        "chunks_created": stats["chunks"],
        "message": "Document processed and indexed. You can now ask questions about it.",
    }


@router.get("/schemes")
def list_schemes() -> dict:
    kg = get_knowledge_graph()
    schemes = kg.nodes_of_type("Scheme")
    return {
        "count": len(schemes),
        "schemes": [
            {"name": kg.graph.nodes[s].get("name", s), "node_id": s}
            for s in schemes
        ],
    }


@router.get("/stats")
def stats() -> dict:
    store = get_vector_store()
    kg = get_knowledge_graph()
    return {
        "chunks_indexed": store.count(),
        "schemes_in_graph": len(kg.nodes_of_type("Scheme")),
        "documents_in_graph": len(kg.nodes_of_type("Document")),
        "graph_nodes": kg.node_count(),
        "graph_edges": kg.edge_count(),
        "known_crops_tracked": len(CROPS),
    }
