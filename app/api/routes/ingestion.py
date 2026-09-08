from __future__ import annotations

from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.core.config import get_settings
from app.core.logging import get_logger
from app.evaluation.evaluator import run_evaluation
from app.ingestion.pipeline import ingest_documents
from app.rag.metadata import RawDocument

logger = get_logger(__name__)
router = APIRouter()


class IngestRequest(BaseModel):
    # Points at a JSON file of RawDocuments already produced by an OFFLINE crawl
    # (scripts/crawl_vikaspedia.py). This endpoint never scrapes live — it only
    # indexes what's already been fetched, matching spec section 6/15.
    documents_path: str = "data/raw/vikaspedia_documents.json"
    update_graph: bool = True


class EvaluateRequest(BaseModel):
    limit: int | None = None
    run_deepeval: bool = False


@router.post("/ingest")
def ingest(request: IngestRequest) -> dict:
    settings = get_settings()
    path = settings.resolve_path(request.documents_path)
    if not path.exists():
        raise HTTPException(status_code=404, detail=f"Documents file not found: {request.documents_path}")

    import json

    data = json.loads(path.read_text(encoding="utf-8"))
    documents = [RawDocument(**d) for d in data]
    stats = ingest_documents(documents, update_graph=request.update_graph)
    return {"status": "ingested", **stats}


@router.post("/evaluate")
def evaluate(request: EvaluateRequest) -> dict:
    return run_evaluation(limit=request.limit, run_deepeval=request.run_deepeval)
