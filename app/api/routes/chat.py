from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.core.config import get_settings
from app.core.logging import get_logger
from app.llm.answer_generator import generate_answer
from app.llm.schemas import ChatResponse, FarmerProfile
from app.websearch.search import search_web

logger = get_logger(__name__)
router = APIRouter()


class ChatRequest(BaseModel):
    query: str
    profile: FarmerProfile | None = None
    chat_history: list[dict] | None = None
    debug: bool = False
    web_search: bool = False


class SchemeCompareRequest(BaseModel):
    scheme_names: list[str] = Field(min_length=2, max_length=4)
    profile: FarmerProfile | None = None


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    response = generate_answer(
        query=request.query,
        profile=request.profile,
        chat_history=request.chat_history,
        debug=request.debug,
    )
    _maybe_attach_web_results(response, request.query, request.web_search)
    return response


def _maybe_attach_web_results(response: ChatResponse, query: str, requested: bool) -> None:
    """Opt-in, best-effort live web search — see app/websearch/search.py.

    Off unless BOTH the operator has enabled it (Settings.web_search_enabled)
    AND the caller explicitly asked for it on this message. Always a
    separate, unverified channel — never blended into `response.answer`.
    """
    if not requested or not get_settings().web_search_enabled:
        return
    try:
        results = search_web(query)
        response.web_results = results
        response.web_search_used = bool(results)
    except Exception:  # noqa: BLE001 — a broken web search must never break the chat answer
        logger.warning("Live web search failed; continuing without it", extra={"query": query})


@router.post("/scheme/compare")
def compare_schemes(request: SchemeCompareRequest) -> dict:
    """Runs one retrieval+answer pass per scheme name and returns them side by side."""
    results = []
    for name in request.scheme_names:
        response = generate_answer(query=f"Tell me about {name}: benefits and eligibility", profile=request.profile)
        results.append({"scheme_name": name, "response": response})
    return {"comparison": results}
