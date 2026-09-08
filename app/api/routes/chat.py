from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.core.logging import get_logger
from app.llm.answer_generator import generate_answer
from app.llm.schemas import ChatResponse, FarmerProfile

logger = get_logger(__name__)
router = APIRouter()


class ChatRequest(BaseModel):
    query: str
    profile: FarmerProfile | None = None
    chat_history: list[dict] | None = None
    debug: bool = False


class SchemeCompareRequest(BaseModel):
    scheme_names: list[str] = Field(min_length=2, max_length=4)
    profile: FarmerProfile | None = None


@router.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    return generate_answer(
        query=request.query,
        profile=request.profile,
        chat_history=request.chat_history,
        debug=request.debug,
    )


@router.post("/scheme/compare")
def compare_schemes(request: SchemeCompareRequest) -> dict:
    """Runs one retrieval+answer pass per scheme name and returns them side by side."""
    results = []
    for name in request.scheme_names:
        response = generate_answer(query=f"Tell me about {name}: benefits and eligibility", profile=request.profile)
        results.append({"scheme_name": name, "response": response})
    return {"comparison": results}
