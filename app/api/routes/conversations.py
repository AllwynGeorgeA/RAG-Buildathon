from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from app.conversations import store
from app.core.logging import get_logger
from app.llm.answer_generator import generate_answer
from app.llm.schemas import ChatResponse, FarmerProfile

logger = get_logger(__name__)
router = APIRouter(prefix="/conversations", tags=["conversations"])


class CreateConversationRequest(BaseModel):
    title: str = store.DEFAULT_TITLE


class RenameRequest(BaseModel):
    title: str


class PinRequest(BaseModel):
    pinned: bool


class PostMessageRequest(BaseModel):
    query: str
    profile: FarmerProfile | None = None
    debug: bool = False


@router.get("")
def list_conversations() -> list[dict]:
    return store.list_conversations()


@router.post("")
def create_conversation(request: CreateConversationRequest) -> dict:
    return store.create_conversation(title=request.title)


@router.get("/{conversation_id}")
def get_conversation(conversation_id: str) -> dict:
    conv = store.get_conversation(conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")
    return conv


@router.patch("/{conversation_id}")
def rename_conversation(conversation_id: str, request: RenameRequest) -> dict:
    if not store.rename_conversation(conversation_id, request.title):
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"ok": True}


@router.patch("/{conversation_id}/pin")
def pin_conversation(conversation_id: str, request: PinRequest) -> dict:
    if not store.set_pinned(conversation_id, request.pinned):
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"ok": True}


@router.delete("/{conversation_id}")
def delete_conversation(conversation_id: str) -> dict:
    if not store.delete_conversation(conversation_id):
        raise HTTPException(status_code=404, detail="Conversation not found")
    return {"ok": True}


@router.post("/{conversation_id}/messages", response_model=ChatResponse)
def post_message(conversation_id: str, request: PostMessageRequest) -> ChatResponse:
    conv = store.get_conversation(conversation_id)
    if conv is None:
        raise HTTPException(status_code=404, detail="Conversation not found")

    store.add_message(conversation_id, "user", request.query)

    chat_history = [{"role": m["role"], "content": m["content"]} for m in conv["messages"]]
    response = generate_answer(
        query=request.query,
        profile=request.profile,
        chat_history=chat_history,
        debug=request.debug,
    )
    store.add_message(conversation_id, "assistant", response.answer, response=response)
    return response
