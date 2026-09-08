"""
FastAPI application entry point.

    uvicorn app.api.main:app --reload --port 8000

Wraps the exact same `generate_answer` pipeline the Streamlit UI calls —
one behaviour, two front ends.
"""
from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import chat, conversations, documents, health, ingestion
from app.core.config import get_settings
from app.core.exceptions import KrishiMitraError
from app.core.logging import configure_logging, get_logger

configure_logging()
logger = get_logger(__name__)
settings = get_settings()

app = FastAPI(
    title="KrishiMitra AI",
    description="Evidence-first AI for discovering and understanding government schemes for farmers.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if settings.app_env == "development" else [],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router, tags=["health"])
app.include_router(chat.router, tags=["chat"])
app.include_router(conversations.router)
app.include_router(documents.router, tags=["documents"])
app.include_router(ingestion.router, tags=["ingestion"])


@app.exception_handler(KrishiMitraError)
async def krishimitra_error_handler(request: Request, exc: KrishiMitraError) -> JSONResponse:
    logger.warning("Handled application error", extra={"path": str(request.url), "error": str(exc)})
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception) -> JSONResponse:
    logger.error("Unhandled exception", extra={"path": str(request.url), "error": str(exc)})
    return JSONResponse(
        status_code=500,
        content={"detail": "Something went wrong while searching the knowledge base. Please try again."},
    )


# ── Web UI ────────────────────────────────────────────────────────────────
# Serves the green-themed static UI (web/) that calls this same API. Mounted
# under /ui rather than "/" so it can never shadow an API route.
_WEB_DIR = Path(__file__).resolve().parent.parent.parent / "web"
if _WEB_DIR.is_dir():
    app.mount("/ui", StaticFiles(directory=str(_WEB_DIR), html=True), name="ui")

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/ui/chatbot-ui-green.html")
