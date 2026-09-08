"""
Central application configuration.

Every module reads settings from here (via `get_settings()`), never from
`os.environ` directly, and never hardcodes a secret or a path. Backed by
Pydantic Settings + a `.env` file.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Repository root = two levels up from this file (app/core/config.py -> app/ -> root)
REPO_ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(REPO_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # ---- LLM ----
    openai_api_key: str = ""
    openai_chat_model: str = "gpt-4o-mini"
    openai_audio_model: str = "whisper-1"
    openai_request_timeout_seconds: int = 30

    # ---- Embeddings ----
    embedding_provider: str = "sentence_transformers"
    embedding_model: str = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"
    embedding_device: str = "cpu"

    # ---- Vector store ----
    vector_store_provider: str = "chroma"
    vector_store_path: str = "data/vector_store"
    vector_store_collection: str = "krishimitra_schemes"

    # ---- Knowledge graph ----
    knowledge_graph_path: str = "data/knowledge_graph/graph.gpickle"

    # ---- Conversation history ----
    conversations_db_path: str = "data/conversations.db"

    # ---- Retrieval ----
    retrieval_top_k: int = 8
    retrieval_score_threshold: float = 0.32
    rerank_top_k: int = 5

    # ---- OCR / STT ----
    ocr_provider: str = "pytesseract"
    ocr_min_confidence: int = 45
    stt_provider: str = "openai"
    tesseract_cmd: str = ""

    # ---- Crawler ----
    crawl_config_path: str = "config/crawl.yaml"
    crawl_user_agent: str = "KrishiMitraAI-Bot/1.0 (+educational-research)"
    crawl_rate_limit_seconds: float = 2.0
    crawl_max_retries: int = 3
    crawl_request_timeout_seconds: int = 30

    # ---- Optional supplementary guardrail/eval layers ----
    nemo_guardrails_enabled: bool = False
    nemo_guardrails_config_path: str = "config/nemo_guardrails"
    deepeval_enabled: bool = False

    # ---- Uploads / security ----
    max_upload_size_mb: int = 15
    allowed_upload_extensions: str = ".pdf,.png,.jpg,.jpeg,.xlsx,.xls,.csv,.wav,.mp3,.m4a"

    # ---- App behaviour ----
    demo_mode: bool = True
    debug_mode: bool = False
    default_language: str = "en"
    app_env: str = "development"
    log_level: str = "INFO"

    # ---- API ----
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    @property
    def allowed_upload_extensions_list(self) -> list[str]:
        return [e.strip().lower() for e in self.allowed_upload_extensions.split(",") if e.strip()]

    def resolve_path(self, relative: str) -> Path:
        """Resolve a config path relative to the repo root."""
        p = Path(relative)
        return p if p.is_absolute() else (REPO_ROOT / p)


@lru_cache
def get_settings() -> Settings:
    """Cached settings singleton — construct once, reuse everywhere."""
    return Settings()
