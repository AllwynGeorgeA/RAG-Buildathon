"""
Voice input: speech-to-text via a pluggable transcription provider (default:
OpenAI's audio API). The transcribed text is always shown to the user
BEFORE it's used for retrieval (spec section 10) — this module only does
the transcription; the "show transcribed query" step lives in the UI/API
layer so the user can confirm/edit it.
"""
from __future__ import annotations

from pathlib import Path

from app.core.config import get_settings
from app.core.exceptions import TranscriptionError
from app.core.logging import get_logger
from app.llm import openai_client

logger = get_logger(__name__)


def transcribe(path: str | Path, language_hint: str | None = None) -> str:
    settings = get_settings()
    path = Path(path)
    if not path.exists():
        raise TranscriptionError(f"Audio file not found: {path}")

    if settings.stt_provider == "openai":
        if not openai_client.is_available():
            raise TranscriptionError(
                "Voice transcription requires OPENAI_API_KEY to be configured. "
                "You can type your question instead."
            )
        try:
            return openai_client.transcribe_audio(str(path), language_hint=language_hint)
        except Exception as exc:  # noqa: BLE001
            raise TranscriptionError(f"Could not transcribe audio: {exc}") from exc

    raise NotImplementedError(f"STT provider '{settings.stt_provider}' is not implemented.")
