"""
Thin, reused wrapper around the OpenAI SDK.

Single client instance per process (perf requirement — never recreate per
request). If OPENAI_API_KEY is not configured, `is_available()` returns
False and callers fall back to the deterministic extractive generator in
`answer_generator.py` — the app never crashes or silently no-ops just
because a key is missing.
"""
from __future__ import annotations

import threading

from app.core.config import get_settings
from app.core.exceptions import LLMError
from app.core.logging import get_logger

logger = get_logger(__name__)

_lock = threading.Lock()
_client = None


def is_available() -> bool:
    return bool(get_settings().openai_api_key)


def get_client():
    global _client
    if _client is not None:
        return _client
    with _lock:
        if _client is not None:
            return _client
        if not is_available():
            raise LLMError("OPENAI_API_KEY is not configured.")
        from openai import OpenAI

        settings = get_settings()
        _client = OpenAI(api_key=settings.openai_api_key, timeout=settings.openai_request_timeout_seconds)
        return _client


def chat_completion_json(system_prompt: str, user_prompt: str, temperature: float = 0.1) -> str:
    """Call the chat model with JSON-object response format. Returns raw JSON text."""
    settings = get_settings()
    client = get_client()
    try:
        response = client.chat.completions.create(
            model=settings.openai_chat_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content or "{}"
    except Exception as exc:  # noqa: BLE001
        logger.error("OpenAI chat completion failed", extra={"error": str(exc)})
        raise LLMError(f"LLM call failed: {exc}") from exc


def transcribe_audio(file_path: str, language_hint: str | None = None) -> str:
    """Speech-to-text via OpenAI's audio transcription API."""
    settings = get_settings()
    client = get_client()
    try:
        with open(file_path, "rb") as f:
            kwargs = {"model": settings.openai_audio_model, "file": f}
            if language_hint:
                kwargs["language"] = language_hint
            transcript = client.audio.transcriptions.create(**kwargs)
        return transcript.text
    except Exception as exc:  # noqa: BLE001
        logger.error("Audio transcription failed", extra={"error": str(exc)})
        raise LLMError(f"Transcription failed: {exc}") from exc
