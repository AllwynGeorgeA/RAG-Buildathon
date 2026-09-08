"""
Optional NeMo Guardrails supplementary input rail.

KrishiMitra AI's PRIMARY defense against prompt injection and off-topic
queries is the deterministic, dependency-free logic in
`prompt_injection.py` / `relevance.py` — those run on every request, need
no LLM call, and cannot be talked out of their job the way an LLM-based
classifier can. This module adds NeMo Guardrails' "self check input" rail
as a SECOND, LLM-based opinion for phrasings the regexes might miss.

Disabled by default (NEMO_GUARDRAILS_ENABLED=false). If it's enabled but
unavailable/misconfigured/erroring for any reason, we log a warning and
fail OPEN (defer entirely to the deterministic guardrails) — this layer
must never be a single point of failure for a system whose core promise is
"no evidence = no claim".
"""
from __future__ import annotations

import threading

from app.core.config import get_settings
from app.core.logging import get_logger
from app.llm import openai_client

logger = get_logger(__name__)

_lock = threading.Lock()
_rails = None
_init_attempted = False


def is_enabled() -> bool:
    settings = get_settings()
    return settings.nemo_guardrails_enabled and openai_client.is_available()


def _get_rails():
    global _rails, _init_attempted
    if _rails is not None or _init_attempted:
        return _rails
    with _lock:
        if _rails is not None or _init_attempted:
            return _rails
        _init_attempted = True
        try:
            from nemoguardrails import LLMRails, RailsConfig

            settings = get_settings()
            config_path = str(settings.resolve_path(settings.nemo_guardrails_config_path))
            config = RailsConfig.from_path(config_path)
            _rails = LLMRails(config)
            logger.info("NeMo Guardrails initialized", extra={"config_path": config_path})
        except Exception as exc:  # noqa: BLE001 — optional layer, must fail open
            logger.warning("NeMo Guardrails unavailable — continuing with deterministic guardrails only", extra={"error": str(exc)})
            _rails = None
    return _rails


def nemo_check_input(text: str) -> tuple[bool, str | None]:
    """Returns (allowed, reason_if_blocked). Fails open on any error."""
    if not is_enabled():
        return True, None

    rails = _get_rails()
    if rails is None:
        return True, None

    try:
        response = rails.generate(messages=[{"role": "user", "content": text}])
        content = (response or {}).get("content", "") if isinstance(response, dict) else str(response or "")
        # NeMo's default self-check-input refusal flow returns a fixed
        # "can't respond" style message instead of an actual answer.
        blocked_markers = ["i'm sorry, i can't respond", "cannot assist with that", "can't help with that"]
        if any(marker in content.lower() for marker in blocked_markers):
            return False, (
                "I can help with farmer and agriculture schemes from the approved knowledge base. "
                "I can't help with that request."
            )
        return True, None
    except Exception as exc:  # noqa: BLE001 — optional layer, must fail open
        logger.warning("NeMo Guardrails check failed — failing open", extra={"error": str(exc)})
        return True, None
