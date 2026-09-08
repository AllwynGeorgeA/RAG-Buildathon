"""
Structured logging setup.

Uses stdlib `logging` with a JSON-ish structured formatter so logs are
greppable and safe: we never log raw uploaded file contents or API keys.
"""
from __future__ import annotations

import json
import logging
import sys
import time
from typing import Any

from app.core.config import get_settings

_CONFIGURED = False

_REDACT_KEYS = {"api_key", "openai_api_key", "authorization", "password", "token"}


class StructuredFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(record.created)),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        extra = getattr(record, "extra_fields", None)
        if extra:
            for k, v in extra.items():
                if k.lower() in _REDACT_KEYS:
                    v = "***redacted***"
                payload[k] = v
        if record.exc_info:
            payload["exc_info"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str, ensure_ascii=False)


class _AdapterLogger(logging.LoggerAdapter):
    def process(self, msg, kwargs):
        extra = kwargs.pop("extra", {}) or {}
        kwargs["extra"] = {"extra_fields": extra}
        return msg, kwargs


def configure_logging() -> None:
    global _CONFIGURED
    if _CONFIGURED:
        return
    settings = get_settings()
    root = logging.getLogger()
    root.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(StructuredFormatter())
    root.handlers = [handler]
    _CONFIGURED = True


def get_logger(name: str) -> _AdapterLogger:
    configure_logging()
    return _AdapterLogger(logging.getLogger(name), {})
