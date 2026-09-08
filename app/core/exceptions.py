"""
Application-level exceptions.

Kept small and explicit so callers can catch precisely and the API/UI layers
can translate them into user-friendly messages without leaking internals.
"""
from __future__ import annotations


class KrishiMitraError(Exception):
    """Base class for all application errors."""


class ConfigurationError(KrishiMitraError):
    """Raised when required configuration is missing or invalid."""


class IngestionError(KrishiMitraError):
    """Raised when a document/crawl page cannot be processed."""


class UnsupportedFileTypeError(IngestionError):
    """Raised when an uploaded file's extension/MIME type is not allowed."""


class FileTooLargeError(IngestionError):
    """Raised when an uploaded file exceeds the configured size limit."""


class OCRLowConfidenceError(IngestionError):
    """Raised when OCR confidence is below the configured threshold."""


class TranscriptionError(IngestionError):
    """Raised when speech-to-text fails or is unavailable."""


class RetrievalError(KrishiMitraError):
    """Raised when vector/graph retrieval fails."""


class GuardrailRejection(KrishiMitraError):
    """Raised when a guardrail blocks a request. Carries a user-safe reason."""

    def __init__(self, reason: str, guardrail: str):
        super().__init__(reason)
        self.reason = reason
        self.guardrail = guardrail


class InsufficientEvidenceError(KrishiMitraError):
    """Raised when retrieval evidence is below the confidence threshold."""


class LLMError(KrishiMitraError):
    """Raised when the LLM provider fails or returns an unparsable response."""


class CrawlPolicyError(KrishiMitraError):
    """Raised when a crawl target violates the allowlist/robots policy."""
