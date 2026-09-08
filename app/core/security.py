"""
Security helpers: upload validation, safe temp-file handling, path-traversal
guards. Centralised here so every ingestion entry point (API, Streamlit,
scripts) applies the same rules.
"""
from __future__ import annotations

import mimetypes
import os
import tempfile
import uuid
from pathlib import Path

from app.core.config import get_settings
from app.core.exceptions import FileTooLargeError, UnsupportedFileTypeError
from app.core.logging import get_logger

logger = get_logger(__name__)

_EXT_MIME_MAP = {
    ".pdf": {"application/pdf"},
    ".png": {"image/png"},
    ".jpg": {"image/jpeg"},
    ".jpeg": {"image/jpeg"},
    ".xlsx": {"application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", "application/zip"},
    ".xls": {"application/vnd.ms-excel"},
    ".csv": {"text/csv", "text/plain", "application/vnd.ms-excel"},
    ".wav": {"audio/wav", "audio/x-wav"},
    ".mp3": {"audio/mpeg"},
    ".m4a": {"audio/mp4", "audio/x-m4a"},
}


def validate_filename(filename: str) -> str:
    """Strip any path components to prevent path traversal; keep basename only."""
    safe = os.path.basename(filename).replace("\x00", "")
    if not safe or safe in {".", ".."}:
        raise UnsupportedFileTypeError("Invalid filename.")
    return safe


def validate_upload(filename: str, size_bytes: int, declared_mime: str | None = None) -> str:
    """
    Validate an uploaded file's extension, size, and (best-effort) MIME type.

    Returns the normalized lowercase extension (e.g. ".pdf") on success, raises
    UnsupportedFileTypeError / FileTooLargeError otherwise.
    """
    settings = get_settings()
    safe_name = validate_filename(filename)
    ext = Path(safe_name).suffix.lower()

    allowed = settings.allowed_upload_extensions_list
    if ext not in allowed:
        raise UnsupportedFileTypeError(
            f"File type '{ext or 'unknown'}' is not supported. Allowed types: {', '.join(allowed)}"
        )

    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    if size_bytes > max_bytes:
        raise FileTooLargeError(
            f"File exceeds the {settings.max_upload_size_mb} MB upload limit."
        )

    guessed_mime, _ = mimetypes.guess_type(safe_name)
    expected = _EXT_MIME_MAP.get(ext, set())
    if declared_mime and expected and declared_mime not in expected and guessed_mime not in expected:
        logger.warning(
            "Upload MIME mismatch — proceeding on extension allowlist only",
            extra={"filename": safe_name, "declared_mime": declared_mime},
        )

    return ext


def safe_temp_path(suffix: str) -> Path:
    """Return a unique path inside the system temp dir for transient processing."""
    name = f"krishimitra_{uuid.uuid4().hex}{suffix}"
    return Path(tempfile.gettempdir()) / name


def sanitize_for_log(text: str, max_len: int = 200) -> str:
    """Truncate/redact text before it goes into logs — never log raw file contents."""
    if text is None:
        return ""
    flat = " ".join(text.split())
    return flat[:max_len] + ("…" if len(flat) > max_len else "")
