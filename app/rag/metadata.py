"""
Canonical document/chunk data structures shared by ingestion and retrieval.

Every ingested unit — a crawled Vikaspedia page, an uploaded PDF, an Excel
sheet, an OCR'd image — is normalized into a `RawDocument` before chunking,
and every chunk carries the full provenance fields required by the source
policy (section 16 of the spec): source_type, source_url, section, language,
timestamp.
"""
from __future__ import annotations

import hashlib
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone


def content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()


def new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:12]}"


@dataclass
class RawDocument:
    """A single ingested source unit (a webpage, an uploaded file, a sheet)."""

    document_id: str
    title: str
    content: str
    source_type: str  # "vikaspedia" | "trusted_document" | "user_upload"
    source_url: str = ""
    language: str = "en"
    sector: str = "agriculture"
    category: str = ""
    section: str = ""
    file_type: str = ""  # pdf/image/excel/audio/html
    crawl_timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_modified: str | None = None
    content_hash_value: str = ""
    extra_metadata: dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.content_hash_value:
            self.content_hash_value = content_hash(self.content)


@dataclass
class Chunk:
    """A retrievable unit of text with full provenance for citation."""

    chunk_id: str
    document_id: str
    title: str
    text: str
    source_type: str
    source_url: str = ""
    section: str = ""
    language: str = "en"
    category: str = ""
    crawl_timestamp: str = ""
    file_type: str = ""
    chunk_index: int = 0
    extra_metadata: dict = field(default_factory=dict)

    def to_vector_metadata(self) -> dict:
        """Flat dict suitable for a vector store's metadata payload (no nested objects)."""
        return {
            "document_id": self.document_id,
            "title": self.title,
            "source_type": self.source_type,
            "source_url": self.source_url,
            "section": self.section,
            "language": self.language,
            "category": self.category,
            "crawl_timestamp": self.crawl_timestamp,
            "file_type": self.file_type,
            "chunk_index": self.chunk_index,
        }
