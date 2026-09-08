"""
Extension-based dispatcher: routes an uploaded file to the right loader and
returns a uniform list[RawDocument], regardless of file type.
"""
from __future__ import annotations

from pathlib import Path

from app.core.exceptions import UnsupportedFileTypeError
from app.ingestion.excel_loader import load_excel
from app.ingestion.image_loader import load_image
from app.ingestion.metadata_extractor import enrich_metadata
from app.ingestion.pdf_loader import load_pdf
from app.rag.metadata import RawDocument

_IMAGE_EXTS = {".png", ".jpg", ".jpeg"}
_EXCEL_EXTS = {".xlsx", ".xls", ".csv"}


def load_document(path: str | Path, source_type: str = "user_upload", title: str | None = None) -> list[RawDocument]:
    """Loads any supported file type into one or more RawDocuments."""
    path = Path(path)
    ext = path.suffix.lower()

    if ext == ".pdf":
        docs = [load_pdf(path, source_type=source_type, title=title)]
    elif ext in _IMAGE_EXTS:
        docs = [load_image(path, source_type=source_type, title=title)]
    elif ext in _EXCEL_EXTS:
        docs = load_excel(path, source_type=source_type, title=title)
    else:
        raise UnsupportedFileTypeError(f"No loader registered for file type '{ext}'.")

    return [enrich_metadata(d) for d in docs]
