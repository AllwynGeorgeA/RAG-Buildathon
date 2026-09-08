"""PDF text extraction via PyMuPDF (primary) with pypdf as a fallback."""
from __future__ import annotations

from pathlib import Path

from app.core.exceptions import IngestionError
from app.core.logging import get_logger
from app.ingestion.cleaner import clean_text
from app.rag.metadata import RawDocument, new_id

logger = get_logger(__name__)


def _extract_with_pymupdf(path: Path) -> str:
    import fitz  # PyMuPDF

    text_parts = []
    with fitz.open(path) as doc:
        for page in doc:
            text_parts.append(page.get_text())
    return "\n\n".join(text_parts)


def _extract_with_pypdf(path: Path) -> str:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    return "\n\n".join(page.extract_text() or "" for page in reader.pages)


def load_pdf(path: str | Path, source_type: str = "user_upload", title: str | None = None) -> RawDocument:
    path = Path(path)
    if not path.exists():
        raise IngestionError(f"PDF not found: {path}")

    try:
        raw_text = _extract_with_pymupdf(path)
    except Exception as exc:  # noqa: BLE001
        logger.warning("PyMuPDF extraction failed, trying pypdf", extra={"error": str(exc)})
        try:
            raw_text = _extract_with_pypdf(path)
        except Exception as exc2:  # noqa: BLE001
            raise IngestionError(f"Could not extract text from PDF: {exc2}") from exc2

    text = clean_text(raw_text)
    if not text:
        raise IngestionError("No extractable text found in PDF (it may be a scanned image — try image upload/OCR).")

    return RawDocument(
        document_id=new_id("pdf"),
        title=title or path.stem,
        content=text,
        source_type=source_type,
        source_url="",
        file_type="pdf",
        category="user-provided-document",
    )
