"""
Image ingestion via a pluggable OCR interface. Default provider:
pytesseract (wraps the Tesseract OCR engine). If OCR confidence is below
the configured threshold, we refuse rather than pass low-quality/garbled
text into the knowledge base (spec section 8).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

from app.core.config import get_settings
from app.core.exceptions import IngestionError, OCRLowConfidenceError
from app.core.logging import get_logger
from app.ingestion.cleaner import clean_text
from app.rag.metadata import RawDocument, new_id

logger = get_logger(__name__)


class OCRProvider(ABC):
    @abstractmethod
    def extract(self, path: Path) -> tuple[str, float]:
        """Return (text, confidence_0_to_100)."""


class PytesseractOCR(OCRProvider):
    def extract(self, path: Path) -> tuple[str, float]:
        import pytesseract
        from PIL import Image

        settings = get_settings()
        if settings.tesseract_cmd:
            pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd

        image = Image.open(path)
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        words = [w for w in data.get("text", []) if w.strip()]
        confidences = [int(c) for c, w in zip(data.get("conf", []), data.get("text", [])) if w.strip() and c != "-1"]
        text = " ".join(words)
        avg_confidence = (sum(confidences) / len(confidences)) if confidences else 0.0
        return text, avg_confidence


def get_ocr_provider() -> OCRProvider:
    settings = get_settings()
    if settings.ocr_provider == "pytesseract":
        return PytesseractOCR()
    raise NotImplementedError(f"OCR provider '{settings.ocr_provider}' is not implemented.")


def load_image(path: str | Path, source_type: str = "user_upload", title: str | None = None) -> RawDocument:
    path = Path(path)
    if not path.exists():
        raise IngestionError(f"Image not found: {path}")

    settings = get_settings()
    provider = get_ocr_provider()
    try:
        raw_text, confidence = provider.extract(path)
    except Exception as exc:  # noqa: BLE001
        logger.error("OCR extraction failed", extra={"error": str(exc)})
        raise IngestionError(
            "I could not reliably read this image. Please upload a clearer image or provide the text."
        ) from exc

    if confidence < settings.ocr_min_confidence or not raw_text.strip():
        raise OCRLowConfidenceError(
            "I could not reliably read this image. Please upload a clearer image or provide the text."
        )

    text = clean_text(raw_text)
    return RawDocument(
        document_id=new_id("image"),
        title=title or path.stem,
        content=text,
        source_type=source_type,
        source_url="",
        file_type="image",
        category="user-provided-document",
        extra_metadata={"ocr_confidence": confidence},
    )
