"""
Embedding provider abstraction.

Default: Hugging Face sentence-transformers, configurable via
EMBEDDING_MODEL (defaults to a multilingual model suitable for Indian
languages). If the model cannot be loaded (no network on first run, disk
issue, etc.) we fall back to a deterministic hashing-based embedder so the
rest of the system keeps working — clearly logged as degraded, never
silently pretending to be the real model.
"""
from __future__ import annotations

import hashlib
import threading
from abc import ABC, abstractmethod

import numpy as np

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class EmbeddingProvider(ABC):
    name: str = "base"

    @abstractmethod
    def embed(self, texts: list[str]) -> list[list[float]]:
        ...

    @abstractmethod
    def dimension(self) -> int:
        ...


class SentenceTransformerEmbedder(EmbeddingProvider):
    name = "sentence_transformers"

    def __init__(self, model_name: str, device: str = "cpu"):
        from sentence_transformers import SentenceTransformer  # local import: heavy dep

        self.model_name = model_name
        self._model = SentenceTransformer(model_name, device=device)
        get_dim = getattr(self._model, "get_embedding_dimension", None) or self._model.get_sentence_embedding_dimension
        self._dim = get_dim()

    def embed(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [v.tolist() for v in vectors]

    def dimension(self) -> int:
        return self._dim


class HashingFallbackEmbedder(EmbeddingProvider):
    """
    Deterministic, dependency-free fallback embedder.

    Not a semantic embedding — it hashes character n-grams into a fixed-size
    bag-of-features vector. Retrieval quality is materially lower than a real
    sentence-transformers model, but it never fabricates and never crashes
    the pipeline. Used only when the configured model cannot be loaded.
    """

    name = "hashing_fallback"

    def __init__(self, dim: int = 384, ngram: int = 3):
        self._dim = dim
        self._ngram = ngram

    def _vector(self, text: str) -> list[float]:
        vec = np.zeros(self._dim, dtype=np.float32)
        text = text.lower()
        n = self._ngram
        grams = [text[i : i + n] for i in range(max(len(text) - n + 1, 1))] or [text]
        for g in grams:
            h = int(hashlib.md5(g.encode("utf-8", errors="ignore")).hexdigest(), 16)
            vec[h % self._dim] += 1.0
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec = vec / norm
        return vec.tolist()

    def embed(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t) for t in texts]

    def dimension(self) -> int:
        return self._dim


_lock = threading.Lock()
_provider: EmbeddingProvider | None = None


def get_embedder() -> EmbeddingProvider:
    """Process-wide singleton embedder — loaded once, reused everywhere (perf requirement)."""
    global _provider
    if _provider is not None:
        return _provider
    with _lock:
        if _provider is not None:
            return _provider
        settings = get_settings()
        try:
            _provider = SentenceTransformerEmbedder(settings.embedding_model, settings.embedding_device)
            logger.info("Loaded embedding model", extra={"model": settings.embedding_model})
        except Exception as exc:  # noqa: BLE001 — deliberate broad fallback boundary
            logger.warning(
                "Falling back to hashing embedder — embedding model unavailable",
                extra={"error": str(exc), "model": settings.embedding_model},
            )
            _provider = HashingFallbackEmbedder()
        return _provider
