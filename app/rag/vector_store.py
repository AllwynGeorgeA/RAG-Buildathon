"""
Vector store abstraction.

Kept behind `VectorStoreInterface` so the backend (Chroma today, Qdrant
tomorrow) can be swapped without touching retrieval/ingestion code. Default
implementation: Chroma in persistent local mode (no server process needed —
good fit for an offline, locally-stored knowledge base).
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from app.core.config import get_settings
from app.core.logging import get_logger
from app.rag.embeddings import get_embedder
from app.rag.metadata import Chunk

logger = get_logger(__name__)


@dataclass
class VectorHit:
    chunk_id: str
    text: str
    metadata: dict
    score: float  # cosine similarity, higher is better, range ~[-1, 1]


class VectorStoreInterface(ABC):
    @abstractmethod
    def upsert(self, chunks: list[Chunk]) -> int:
        ...

    @abstractmethod
    def search(self, query: str, top_k: int = 8, where: dict | None = None) -> list[VectorHit]:
        ...

    @abstractmethod
    def count(self) -> int:
        ...

    @abstractmethod
    def delete_by_document(self, document_id: str) -> int:
        ...

    @abstractmethod
    def get_all(self) -> tuple[list[str], list[str], list[dict]]:
        """Returns (chunk_ids, texts, metadatas) for every indexed chunk — used to
        build the BM25 lexical index without maintaining a second corpus."""
        ...


class ChromaVectorStore(VectorStoreInterface):
    def __init__(self, path: str | None = None, collection_name: str | None = None):
        import chromadb

        settings = get_settings()
        self._path = str(settings.resolve_path(path or settings.vector_store_path))
        self._collection_name = collection_name or settings.vector_store_collection
        self._client = chromadb.PersistentClient(path=self._path)
        self._collection = self._client.get_or_create_collection(
            name=self._collection_name, metadata={"hnsw:space": "cosine"}
        )
        self._embedder = get_embedder()

    def upsert(self, chunks: list[Chunk]) -> int:
        if not chunks:
            return 0
        texts = [c.text for c in chunks]
        embeddings = self._embedder.embed(texts)
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=texts,
            metadatas=[c.to_vector_metadata() for c in chunks],
        )
        logger.info("Upserted chunks into vector store", extra={"count": len(chunks)})
        return len(chunks)

    def search(self, query: str, top_k: int = 8, where: dict | None = None) -> list[VectorHit]:
        if self.count() == 0:
            return []
        query_embedding = self._embedder.embed([query])[0]
        kwargs = {"query_embeddings": [query_embedding], "n_results": min(top_k, self.count())}
        if where:
            kwargs["where"] = where
        results = self._collection.query(**kwargs)
        hits: list[VectorHit] = []
        ids = results.get("ids", [[]])[0]
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        dists = results.get("distances", [[]])[0]
        for i, doc, meta, dist in zip(ids, docs, metas, dists):
            # Chroma cosine "distance" = 1 - cosine_similarity
            similarity = 1.0 - dist
            hits.append(VectorHit(chunk_id=i, text=doc, metadata=meta or {}, score=similarity))
        return hits

    def count(self) -> int:
        return self._collection.count()

    def delete_by_document(self, document_id: str) -> int:
        existing = self._collection.get(where={"document_id": document_id})
        ids = existing.get("ids", [])
        if ids:
            self._collection.delete(ids=ids)
        return len(ids)

    def get_all(self) -> tuple[list[str], list[str], list[dict]]:
        if self.count() == 0:
            return [], [], []
        raw = self._collection.get(include=["documents", "metadatas"])
        return raw.get("ids", []), raw.get("documents", []), raw.get("metadatas", [])


_store: VectorStoreInterface | None = None


def get_vector_store() -> VectorStoreInterface:
    """Process-wide singleton — vector DB loaded once (perf requirement)."""
    global _store
    if _store is None:
        settings = get_settings()
        if settings.vector_store_provider == "chroma":
            _store = ChromaVectorStore()
        else:
            raise NotImplementedError(
                f"Vector store provider '{settings.vector_store_provider}' is not implemented. "
                "Implement VectorStoreInterface and register it here (e.g. Qdrant local mode)."
            )
    return _store


def reset_vector_store_singleton() -> None:
    """Used by tests to force re-initialisation against a fresh path."""
    global _store
    _store = None
