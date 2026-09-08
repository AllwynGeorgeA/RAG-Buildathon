"""Shared pytest fixtures — an isolated knowledge base per test so tests
never touch the real data/vector_store or data/knowledge_graph."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pytest

from app.core.config import get_settings
from app.graph.knowledge_graph import reset_knowledge_graph_singleton
from app.rag.bm25_index import reset_bm25_index_singleton
from app.rag.vector_store import reset_vector_store_singleton


@pytest.fixture
def isolated_kb(tmp_path, monkeypatch):
    """Points the vector store / knowledge graph at a fresh tmp_path for this test."""
    monkeypatch.setenv("VECTOR_STORE_PATH", str(tmp_path / "vector_store"))
    monkeypatch.setenv("KNOWLEDGE_GRAPH_PATH", str(tmp_path / "graph.gpickle"))
    get_settings.cache_clear()
    reset_vector_store_singleton()
    reset_knowledge_graph_singleton()
    reset_bm25_index_singleton()
    yield
    reset_vector_store_singleton()
    reset_knowledge_graph_singleton()
    reset_bm25_index_singleton()
    get_settings.cache_clear()
