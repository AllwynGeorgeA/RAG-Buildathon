"""
NetworkX-backed knowledge graph wrapper.

Node types: Scheme, Benefit, Eligibility, FarmerType, Crop, State, District,
Ministry, Department, Document, Source, Requirement, ApplicationMethod,
Contact, Category.

Edge types: HAS_BENEFIT, HAS_ELIGIBILITY, FOR_CROP, AVAILABLE_IN,
MANAGED_BY, HAS_REQUIREMENT, SOURCE, MENTIONED_IN, HAS_APPLICATION_METHOD,
HAS_CONTACT, IN_CATEGORY, FOR_FARMER_TYPE.

The graph is built offline by `app/graph/graph_builder.py` from ingested
documents and persisted to disk (gpickle) — never mutated during a chat
turn.
"""
from __future__ import annotations

import pickle
from pathlib import Path

import networkx as nx

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)

NODE_TYPES = {
    "Scheme", "Benefit", "Eligibility", "FarmerType", "Crop", "State", "District",
    "Ministry", "Department", "Document", "Source", "Requirement",
    "ApplicationMethod", "Contact", "Category",
}

EDGE_TYPES = {
    "HAS_BENEFIT", "HAS_ELIGIBILITY", "FOR_CROP", "AVAILABLE_IN", "MANAGED_BY",
    "HAS_REQUIREMENT", "SOURCE", "MENTIONED_IN", "HAS_APPLICATION_METHOD",
    "HAS_CONTACT", "IN_CATEGORY", "FOR_FARMER_TYPE",
}


class KnowledgeGraph:
    def __init__(self, graph: nx.MultiDiGraph | None = None):
        self.graph = graph if graph is not None else nx.MultiDiGraph()

    # ---- mutation (ingestion-time only) ----
    def add_node(self, node_id: str, node_type: str, **attrs) -> None:
        if node_type not in NODE_TYPES:
            raise ValueError(f"Unknown node type: {node_type}")
        if self.graph.has_node(node_id):
            self.graph.nodes[node_id].update(attrs)
        else:
            self.graph.add_node(node_id, type=node_type, **attrs)

    def add_edge(self, src: str, relation: str, dst: str, **attrs) -> None:
        if relation not in EDGE_TYPES:
            raise ValueError(f"Unknown relation type: {relation}")
        self.graph.add_edge(src, dst, key=relation, relation=relation, **attrs)

    # ---- read-only query surface ----
    def neighbors(self, node_id: str, relation: str | None = None) -> list[tuple[str, dict]]:
        if node_id not in self.graph:
            return []
        results = []
        for _, target, data in self.graph.out_edges(node_id, data=True):
            if relation is None or data.get("relation") == relation:
                results.append((target, data))
        return results

    def node_type(self, node_id: str) -> str | None:
        return self.graph.nodes[node_id].get("type") if node_id in self.graph else None

    def nodes_of_type(self, node_type: str) -> list[str]:
        return [n for n, d in self.graph.nodes(data=True) if d.get("type") == node_type]

    def node_count(self) -> int:
        return self.graph.number_of_nodes()

    def edge_count(self) -> int:
        return self.graph.number_of_edges()

    # ---- persistence ----
    def save(self, path: str | Path | None = None) -> Path:
        settings = get_settings()
        target = Path(path) if path else settings.resolve_path(settings.knowledge_graph_path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "wb") as f:
            pickle.dump(self.graph, f)
        logger.info("Saved knowledge graph", extra={"path": str(target), "nodes": self.node_count()})
        return target

    @classmethod
    def load(cls, path: str | Path | None = None) -> "KnowledgeGraph":
        settings = get_settings()
        source = Path(path) if path else settings.resolve_path(settings.knowledge_graph_path)
        if not source.exists():
            logger.warning("Knowledge graph file not found — starting empty", extra={"path": str(source)})
            return cls()
        with open(source, "rb") as f:
            graph = pickle.load(f)
        return cls(graph)


_kg: KnowledgeGraph | None = None


def get_knowledge_graph() -> KnowledgeGraph:
    """Process-wide singleton — graph loaded once from disk (perf requirement)."""
    global _kg
    if _kg is None:
        _kg = KnowledgeGraph.load()
    return _kg


def reset_knowledge_graph_singleton() -> None:
    global _kg
    _kg = None
