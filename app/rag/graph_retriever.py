"""
Knowledge-graph-based retrieval: turns query entities into scheme candidates
and structured facts, independent of vector similarity.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.graph.graph_queries import GraphMatch, extract_entities_from_query, find_schemes_by_entities, get_scheme_profile
from app.graph.knowledge_graph import KnowledgeGraph, get_knowledge_graph


@dataclass
class GraphRetrievalResult:
    matches: list[GraphMatch] = field(default_factory=list)
    profiles: dict[str, dict] = field(default_factory=dict)  # scheme_node -> profile


def retrieve_from_graph(query: str, kg: KnowledgeGraph | None = None) -> GraphRetrievalResult:
    kg = kg or get_knowledge_graph()
    entities = extract_entities_from_query(query)
    matches = find_schemes_by_entities(kg, entities)
    profiles = {m.scheme_node: get_scheme_profile(kg, m.scheme_node) for m in matches}
    return GraphRetrievalResult(matches=matches, profiles=profiles)
