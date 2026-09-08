"""
Read-only graph traversal queries used at retrieval time.

These never mutate the graph. They translate extracted query entities
(crop, state, farmer type) into candidate scheme nodes and the evidence
(document_ids) that support each relationship, so hybrid retrieval can
combine graph-derived candidates with vector hits.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.graph.graph_builder import CROPS, FARMER_TYPES, INDIAN_STATES, slug
from app.graph.knowledge_graph import KnowledgeGraph

# Vikaspedia content sometimes uses one term where a farmer's query uses its
# common synonym (e.g. "rice" vs the agronomic term "paddy"). The graph is
# built from the literal words in the text (so it stays faithful to the
# source), but query-time matching should still find the right scheme.
CROP_SYNONYMS: dict[str, list[str]] = {
    "rice": ["paddy"],
    "paddy": ["rice"],
    "groundnut": ["peanut"],
    "peanut": ["groundnut"],
}


@dataclass
class GraphEntities:
    crops: list[str] = field(default_factory=list)
    states: list[str] = field(default_factory=list)
    farmer_types: list[str] = field(default_factory=list)


@dataclass
class GraphMatch:
    scheme_name: str
    scheme_node: str
    matched_via: list[str]  # e.g. ["crop:rice", "state:tamil_nadu"]
    supporting_document_ids: set[str]
    graph_score: float


def extract_entities_from_query(query: str) -> GraphEntities:
    q = query.lower()
    crops = [c for c in CROPS if re.search(rf"\b{re.escape(c)}\b", q)]
    states = [s for s in INDIAN_STATES if s.lower() in q]
    farmer_types = [f for f in FARMER_TYPES if f in q]
    return GraphEntities(crops=crops, states=states, farmer_types=farmer_types)


def find_schemes_by_entities(kg: KnowledgeGraph, entities: GraphEntities) -> list[GraphMatch]:
    """Traverse the graph to find schemes connected to the query's crop/state/farmer-type."""
    candidates: dict[str, GraphMatch] = {}

    def _register(scheme_node: str, via: str, doc_id: str | None):
        name = kg.graph.nodes[scheme_node].get("name", scheme_node)
        if scheme_node not in candidates:
            candidates[scheme_node] = GraphMatch(
                scheme_name=name, scheme_node=scheme_node, matched_via=[], supporting_document_ids=set(), graph_score=0.0
            )
        match = candidates[scheme_node]
        if via not in match.matched_via:
            match.matched_via.append(via)
        if doc_id:
            match.supporting_document_ids.add(doc_id)

    for crop in entities.crops:
        for candidate in [crop] + CROP_SYNONYMS.get(crop, []):
            crop_node = f"crop:{slug(candidate)}"
            if crop_node not in kg.graph:
                continue
            for scheme_node, _, data in kg.graph.in_edges(crop_node, data=True):
                if data.get("relation") == "FOR_CROP":
                    _register(scheme_node, f"crop:{crop}", data.get("document_id"))

    for state in entities.states:
        state_node = f"state:{slug(state)}"
        if state_node not in kg.graph:
            continue
        for scheme_node, _, data in kg.graph.in_edges(state_node, data=True):
            if data.get("relation") == "AVAILABLE_IN":
                _register(scheme_node, f"state:{state}", data.get("document_id"))

    for ftype in entities.farmer_types:
        ft_node = f"farmer_type:{slug(ftype)}"
        if ft_node not in kg.graph:
            continue
        for scheme_node, _, data in kg.graph.in_edges(ft_node, data=True):
            if data.get("relation") == "FOR_FARMER_TYPE":
                _register(scheme_node, f"farmer_type:{ftype}", data.get("document_id"))

    total_signals = max(len(entities.crops) + len(entities.states) + len(entities.farmer_types), 1)
    for match in candidates.values():
        match.graph_score = min(len(match.matched_via) / total_signals, 1.0)

    return sorted(candidates.values(), key=lambda m: m.graph_score, reverse=True)


def find_schemes_by_document_ids(kg: KnowledgeGraph, document_ids: list[str]) -> list[GraphMatch]:
    """
    Evidence-grounded scheme lookup: given the document_ids that retrieval
    actually found strong evidence in, reverse-walk MENTIONED_IN to find
    which scheme(s) those documents are about.

    This matters for schemes with broad, non-crop-specific eligibility
    (e.g. PM-KISAN applies to "all land holding farmer families" — it has
    no FOR_CROP/AVAILABLE_IN edges to match against a query's crop/state,
    and correctly shouldn't). Crop/state/farmer-type graph matching alone
    would then never surface it even when retrieval clearly found it, so
    this complements (not replaces) `find_schemes_by_entities`.
    """
    candidates: dict[str, GraphMatch] = {}
    doc_id_set = set(document_ids)
    for doc_id in doc_id_set:
        doc_node = f"document:{doc_id}"
        if doc_node not in kg.graph:
            continue
        for scheme_node, _, data in kg.graph.in_edges(doc_node, data=True):
            if data.get("relation") != "MENTIONED_IN" or kg.node_type(scheme_node) != "Scheme":
                continue
            name = kg.graph.nodes[scheme_node].get("name", scheme_node)
            if scheme_node not in candidates:
                candidates[scheme_node] = GraphMatch(
                    scheme_name=name, scheme_node=scheme_node, matched_via=["retrieved_evidence"],
                    supporting_document_ids=set(), graph_score=0.0,
                )
            candidates[scheme_node].supporting_document_ids.add(doc_id)

    for match in candidates.values():
        match.graph_score = min(len(match.supporting_document_ids) / max(len(doc_id_set), 1), 1.0)

    return sorted(candidates.values(), key=lambda m: m.graph_score, reverse=True)


def get_scheme_profile(kg: KnowledgeGraph, scheme_node: str) -> dict:
    """Collect all structured facts the graph holds about one scheme, with document provenance."""
    profile: dict[str, list] = {
        "benefits": [], "eligibility": [], "requirements": [], "application_methods": [],
        "contacts": [], "crops": [], "states": [], "farmer_types": [], "departments": [],
    }
    relation_map = {
        "HAS_BENEFIT": "benefits", "HAS_ELIGIBILITY": "eligibility",
        "HAS_REQUIREMENT": "requirements", "HAS_APPLICATION_METHOD": "application_methods",
        "HAS_CONTACT": "contacts", "FOR_CROP": "crops", "AVAILABLE_IN": "states",
        "FOR_FARMER_TYPE": "farmer_types", "MANAGED_BY": "departments",
    }
    for target, data in kg.neighbors(scheme_node):
        relation = data.get("relation")
        key = relation_map.get(relation)
        if not key or target not in kg.graph:
            continue
        node_data = kg.graph.nodes[target]
        text = node_data.get("text") or node_data.get("name") or node_data.get("value") or target
        profile[key].append({"text": text, "document_id": data.get("document_id")})
    return profile
