from __future__ import annotations

from app.graph.graph_builder import build_graph_from_documents, extract_scheme_names, slug
from app.graph.graph_queries import extract_entities_from_query, find_schemes_by_entities
from app.graph.knowledge_graph import KnowledgeGraph
from app.rag.metadata import RawDocument


def _doc(document_id, content, section, category="policies-and-schemes"):
    return RawDocument(
        document_id=document_id, title="PM-KISAN", content=content, source_type="vikaspedia",
        source_url="https://vikaspedia.in/agriculture/pm-kisan", section=section, category=category,
    )


class TestSchemeExtraction:
    def test_extracts_known_acronym(self):
        doc = _doc("d1", "PM-KISAN provides income support to farmers.", "Objective")
        assert "PM-KISAN" in extract_scheme_names(doc)

    def test_extracts_pradhan_mantri_pattern(self):
        doc = _doc("d1", "The Pradhan Mantri Fasal Bima Yojana covers crop losses.", "Objective")
        names = extract_scheme_names(doc)
        assert any("Fasal Bima" in n for n in names)


class TestGraphBuilder:
    def test_builds_benefit_and_eligibility_edges(self):
        docs = [
            _doc("d1", "PM-KISAN provides Rs 6000 per year to farmers.", "What are the benefits?"),
            _doc("d2", "Small and marginal farmers with landholding up to 2 hectares are eligible.", "Eligibility"),
            _doc("d3", "PM-KISAN is available for farmers growing rice in Tamil Nadu.", "Coverage"),
        ]
        kg = build_graph_from_documents(docs)
        scheme_node = f"scheme:{slug('PM-KISAN')}"
        assert scheme_node in kg.graph
        benefits = kg.neighbors(scheme_node, relation="HAS_BENEFIT")
        eligibility = kg.neighbors(scheme_node, relation="HAS_ELIGIBILITY")
        crops = kg.neighbors(scheme_node, relation="FOR_CROP")
        states = kg.neighbors(scheme_node, relation="AVAILABLE_IN")
        assert len(benefits) >= 1
        assert len(eligibility) >= 1
        assert any(kg.graph.nodes[n].get("name") == "rice" for n, _ in crops)
        assert any(kg.graph.nodes[n].get("name") == "Tamil Nadu" for n, _ in states)

    def test_no_scheme_no_crash(self):
        docs = [_doc("d1", "Just some generic agriculture text with no scheme name.", "Intro")]
        kg = build_graph_from_documents(docs)
        assert isinstance(kg, KnowledgeGraph)


class TestGraphQueries:
    def test_extract_entities_from_query(self):
        entities = extract_entities_from_query("I grow rice in Tamil Nadu, am I a small farmer eligible?")
        assert "rice" in entities.crops
        assert "Tamil Nadu" in entities.states

    def test_find_schemes_by_entities(self):
        docs = [_doc("d1", "PM-KISAN is available for farmers growing rice in Tamil Nadu.", "Coverage")]
        kg = build_graph_from_documents(docs)
        entities = extract_entities_from_query("rice farmer in Tamil Nadu")
        matches = find_schemes_by_entities(kg, entities)
        assert len(matches) >= 1
        assert matches[0].scheme_name == "PM-KISAN"
        assert "d1" in matches[0].supporting_document_ids
