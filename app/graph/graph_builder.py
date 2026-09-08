"""
Builds the NetworkX knowledge graph from ingested documents.

Deterministic, rule-based entity extraction — NO LLM calls, so the graph
never contains anything not literally present in the ingested text. Every
edge is traceable back to a `document_id` via a MENTIONED_IN / SOURCE edge,
which is what lets graph-based retrieval hand back citable evidence instead
of unattributed "facts".

This is intentionally a lightweight, explainable extractor rather than a
general-purpose NER pipeline: it is tuned to the structure Vikaspedia scheme
pages actually use (section headings like "What are the benefits?",
"Eligibility", "How to apply?") and to the vocabulary of Indian agricultural
schemes (crops, states, scheme-name patterns, ministries).
"""
from __future__ import annotations

import re

from app.core.logging import get_logger
from app.graph.knowledge_graph import KnowledgeGraph
from app.rag.metadata import RawDocument

logger = get_logger(__name__)

INDIAN_STATES = [
    "Andhra Pradesh", "Arunachal Pradesh", "Assam", "Bihar", "Chhattisgarh", "Goa",
    "Gujarat", "Haryana", "Himachal Pradesh", "Jharkhand", "Karnataka", "Kerala",
    "Madhya Pradesh", "Maharashtra", "Manipur", "Meghalaya", "Mizoram", "Nagaland",
    "Odisha", "Punjab", "Rajasthan", "Sikkim", "Tamil Nadu", "Telangana", "Tripura",
    "Uttar Pradesh", "Uttarakhand", "West Bengal", "Delhi", "Jammu and Kashmir",
    "Puducherry", "Chandigarh",
]

CROPS = [
    "rice", "paddy", "wheat", "maize", "millet", "bajra", "jowar", "ragi", "cotton",
    "sugarcane", "groundnut", "soybean", "pulses", "gram", "mustard", "oilseeds",
    "tea", "coffee", "coconut", "banana", "mango", "vegetables", "fruits", "tobacco",
    "jute", "barley", "sunflower", "turmeric", "spices", "onion", "potato", "tomato",
]

FARMER_TYPES = [
    "small farmer", "marginal farmer", "small and marginal farmer", "large farmer",
    "landless farmer", "tenant farmer", "sharecropper", "women farmer", "small farmers",
    "marginal farmers", "large farmers", "landless labourer",
]

DEPARTMENTS = [
    "Ministry of Agriculture and Farmers Welfare", "Department of Agriculture",
    "Ministry of Rural Development", "Department of Animal Husbandry",
    "Ministry of Fisheries, Animal Husbandry and Dairying", "NABARD",
    "Department of Agriculture and Cooperation", "Ministry of Rural Development",
]

# Scheme-name heuristics: acronyms commonly used for central schemes, and a
# generic pattern for "Pradhan Mantri ... Yojana/Scheme/Mission/Card/Nidhi/Fund"
SCHEME_ACRONYMS = [
    "PM-KISAN", "PMFBY", "PMKSY", "PM-KUSUM", "PM KUSUM", "KCC", "SHC", "RKVY", "NMSA", "NFSM",
    "PKVY", "eNAM", "e-NAM", "ATMA", "MIDH", "PMEGP", "NHM",
]
SCHEME_NAME_PATTERN = re.compile(
    r"\b(Pradhan Mantri [A-Z][\w \-]{3,60}?(?:Yojana|Scheme|Mission|Nidhi|Card|Fund|Bima|Yojna))\b"
    r"|\b([A-Z][\w]*(?: [A-Z][\w]*){1,6} (?:Yojana|Scheme|Mission|Nidhi|Card|Fund|Bima Yojana))\b"
)

BENEFIT_HEADERS = ["benefit", "what can you get", "what will you get", "assistance provided", "subsidy"]
ELIGIBILITY_HEADERS = ["eligib", "who can apply", "who is eligible", "criteria"]
REQUIREMENT_HEADERS = ["document", "requirement", "what do you need"]
APPLICATION_HEADERS = ["how to apply", "application process", "how can you avail", "procedure"]
CONTACT_PATTERN = re.compile(r"(\b1800[\d\-\s]{6,}\b)|(\b\d{4}-\d{6,7}\b)|([\w.+-]+@[\w-]+\.[\w.-]+)")


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def _clean_page_title(title: str) -> str:
    """Strips Vikaspedia's own site-branding suffix ('| Vikaspedia - English - ...')
    before a raw page title is used as a fallback scheme name."""
    return title.split("|")[0].strip()


def _find_scheme_names(text: str) -> list[str]:
    found: set[str] = set()
    for acr in SCHEME_ACRONYMS:
        if re.search(rf"\b{re.escape(acr)}\b", text):
            found.add(acr)
    for match in SCHEME_NAME_PATTERN.finditer(text):
        name = match.group(1) or match.group(2)
        if name:
            found.add(name.strip())
    return sorted(found)


def _bullets(text: str, limit: int = 6) -> list[str]:
    lines = [l.strip("-•* \t") for l in text.splitlines() if l.strip()]
    lines = [l for l in lines if len(l) > 3]
    return lines[:limit] if lines else ([text[:300]] if text.strip() else [])


def extract_scheme_names(document: RawDocument) -> list[str]:
    """Public helper: which scheme names appear in a document (title + content)."""
    return _find_scheme_names(f"{document.title}\n{document.content}")


def build_graph_from_documents(documents: list[RawDocument], kg: KnowledgeGraph | None = None) -> KnowledgeGraph:
    kg = kg or KnowledgeGraph()

    for doc in documents:
        doc_node = f"document:{doc.document_id}"
        kg.add_node(doc_node, "Document", title=doc.title, source_url=doc.source_url)

        source_node = f"source:{doc.source_type}"
        kg.add_node(source_node, "Source", name=doc.source_type)

        if doc.category:
            cat_node = f"category:{slug(doc.category)}"
            kg.add_node(cat_node, "Category", name=doc.category)

        scheme_names = _find_scheme_names(f"{doc.title}\n{doc.content}")
        if not scheme_names and doc.category and "scheme" in doc.category.lower():
            # Fall back to using the page title itself as the scheme entity
            scheme_names = [_clean_page_title(doc.title)]

        text_lower = doc.content.lower()
        section_lower = (doc.section or "").lower()

        for scheme_name in scheme_names:
            scheme_node = f"scheme:{slug(scheme_name)}"
            kg.add_node(scheme_node, "Scheme", name=scheme_name)
            kg.add_edge(scheme_node, "MENTIONED_IN", doc_node, document_id=doc.document_id)
            kg.add_edge(scheme_node, "SOURCE", source_node)
            if doc.category:
                kg.add_edge(scheme_node, "IN_CATEGORY", cat_node)

            if any(h in section_lower for h in BENEFIT_HEADERS):
                for b in _bullets(doc.content):
                    bnode = f"benefit:{slug(scheme_name)}:{slug(b)[:40]}"
                    kg.add_node(bnode, "Benefit", text=b, document_id=doc.document_id)
                    kg.add_edge(scheme_node, "HAS_BENEFIT", bnode, document_id=doc.document_id)

            if any(h in section_lower for h in ELIGIBILITY_HEADERS):
                for e in _bullets(doc.content):
                    enode = f"eligibility:{slug(scheme_name)}:{slug(e)[:40]}"
                    kg.add_node(enode, "Eligibility", text=e, document_id=doc.document_id)
                    kg.add_edge(scheme_node, "HAS_ELIGIBILITY", enode, document_id=doc.document_id)

            if any(h in section_lower for h in REQUIREMENT_HEADERS):
                for r in _bullets(doc.content):
                    rnode = f"requirement:{slug(scheme_name)}:{slug(r)[:40]}"
                    kg.add_node(rnode, "Requirement", text=r, document_id=doc.document_id)
                    kg.add_edge(scheme_node, "HAS_REQUIREMENT", rnode, document_id=doc.document_id)

            if any(h in section_lower for h in APPLICATION_HEADERS):
                for a in _bullets(doc.content, limit=3):
                    anode = f"application:{slug(scheme_name)}:{slug(a)[:40]}"
                    kg.add_node(anode, "ApplicationMethod", text=a, document_id=doc.document_id)
                    kg.add_edge(scheme_node, "HAS_APPLICATION_METHOD", anode, document_id=doc.document_id)

            for contact_match in CONTACT_PATTERN.finditer(doc.content):
                contact = next(g for g in contact_match.groups() if g)
                cnode = f"contact:{slug(contact)}"
                kg.add_node(cnode, "Contact", value=contact, document_id=doc.document_id)
                kg.add_edge(scheme_node, "HAS_CONTACT", cnode, document_id=doc.document_id)

            for crop in CROPS:
                if re.search(rf"\b{re.escape(crop)}\b", text_lower):
                    cnode = f"crop:{slug(crop)}"
                    kg.add_node(cnode, "Crop", name=crop)
                    kg.add_edge(scheme_node, "FOR_CROP", cnode, document_id=doc.document_id)

            for state in INDIAN_STATES:
                if state.lower() in text_lower:
                    snode = f"state:{slug(state)}"
                    kg.add_node(snode, "State", name=state)
                    kg.add_edge(scheme_node, "AVAILABLE_IN", snode, document_id=doc.document_id)

            for ftype in FARMER_TYPES:
                if ftype in text_lower:
                    fnode = f"farmer_type:{slug(ftype)}"
                    kg.add_node(fnode, "FarmerType", name=ftype)
                    kg.add_edge(scheme_node, "FOR_FARMER_TYPE", fnode, document_id=doc.document_id)

            for dept in DEPARTMENTS:
                if dept.lower() in text_lower:
                    dnode = f"department:{slug(dept)}"
                    kg.add_node(dnode, "Department", name=dept)
                    kg.add_edge(scheme_node, "MANAGED_BY", dnode, document_id=doc.document_id)

    logger.info(
        "Knowledge graph build complete",
        extra={"nodes": kg.node_count(), "edges": kg.edge_count(), "documents": len(documents)},
    )
    return kg
