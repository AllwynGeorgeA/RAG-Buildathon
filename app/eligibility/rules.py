"""
Deterministic parsing of eligibility-criteria *text* (as extracted from the
knowledge graph / evidence) into structured, checkable rules.

Nothing here invents a criterion — every rule is parsed from a real
eligibility-section sentence already stored in the graph/evidence.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from app.graph.graph_builder import CROPS, FARMER_TYPES, INDIAN_STATES

_LAND_CEILING_RE = re.compile(
    r"(?:up\s*to|maximum of|not\s*(?:exceed|more than))\s*(\d+(?:\.\d+)?)\s*(hectare|ha\b|acre)",
    re.IGNORECASE,
)


@dataclass
class ParsedCriteria:
    land_ceiling_hectares: float | None = None
    required_farmer_types: list[str] = field(default_factory=list)
    required_states: list[str] = field(default_factory=list)
    required_crops: list[str] = field(default_factory=list)
    raw_texts: list[str] = field(default_factory=list)


def parse_eligibility_texts(texts: list[str]) -> ParsedCriteria:
    criteria = ParsedCriteria(raw_texts=list(texts))
    for text in texts:
        lower = text.lower()

        land_match = _LAND_CEILING_RE.search(lower)
        if land_match:
            value = float(land_match.group(1))
            unit = land_match.group(2)
            if unit.startswith("acre"):
                value *= 0.4047
            if criteria.land_ceiling_hectares is None or value < criteria.land_ceiling_hectares:
                criteria.land_ceiling_hectares = round(value, 3)

        for ftype in FARMER_TYPES:
            if ftype in lower and ftype not in criteria.required_farmer_types:
                criteria.required_farmer_types.append(ftype)

        for state in INDIAN_STATES:
            if state.lower() in lower and state not in criteria.required_states:
                criteria.required_states.append(state)

        for crop in CROPS:
            if re.search(rf"\b{re.escape(crop)}\b", lower) and crop not in criteria.required_crops:
                criteria.required_crops.append(crop)

    return criteria
