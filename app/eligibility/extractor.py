"""
Extracts a `FarmerProfile` (state, crop, landholding, farmer type) from free
text — the user's query, or an explicit "Farmer Profile" panel in the UI.

Deterministic keyword/regex extraction, not an LLM call: the eligibility
engine must never depend on an LLM guessing at the user's situation.
"""
from __future__ import annotations

import re

from app.graph.graph_builder import CROPS, FARMER_TYPES, INDIAN_STATES
from app.llm.schemas import FarmerProfile

ACRE_TO_HECTARE = 0.4047

_LAND_RE = re.compile(
    r"(\d+(?:\.\d+)?)\s*(hectare|hectares|ha\b|acre|acres|bigha)", re.IGNORECASE
)

_FARMER_TYPE_NORMALIZE = {
    "small and marginal farmer": "small_marginal",
    "small and marginal farmers": "small_marginal",
    "small farmer": "small",
    "small farmers": "small",
    "marginal farmer": "marginal",
    "marginal farmers": "marginal",
    "large farmer": "large",
    "large farmers": "large",
    "landless farmer": "landless",
    "landless labourer": "landless",
    "tenant farmer": "tenant",
    "sharecropper": "tenant",
    "women farmer": "women",
}


def extract_land_size_hectares(text: str) -> float | None:
    match = _LAND_RE.search(text)
    if not match:
        return None
    value = float(match.group(1))
    unit = match.group(2).lower()
    if unit.startswith("acre"):
        return round(value * ACRE_TO_HECTARE, 3)
    if unit == "bigha":
        return None  # bigha varies too much by region to convert reliably — don't guess
    return round(value, 3)


def extract_state(text: str) -> str | None:
    lower = text.lower()
    for state in INDIAN_STATES:
        if state.lower() in lower:
            return state
    return None


def extract_crop(text: str) -> str | None:
    lower = text.lower()
    for crop in CROPS:
        if re.search(rf"\b{re.escape(crop)}\b", lower):
            return crop
    return None


def extract_farmer_type(text: str) -> str | None:
    lower = text.lower()
    # match longest phrases first so "small and marginal farmer" wins over "small farmer"
    for phrase in sorted(_FARMER_TYPE_NORMALIZE, key=len, reverse=True):
        if phrase in lower:
            return _FARMER_TYPE_NORMALIZE[phrase]
    for phrase in FARMER_TYPES:
        if phrase in lower:
            return _FARMER_TYPE_NORMALIZE.get(phrase, phrase.replace(" ", "_"))
    return None


def extract_profile_from_text(text: str, base: FarmerProfile | None = None) -> FarmerProfile:
    """Extract whatever attributes are mentioned; merge onto an existing profile if given."""
    data = base.model_dump() if base else {}
    state = extract_state(text)
    crop = extract_crop(text)
    land = extract_land_size_hectares(text)
    farmer_type = extract_farmer_type(text)

    if state:
        data["state"] = state
    if crop:
        data["crop"] = crop
    if land is not None:
        data["land_size_hectares"] = land
    if farmer_type:
        data["farmer_type"] = farmer_type

    return FarmerProfile(**data)
