"""
Compares a `FarmerProfile` against parsed scheme eligibility criteria and
returns a deterministic `SchemeEligibilityStatus` — never a definitive
"eligible" unless the evidence provides sufficient criteria AND every
required attribute is known and satisfied (spec section 17).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from app.eligibility.rules import ParsedCriteria, parse_eligibility_texts
from app.graph.graph_queries import CROP_SYNONYMS
from app.llm.schemas import FarmerProfile, SchemeEligibilityStatus

_FARMER_TYPE_COMPATIBLE = {
    "small_marginal": {"small and marginal farmer", "small and marginal farmers", "small farmer", "small farmers",
                        "marginal farmer", "marginal farmers"},
    "small": {"small farmer", "small farmers", "small and marginal farmer", "small and marginal farmers"},
    "marginal": {"marginal farmer", "marginal farmers", "small and marginal farmer", "small and marginal farmers"},
    "large": {"large farmer", "large farmers"},
    "landless": {"landless farmer", "landless labourer"},
    "tenant": {"tenant farmer", "sharecropper"},
    "women": {"women farmer"},
}


@dataclass
class EligibilityAssessment:
    status: SchemeEligibilityStatus
    matched_criteria: list[str] = field(default_factory=list)
    contradicted_criteria: list[str] = field(default_factory=list)
    missing_information: list[str] = field(default_factory=list)
    criteria: ParsedCriteria | None = None


def assess_eligibility(profile: FarmerProfile, eligibility_texts: list[str]) -> EligibilityAssessment:
    criteria = parse_eligibility_texts(eligibility_texts)

    if not eligibility_texts:
        return EligibilityAssessment(
            status=SchemeEligibilityStatus.INSUFFICIENT_INFORMATION,
            missing_information=["scheme eligibility text was not found in the evidence"],
            criteria=criteria,
        )

    matched: list[str] = []
    contradicted: list[str] = []
    missing: list[str] = []

    # Landholding ceiling
    if criteria.land_ceiling_hectares is not None:
        if profile.land_size_hectares is None:
            missing.append("landholding size")
        elif profile.land_size_hectares <= criteria.land_ceiling_hectares:
            matched.append(f"landholding within {criteria.land_ceiling_hectares} ha ceiling")
        else:
            contradicted.append(f"landholding exceeds the {criteria.land_ceiling_hectares} ha ceiling stated for this scheme")

    # Farmer type
    if criteria.required_farmer_types:
        if not profile.farmer_type:
            missing.append("farmer category (small/marginal/large/tenant/women)")
        else:
            compatible = _FARMER_TYPE_COMPATIBLE.get(profile.farmer_type, {profile.farmer_type})
            if compatible & set(criteria.required_farmer_types):
                matched.append("farmer category matches scheme's stated eligible category")
            else:
                contradicted.append(
                    f"scheme criteria mention {', '.join(criteria.required_farmer_types)}, "
                    f"which does not match the provided farmer category"
                )

    # State restriction — only meaningful if the evidence names specific states
    if criteria.required_states:
        if not profile.state:
            missing.append("state")
        elif profile.state in criteria.required_states:
            matched.append(f"available in {profile.state}")
        else:
            contradicted.append(
                f"the indexed evidence names {', '.join(criteria.required_states)}; "
                f"{profile.state} was not found there"
            )

    # Crop restriction — only meaningful if the evidence names specific crops
    if criteria.required_crops:
        if not profile.crop:
            missing.append("crop grown")
        elif profile.crop in criteria.required_crops or any(
            syn in criteria.required_crops for syn in CROP_SYNONYMS.get(profile.crop, [])
        ):
            matched.append(f"crop '{profile.crop}' matches scheme's stated crop(s)")
        else:
            contradicted.append(
                f"scheme evidence mentions {', '.join(criteria.required_crops)}; '{profile.crop}' was not found there"
            )

    if contradicted:
        status = SchemeEligibilityStatus.LIKELY_NOT_ELIGIBLE
    elif missing:
        status = SchemeEligibilityStatus.POSSIBLY_ELIGIBLE if matched else SchemeEligibilityStatus.INSUFFICIENT_INFORMATION
    elif matched:
        status = SchemeEligibilityStatus.LIKELY_ELIGIBLE
    else:
        # No checkable criteria were found in the text at all (e.g. eligibility text
        # was descriptive prose without a parseable ceiling/category/state/crop).
        status = SchemeEligibilityStatus.INSUFFICIENT_INFORMATION

    return EligibilityAssessment(
        status=status,
        matched_criteria=matched,
        contradicted_criteria=contradicted,
        missing_information=missing,
        criteria=criteria,
    )
