"""
Turns an `EligibilityAssessment` into cautious, human-readable language.
Never asserts definitive eligibility — mirrors the phrasing mandated in
spec section 17.
"""
from __future__ import annotations

from app.eligibility.matcher import EligibilityAssessment
from app.llm.schemas import SchemeEligibilityStatus

_STATUS_LABEL = {
    SchemeEligibilityStatus.LIKELY_ELIGIBLE: "🟢 Likely eligible",
    SchemeEligibilityStatus.POSSIBLY_ELIGIBLE: "🟡 Possibly eligible",
    SchemeEligibilityStatus.INSUFFICIENT_INFORMATION: "⚪ Insufficient information",
    SchemeEligibilityStatus.LIKELY_NOT_ELIGIBLE: "🔴 Likely not eligible",
}


def status_label(status: SchemeEligibilityStatus) -> str:
    return _STATUS_LABEL[status]


def explain(assessment: EligibilityAssessment) -> str:
    parts: list[str] = []
    if assessment.status == SchemeEligibilityStatus.LIKELY_ELIGIBLE:
        parts.append(
            "Based on the information provided and the indexed scheme criteria, this appears relevant "
            "and your details match the documented conditions."
        )
    elif assessment.status == SchemeEligibilityStatus.POSSIBLY_ELIGIBLE:
        parts.append(
            "Based on the information provided and the indexed scheme criteria, this appears relevant, "
            "but some details are still needed to confirm eligibility."
        )
    elif assessment.status == SchemeEligibilityStatus.LIKELY_NOT_ELIGIBLE:
        parts.append(
            "Based on the indexed scheme criteria, your provided details do not appear to match this scheme's "
            "stated conditions."
        )
    else:
        parts.append(
            "There isn't enough information yet — either about your profile or in the indexed scheme criteria "
            "— to assess eligibility for this scheme."
        )

    if assessment.matched_criteria:
        parts.append("Matches: " + "; ".join(assessment.matched_criteria) + ".")
    if assessment.contradicted_criteria:
        parts.append("Does not match: " + "; ".join(assessment.contradicted_criteria) + ".")
    if assessment.missing_information:
        parts.append("Still needed: " + ", ".join(assessment.missing_information) + ".")

    return " ".join(parts)
