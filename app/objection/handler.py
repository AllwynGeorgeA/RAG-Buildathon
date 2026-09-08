"""
Objection & misconception detection.

Farmers often arrive with a second-hand claim ("my neighbor said...", "I
heard only large farmers qualify"). This module detects that pattern so the
pipeline can switch into "objection mode": address only the stated claim,
using evidence, without arguing or speculating (spec section 18).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

_OBJECTION_PATTERNS = {
    "eligibility_misconception": [
        r"only (for )?large farmers", r"not (for|available to) small farmers",
        r"i don'?t think i qualify", r"i don'?t qualify", r"someone told me",
        r"my (friend|neighbou?r|relative) says", r"heard that (only|this)",
    ],
    "already_benefited": [
        r"already (received|got|availed)", r"can'?t (get|receive|apply for) (it|this) again",
        r"already (getting|receiving) (a|another) (government )?benefit",
    ],
    "state_availability_doubt": [
        r"is (this|it) (really |actually )?available in my state",
        r"not available in (my|our) state", r"only available in",
    ],
    "peer_comparison": [
        r"my (neighbou?r|friend|relative) (received|got)", r"but i didn'?t",
    ],
    "general_skepticism": [
        r"why should i apply", r"is (this|it) worth (it|applying)", r"is (this|it) real",
        r"is (this|it) a scam",
    ],
}

_COMPILED = {
    label: [re.compile(p, re.IGNORECASE) for p in patterns]
    for label, patterns in _OBJECTION_PATTERNS.items()
}


@dataclass
class ObjectionDetection:
    is_objection: bool
    objection_type: str | None
    matched_pattern: str | None


def detect_objection(text: str) -> ObjectionDetection:
    for label, patterns in _COMPILED.items():
        for pattern in patterns:
            if pattern.search(text):
                return ObjectionDetection(is_objection=True, objection_type=label, matched_pattern=pattern.pattern)
    return ObjectionDetection(is_objection=False, objection_type=None, matched_pattern=None)
