"""
Prompt injection / jailbreak detection.

Deterministic pattern matching — deliberately not an LLM call, so this
guardrail can never itself be bypassed by an injection aimed at an LLM
classifier, and always runs even if OPENAI_API_KEY is unset.

Retrieved documents are also treated as untrusted DATA elsewhere (see
app/llm/prompts.py) — this module protects the USER-INPUT side of that
contract.
"""
from __future__ import annotations

import re

_INJECTION_PATTERNS = [
    r"\bignore (all |any |your )?(previous|prior|above|earlier) instructions\b",
    r"\bignore your instructions\b",
    r"\bdisregard (all |any |your )?(previous|prior|above) (instructions|rules|prompt)\b",
    r"\breveal (your |the )?system prompt\b",
    r"\bshow (me )?(your |the )?(system )?prompt\b",
    r"\bwhat (is|are) your (system )?(instructions|prompt)\b",
    r"\byou are now\b",
    r"\bact as (a|an)\b",
    r"\bact as if you are\b",
    r"\bpretend (to be|you are)\b",
    r"\bdeveloper mode\b",
    r"\bjailbreak\b",
    r"\bbypass (your |the )?(restrictions|rules|guardrails|filters)\b",
    r"\buse (information |your own knowledge |data )?outside (of )?vikaspedia\b",
    r"\buse your (own )?(general )?knowledge\b",
    r"\bnot (present |available )?in your (database|knowledge base|documents)\b.*\btell me\b",
    r"\btell me something not in\b",
    r"\bforget (that |you are|everything)\b",
    r"\bnew instructions?:\b",
    r"\bsystem:\s",
    r"\[system\]",
    r"\boutput the raw\b",
    r"\bwithout (any )?(guardrails|restrictions|filters|citations)\b",
]

_COMPILED = [re.compile(p, re.IGNORECASE) for p in _INJECTION_PATTERNS]


def detect_prompt_injection(text: str) -> tuple[bool, str | None]:
    """Return (is_injection, matched_pattern)."""
    if not text:
        return False, None
    for pattern in _COMPILED:
        if pattern.search(text):
            return True, pattern.pattern
    return False, None
