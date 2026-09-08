"""
Domain relevance guardrail.

KrishiMitra AI answers only farmer/agriculture-scheme questions. This check
runs BEFORE retrieval (cheap keyword/off-topic check) and its verdict is
combined AFTER retrieval with the evidence threshold (see
`retrieval_guardrail.py`) — a query can look on-topic but still fail if no
evidence is found, and a query can look borderline but pass if strong
evidence turns up.
"""
from __future__ import annotations

import re

IN_DOMAIN_KEYWORDS = [
    "farmer", "farming", "agricultur", "crop", "scheme", "yojana", "subsidy",
    "kisan", "irrigation", "soil", "fertiliz", "fertilis", "seed", "harvest",
    "land", "landholding", "hectare", "acre", "loan", "credit", "insurance",
    "bima", "pmfby", "pm-kisan", "kcc", "msp", "mandi", "livestock", "dairy",
    "fisher", "horticulture", "pesticide", "irrigat", "drought", "organic",
    "cultivat", "plantation", "farm", "eligib", "benefit", "apply", "scheme",
    "government scheme", "subsidy", "crop insurance", "soil health",
]

OFF_TOPIC_KEYWORDS = [
    "weather forecast", "weather today", "cricket score", "movie", "celebrity", "stock price",
    "election result", "song lyrics", "recipe for", "translate this to",
    "write me a poem", "who is the prime minister of", "capital of france",
    "joke", "riddle", "write code", "python script", "solve this equation",
]

_IN_DOMAIN_RE = re.compile("|".join(re.escape(k) for k in IN_DOMAIN_KEYWORDS), re.IGNORECASE)
_OFF_TOPIC_RE = re.compile("|".join(re.escape(k) for k in OFF_TOPIC_KEYWORDS), re.IGNORECASE)


def looks_in_domain(query: str) -> bool:
    """Cheap heuristic check: does the query mention agriculture/scheme vocabulary?"""
    if not query or not query.strip():
        return False
    if _OFF_TOPIC_RE.search(query):
        return False
    return bool(_IN_DOMAIN_RE.search(query))


def looks_clearly_off_topic(query: str) -> bool:
    """Explicit off-topic patterns (weather, entertainment, general trivia, code)."""
    return bool(_OFF_TOPIC_RE.search(query)) and not _IN_DOMAIN_RE.search(query)
