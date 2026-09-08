"""
Guardrail module — checks whether a user query is agriculture/farming related.
Off-topic queries are blocked before hitting the RAG pipeline.
"""
from __future__ import annotations
import os
from openai import OpenAI

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    return _client


SYSTEM_PROMPT = """You are a topic classifier for an Indian agriculture assistant.
Determine whether the user's query is related to:
- Indian farming, crops, irrigation, soil, fertilizers
- Government agricultural schemes, subsidies, loans for farmers
- Livestock, fisheries, horticulture
- Agricultural markets, MSP, mandi prices
- Weather or climate as it affects farming

Reply with exactly one word: ALLOWED or BLOCKED.
If the query is agriculture-related: ALLOWED
If the query is off-topic (politics, entertainment, general chat, etc.): BLOCKED
"""


def check(query: str) -> tuple[bool, str]:
    """Return (is_allowed, reason_if_blocked)."""
    try:
        client = _get_client()
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
            max_tokens=5,
            temperature=0,
        )
        verdict = resp.choices[0].message.content.strip().upper()
        if verdict == "ALLOWED":
            return True, ""
        return False, (
            "I can only answer questions about Indian agriculture and government "
            "schemes for farmers. Please ask something related to farming, crops, "
            "subsidies, or agricultural schemes."
        )
    except Exception as exc:
        # On API error, allow the query through (fail-open)
        return True, ""
