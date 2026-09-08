"""
Optional live web search — explicitly separate from the offline, cited RAG
pipeline everywhere else in this app.

KrishiMitra AI's core guarantee is: answers come only from a vetted,
offline knowledge base, every claim cited, no live scraping during chat
(see README "Solution & differentiators"). This module exists for users who
explicitly want a *supplementary*, clearly-unverified live web lookup
alongside that guaranteed answer — never blended into it, never run through
the hallucination guard (there's nothing to verify a live result against),
and off by default (`Settings.web_search_enabled`).

Uses DuckDuckGo's HTML endpoint (no API key required). Best-effort: any
failure (network, parsing, rate-limiting) returns an empty list rather than
raising — a broken web search must never break the underlying answer.
"""
from __future__ import annotations

from urllib.parse import parse_qs, unquote, urlparse

import requests
from bs4 import BeautifulSoup

from app.core.config import get_settings
from app.core.logging import get_logger
from app.llm.schemas import WebResult

logger = get_logger(__name__)

_SEARCH_URL = "https://html.duckduckgo.com/html/"
_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"
    )
}


def _unwrap_redirect(href: str) -> str:
    """DuckDuckGo's HTML results wrap real URLs behind /l/?uddg=<encoded>."""
    if href.startswith("//"):
        href = "https:" + href
    parsed = urlparse(href)
    if "duckduckgo.com" in parsed.netloc and parsed.path.startswith("/l/"):
        qs = parse_qs(parsed.query)
        real = qs.get("uddg", [None])[0]
        if real:
            return unquote(real)
    return href


def search_web(query: str, max_results: int | None = None) -> list[WebResult]:
    """Best-effort live web search. Returns [] on any failure — never raises."""
    settings = get_settings()
    limit = max_results or settings.web_search_max_results

    query = (query or "").strip()
    if not query:
        return []

    try:
        resp = requests.post(
            _SEARCH_URL,
            data={"q": query},
            headers=_HEADERS,
            timeout=settings.web_search_timeout_seconds,
        )
        resp.raise_for_status()
    except Exception as exc:  # noqa: BLE001 — a failed web search must never break chat
        logger.warning("Live web search request failed", extra={"query": query, "error": str(exc)})
        return []

    try:
        soup = BeautifulSoup(resp.text, "html.parser")
        results: list[WebResult] = []
        for row in soup.select(".result"):
            link = row.select_one(".result__a")
            if not link or not link.get("href"):
                continue
            snippet_el = row.select_one(".result__snippet")
            results.append(
                WebResult(
                    title=link.get_text(strip=True) or link["href"],
                    url=_unwrap_redirect(link["href"]),
                    snippet=snippet_el.get_text(strip=True) if snippet_el else "",
                )
            )
            if len(results) >= limit:
                break
        return results
    except Exception as exc:  # noqa: BLE001
        logger.warning("Live web search parsing failed", extra={"query": query, "error": str(exc)})
        return []
