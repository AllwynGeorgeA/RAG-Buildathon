"""Tests for the optional live web search module — network calls are always
mocked here (this test suite stays fully offline/deterministic, consistent
with the rest of the project, even though the *app* now does live scraping
when a user explicitly opts in)."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from app.api.routes.chat import _maybe_attach_web_results
from app.core.config import get_settings
from app.llm.schemas import ChatResponse, TrustCheck, WebResult
from app.websearch.search import _unwrap_redirect, search_web

SAMPLE_DDG_HTML = """
<div class="result results_links results_links_deep web-result">
  <a class="result__a" href="//duckduckgo.com/l/?uddg=https%3A%2F%2Fpmkisan.gov.in%2F&rut=abc">PM-KISAN Official Portal</a>
  <a class="result__snippet">Official portal for the PM-KISAN scheme.</a>
</div>
<div class="result results_links results_links_deep web-result">
  <a class="result__a" href="https://example.com/scheme">Example Scheme Page</a>
  <a class="result__snippet">A generic example result.</a>
</div>
"""


def _fresh_settings(monkeypatch, **overrides):
    for key, value in overrides.items():
        monkeypatch.setenv(key.upper(), str(value))
    get_settings.cache_clear()
    return get_settings()


def test_unwrap_redirect_extracts_real_url():
    wrapped = "//duckduckgo.com/l/?uddg=https%3A%2F%2Fpmkisan.gov.in%2F&rut=abc"
    assert _unwrap_redirect(wrapped) == "https://pmkisan.gov.in/"


def test_unwrap_redirect_passthrough_for_plain_url():
    assert _unwrap_redirect("https://example.com/page") == "https://example.com/page"


@patch("app.websearch.search.requests.post")
def test_search_web_parses_results(mock_post, monkeypatch):
    _fresh_settings(monkeypatch)
    mock_response = MagicMock()
    mock_response.text = SAMPLE_DDG_HTML
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response

    results = search_web("PM-KISAN eligibility", max_results=4)

    assert len(results) == 2
    assert results[0].title == "PM-KISAN Official Portal"
    assert results[0].url == "https://pmkisan.gov.in/"
    assert "Official portal" in results[0].snippet
    assert results[1].url == "https://example.com/scheme"


@patch("app.websearch.search.requests.post")
def test_search_web_respects_max_results(mock_post, monkeypatch):
    _fresh_settings(monkeypatch)
    mock_response = MagicMock()
    mock_response.text = SAMPLE_DDG_HTML
    mock_response.raise_for_status = MagicMock()
    mock_post.return_value = mock_response

    results = search_web("anything", max_results=1)
    assert len(results) == 1


@patch("app.websearch.search.requests.post", side_effect=ConnectionError("network down"))
def test_search_web_returns_empty_on_network_failure(mock_post, monkeypatch):
    _fresh_settings(monkeypatch)
    assert search_web("anything") == []


def test_search_web_returns_empty_for_blank_query():
    assert search_web("   ") == []


def _dummy_response() -> ChatResponse:
    return ChatResponse(
        answer="An answer.",
        relevant=True,
        confidence="medium",
        trust_check=TrustCheck(
            evidence_found=True, source_verified=True, hallucination_checked=True,
            eligibility_complete=True, confidence="medium",
        ),
    )


@patch("app.api.routes.chat.search_web")
def test_attach_web_results_noop_when_not_requested(mock_search, monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_ENABLED", "true")
    get_settings.cache_clear()

    response = _dummy_response()
    _maybe_attach_web_results(response, "some query", requested=False)

    mock_search.assert_not_called()
    assert response.web_results == []
    assert response.web_search_used is False
    get_settings.cache_clear()


@patch("app.api.routes.chat.search_web")
def test_attach_web_results_noop_when_globally_disabled(mock_search, monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_ENABLED", "false")
    get_settings.cache_clear()

    response = _dummy_response()
    _maybe_attach_web_results(response, "some query", requested=True)

    mock_search.assert_not_called()
    assert response.web_results == []
    get_settings.cache_clear()


@patch("app.api.routes.chat.search_web")
def test_attach_web_results_when_requested_and_enabled(mock_search, monkeypatch):
    monkeypatch.setenv("WEB_SEARCH_ENABLED", "true")
    get_settings.cache_clear()
    mock_search.return_value = [WebResult(title="T", url="https://x.test", snippet="s")]

    response = _dummy_response()
    _maybe_attach_web_results(response, "some query", requested=True)

    mock_search.assert_called_once_with("some query")
    assert response.web_search_used is True
    assert len(response.web_results) == 1
    get_settings.cache_clear()
