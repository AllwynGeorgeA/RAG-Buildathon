"""Tests for the SQLite-backed conversation history store."""
from __future__ import annotations

import pytest

from app.conversations import store
from app.core.config import get_settings


@pytest.fixture
def isolated_conversations(tmp_path, monkeypatch):
    monkeypatch.setenv("CONVERSATIONS_DB_PATH", str(tmp_path / "conversations.db"))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def test_create_and_list_conversation(isolated_conversations):
    conv = store.create_conversation()
    assert conv["title"] == store.DEFAULT_TITLE
    assert conv["pinned"] is False

    listed = store.list_conversations()
    assert len(listed) == 1
    assert listed[0]["id"] == conv["id"]


def test_add_message_auto_titles_conversation(isolated_conversations):
    conv = store.create_conversation()
    store.add_message(conv["id"], "user", "Am I eligible for PM-KISAN with 2 acres?")

    loaded = store.get_conversation(conv["id"])
    assert loaded["title"] == "Am I eligible for PM-KISAN with 2 acres?"
    assert len(loaded["messages"]) == 1
    assert loaded["messages"][0]["role"] == "user"


def test_get_missing_conversation_returns_none(isolated_conversations):
    assert store.get_conversation("does-not-exist") is None


def test_rename_and_pin_conversation(isolated_conversations):
    conv = store.create_conversation()

    assert store.rename_conversation(conv["id"], "My renamed chat") is True
    assert store.set_pinned(conv["id"], True) is True

    loaded = store.get_conversation(conv["id"])
    assert loaded["title"] == "My renamed chat"
    assert loaded["pinned"] is True


def test_rename_missing_conversation_returns_false(isolated_conversations):
    assert store.rename_conversation("nope", "x") is False
    assert store.set_pinned("nope", True) is False


def test_delete_conversation_removes_messages(isolated_conversations):
    conv = store.create_conversation()
    store.add_message(conv["id"], "user", "hello")

    assert store.delete_conversation(conv["id"]) is True
    assert store.get_conversation(conv["id"]) is None
    assert store.list_conversations() == []


def test_delete_missing_conversation_returns_false(isolated_conversations):
    assert store.delete_conversation("nope") is False


def test_list_conversations_orders_by_most_recently_updated(isolated_conversations):
    first = store.create_conversation(title="First")
    second = store.create_conversation(title="Second")

    store.add_message(first["id"], "user", "bumping the first conversation")

    ordered = store.list_conversations()
    assert [c["id"] for c in ordered] == [first["id"], second["id"]]
