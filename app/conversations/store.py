"""
Lightweight SQLite-backed conversation history.

Each conversation is a thread of user/assistant turns, persisted across
server restarts (default: data/conversations.db — see
Settings.conversations_db_path). Deliberately no ORM: every other persisted
artifact in this repo (vector store, knowledge graph) is already a plain
file on disk, and a demo-scale conversation list doesn't need one either.

Concurrency note: each function opens and closes its own short-lived
connection rather than sharing one across requests — simple and safe under
FastAPI's threadpool-per-sync-request model, at the cost of not being built
for high write concurrency (fine for a single-user/demo deployment).
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator

from app.core.config import get_settings
from app.core.logging import get_logger
from app.llm.schemas import ChatResponse

logger = get_logger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    pinned INTEGER NOT NULL DEFAULT 0,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS messages (
    id TEXT PRIMARY KEY,
    conversation_id TEXT NOT NULL,
    role TEXT NOT NULL,
    content TEXT NOT NULL,
    response_json TEXT,
    created_at TEXT NOT NULL,
    FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_messages_conversation ON messages(conversation_id);
"""

DEFAULT_TITLE = "New chat"


def _db_path() -> Path:
    settings = get_settings()
    path = settings.resolve_path(settings.conversations_db_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


@contextmanager
def _connect() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(_db_path())
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.executescript(_SCHEMA)
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _row_to_conversation(row: sqlite3.Row) -> dict:
    d = dict(row)
    d["pinned"] = bool(d["pinned"])
    return d


def create_conversation(title: str = DEFAULT_TITLE) -> dict:
    conv_id = uuid.uuid4().hex[:16]
    now = _now()
    with _connect() as conn:
        conn.execute(
            "INSERT INTO conversations (id, title, pinned, created_at, updated_at) VALUES (?, ?, 0, ?, ?)",
            (conv_id, title or DEFAULT_TITLE, now, now),
        )
    logger.info("Created conversation", extra={"conversation_id": conv_id})
    return {"id": conv_id, "title": title or DEFAULT_TITLE, "pinned": False, "created_at": now, "updated_at": now}


def list_conversations() -> list[dict]:
    with _connect() as conn:
        rows = conn.execute(
            "SELECT id, title, pinned, created_at, updated_at FROM conversations ORDER BY updated_at DESC"
        ).fetchall()
    return [_row_to_conversation(r) for r in rows]


def get_conversation(conversation_id: str) -> dict | None:
    with _connect() as conn:
        row = conn.execute(
            "SELECT id, title, pinned, created_at, updated_at FROM conversations WHERE id = ?",
            (conversation_id,),
        ).fetchone()
        if row is None:
            return None
        message_rows = conn.execute(
            "SELECT id, role, content, response_json, created_at FROM messages "
            "WHERE conversation_id = ? ORDER BY created_at ASC",
            (conversation_id,),
        ).fetchall()

    conv = _row_to_conversation(row)
    conv["messages"] = [
        {
            "id": m["id"],
            "role": m["role"],
            "content": m["content"],
            "response": json.loads(m["response_json"]) if m["response_json"] else None,
            "created_at": m["created_at"],
        }
        for m in message_rows
    ]
    return conv


def add_message(conversation_id: str, role: str, content: str, response: ChatResponse | None = None) -> dict:
    msg_id = uuid.uuid4().hex[:16]
    now = _now()
    response_json = response.model_dump_json() if response is not None else None
    with _connect() as conn:
        conn.execute(
            "INSERT INTO messages (id, conversation_id, role, content, response_json, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (msg_id, conversation_id, role, content, response_json, now),
        )
        conn.execute("UPDATE conversations SET updated_at = ? WHERE id = ?", (now, conversation_id))

        # First user message renames the conversation away from the generic default.
        if role == "user":
            row = conn.execute("SELECT title FROM conversations WHERE id = ?", (conversation_id,)).fetchone()
            if row and row["title"] == DEFAULT_TITLE:
                title = content.strip().replace("\n", " ")[:60] or DEFAULT_TITLE
                conn.execute("UPDATE conversations SET title = ? WHERE id = ?", (title, conversation_id))

    return {"id": msg_id, "role": role, "content": content, "created_at": now}


def rename_conversation(conversation_id: str, title: str) -> bool:
    clean_title = title.strip()[:120] or DEFAULT_TITLE
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE conversations SET title = ?, updated_at = ? WHERE id = ?",
            (clean_title, _now(), conversation_id),
        )
        changed = cur.rowcount > 0
    return changed


def set_pinned(conversation_id: str, pinned: bool) -> bool:
    with _connect() as conn:
        cur = conn.execute(
            "UPDATE conversations SET pinned = ?, updated_at = ? WHERE id = ?",
            (1 if pinned else 0, _now(), conversation_id),
        )
        changed = cur.rowcount > 0
    return changed


def delete_conversation(conversation_id: str) -> bool:
    with _connect() as conn:
        conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conversation_id,))
        cur = conn.execute("DELETE FROM conversations WHERE id = ?", (conversation_id,))
        changed = cur.rowcount > 0
    return changed
