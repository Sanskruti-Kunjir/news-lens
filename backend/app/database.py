"""SQLite prediction history, user preferences, and bookmarks (parameterised queries only)."""
from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

# ------------------------------------------------------------------ schema v1
SCHEMA = """
CREATE TABLE IF NOT EXISTS prediction_history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    headline TEXT NOT NULL,
    short_description TEXT,
    predicted_category TEXT NOT NULL,
    candidate_scores TEXT NOT NULL,
    score_type TEXT,
    model_version TEXT,
    input_mode TEXT,
    latency_ms REAL,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_history_created ON prediction_history(created_at);
CREATE INDEX IF NOT EXISTS idx_history_category ON prediction_history(predicted_category);
"""

# ------------------------------------------------------------------ schema v2
SCHEMA_V2 = """
CREATE TABLE IF NOT EXISTS user_preferences (
    user_id TEXT PRIMARY KEY,
    preferred_categories TEXT NOT NULL DEFAULT '[]',
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS bookmarks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id TEXT NOT NULL DEFAULT 'local',
    article_id INTEGER NOT NULL,
    headline TEXT NOT NULL,
    category TEXT,
    created_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_bookmarks_user ON bookmarks(user_id);
"""


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    return conn


def _migrate(conn: sqlite3.Connection) -> None:
    """Add columns introduced after v1.  Safe to call multiple times."""
    for col, typ in [("sentiment_label", "TEXT"), ("sentiment_scores", "TEXT")]:
        try:
            conn.execute(f"ALTER TABLE prediction_history ADD COLUMN {col} {typ}")
        except sqlite3.OperationalError:
            pass  # column already exists


def init_db(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(_connect(path)) as conn:
        conn.executescript(SCHEMA)
        conn.executescript(SCHEMA_V2)
        _migrate(conn)
        conn.commit()


# ===================================================================== history
def insert(path: Path, headline: str, description: str, result: dict,
           sentiment: dict | None = None) -> int:
    sent_label = sentiment.get("sentiment") if sentiment else None
    sent_scores = json.dumps(sentiment.get("scores")) if sentiment and sentiment.get("scores") else None
    with closing(_connect(path)) as conn:
        cur = conn.execute(
            "INSERT INTO prediction_history (headline, short_description, predicted_category, candidate_scores,"
            " score_type, model_version, input_mode, latency_ms, created_at, sentiment_label, sentiment_scores)"
            " VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            (headline, description or None, result["predicted_category"], json.dumps(result["candidates"]),
             result["score_type"], result["model_version"], result["input_mode"], result["latency_ms"],
             datetime.now(timezone.utc).isoformat(), sent_label, sent_scores),
        )
        conn.commit()
        return int(cur.lastrowid)


def _row(r: sqlite3.Row) -> dict:
    d = dict(r)
    d["candidate_scores"] = json.loads(d["candidate_scores"])
    if d.get("sentiment_scores"):
        d["sentiment_scores"] = json.loads(d["sentiment_scores"])
    return d


def list_history(path: Path, page: int, page_size: int, search: str | None, category: str | None, newest_first: bool):
    where, params = [], []
    if search:
        where.append("headline LIKE ?")
        params.append(f"%{search}%")
    if category:
        where.append("predicted_category = ?")
        params.append(category)
    clause = ("WHERE " + " AND ".join(where)) if where else ""
    order = "DESC" if newest_first else "ASC"
    with closing(_connect(path)) as conn:
        total = conn.execute(f"SELECT COUNT(*) FROM prediction_history {clause}", params).fetchone()[0]
        rows = conn.execute(
            f"SELECT * FROM prediction_history {clause} ORDER BY created_at {order}, id {order} LIMIT ? OFFSET ?",
            [*params, page_size, (page - 1) * page_size],
        ).fetchall()
    return [_row(r) for r in rows], int(total)


def get_one(path: Path, record_id: int):
    with closing(_connect(path)) as conn:
        r = conn.execute("SELECT * FROM prediction_history WHERE id = ?", (record_id,)).fetchone()
    return _row(r) if r else None


def delete_one(path: Path, record_id: int) -> bool:
    with closing(_connect(path)) as conn:
        cur = conn.execute("DELETE FROM prediction_history WHERE id = ?", (record_id,))
        conn.commit()
        return cur.rowcount > 0


def clear(path: Path) -> int:
    with closing(_connect(path)) as conn:
        cur = conn.execute("DELETE FROM prediction_history")
        conn.commit()
        return cur.rowcount


# ================================================================ preferences
def get_preferences(path: Path, user_id: str = "local") -> dict:
    with closing(_connect(path)) as conn:
        r = conn.execute("SELECT * FROM user_preferences WHERE user_id = ?", (user_id,)).fetchone()
    if r is None:
        return {"user_id": user_id, "preferred_categories": [], "updated_at": None}
    return {"user_id": r["user_id"],
            "preferred_categories": json.loads(r["preferred_categories"]),
            "updated_at": r["updated_at"]}


def upsert_preferences(path: Path, user_id: str, categories: list[str]) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    cats_json = json.dumps(categories)
    with closing(_connect(path)) as conn:
        conn.execute(
            "INSERT INTO user_preferences (user_id, preferred_categories, updated_at)"
            " VALUES (?, ?, ?) ON CONFLICT(user_id) DO UPDATE SET preferred_categories=excluded.preferred_categories,"
            " updated_at=excluded.updated_at",
            (user_id, cats_json, now),
        )
        conn.commit()
    return {"user_id": user_id, "preferred_categories": categories, "updated_at": now}


# =================================================================== bookmarks
def list_bookmarks(path: Path, user_id: str = "local") -> list[dict]:
    with closing(_connect(path)) as conn:
        rows = conn.execute(
            "SELECT * FROM bookmarks WHERE user_id = ? ORDER BY created_at DESC", (user_id,)
        ).fetchall()
    return [dict(r) for r in rows]


def add_bookmark(path: Path, user_id: str, article_id: int,
                 headline: str, category: str | None) -> dict:
    now = datetime.now(timezone.utc).isoformat()
    with closing(_connect(path)) as conn:
        # Prevent duplicates
        existing = conn.execute(
            "SELECT id FROM bookmarks WHERE user_id = ? AND article_id = ?",
            (user_id, article_id)
        ).fetchone()
        if existing:
            return {"id": existing["id"], "user_id": user_id, "article_id": article_id,
                    "headline": headline, "category": category, "created_at": now,
                    "already_existed": True}
        cur = conn.execute(
            "INSERT INTO bookmarks (user_id, article_id, headline, category, created_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (user_id, article_id, headline, category, now),
        )
        conn.commit()
        return {"id": int(cur.lastrowid), "user_id": user_id, "article_id": article_id,
                "headline": headline, "category": category, "created_at": now}


def remove_bookmark(path: Path, bookmark_id: int, user_id: str = "local") -> bool:
    with closing(_connect(path)) as conn:
        cur = conn.execute(
            "DELETE FROM bookmarks WHERE id = ? AND user_id = ?", (bookmark_id, user_id)
        )
        conn.commit()
        return cur.rowcount > 0
