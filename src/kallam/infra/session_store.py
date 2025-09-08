# infra/session_store.py
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, Dict, Any, List
from kallam.infra.db import sqlite_conn
import uuid

@dataclass
class SessionMeta:
    session_id: str
    total_user_messages: int

class SessionStore:
    """
    Storage facade expected by ChatbotManager.
    Accepts either 'sqlite:///path/to.db' or 'path/to.db' and normalizes to file path.
    """
    def __init__(self, db_path: str):
        # ChatbotManager passes f"sqlite:///{Path(...)}"
        self.db_path = db_path.replace("sqlite:///", "")

    # ----------------- create -----------------
    def create(self, saved_memories: Optional[str]) -> str:
        sid = f"ID-{uuid.uuid4().hex[:8].upper()}"
        now = datetime.now().isoformat()
        with sqlite_conn(self.db_path) as c:
            c.execute(
                """
                INSERT INTO sessions (
                    session_id, timestamp, last_activity, saved_memories,
                    total_messages, total_user_messages, total_assistant_messages,
                    total_summaries, is_active
                )
                VALUES (?, ?, ?, ?, 0, 0, 0, 0, 1)
                """,
                (sid, now, now, saved_memories),
            )
        return sid

    # ----------------- read (typed) -----------------
    def get(self, session_id: str) -> Optional[SessionMeta]:
        with sqlite_conn(self.db_path) as c:
            r = c.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            if not r:
                return None
            return SessionMeta(
                session_id=r["session_id"],
                total_user_messages=r["total_user_messages"] or 0,
            )

    # ----------------- read (raw dict) -----------------
    def get_raw(self, session_id: str) -> Optional[Dict[str, Any]]:
        with sqlite_conn(self.db_path) as c:
            r = c.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone()
            return dict(r) if r else None

    # ----------------- meta subset for manager -----------------
    def get_meta(self, session_id: str) -> Optional[Dict[str, Any]]:
        with sqlite_conn(self.db_path) as c:
            r = c.execute(
                """
                SELECT session_id, total_messages, total_user_messages,
                       total_assistant_messages, total_summaries, last_activity, is_active
                FROM sessions
                WHERE session_id = ?
                """,
                (session_id,),
            ).fetchone()
            return dict(r) if r else None

    # ----------------- list -----------------
    def list(self, active_only: bool = True, limit: int = 50) -> List[Dict[str, Any]]:
        sql = "SELECT * FROM sessions"
        params: List[Any] = []
        if active_only:
            sql += " WHERE is_active = 1"
        sql += " ORDER BY last_activity DESC LIMIT ?"
        params.append(limit)
        with sqlite_conn(self.db_path) as c:
            rows = c.execute(sql, params).fetchall()
            return [dict(r) for r in rows]

    # ----------------- close -----------------
    def close(self, session_id: str) -> bool:
        with sqlite_conn(self.db_path) as c:
            res = c.execute(
                "UPDATE sessions SET is_active = 0, last_activity = ? WHERE session_id = ?",
                (datetime.now().isoformat(), session_id),
            )
            return res.rowcount > 0

    # ----------------- delete -----------------
    def delete(self, session_id: str) -> bool:
        # messages/summaries are ON DELETE CASCADE according to your schema
        with sqlite_conn(self.db_path) as c:
            res = c.execute("DELETE FROM sessions WHERE session_id = ?", (session_id,))
            return res.rowcount > 0

    # ----------------- cleanup -----------------
    def cleanup_before(self, cutoff_iso: str) -> int:
        with sqlite_conn(self.db_path) as c:
            res = c.execute(
                "DELETE FROM sessions WHERE last_activity < ?",
                (cutoff_iso,),
            )
            return res.rowcount

    # Optional: utility to bump last_activity
    def touch(self, session_id: str) -> None:
        with sqlite_conn(self.db_path) as c:
            c.execute(
                "UPDATE sessions SET last_activity = ? WHERE session_id = ?",
                (datetime.now().isoformat(), session_id),
            )
