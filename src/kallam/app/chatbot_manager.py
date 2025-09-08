# src/your_pkg/app/chatbot_manager.py
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any

# Keep your orchestrator import AS-IS to avoid ripples.
# If your package path differs, adjust here only.
from kallam.domain.agents.orchestrator import Orchestrator

from kallam.infra.session_store import SessionStore
from kallam.infra.message_store import MessageStore
from kallam.infra.summary_store import SummaryStore
from kallam.infra.exporter import JsonExporter
from kallam.infra.token_counter import TokenCounter
from kallam.infra.db import sqlite_conn  # for the cleanup method

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

EXPORT_FOLDER = "exported_sessions"

@dataclass
class SessionStats:
    message_count: int = 0
    total_tokens_in: int = 0
    total_tokens_out: int = 0
    avg_latency: float = 0.0
    first_message: Optional[str] = None
    last_message: Optional[str] = None

class ChatbotManager:
    """
    Backward-compatible facade. Same constructor and methods as your original class.
    Under the hood we delegate to infra stores and the orchestrator.
    """

    def __init__(self,
                 db_path: str = "chatbot_data.db",
                 summarize_every_n_messages: int = 10,
                 message_limit: int = 20):
        if summarize_every_n_messages <= 0:
            raise ValueError("summarize_every_n_messages must be positive")
        if message_limit <= 0:
            raise ValueError("message_limit must be positive")

        self.orchestrator = Orchestrator()
        self.sum_every_n = summarize_every_n_messages
        self.message_limit = message_limit
        self.db_path = Path(db_path)  # still accept same arg
        self.lock = threading.RLock()
        self.tokens = TokenCounter(capacity=1000)

        # wire infra
        # Note: infra stores expect a sqlite URL or a file path; we normalize to file path.
        db_url = f"sqlite:///{self.db_path}"
        self.sessions = SessionStore(db_url)
        self.messages = MessageStore(db_url)
        self.summaries = SummaryStore(db_url)
        self.exporter = JsonExporter(db_url, out_dir=EXPORT_FOLDER)

        # ensure schema exists (stores already init; but we force create indexes equivalent to old)
        self._ensure_schema()

        logger.info(f"ChatbotManager initialized with database: {self.db_path}")

    # ---------- schema bootstrap (keeps your original tables/indexes) ----------
    def _ensure_schema(self) -> None:
        ddl_sessions = """
        CREATE TABLE IF NOT EXISTS sessions (
            session_id TEXT PRIMARY KEY,
            timestamp TEXT NOT NULL,
            last_activity TEXT NOT NULL,
            saved_memories TEXT,
            total_messages INTEGER DEFAULT 0,
            total_user_messages INTEGER DEFAULT 0,
            total_assistant_messages INTEGER DEFAULT 0,
            total_summaries INTEGER DEFAULT 0,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            is_active BOOLEAN DEFAULT 1
        );
        """
        ddl_messages = """
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            message_id TEXT UNIQUE NOT NULL,
            timestamp TEXT NOT NULL,
            role TEXT NOT NULL CHECK(role IN ('user','assistant','system')),
            content TEXT NOT NULL,
            translated_content TEXT,
            chain_of_thoughts TEXT,
            tokens_input INTEGER DEFAULT 0,
            tokens_output INTEGER DEFAULT 0,
            latency_ms INTEGER,
            flags TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
        );
        """
        ddl_summaries = """
        CREATE TABLE IF NOT EXISTS summaries (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            timestamp TEXT NOT NULL,
            summary TEXT NOT NULL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY(session_id) REFERENCES sessions(session_id) ON DELETE CASCADE
        );
        """
        idx = [
            "CREATE INDEX IF NOT EXISTS idx_messages_session_id ON messages(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp DESC)",
            "CREATE INDEX IF NOT EXISTS idx_sessions_last_activity ON sessions(last_activity DESC)",
            "CREATE INDEX IF NOT EXISTS idx_summaries_session_id ON summaries(session_id)",
            "CREATE INDEX IF NOT EXISTS idx_messages_role ON messages(role)"
        ]
        with sqlite_conn(str(self.db_path)) as c:
            c.execute(ddl_sessions)
            c.execute(ddl_messages)
            c.execute(ddl_summaries)
            for q in idx:
                c.execute(q)

    # ---------- validation/util ----------
    def _validate_inputs(self, **kwargs):
        validators = {
            'user_message': lambda x: bool(x and str(x).strip()),
            'session_id': lambda x: bool(x),
            'role': lambda x: x in ('user', 'assistant', 'system'),
        }
        for k, v in kwargs.items():
            fn = validators.get(k)
            if fn and not fn(v):
                raise ValueError(f"Invalid {k}: {v}")

    # ---------- public API: identical names & behavior ----------
    def start_session(self, saved_memories: Optional[str] = None) -> str:
        return self.sessions.create(saved_memories=saved_memories)

    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        # Prefer a stable API; get_meta should return a dict-like row
        return self.sessions.get_meta(session_id)

    def list_sessions(self, active_only: bool = True, limit: int = 50) -> List[Dict[str, Any]]:
        return self.sessions.list(active_only=active_only, limit=limit)

    def close_session(self, session_id: str) -> bool:
        return self.sessions.close(session_id)

    def delete_session(self, session_id: str) -> bool:
        return self.sessions.delete(session_id)

    def cleanup_old_sessions(self, days_old: int = 30) -> int:
        if days_old <= 0:
            raise ValueError("days_old must be positive")
        cutoff = (datetime.now() - timedelta(days=days_old)).isoformat()
        return self.sessions.cleanup_before(cutoff)

    def handle_message(self, session_id: str, user_message: str,
                       health_status: Optional[str] = None) -> str:
        self._validate_inputs(session_id=session_id, user_message=user_message)
        with self.lock:
            t0 = time.time()

            # ensure session exists
            if not self.get_session(session_id):
                raise ValueError(f"Session {session_id} not found")

            # fetch context
            eng_history = self.messages.get_translated_history(session_id, limit=self.message_limit)
            eng_summaries = self.summaries.list(session_id)
            chain = self.messages.get_reasoning_traces(session_id, limit=10)

            meta = self.sessions.get_meta(session_id) or {}
            memory_context = (meta.get("saved_memories") or "") if isinstance(meta, dict) else ""

            # flags and translation
            flags = self._get_flags_dict(session_id, user_message)
            eng_msg = self.orchestrator.get_translation(
                message=user_message, flags=flags, translation_type="forward"
            )

            # respond
            result = self.orchestrator.get_response(
                chat_history=eng_history,
                user_message=eng_msg,
                flags=flags,
                chain_of_thoughts=chain,
                memory_context=memory_context,
                summarized_histories=eng_summaries,
            )

            bot_eng = result["final_output"]
            bot_reply = self.orchestrator.get_translation(
                message=bot_eng, flags=flags, translation_type="backward"
            )
            latency_ms = int((time.time() - t0) * 1000)

            # persist
            tok_user = self.tokens.count(user_message)
            tok_bot = self.tokens.count(bot_reply)
            self.messages.append_user(session_id, content=user_message,
                                      translated=eng_msg, flags=flags, tokens_in=tok_user)
            self.messages.append_assistant(session_id, content=bot_reply,
                                           translated=bot_eng, reasoning=result,
                                           tokens_out=tok_bot, latency_ms=latency_ms)

            # summarize checkpoint
            meta = self.sessions.get_meta(session_id)
            if meta and (meta["total_user_messages"] % self.sum_every_n == 0):
                self.summarize_session(session_id)

            return bot_reply

    def _get_flags_dict(self, session_id: str, user_message: str) -> Dict[str, Any]:
        self._validate_inputs(session_id=session_id)
        try:
            # Build context Supervisor expects
            chat_history = self.messages.get_translated_history(session_id, limit=self.message_limit) or []
            summaries = self.summaries.list(session_id) or []
            meta = self.sessions.get_meta(session_id) or {}
            memory_context = (meta.get("saved_memories") or "") if isinstance(meta, dict) else ""

            flags = self.orchestrator.get_flags_from_supervisor(
                chat_history=chat_history,
                user_message=user_message,
                memory_context=memory_context,
                task="flag",
                summarized_histories=summaries
            )
            # Ensure language exists and is supported
            lang = (flags or {}).get("language") or "english"
            if lang not in {"thai", "english"}:
                lang = "english"
            flags["language"] = lang
            return flags
        except Exception as e:
            logger.warning(f"Failed to get flags from supervisor: {e}, using safe defaults")
            # Safe defaults keep the pipeline alive
            return {"language": "english", "doctor": False, "psychologist": False}


    def summarize_session(self, session_id: str) -> str:
        with self.lock:
            eng_history = self.messages.get_translated_history(session_id, limit=self.message_limit)
            if not eng_history:
                raise ValueError("No chat history found for session")
            eng_summaries = self.summaries.list(session_id)
            summary = self.orchestrator.summarize_history(
                chat_history=eng_history, eng_summaries=eng_summaries
            )
            self.summaries.add(session_id, summary)
            return summary

    def get_session_stats(self, session_id: str) -> dict:
        stats, session = self.messages.aggregate_stats(session_id)

        stats_dict = {
            "message_count": stats.get("message_count") or 0,
            "total_tokens_in": stats.get("total_tokens_in") or 0,
            "total_tokens_out": stats.get("total_tokens_out") or 0,
            "avg_latency": float(stats.get("avg_latency") or 0),
            "first_message": stats.get("first_message"),
            "last_message": stats.get("last_message"),
        }

        return {
            "session_info": session,   # already a dict from MessageStore
            "stats": stats_dict,       # plain dict, UI can call .get()
        }

    def get_original_chat_history(self, session_id: str, limit: int | None = None) -> list[dict]:
        self._validate_inputs(session_id=session_id)
        if limit is None:
            limit = self.message_limit

        if hasattr(self.messages, "get_history"):
            return self.messages.get_history(session_id=session_id, limit=limit)

        # Fallback: direct query
        with sqlite_conn(str(self.db_path)) as c:
            rows = c.execute(
                """
                SELECT role, content, timestamp
                FROM messages
                WHERE session_id = ?
                ORDER BY id ASC
                LIMIT ?
                """,
                (session_id, limit),
            ).fetchall()
            return [dict(r) for r in rows]

    def export_session_json(self, session_id: str) -> str:
        return self.exporter.export_session_json(session_id)
