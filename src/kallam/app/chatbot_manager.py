# src/your_pkg/app/chatbot_manager.py
from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Any
from functools import wraps
from contextvars import ContextVar

# Keep your orchestrator import AS-IS to avoid ripples.
from kallam.domain.agents.Unified_Dataset_Orchestrator import UnifiedDatasetOrchestrator as Orchestrator

from kallam.infra.session_store import SessionStore
from kallam.infra.message_store import MessageStore
from kallam.infra.summary_store import SummaryStore
from kallam.infra.exporter import JsonExporter
from kallam.infra.token_counter import TokenCounter
from kallam.infra.db import sqlite_conn  # for the cleanup method

# -----------------------------------------------------------------------------
# Logging setup (configurable)
# -----------------------------------------------------------------------------

_request_id: ContextVar[str] = ContextVar("_request_id", default="-")

class RequestIdFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        record.request_id = _request_id.get()
        return True

def _setup_logging(level: Optional[str] = None, json_mode: bool = False, logger_name: str = "kallam.chatbot"):
    lvl = (level or os.getenv("LOG_LEVEL", "INFO")).upper()
    root = logging.getLogger()
    # Avoid duplicate handlers if constructed multiple times in REPL/tests
    if not any(isinstance(h, logging.StreamHandler) for h in root.handlers):
        handler = logging.StreamHandler()
        if json_mode or os.getenv("LOG_JSON", "0") in {"1", "true", "True"}:
            fmt = '{"ts":"%(asctime)s","lvl":"%(levelname)s","logger":"%(name)s","req":"%(request_id)s","msg":"%(message)s"}'
        else:
            fmt = "%(asctime)s | %(levelname)-7s | %(name)s | req=%(request_id)s | %(message)s"
        handler.setFormatter(logging.Formatter(fmt))
        handler.addFilter(RequestIdFilter())
        root.addHandler(handler)
    root.setLevel(lvl)
    return logging.getLogger(logger_name)

logger = _setup_logging()  # default init; can be reconfigured via ChatbotManager args

def _with_trace(level: int = logging.INFO):
    """
    Decorator to visualize call sequence with timing and exceptions.
    Uses the same request_id context if already set, or creates one.
    """
    def deco(fn):
        @wraps(fn)
        def wrapper(self, *args, **kwargs):
            rid = _request_id.get()
            created_here = False
            if rid == "-" and fn.__name__ in {"handle_message", "start_session"}:
                rid = uuid.uuid4().hex[:8]
                _request_id.set(rid)
                created_here = True
            logger.log(level, f"→ {fn.__name__}")
            t0 = time.time()
            try:
                out = fn(self, *args, **kwargs)
                dt = int((time.time() - t0) * 1000)
                logger.log(level, f"← {fn.__name__} done in {dt} ms")
                return out
            except Exception:
                logger.exception(f"✖ {fn.__name__} failed")
                raise
            finally:
                # Reset the request id when we originated it here
                if created_here:
                    _request_id.set("-")
        return wrapper
    return deco


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
                 db_path: str = "chatbot_single_llm_data.db",
                 summarize_every_n_messages: int = 10,
                 message_limit: int = 10,
                 sunmmary_limit: int = 20,
                 chain_of_thoughts_limit: int = 5,
                 # logging knobs
                 log_level: Optional[str] = None,
                 log_json: bool = False,
                 log_name: str = "kallam.chatbot",
                 trace_level: int = logging.INFO):
        if summarize_every_n_messages <= 0:
            raise ValueError("summarize_every_n_messages must be positive")
        if message_limit <= 0:
            raise ValueError("message_limit must be positive")

        # Reconfigure logger per instance if caller wants
        global logger
        logger = _setup_logging(level=log_level, json_mode=log_json, logger_name=log_name)
        self._trace_level = trace_level

        self.orchestrator = Orchestrator()
        self.sum_every_n = summarize_every_n_messages
        self.message_limit = message_limit
        self.summary_limit = sunmmary_limit
        self.chain_of_thoughts_limit = chain_of_thoughts_limit
        self.db_path = Path(db_path)
        self.lock = threading.RLock()
        self.tokens = TokenCounter(capacity=1000)

        # wire infra
        db_url = f"sqlite:///{self.db_path}"
        self.sessions = SessionStore(db_url)
        self.messages = MessageStore(db_url)
        self.summaries = SummaryStore(db_url)
        self.exporter = JsonExporter(db_url, out_dir=EXPORT_FOLDER)

        # ensure schema exists
        self._ensure_schema()

        logger.info(f"ChatbotManager initialized with database: {self.db_path}")

    # ---------- schema bootstrap ----------
    @_with_trace()
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

    # ---------- public API ----------
    @_with_trace()
    def start_session(self, saved_memories: Optional[str] = None) -> str:
        sid = self.sessions.create(saved_memories=saved_memories)
        logger.debug(f"start_session: saved_memories_len={len(saved_memories or '')} session_id={sid}")
        return sid

    @_with_trace()
    def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self.sessions.get_meta(session_id)

    @_with_trace()
    def list_sessions(self, active_only: bool = True, limit: int = 50) -> List[Dict[str, Any]]:
        return self.sessions.list(active_only=active_only, limit=limit)

    @_with_trace()
    def close_session(self, session_id: str) -> bool:
        return self.sessions.close(session_id)

    @_with_trace()
    def delete_session(self, session_id: str) -> bool:
        return self.sessions.delete(session_id)

    @_with_trace()
    def cleanup_old_sessions(self, days_old: int = 30) -> int:
        if days_old <= 0:
            raise ValueError("days_old must be positive")
        cutoff = (datetime.now() - timedelta(days=days_old)).isoformat()
        return self.sessions.cleanup_before(cutoff)

    @_with_trace()
    def handle_message(self, session_id: str, user_message: str) -> str:
        # Ensure one correlation id per request flow
        if _request_id.get() == "-":
            _request_id.set(uuid.uuid4().hex[:8])

        self._validate_inputs(session_id=session_id, user_message=user_message)
        with self.lock:
            t0 = time.time()

            # ensure session exists
            if not self.get_session(session_id):
                raise ValueError(f"Session {session_id} not found")

            # fetch context
            original_history = self.messages.get_original_history(session_id, limit=self.message_limit)
            eng_history = self.messages.get_translated_history(session_id, limit=self.message_limit)
            eng_summaries = self.summaries.list(session_id, limit=self.summary_limit)
            chain = self.messages.get_reasoning_traces(session_id, limit=self.chain_of_thoughts_limit)

            meta = self.sessions.get_meta(session_id) or {}
            memory_context = (meta.get("saved_memories") or "") if isinstance(meta, dict) else ""

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "context pulled: history=%d summaries=%d chain=%d mem_len=%d",
                    len(eng_history or []), len(eng_summaries or []), len(chain or []), len(memory_context),
                )

            # flags and translation
            flags = self._get_flags_dict(session_id, user_message)
            if logger.isEnabledFor(logging.DEBUG):
                # keep flags concise if large
                short_flags = {k: (v if isinstance(v, (int, float, bool, str)) else "…") for k, v in (flags or {}).items()}
                logger.debug(f"flags: {short_flags}")

            eng_msg = self.orchestrator.get_translation(
                message=user_message, flags=flags, translation_type="forward"
            )

            # respond
            response_commentary = self.orchestrator.get_commented_response(
                original_history=original_history,
                original_message=user_message,
                eng_history=eng_history,
                eng_message=eng_msg,
                flags=flags,
                chain_of_thoughts=chain,
                memory_context=memory_context,
                summarized_histories=eng_summaries,
            )

            bot_message = response_commentary["final_output"]
            bot_eng = self.orchestrator.get_translation(
                message=bot_message, flags=flags, translation_type="forward"
            )
            latency_ms = int((time.time() - t0) * 1000)

            # persist
            tok_user = self.tokens.count(user_message)
            tok_bot = self.tokens.count(bot_message)

            self.messages.append_user(session_id, content=user_message,
                                      translated=eng_msg, flags=flags, tokens_in=tok_user)
            self.messages.append_assistant(session_id, content=bot_message,
                                           translated=bot_eng, reasoning=response_commentary,
                                           tokens_out=tok_bot)

            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(
                    "persisted: tokens_in=%d tokens_out=%d latency_ms=%d", tok_user, tok_bot, latency_ms
                )

            # summarize checkpoint
            meta = self.sessions.get_meta(session_id)
            if meta and (meta["total_user_messages"] % self.sum_every_n == 0) and (meta["total_user_messages"] > 0):
                logger.log(self._trace_level, f"checkpoint: summarizing session {session_id}")
                self.summarize_session(session_id)

            return bot_message

    @_with_trace()
    def _get_flags_dict(self, session_id: str, user_message: str) -> Dict[str, Any]:
        self._validate_inputs(session_id=session_id)
        try:
            # Build context Supervisor expects
            chat_history = self.messages.get_translated_history(session_id, limit=self.message_limit) or []
            summaries = self.summaries.list(session_id, limit=self.message_limit) or []
            meta = self.sessions.get_meta(session_id) or {}
            memory_context = (meta.get("saved_memories") or "") if isinstance(meta, dict) else ""

            flags = self.orchestrator.get_flags_from_supervisor(
                chat_history=chat_history,
                user_message=user_message,
                memory_context=memory_context,
                summarized_histories=summaries
            )
            return flags
        except Exception as e:
            logger.warning(f"Failed to get flags from supervisor: {e}, using safe defaults")
            # Safe defaults keep the pipeline alive
            return {"language": "invalid", "doctor": False, "psychologist": False}

    @_with_trace()
    def summarize_session(self, session_id: str) -> str:
        eng_history = self.messages.get_translated_history(session_id, limit=self.message_limit)
        if not eng_history:
            raise ValueError("No chat history found for session")
        eng_summaries = self.summaries.list(session_id)
        eng_summary = self.orchestrator.summarize_history(
            chat_history=eng_history, eng_summaries=eng_summaries
        )
        self.summaries.add(session_id, eng_summary)
        logger.debug("summary_len=%d total_summaries=%d", len(eng_summary or ""), len(eng_summaries or []) + 1)
        return eng_summary

    @_with_trace()
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
        logger.debug("stats: %s", stats_dict)
        return {
            "session_info": session,   # already a dict from MessageStore
            "stats": stats_dict,       # plain dict
        }

    @_with_trace()
    def get_original_chat_history(self, session_id: str, limit: int | None = None) -> list[dict]:
        self._validate_inputs(session_id=session_id)
        if limit is None:
            limit = self.message_limit

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

    @_with_trace()
    def export_session_json(self, session_id: str) -> str:
        path = self.exporter.export_session_json(session_id)
        logger.info(f"exported session to {path}")
        return path
    
    @_with_trace()
    def export_all_sessions_json(self) -> str:
        path = self.exporter.export_all_sessions_json()
        logger.info(f"exported session to {path}")
        return path
