# infra/message_store.py
from typing import Any, Dict, List, Tuple, Optional
import json
from kallam.infra.db import sqlite_conn
from datetime import datetime
import uuid

class MessageStore:
    def __init__(self, db_path: str): 
        self.db_path = db_path.replace("sqlite:///", "")

    def get_translated_history(self, session_id: str, limit: int) -> List[Dict[str, str]]:
        with sqlite_conn(self.db_path) as c:
            rows = c.execute("""
                select role, coalesce(translated_content, content) as content
                from messages where session_id=? and role in ('user','assistant')
                order by id desc limit ?""", (session_id, limit)).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    def get_reasoning_traces(self, session_id: str, limit: int = 10) -> List[Dict[str, Any]]:
        with sqlite_conn(self.db_path) as c:
            rows = c.execute("""
                select message_id, chain_of_thoughts from messages
                where session_id=? and chain_of_thoughts is not null
                order by id desc limit ?""", (session_id, limit)).fetchall()
        out = []
        for r in rows:
            try:
                out.append({"message_id": r["message_id"], "contents": json.loads(r["chain_of_thoughts"])})
            except json.JSONDecodeError:
                continue
        return out

    def append_user(self, session_id: str, content: str, translated: str | None,
                    flags: Dict[str, Any] | None, tokens_in: int) -> None:
        self._append(session_id, "user", content, translated, None, None, flags, tokens_in, 0)

    def append_assistant(self, session_id: str, content: str, translated: str | None,
                         reasoning: Dict[str, Any] | None, tokens_out: int) -> None:
        self._append(session_id, "assistant", content, translated, reasoning, None, None, 0, tokens_out)

    def _append(self, session_id, role, content, translated, reasoning, latency_ms, flags, tok_in, tok_out):
        mid = f"MSG-{uuid.uuid4().hex[:8].upper()}"
        now = datetime.now().isoformat()
        with sqlite_conn(self.db_path) as c:
            c.execute("""insert into messages (session_id,message_id,timestamp,role,content,
                         translated_content,chain_of_thoughts,tokens_input,tokens_output,latency_ms,flags)
                         values (?,?,?,?,?,?,?,?,?,?,?)""",
                      (session_id, mid, now, role, content,
                       translated, json.dumps(reasoning) if reasoning else None,
                       tok_in, tok_out, latency_ms, json.dumps(flags) if flags else None))
            if role == "user":
                c.execute("""update sessions set total_messages=total_messages+1,
                             total_user_messages=coalesce(total_user_messages,0)+1,
                             last_activity=? where session_id=?""", (now, session_id))
            elif role == "assistant":
                c.execute("""update sessions set total_messages=total_messages+1,
                             total_assistant_messages=coalesce(total_assistant_messages,0)+1,
                             last_activity=? where session_id=?""", (now, session_id))

    def aggregate_stats(self, session_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Returns:
          stats: {
            "message_count": int,
            "total_tokens_in": int,
            "total_tokens_out": int,
            "avg_latency": float | None,
            "first_message": str | None,  # ISO timestamp
            "last_message": str | None,   # ISO timestamp
          }
          session: dict  # full row from sessions table (as a mapping)
        """
        with sqlite_conn(self.db_path) as c:
            # Roll up message stats
            row = c.execute(
                """
                SELECT
                  COUNT(*) AS message_count,
                  COALESCE(SUM(tokens_input), 0)  AS total_tokens_in,
                  COALESCE(SUM(tokens_output), 0) AS total_tokens_out,
                  AVG(CASE WHEN role='assistant' THEN latency_ms END) AS avg_latency,
                  MIN(timestamp)                  AS first_message,
                  MAX(timestamp)                  AS last_message
                FROM messages
                WHERE session_id = ?
                  AND role IN ('user','assistant')
                """,
                (session_id,),
            ).fetchone()

            stats = {
                "message_count": row["message_count"] or 0,
                "total_tokens_in": row["total_tokens_in"] or 0,
                "total_tokens_out": row["total_tokens_out"] or 0,
                # Normalize avg_latency to float if not None
                "avg_latency": float(row["avg_latency"]) if row["avg_latency"] is not None else None,
                "first_message": row["first_message"],
                "last_message": row["last_message"],
            }

            # Fetch session info (entire row)
            session = c.execute(
                "SELECT * FROM sessions WHERE session_id = ?",
                (session_id,),
            ).fetchone() or {}

        # Convert sqlite Row to plain dict if needed
        if hasattr(session, "keys"):
            session = {k: session[k] for k in session.keys()}

        return stats, session
