# infra/summary_store.py
from datetime import datetime
from kallam.infra.db import sqlite_conn

class SummaryStore:
    def __init__(self, db_path: str): self.db_path = db_path.replace("sqlite:///", "")

    def list(self, session_id: str, limit: int | None = None):
        q = "select timestamp, summary from summaries where session_id=? order by id desc"
        params = [session_id]
        if limit: q += " limit ?"; params.append(limit)
        with sqlite_conn(self.db_path) as c:
            rows = c.execute(q, params).fetchall()
        return [{"timestamp": r["timestamp"], "summary": r["summary"]} for r in rows]

    def add(self, session_id: str, summary: str):
        now = datetime.now().isoformat()
        with sqlite_conn(self.db_path) as c:
            c.execute("insert into summaries (session_id, timestamp, summary) values (?,?,?)",
                      (session_id, now, summary))
            c.execute("update sessions set total_summaries = total_summaries + 1, last_activity=? where session_id=?",
                      (now, session_id))
