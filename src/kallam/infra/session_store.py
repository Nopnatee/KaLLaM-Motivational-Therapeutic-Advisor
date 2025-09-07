# infra/session_store.py
from dataclasses import dataclass
from datetime import datetime, timedelta
from kallam.infra.db import sqlite_conn
import uuid

@dataclass
class SessionMeta:
    session_id: str
    total_user_messages: int
    # ... add fields as needed

class SessionStore:
    def __init__(self, db_path: str): self.db_path = db_path.replace("sqlite:///", "")

    def create(self, saved_memories: str | None) -> str:
        sid = f"ID-{uuid.uuid4().hex[:8].upper()}"
        now = datetime.now().isoformat()
        with sqlite_conn(self.db_path) as c:
            c.execute("""insert into sessions (session_id,timestamp,last_activity,saved_memories)
                         values (?,?,?,?)""", (sid, now, now, saved_memories))
        return sid

    def get(self, session_id: str) -> SessionMeta | None:
        with sqlite_conn(self.db_path) as c:
            r = c.execute("select * from sessions where session_id=?", (session_id,)).fetchone()
            if not r: return None
            return SessionMeta(session_id=r["session_id"],
                               total_user_messages=r["total_user_messages"] or 0)

    # add list/close/delete/cleanup using same pattern
