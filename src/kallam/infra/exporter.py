# infra/exporter.py
import json
from pathlib import Path
from kallam.infra.db import sqlite_conn

class JsonExporter:
    def __init__(self, db_path: str, out_dir: str = "exported_sessions"):
        self.db_path = db_path.replace("sqlite:///", "")
        self.out_dir = Path(out_dir); self.out_dir.mkdir(parents=True, exist_ok=True)

    def export_session_json(self, session_id: str) -> str:
        with sqlite_conn(self.db_path) as c:
            s = c.execute("select * from sessions where session_id=?", (session_id,)).fetchone()
            if not s: raise ValueError(f"Session {session_id} does not exist")
            msgs = [dict(r) for r in c.execute("select * from messages where session_id=? order by id", (session_id,))]
            sums = [dict(r) for r in c.execute("select * from summaries where session_id=? order by id", (session_id,))]
        data = {"session_info": dict(s), "summaries": sums, "chat_history": msgs,
                "export_metadata": {"exported_at_version": "2.1"}}
        out = self.out_dir / f"{session_id}.json"
        out.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(out)

    def export_all_sessions_json(self) -> str:
        """Export *all* sessions into a single JSON file."""
        all_data = []
        with sqlite_conn(self.db_path) as c:
            session_ids = [row["session_id"] for row in c.execute("select session_id from sessions")]
            for sid in session_ids:
                s = c.execute("select * from sessions where session_id=?", (sid,)).fetchone()
                msgs = [dict(r) for r in c.execute(
                    "select * from messages where session_id=? order by id", (sid,)
                )]
                sums = [dict(r) for r in c.execute(
                    "select * from summaries where session_id=? order by id", (sid,)
                )]
                data = {
                    "session_info": dict(s),
                    "summaries": sums,
                    "chat_history": msgs,
                    "export_metadata": {"exported_at_version": "2.1"},
                }
                all_data.append(data)

        out = self.out_dir / "all_sessions.json"
        out.write_text(json.dumps(all_data, ensure_ascii=False, indent=2), encoding="utf-8")
        return str(out)