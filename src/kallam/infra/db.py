# infra/db.py
from contextlib import contextmanager
import sqlite3

@contextmanager
def sqlite_conn(path: str):
    conn = sqlite3.connect(path, timeout=30.0, check_same_thread=False)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        conn.execute("PRAGMA journal_mode = WAL")
        conn.row_factory = sqlite3.Row
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
