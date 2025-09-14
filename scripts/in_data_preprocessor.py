# make_test_from_all_sessions.py
# Usage: python make_test_from_all_sessions.py
# Output: test_from_sessions.jsonl in the same folder

import json
import re
from pathlib import Path
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parents[1]
IN_PATH = REPO_ROOT / "exported_sessions" / "all_sessions.json"
OUT_PATH = REPO_ROOT / "data" / "orchestrated" / "pre_annotate.jsonl"

# Map your roles to the target schema
ROLE_MAP = {
    "user": "Client",
    "assistant": "Therapist",
}

# Strip common speaker prefixes if they slipped into the text
PREFIX_RE = re.compile(r'^\s*(?:User|Bot|Client|Therapist)\s*:\s*', re.IGNORECASE)

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.strip()
    text = PREFIX_RE.sub("", text)
    # Optionally collapse internal whitespace a bit
    return text

def iso_to_dt(s):
    # best effort parser; fall back to created_at
    try:
        return datetime.fromisoformat(s.replace("Z",""))
    except Exception:
        return None

def iter_messages(all_sessions):
    """
    Yield normalized (timestamp, role, text) across all sessions in order.
    Within each session, we sort by timestamp then id for stability.
    """
    for sess in all_sessions:
        history = sess.get("chat_history", []) or []
        # sort robustly by timestamp then id
        def sort_key(m):
            ts = m.get("timestamp") or m.get("created_at") or ""
            dt = iso_to_dt(ts) or datetime.max
            return (dt, m.get("id", 10**12))
        history = sorted(history, key=sort_key)

        for m in history:
            role = (m.get("role") or "").lower()
            if role not in ROLE_MAP:
                continue  # ignore system or unknown
            text = m.get("content") or ""
            text = clean_text(text)
            if not text:
                continue
            yield {
                "role": ROLE_MAP[role],
                "text": text
            }

def main():
    if not IN_PATH.exists():
        raise FileNotFoundError(f"Missing {IN_PATH}")
    with IN_PATH.open("r", encoding="utf-8") as f:
        all_sessions = json.load(f)

    # Build rolling examples
    rolling_history = []
    n_written = 0
    with OUT_PATH.open("w", encoding="utf-8") as out:
        for msg in iter_messages(all_sessions):
            # Current example: previous turns go to "history", current goes to "utterance_*"
            example = {
                "history": rolling_history.copy(),
                "utterance_role": msg["role"],
                "utterance_text": msg["text"],
            }
            out.write(json.dumps(example, ensure_ascii=False) + "\n")
            n_written += 1
            # Update rolling history with the current utterance
            rolling_history.append({"role": msg["role"], "text": msg["text"]})

    print(f"Wrote {n_written} lines to {OUT_PATH}")

if __name__ == "__main__":
    main()
