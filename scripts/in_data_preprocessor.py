# make_test_from_all_sessions.py
# Usage from CLI (still works): python make_test_from_all_sessions.py
# Usage from Python: main("path/to/input.json", "path/to/output.jsonl")

import json
import re
from pathlib import Path
from datetime import datetime

# Defaults 
DEFAULT_IN = Path("exported_sessions/raw_gemini.json")
DEFAULT_OUT = Path("data/gemini/pre_annotate.json")

ROLE_MAP = {
    "user": "Client",
    "assistant": "Therapist",
}

PREFIX_RE = re.compile(r'^\s*(?:User|Bot|Client|Therapist)\s*:\s*', re.IGNORECASE)

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return PREFIX_RE.sub("", text.strip())

def iso_to_dt(s):
    try:
        return datetime.fromisoformat(s.replace("Z",""))
    except Exception:
        return None

def iter_messages(all_sessions):
    for sess in all_sessions:
        history = sess.get("chat_history", []) or []

        def sort_key(m):
            ts = m.get("timestamp") or m.get("created_at") or ""
            dt = iso_to_dt(ts) or datetime.max
            return (dt, m.get("id", 10**12))
        history = sorted(history, key=sort_key)

        for m in history:
            role = (m.get("role") or "").lower()
            if role not in ROLE_MAP:
                continue
            text = clean_text(m.get("content") or "")
            if not text:
                continue
            yield {"role": ROLE_MAP[role], "text": text}

def main(in_path: Path = DEFAULT_IN, out_path: Path = DEFAULT_OUT):
    in_path = Path(in_path)
    out_path = Path(out_path)

    if not in_path.exists():
        raise FileNotFoundError(f"Missing {in_path}")
    with in_path.open("r", encoding="utf-8") as f:
        all_sessions = json.load(f)

    rolling_history = []
    n_written = 0
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as out:
        for msg in iter_messages(all_sessions):
            example = {
                "history": rolling_history.copy(),
                "utterance_role": msg["role"],
                "utterance_text": msg["text"],
            }
            out.write(json.dumps(example, ensure_ascii=False) + "\n")
            n_written += 1
            rolling_history.append({"role": msg["role"], "text": msg["text"]})

    print(f"Wrote {n_written} lines to {out_path}")

if __name__ == "__main__":
    # Still works from CLI with defaults
    main()
