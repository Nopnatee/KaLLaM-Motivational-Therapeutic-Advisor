#!/usr/bin/env python3
# ChatbotManager SQLite smoketest with explicit I/O visualization and signature-safe store calls.
# No Orchestrator paths are ever called. Launches local, auto-opens browser.

from __future__ import annotations

import inspect
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import gradio as gr

# Monkeypatch Orchestrator BEFORE importing ChatbotManager to block any agent init.
from kallam.app import chatbot_manager as cm_mod  # <-- adjust only if your package root differs

class _DummyOrchestrator:
    def __init__(self, *a, **k): ...
    def __getattr__(self, name):
        raise RuntimeError(f"Orchestrator method '{name}' should NOT be called in this test.")

cm_mod.Orchestrator = _DummyOrchestrator

from kallam.app.chatbot_manager import ChatbotManager  # now safe to import

DB_FILE = Path("./.smoketest.db").resolve()
MGR = ChatbotManager(db_path=str(DB_FILE), summarize_every_n_messages=10, message_limit=50)
MGR.orchestrator = _DummyOrchestrator()  # belt-and-suspenders

# ---------- helpers ----------
def J(x: Any) -> str:
    return json.dumps(x, indent=2, ensure_ascii=False)

def _call_store(func, *, session_id: str, **kwargs):
    """
    Adapt kwargs to the store method's real signature.
    Maps common synonyms and silently drops unknowns.
    """
    sig = inspect.signature(func)
    params = sig.parameters
    out = {}

    def allow(names: List[str], val: Any):
        for n in names:
            if n in params:
                out[n] = val
                return True
        return False

    if "content" in kwargs: allow(["content"], kwargs["content"])
    if "translated" in kwargs: allow(["translated", "translated_content"], kwargs["translated"])
    if "reasoning" in kwargs: allow(["reasoning", "chain_of_thoughts"], kwargs["reasoning"])
    if "flags" in kwargs: allow(["flags"], kwargs["flags"])

    if "tokens_in" in kwargs: allow(["tokens_in", "tokens_input", "tokens"], kwargs["tokens_in"])
    if "tokens_out" in kwargs: allow(["tokens_out", "tokens_output", "tokens"], kwargs["tokens_out"])

    if "latency_ms" in kwargs: allow(["latency_ms", "latency"], kwargs["latency_ms"])

    return func(session_id, **out)

# ---------- session ops (return Tuple[input_json, output_json_or_str]) ----------
def create_session_fn(saved_memories: str) -> Tuple[str, str]:
    _in = {"saved_memories": saved_memories}
    sid = MGR.start_session(saved_memories or None)
    return J(_in), J({"status": "created", "session_id": sid})

def list_sessions_fn(active_only: bool) -> Tuple[str, str]:
    _in = {"active_only": active_only}
    rows = MGR.list_sessions(active_only=active_only, limit=200) or []
    return J(_in), J({"count": len(rows), "items": rows})

def get_session_meta_fn(session_id: str) -> Tuple[str, str]:
    _in = {"session_id": session_id}
    if not session_id.strip():
        return J(_in), "Provide session_id."
    meta = MGR.get_session(session_id.strip())
    return J(_in), J(meta or {})

def close_session_fn(session_id: str) -> Tuple[str, str]:
    _in = {"session_id": session_id}
    if not session_id.strip():
        return J(_in), "Provide session_id."
    ok = MGR.close_session(session_id.strip())
    return J(_in), J({"closed": ok, "session_id": session_id})

def delete_session_fn(session_id: str) -> Tuple[str, str]:
    _in = {"session_id": session_id}
    if not session_id.strip():
        return J(_in), "Provide session_id."
    ok = MGR.delete_session(session_id.strip())
    return J(_in), J({"deleted": ok, "session_id": session_id})

def cleanup_old_fn(days_old: float) -> Tuple[str, str]:
    _in = {"days_old": days_old}
    try:
        d = int(days_old)
    except Exception:
        return J(_in), "days_old must be an integer."
    if d <= 0:
        return J(_in), "days_old must be positive."
    n = MGR.cleanup_old_sessions(days_old=d)
    return J(_in), J({"cleaned_sessions": n, "threshold_days": d})

# ---------- messages (echo; never hits Orchestrator) ----------
def append_user_fn(session_id: str, content: str) -> Tuple[str, str]:
    _in = {"session_id": session_id, "content": content}
    if not session_id.strip():
        return J(_in), "Provide session_id."
    if not content.strip():
        return J(_in), "Provide content."
    _call_store(
        MGR.messages.append_user,
        session_id=session_id.strip(),
        content=content,
        translated=content,
        flags={"language": "english", "doctor": False, "psychologist": False},
        tokens_in=len(content.split()),
    )
    tail = MGR.get_original_chat_history(session_id.strip(), limit=10)
    return J(_in), J({"status": "user_appended", "tail_history": tail})

def append_assistant_fn(session_id: str, content: str) -> Tuple[str, str]:
    _in = {"session_id": session_id, "content": content}
    if not session_id.strip():
        return J(_in), "Provide session_id."
    if not content.strip():
        return J(_in), "Provide content."
    _call_store(
        MGR.messages.append_assistant,
        session_id=session_id.strip(),
        content=content,
        translated=content,
        reasoning={"final_output": content, "note": "echo_sample"},
        tokens_out=len(content.split()),
        latency_ms=0,  # will be ignored if store doesn't accept it
    )
    tail = MGR.get_original_chat_history(session_id.strip(), limit=10)
    return J(_in), J({"status": "assistant_appended", "tail_history": tail})

# ---------- summaries ----------
def add_summary_fn(session_id: str, text: str) -> Tuple[str, str]:
    _in = {"session_id": session_id, "summary": text}
    if not session_id.strip():
        return J(_in), "Provide session_id."
    if not text.strip():
        return J(_in), "Provide summary text."
    MGR.summaries.add(session_id.strip(), text)
    return J(_in), J({"status": "summary_added"})

def list_summaries_fn(session_id: str) -> Tuple[str, str]:
    _in = {"session_id": session_id}
    if not session_id.strip():
        return J(_in), "Provide session_id."
    lst = MGR.summaries.list(session_id.strip()) or []
    return J(_in), J({"count": len(lst), "items": lst})

# ---------- inspect ----------
def get_history_fn(session_id: str, limit: float) -> Tuple[str, str]:
    _in = {"session_id": session_id, "limit": limit}
    if not session_id.strip():
        return J(_in), "Provide session_id."
    try:
        n = max(1, min(500, int(limit)))
    except Exception:
        n = 50
    rows = MGR.get_original_chat_history(session_id.strip(), limit=n)
    return J(_in), J(rows or [])

def get_stats_fn(session_id: str) -> Tuple[str, str]:
    _in = {"session_id": session_id}
    if not session_id.strip():
        return J(_in), "Provide session_id."
    payload = MGR.get_session_stats(session_id.strip())
    return J(_in), J(payload or {})

def export_json_fn(session_id: str) -> Tuple[str, str]:
    _in = {"session_id": session_id}
    if not session_id.strip():
        return J(_in), "Provide session_id."
    path = MGR.export_session_json(session_id.strip())
    return J(_in), f"Exported to: {path}"

# ---------- UI (TabbedInterface: each tab shows Inputs + Output) ----------
io_in = "Inputs (echoed JSON)"
io_out = "Output (raw JSON / text)"

tabs = [
    gr.Interface(create_session_fn, [gr.Textbox(label="Saved memories (optional)")],
                 [gr.Code(label=io_in, language="json"), gr.Code(label=io_out, language="json")],
                 title="Create Session"),
    gr.Interface(list_sessions_fn, [gr.Checkbox(label="Active only", value=False)],
                 [gr.Code(label=io_in, language="json"), gr.Code(label=io_out, language="json")],
                 title="List Sessions"),
    gr.Interface(get_session_meta_fn, [gr.Textbox(label="Session ID")],
                 [gr.Code(label=io_in, language="json"), gr.Code(label=io_out, language="json")],
                 title="Get Session Meta"),
    gr.Interface(cleanup_old_fn, [gr.Number(label="days_old", value=30, precision=0)],
                 [gr.Code(label=io_in, language="json"), gr.Code(label=io_out, language="json")],
                 title="Cleanup Old Sessions"),
    gr.Interface(close_session_fn, [gr.Textbox(label="Session ID")],
                 [gr.Code(label=io_in, language="json"), gr.Code(label=io_out, language="json")],
                 title="Close Session"),
    gr.Interface(delete_session_fn, [gr.Textbox(label="Session ID")],
                 [gr.Code(label=io_in, language="json"), gr.Code(label=io_out, language="json")],
                 title="Delete Session"),
    gr.Interface(append_user_fn, [gr.Textbox(label="Session ID"), gr.Textbox(label="User content")],
                 [gr.Code(label=io_in, language="json"), gr.Code(label=io_out, language="json")],
                 title="Append User (echo)"),
    gr.Interface(append_assistant_fn, [gr.Textbox(label="Session ID"), gr.Textbox(label="Assistant content")],
                 [gr.Code(label=io_in, language="json"), gr.Code(label=io_out, language="json")],
                 title="Append Assistant (echo)"),
    gr.Interface(add_summary_fn, [gr.Textbox(label="Session ID"), gr.Textbox(label="Summary text")],
                 [gr.Code(label=io_in, language="json"), gr.Code(label=io_out, language="json")],
                 title="Add Summary"),
    gr.Interface(list_summaries_fn, [gr.Textbox(label="Session ID")],
                 [gr.Code(label=io_in, language="json"), gr.Code(label=io_out, language="json")],
                 title="List Summaries"),
    gr.Interface(get_history_fn, [gr.Textbox(label="Session ID"), gr.Number(label="Limit", value=50, precision=0)],
                 [gr.Code(label=io_in, language="json"), gr.Code(label=io_out, language="json")],
                 title="Get Original History"),
    gr.Interface(get_stats_fn, [gr.Textbox(label="Session ID")],
                 [gr.Code(label=io_in, language="json"), gr.Code(label=io_out, language="json")],
                 title="Get Session Stats"),
    gr.Interface(export_json_fn, [gr.Textbox(label="Session ID")],
                 [gr.Code(label=io_in, language="json"), gr.Markdown(label=io_out)],
                 title="Export Session JSON"),
]

demo = gr.TabbedInterface(
    tabs,
    tab_names=[
        "Create", "List", "Meta", "Cleanup", "Close", "Delete",
        "User Msg", "Assistant Msg", "Add Summary", "List Summaries",
        "History", "Stats", "Export"
    ],
    title="ChatbotManager SQLite Smoketest (No Orchestrator)"
)

if __name__ == "__main__":
    demo.launch(share=False, inbrowser=True)
