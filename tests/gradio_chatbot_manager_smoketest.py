#!/usr/bin/env python3
# Minimal Gradio smoketest for ChatbotManager's SQLite paths (no agents).
# Run: python tests/gradio_chatbot_manager_smoketest.py
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import gradio as gr

# 1) Import the module to monkeypatch Orchestrator BEFORE ChatbotManager is constructed.
from kallam.app import chatbot_manager as cm_mod  # adjust package root if needed

class _DummyOrchestrator:
    def __init__(self, *a, **k): pass
    def __getattr__(self, name):
        raise RuntimeError(f"Orchestrator method '{name}' should NOT be called in this test app.")

# Monkeypatch the symbol ChatbotManager imports at module scope.
cm_mod.Orchestrator = _DummyOrchestrator  # stop real init chatter/logs

# 2) Now import the class
from kallam.app.chatbot_manager import ChatbotManager

DB_FILE = Path("./.smoketest.db").resolve()

def init_manager(db_path: str) -> ChatbotManager:
    mgr = ChatbotManager(db_path=db_path, summarize_every_n_messages=10, message_limit=50)
    # Belt-and-suspenders: if someone rebinds later, keep it dummy.
    mgr.orchestrator = _DummyOrchestrator()
    return mgr

MGR = init_manager(str(DB_FILE))

def _choices() -> List[str]:
    rows = MGR.list_sessions(active_only=False, limit=200) or []
    try:
        rows.sort(key=lambda r: (r.get("last_activity") or r.get("timestamp") or ""), reverse=True)
    except Exception:
        pass
    return [r["session_id"] for r in rows if r.get("session_id")]

# Helpers that return proper UI updates (no more list-into-Markdown crimes)
def dd_update(value: Optional[str] = None) -> gr.Dropdown:
    choices = _choices()
    if value not in choices:
        value = None
    return gr.Dropdown.update(choices=choices, value=value)

# ---------- Session ops ----------
def create_session(saved_memories: str):
    sid = MGR.start_session(saved_memories or None)
    return dd_update(value=sid), f"Created session: {sid}"

def refresh_sessions():
    return dd_update()

def get_session_meta(session_id: str):
    if not session_id:
        return "Select a session."
    meta = MGR.get_session(session_id)
    return json.dumps(meta or {}, indent=2, ensure_ascii=False)

def close_session(session_id: str):
    if not session_id:
        return dd_update(), "Select a session."
    ok = MGR.close_session(session_id)
    return dd_update(), f"Closed: {session_id} -> {ok}"

def delete_session(session_id: str):
    if not session_id:
        return dd_update(), "Select a session."
    ok = MGR.delete_session(session_id)
    return dd_update(), f"Deleted: {session_id} -> {ok}"

def cleanup_sessions(days_old: float):
    try:
        days = int(days_old)
    except Exception:
        return "days_old must be an integer."
    if days <= 0:
        return "days_old must be positive."
    n = MGR.cleanup_old_sessions(days_old=days)
    return f"Cleaned up {n} session(s) older than {days} day(s)."

# ---------- Message ops (echo, no orchestrator) ----------
def append_user(session_id: str, content: str):
    if not session_id:
        return "Select a session."
    if not content or not content.strip():
        return "Provide user content."
    MGR.messages.append_user(
        session_id,
        content=content,
        translated=content,  # echo
        flags={"language": "english", "doctor": False, "psychologist": False},
        tokens_in=len(content.split()),
    )
    return "User message appended."

def append_assistant(session_id: str, content: str):
    if not session_id:
        return "Select a session."
    if not content or not content.strip():
        return "Provide assistant content."
    MGR.messages.append_assistant(
        session_id,
        content=content,
        translated=content,
        reasoning={"final_output": content, "note": "echo_sample"},
        tokens_out=len(content.split()),
        latency_ms=0,
    )
    return "Assistant message appended."

# ---------- Summaries ----------
def add_summary(session_id: str, text: str):
    if not session_id:
        return "Select a session."
    if not text or not text.strip():
        return "Provide summary text."
    MGR.summaries.add(session_id, text)
    return "Summary added."

def list_summaries(session_id: str):
    if not session_id:
        return "Select a session."
    lst = MGR.summaries.list(session_id) or []
    return json.dumps(lst, indent=2, ensure_ascii=False)

# ---------- Reads / Stats / Export ----------
def get_history(session_id: str, limit: float):
    if not session_id:
        return "Select a session."
    try:
        n = max(1, min(500, int(limit)))
    except Exception:
        n = 50
    rows = MGR.get_original_chat_history(session_id, limit=n)
    return json.dumps(rows or [], indent=2, ensure_ascii=False)

def get_stats(session_id: str):
    if not session_id:
        return "Select a session."
    payload = MGR.get_session_stats(session_id)
    return json.dumps(payload, indent=2, ensure_ascii=False)

def export_json(session_id: str):
    if not session_id:
        return "Select a session."
    path = MGR.export_session_json(session_id)
    return f"Exported to: {path}"

with gr.Blocks(title="ChatbotManager SQLite Smoketest") as demo:
    gr.Markdown("## ChatbotManager SQLite Smoketest (no agents, echo only)")

    with gr.Tab("Sessions"):
        with gr.Row():
            saved_mem = gr.Textbox(label="Saved memories (optional)", placeholder="e.g., prefers Thai; risk low")
            create_btn = gr.Button("Create session")
        with gr.Row():
            session_dd = gr.Dropdown(choices=_choices(), value=None, label="Select session", interactive=True)
            refresh_btn = gr.Button("Refresh list")
        with gr.Row():
            meta_btn = gr.Button("Get session meta")
            close_btn = gr.Button("Close session")
            delete_btn = gr.Button("Delete session")
        with gr.Row():
            days_old = gr.Number(value=30, precision=0, label="Cleanup: days_old")
            cleanup_btn = gr.Button("Run cleanup")
        session_status = gr.Markdown()
        session_meta = gr.Code(label="Session meta (JSON)", language="json")

        create_btn.click(create_session, [saved_mem], [session_dd, session_status])
        refresh_btn.click(refresh_sessions, None, [session_dd])
        meta_btn.click(get_session_meta, [session_dd], [session_meta])
        close_btn.click(close_session, [session_dd], [session_dd, session_status])
        delete_btn.click(delete_session, [session_dd], [session_dd, session_status])
        cleanup_btn.click(cleanup_sessions, [days_old], [session_status])

    with gr.Tab("Messages"):
        with gr.Row():
            session_dd2 = gr.Dropdown(choices=_choices(), value=None, label="Select session", interactive=True)
            sync_btn2 = gr.Button("Sync sessions")
        with gr.Row():
            user_txt = gr.Textbox(label="User content")
            user_add = gr.Button("Append user")
        with gr.Row():
            asst_txt = gr.Textbox(label="Assistant content (echo sample)")
            asst_add = gr.Button("Append assistant")
        msg_status = gr.Markdown()

        sync_btn2.click(refresh_sessions, None, [session_dd2])
        user_add.click(append_user, [session_dd2, user_txt], [msg_status])
        asst_add.click(append_assistant, [session_dd2, asst_txt], [msg_status])

    with gr.Tab("Summaries"):
        with gr.Row():
            session_dd3 = gr.Dropdown(choices=_choices(), value=None, label="Select session", interactive=True)
            sync_btn3 = gr.Button("Sync sessions")
        with gr.Row():
            sum_txt = gr.Textbox(label="Summary text")
            sum_add = gr.Button("Add summary")
            sum_list_btn = gr.Button("List summaries")
        sum_status = gr.Markdown()
        sum_list = gr.Code(label="Summaries (JSON)", language="json")

        sync_btn3.click(refresh_sessions, None, [session_dd3])
        sum_add.click(add_summary, [session_dd3, sum_txt], [sum_status])
        sum_list_btn.click(list_summaries, [session_dd3], [sum_list])

    with gr.Tab("History / Stats / Export"):
        with gr.Row():
            session_dd4 = gr.Dropdown(choices=_choices(), value=None, label="Select session", interactive=True)
            sync_btn4 = gr.Button("Sync sessions")
        with gr.Row():
            limit_num = gr.Number(value=50, precision=0, label="History limit (1-500)")
            hist_btn = gr.Button("Get original history")
            stats_btn = gr.Button("Get session stats")
            export_btn = gr.Button("Export session JSON")
        hist_out = gr.Code(label="Original history (JSON)", language="json")
        stats_out = gr.Code(label="Stats payload (JSON)", language="json")
        export_status = gr.Markdown()

        sync_btn4.click(refresh_sessions, None, [session_dd4])
        hist_btn.click(get_history, [session_dd4, limit_num], [hist_out])
        stats_btn.click(get_stats, [session_dd4], [stats_out])
        export_btn.click(export_json, [session_dd4], [export_status])

if __name__ == "__main__":
    demo.launch(share=False, inbrowser=True)
