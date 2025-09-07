import json
from kallam.app.chatbot_manager import ChatbotManager
from pathlib import Path
from tests.fakes import FakeOrchestrator

def make_manager(db_path: str, n: int = 2):
    mgr = ChatbotManager(db_path=db_path, summarize_every_n_messages=n, message_limit=20)
    # inject fake orchestrator to avoid APIs
    mgr.orchestrator = FakeOrchestrator()
    return mgr

def test_session_lifecycle(db_path):
    mgr = make_manager(db_path)
    sid = mgr.start_session(saved_memories='{"k":"v"}')
    s = mgr.get_session(sid)
    assert s is not None and s["session_id"] == sid
    sessions = mgr.list_sessions(active_only=True)
    assert any(x["session_id"] == sid for x in sessions)
    assert mgr.close_session(sid) is True
    assert mgr.delete_session(sid) is True

def test_handle_message_and_summary_trigger(db_path):
    mgr = make_manager(db_path, n=2)  # summarize every 2 user msgs
    sid = mgr.start_session()
    # 1st message
    out1 = mgr.handle_message(sid, "hello")
    assert out1.startswith("BOT:")
    # 2nd message should trigger a summary
    out2 = mgr.handle_message(sid, "TH: สวัสดี")  # also exercises translate flag
    assert out2.startswith("BOT:")

    # stats shape
    stats = mgr.get_session_stats(sid)
    assert stats["stats"].message_count >= 2
    # ensure summaries table updated via manager call
    summary_text = mgr.summarize_session(sid)
    assert summary_text.startswith("SUMMARY(")

def test_export_json_schema(db_path, tmp_path):
    mgr = make_manager(db_path)
    sid = mgr.start_session()
    mgr.handle_message(sid, "hello")
    path = mgr.export_session_json(sid)
    # file exists and minimal schema checks
    p = tmp_path / path if not path.startswith(str(tmp_path)) else Path(path)
    assert Path(path).exists()
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    assert "session_info" in data and "chat_history" in data and "summaries" in data and "export_metadata" in data
