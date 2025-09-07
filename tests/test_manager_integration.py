from kallam.app.chatbot_manager import ChatbotManager
from tests.fakes import FakeOrchestrator

def test_end_to_end(db_path, tmp_path):
    mgr = ChatbotManager(db_path=db_path, summarize_every_n_messages=3, message_limit=10)
    mgr.orchestrator = FakeOrchestrator()  # still avoid APIs

    sid = mgr.start_session()
    for i in range(5):
        out = mgr.handle_message(sid, f"msg-{i}")
        assert out.startswith("BOT:")

    # export and verify non-empty
    export_path = mgr.export_session_json(sid)
    assert export_path and len(open(export_path, "r", encoding="utf-8").read()) > 10
