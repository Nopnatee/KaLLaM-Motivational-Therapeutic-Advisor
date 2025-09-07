# scripts/smoke_manager.py
from kallam.app.chatbot_manager import ChatbotManager
from tests.fakes import FakeOrchestrator

mgr = ChatbotManager(db_path="data/smoke.db", summarize_every_n_messages=2)
mgr.orchestrator = FakeOrchestrator()

sid = mgr.start_session()
print("SID:", sid)
print("BOT:", mgr.handle_message(sid, "hello"))
print("BOT:", mgr.handle_message(sid, "TH: สวัสดี"))
print("STATS:", mgr.get_session_stats(sid))
print("EXPORT:", mgr.export_session_json(sid))
