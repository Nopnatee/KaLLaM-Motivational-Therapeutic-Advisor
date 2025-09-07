from kallam.infra.session_store import SessionStore
from kallam.infra.message_store import MessageStore
from kallam.infra.summary_store import SummaryStore

def test_stores_roundtrip(db_path):
    url = f"sqlite:///{db_path}"
    sessions = SessionStore(url)
    messages = MessageStore(url)
    summaries = SummaryStore(url)

    sid = sessions.create(saved_memories=None)
    assert sessions.get_raw(sid)["session_id"] == sid

    messages.append_user(sid, content="u1", translated="u1", flags={"translate": False}, tokens_in=1)
    messages.append_assistant(sid, content="a1", translated="a1", reasoning={"r": 1}, tokens_out=1)

    hist = messages.get_translated_history(sid, limit=10)
    assert len(hist) == 2 and hist[0]["role"] == "user" and hist[1]["role"] == "assistant"

    summaries.add(sid, "S1")
    lst = summaries.list(sid)
    assert lst and lst[0]["summary"] == "S1"
