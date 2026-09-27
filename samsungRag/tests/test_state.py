from app.retrieval_controller.state import InMemorySessionStore, SessionState


def test_save_and_get_isolated():
    store = InMemorySessionStore()
    store.save(SessionState(session_id="S1", normalized_query="hello"))
    store.save(SessionState(session_id="S2", normalized_query="world"))
    assert store.get("S1").normalized_query == "hello"
    assert store.get("S2").normalized_query == "world"
    assert store.get("MISSING") is None


def test_no_cross_session_leak():
    store = InMemorySessionStore()
    s = SessionState(session_id="SA", last_retrieval_query="venues in pune")
    store.save(s)
    other = store.get("SB")
    assert other is None
