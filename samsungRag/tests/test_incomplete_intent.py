"""Required incomplete-intent regression tests (bug: 'I need a' -> RETRIEVE).

Covers the 9 mandated cases: WAIT x5, RETRIEVE x2, NO_RETRIEVAL x1,
and Pune->Mumbai new-constraint session flow. Each case uses an
isolated session_id per the spec.
"""

from fastapi.testclient import TestClient

from app.main import app
from app.retrieval_controller.controller import RetrievalController
from app.shared.models import StreamingInput

client = TestClient(app)


def _inp(session, turn, transcript, is_final):
    return StreamingInput(
        session_id=session,
        turn_id=turn,
        transcript=transcript,
        is_final=is_final,
        timestamp=1726800000.0,
        language="en",
        metadata={},
    )


def _decide(session, turn, transcript, is_final):
    return RetrievalController().decide(_inp(session, turn, transcript, is_final))


def test_case1_single_I_wait():
    d = _decide("TEST_WAIT_01", "T001", "I", False)
    assert d.decision.value == "WAIT"
    assert "INCOMPLETE_INTENT" in d.reason_codes


def test_case2_I_need_wait():
    d = _decide("TEST_WAIT_02", "T001", "I need", False)
    assert d.decision.value == "WAIT"
    assert "INCOMPLETE_INTENT" in d.reason_codes


def test_case3_I_need_a_wait_main_bug():
    d = _decide("TEST_WAIT_03", "T001", "I need a", False)
    assert d.decision.value == "WAIT"
    assert "INCOMPLETE_INTENT" in d.reason_codes
    assert d.trigger.value == "NONE"
    assert d.is_final is False
    assert d.query == "I need a"


def test_case4_help_me_with_wait():
    d = _decide("TEST_WAIT_04", "T001", "Can you help me with", False)
    assert d.decision.value == "WAIT"
    assert "INCOMPLETE_INTENT" in d.reason_codes


def test_case5_want_to_know_about_wait():
    d = _decide("TEST_WAIT_05", "T001", "I want to know about", False)
    assert d.decision.value == "WAIT"
    assert "INCOMPLETE_INTENT" in d.reason_codes


def test_case6_complete_request_retrieves():
    d = _decide("TEST_RETRIEVE_01", "T001", "I want to know about workshop venues in Pune", False)
    assert d.decision.value == "RETRIEVE"


def test_case7_final_complete_retrieves():
    d = _decide("TEST_RETRIEVE_02", "T001", "Find workshop venues in Pune", True)
    assert d.decision.value == "RETRIEVE"


def test_case8_presentation_no_retrieval():
    d = _decide("TEST_NO_RETRIEVAL_01", "T001", "Put the previous answer in bullet points", True)
    assert d.decision.value == "NO_RETRIEVAL"


def test_case9_pune_to_mumbai_new_constraint():
    c = RetrievalController()
    c.decide(_inp("TEST_CONSTRAINT_01", "T001", "Find workshop venues in Pune", True))
    d2 = c.decide(_inp("TEST_CONSTRAINT_01", "T002", "Find workshop venues in Mumbai", True))
    assert d2.decision.value == "RETRIEVE"
    assert d2.trigger.value == "NEW_CONSTRAINT"
    assert "NEW_CONSTRAINT" in d2.reason_codes
    assert d2.state_version == 2


def test_case3_via_api():
    resp = client.post(
        "/api/v1/retrieval-controller/decide",
        json={
            "session_id": "TEST_WAIT_03",
            "turn_id": "T001",
            "transcript": "I need a",
            "is_final": False,
            "timestamp": 1726800000.0,
            "language": "en",
            "metadata": {},
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["decision"] == "WAIT"
    assert "INCOMPLETE_INTENT" in body["reason_codes"]
    assert body["trigger"] == "NONE"
    assert body["query"] == "I need a"
