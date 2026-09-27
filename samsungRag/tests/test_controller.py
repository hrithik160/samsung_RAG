import time

import pytest

from app.retrieval_controller.controller import RetrievalController
from app.retrieval_controller.adapters.decomposer_adapter import MockDecomposerAdapter
from app.shared.models import StreamingInput


def make_input(session="S001", turn="T001", transcript="", is_final=False):
    return StreamingInput(
        session_id=session,
        turn_id=turn,
        transcript=transcript,
        is_final=is_final,
        timestamp=1726800000.0,
        language="en",
        metadata={},
    )


def test_example1_incomplete_intent_wait():
    c = RetrievalController()
    d = c.decide(make_input(transcript="I need to..."))
    assert d.decision.value == "WAIT"
    assert "INCOMPLETE_INTENT" in d.reason_codes


def test_example2_stable_retrieve():
    c = RetrievalController()
    d = c.decide(make_input(transcript="Find workshop venues in Pune"))
    assert d.decision.value == "RETRIEVE"
    assert "INTENT_STABLE" in d.reason_codes
    assert d.trigger.value == "PROVISIONAL"


def test_example3_presentation_only():
    c = RetrievalController()
    d = c.decide(make_input(transcript="Put the previous answer in bullet points"))
    assert d.decision.value == "NO_RETRIEVAL"
    assert "PRESENTATION_ONLY" in d.reason_codes


def test_example4_new_constraint():
    c = RetrievalController()
    c.decide(make_input(turn="T001", transcript="Find workshop venues in Pune"))
    d = c.decide(make_input(turn="T002", transcript="Find workshop venues in Pune for 30 people"))
    assert d.decision.value == "RETRIEVE"
    assert "NEW_CONSTRAINT" in d.reason_codes
    assert d.trigger.value == "NEW_CONSTRAINT"


def test_example5_query_modification():
    c = RetrievalController()
    c.decide(make_input(turn="T001", transcript="Explain the travel reimbursement policy"))
    d = c.decide(make_input(turn="T002", transcript="Explain the international travel reimbursement policy"))
    assert d.decision.value == "RETRIEVE"
    assert d.reason_codes == ["NEW_CONSTRAINT"] or d.reason_codes == ["QUERY_CHANGE"]


def test_example6_acknowledgement():
    c = RetrievalController()
    d = c.decide(make_input(transcript="Okay, thanks"))
    assert d.decision.value == "NO_RETRIEVAL"
    assert "CONVERSATIONAL_RESPONSE" in d.reason_codes


def test_example7_repeated_query_suppressed():
    c = RetrievalController()
    c.decide(make_input(turn="T001", transcript="Find workshop venues in Pune"))
    d = c.decide(make_input(turn="T002", transcript="Find workshop venues in Pune"))
    assert d.decision.value == "NO_RETRIEVAL"
    assert "QUERY_UNCHANGED" in d.reason_codes or "ALREADY_RETRIEVED" in d.reason_codes


def test_empty_input_wait():
    c = RetrievalController()
    d = c.decide(make_input(transcript="   "))
    assert d.decision.value == "WAIT"
    assert "EMPTY_INPUT" in d.reason_codes


def test_provisional_vs_final_trigger():
    c = RetrievalController()
    prov = c.decide(make_input(session="S10", transcript="Find workshop venues in Pune", is_final=False))
    assert prov.trigger.value == "PROVISIONAL"
    c2 = RetrievalController()
    final = c2.decide(make_input(session="S11", transcript="Find workshop venues in Pune", is_final=True))
    assert final.trigger.value == "FINAL"


def test_session_isolation():
    c = RetrievalController()
    c.decide(make_input(session="SA", turn="T001", transcript="Find workshop venues in Pune"))
    # Different session, same query -> first-query RETRIEVE, not QUERY_UNCHANGED.
    d = c.decide(make_input(session="SB", turn="T001", transcript="Find workshop venues in Pune"))
    assert d.decision.value == "RETRIEVE"
    assert "INTENT_STABLE" in d.reason_codes


def test_state_version_increments():
    c = RetrievalController()
    d1 = c.decide(make_input(turn="T001", transcript="I need to..."))
    d2 = c.decide(make_input(turn="T002", transcript="Find workshop venues in Pune"))
    assert d2.state_version == d1.state_version + 1 == 2


def test_mock_decomposer_integration():
    c = RetrievalController(decomposer=MockDecomposerAdapter())
    d = c.decide(make_input(transcript="Find workshop venues in Pune"))
    assert d.decision.value == "RETRIEVE"
    preview = c.preview_decomposition(d)
    assert preview["subqueries"] == ["Find workshop venues in Pune"]
    # NO_RETRIEVAL decisions yield no subqueries
    d2 = c.decide(make_input(turn="T002", transcript="Okay, thanks"))
    assert c.preview_decomposition(d2) == {"subqueries": [], "intents": []}


def test_output_contract_fields():
    c = RetrievalController()
    d = c.decide(make_input(transcript="Find workshop venues in Pune"))
    assert d.schema_version == "1.0"
    assert d.session_id == "S001" and d.turn_id == "T001"
    assert 0.0 <= d.confidence <= 1.0
    assert d.state_version >= 1
    assert isinstance(d.timestamp, float)
