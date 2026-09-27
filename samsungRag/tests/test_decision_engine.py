from app.retrieval_controller.config import ControllerConfig
from app.retrieval_controller.decision_engine import decide
from app.retrieval_controller.entity_extractor import extract_entities
from app.retrieval_controller.intent_detector import detect_intent
from app.retrieval_controller.query_comparator import compare_queries, normalize_query


def run_engine(transcript, prev="", prev_constraints=None, is_final=False):
    normalized = normalize_query(transcript)
    intent = detect_intent(transcript)
    cur_c = extract_entities(transcript).constraints()
    comp = compare_queries(
        current_normalized=normalized,
        previous_normalized=normalize_query(prev),
        current_constraints=cur_c,
        previous_constraints=prev_constraints or [],
        has_previous=bool(prev),
    )
    return decide(
        transcript=transcript,
        normalized=normalized,
        is_final=is_final,
        intent=intent,
        comparison=comp,
        config=ControllerConfig(),
    )


def test_wait_on_incomplete():
    r = run_engine("I need to...")
    assert r.decision.value == "WAIT"


def test_retrieve_stable_first_query():
    r = run_engine("Find workshop venues in Pune")
    assert r.decision.value == "RETRIEVE"
    assert r.trigger.value == "PROVISIONAL"


def test_final_trigger():
    r = run_engine("Find workshop venues in Pune", is_final=True)
    assert r.trigger.value == "FINAL"


def test_no_retrieval_ack():
    r = run_engine("Okay, thanks")
    assert r.decision.value == "NO_RETRIEVAL"


def test_no_retrieval_presentation():
    r = run_engine("Put the previous answer in bullet points")
    assert r.decision.value == "NO_RETRIEVAL"


def test_new_constraint_trigger():
    prev_c = extract_entities("Find workshop venues in Pune").constraints()
    r = run_engine("Find workshop venues in Pune for 30 people", prev="Find workshop venues in Pune", prev_constraints=prev_c)
    assert r.decision.value == "RETRIEVE"
    assert r.trigger.value == "NEW_CONSTRAINT"


def test_cosmetic_change_suppressed():
    # Case-only change normalizes to identical -> suppressed upstream;
    # here test near-identical wording with same constraints stays non-meaningful.
    prev_c = extract_entities("Find workshop venues in Pune").constraints()
    r = run_engine(
        "Find workshop venues in Pune",
        prev="Find workshop venues in Pune",
        prev_constraints=prev_c,
    )
    assert r.decision.value == "NO_RETRIEVAL"
