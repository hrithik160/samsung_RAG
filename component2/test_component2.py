"""Run with:  python test_component2.py   (or: pytest -q)
Offline only -- no LLM/API calls needed."""
from decomposer import (
    ComplexityGate, RuleBasedDecomposer, LLMDecomposer, MultiIntentDecomposer,
    classify_intent, resolve_references, validate_subqueries,
)
from schema import SubQueryItem, DecompositionResult


def test_gate_lets_simple_queries_through():
    gate = ComplexityGate()
    assert gate.is_complex("What is the battery capacity of the S24?") is False
    assert gate.is_complex("mac and cheese recipe") is False  # "and" inside a noun phrase


def test_gate_catches_comparatives_and_conjunctions():
    gate = ComplexityGate()
    assert gate.is_complex("S24 vs S24 Ultra battery") is True
    assert gate.is_complex("difference between S24 and S24 Ultra") is True
    assert gate.is_complex("how fast does it charge and is it waterproof") is True
    assert gate.is_complex("what is the screen size? is it AMOLED?") is True


def test_intent_classification():
    assert classify_intent("What is the battery capacity?") == "definition"
    assert classify_intent("S24 vs S24 Ultra camera") == "comparison"
    assert classify_intent("How many mAh is the battery?") == "numerical"
    assert classify_intent("What about in Pune?") == "followup"
    assert classify_intent("Cancellation policy for hotels") == "factual"


def test_reference_resolution():
    assert resolve_references("it charges fast", "S24 Ultra") == "S24 Ultra charges fast"
    assert resolve_references("what about in Pune?", "workshop venue") == "workshop venue in Pune?"
    assert resolve_references("battery life", None) == "battery life"


def test_rule_backend_splits_compound_query():
    items = RuleBasedDecomposer().split("S24 battery capacity vs S24 Ultra battery capacity")
    assert len(items) == 2
    assert all(isinstance(i, SubQueryItem) for i in items)


def test_rule_backend_single_clause_passthrough():
    items = RuleBasedDecomposer().split("What is the battery capacity?")
    assert len(items) == 1


def test_validate_subqueries_rejects_empty_and_lossy():
    assert validate_subqueries("battery capacity of S24", []) is False
    good = [SubQueryItem(sub_query="battery capacity of the S24")]
    assert validate_subqueries("battery capacity of S24", good) is True
    lossy = [SubQueryItem(sub_query="hello there")]
    assert validate_subqueries("battery capacity of S24", lossy) is False


def test_llm_decomposer_parses_valid_json():
    def fake_llm(prompt):
        return (
            '[{"sub_query": "S24 camera specs", "intent_type": "factual", '
            '"depends_on": null, "priority": 1}, '
            '{"sub_query": "S24 Ultra camera specs", "intent_type": "factual", '
            '"depends_on": null, "priority": 1}]'
        )
    items = LLMDecomposer(fake_llm).split("S24 vs S24 Ultra camera")
    assert len(items) == 2 and items[0].sub_query == "S24 camera specs"


def test_llm_decomposer_retries_then_raises_on_malformed_json():
    calls = {"n": 0}

    def bad_llm(prompt):
        calls["n"] += 1
        return "not json at all"

    try:
        LLMDecomposer(bad_llm).split("S24 vs S24 Ultra camera")
        assert False, "expected ValueError"
    except ValueError:
        pass
    assert calls["n"] == 2  # initial + one retry


def test_llm_decomposer_strips_markdown_fences():
    def fenced_llm(prompt):
        return '```json\n[{"sub_query": "battery life", "intent_type": "factual", "depends_on": null, "priority": 1}]\n```'
    items = LLMDecomposer(fenced_llm).split("battery life")
    assert len(items) == 1 and items[0].sub_query == "battery life"


def test_facade_falls_back_to_rules_when_llm_fails():
    def broken_llm(prompt):
        raise RuntimeError("API down")

    decomposer = MultiIntentDecomposer(llm_backend=LLMDecomposer(broken_llm))
    res = decomposer.decompose("S24 vs S24 Ultra battery", session_id="s1", turn_id="T1")
    assert res.gate_triggered is True
    assert res.used_llm is False
    assert res.fallback_used is True
    assert len(res.sub_queries) >= 1


def test_facade_uses_llm_when_valid():
    def good_llm(prompt):
        return (
            '[{"sub_query": "S24 battery capacity", "intent_type": "factual", "depends_on": null, "priority": 1}, '
            '{"sub_query": "S24 Ultra battery capacity", "intent_type": "factual", "depends_on": null, "priority": 1}]'
        )
    decomposer = MultiIntentDecomposer(llm_backend=LLMDecomposer(good_llm))
    res = decomposer.decompose("S24 vs S24 Ultra battery capacity", session_id="s2", turn_id="T1")
    assert res.used_llm is True and res.fallback_used is False
    assert len(res.sub_queries) == 2


def test_facade_resolves_followup_using_session_memory():
    decomposer = MultiIntentDecomposer()
    decomposer.decompose("What is the S24 Ultra battery capacity?", session_id="s3", turn_id="T1")
    res = decomposer.decompose("what about in Pune?", session_id="s3", turn_id="T2")
    assert "Pune" in res.sub_queries[0].sub_query
    assert res.gate_triggered is False


def test_to_component3_subquery_kwargs_shape():
    item = SubQueryItem(sub_query="battery life", intent_type="factual", priority=2)
    kwargs = item.to_subquery_kwargs(qid="T1.0")
    assert kwargs["text"] == "battery life"
    assert kwargs["qid"] == "T1.0"
    assert 0 < kwargs["weight"] <= 1.0


def test_decomposition_result_to_event_is_json_safe():
    import json
    res = DecompositionResult(
        original_query="q", session_id="s", turn_id="T1",
        sub_queries=[SubQueryItem(sub_query="q")],
    )
    json.dumps(res.to_event())  # must not raise


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
