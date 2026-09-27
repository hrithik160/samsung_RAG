"""Run with:  python test_adapter.py"""
from adapter import RealDecomposerAdapter, DecomposerResponse


class _FakeRequest:
    def __init__(self, query, session_id="s1", turn_id="T1"):
        self.query = query
        self.session_id = session_id
        self.turn_id = turn_id


def test_adapter_wraps_simple_query():
    adapter = RealDecomposerAdapter()
    resp = adapter.decompose(_FakeRequest("What is the battery capacity?"))
    assert isinstance(resp, DecomposerResponse)
    assert resp.subqueries == ["What is the battery capacity?"]
    assert resp.intents == ["definition"]


def test_adapter_splits_compound_query():
    adapter = RealDecomposerAdapter()
    resp = adapter.decompose(_FakeRequest("S24 vs S24 Ultra battery capacity"))
    assert len(resp.subqueries) == 2
    assert len(resp.intents) == 2


def test_adapter_empty_query_returns_empty():
    adapter = RealDecomposerAdapter()
    resp = adapter.decompose(_FakeRequest(""))
    assert resp.subqueries == [] and resp.intents == []


def test_adapter_uses_llm_backend_when_given():
    def fake_llm(prompt):
        return (
            '[{"sub_query": "S24 battery capacity", "intent_type": "factual", "depends_on": null, "priority": 1}, '
            '{"sub_query": "S24 Ultra battery capacity", "intent_type": "factual", "depends_on": null, "priority": 1}]'
        )
    adapter = RealDecomposerAdapter(llm_call=fake_llm)
    resp = adapter.decompose(_FakeRequest("S24 vs S24 Ultra battery capacity"))
    assert resp.subqueries == ["S24 battery capacity", "S24 Ultra battery capacity"]


def test_decompose_full_exposes_rich_result():
    adapter = RealDecomposerAdapter()
    result = adapter.decompose_full("S24 vs S24 Ultra battery capacity", session_id="s1", turn_id="T1")
    assert hasattr(result, "sub_queries")
    assert result.sub_queries[0].intent_type in {"factual", "comparison", "definition", "numerical", "followup"}


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
