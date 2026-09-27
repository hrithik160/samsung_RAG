"""Run with:  python test_component4.py   (or: pytest -q)
Offline only -- uses a fake retriever, no real Component 3 index needed."""
import asyncio
from dataclasses import dataclass, field

from schema import CachedChunk, Claim, SessionState
from session_synthesis import (
    SessionStore, TurnClassifier, OfflineSynthesizer,
    validate_and_strip_citations, SessionAwareSynthesizer,
)


@dataclass
class _Hit:
    chunk_id: str
    doc_id: str
    text: str
    meta: dict = field(default_factory=dict)


@dataclass
class _Result:
    hits: list


class _FakeRetriever:
    def __init__(self, hits_by_keyword):
        self.hits_by_keyword = hits_by_keyword
        self.calls = []

    async def retrieve(self, query):
        self.calls.append(query)
        for kw, hits in self.hits_by_keyword.items():
            if kw.lower() in query.lower():
                return _Result(hits=hits)
        return _Result(hits=[])


def test_session_store_creates_and_persists():
    store = SessionStore()
    s = store.get("s1")
    assert s.session_id == "s1" and s.answer_version == 0
    s.answer_version = 3
    store.save(s)
    assert store.get("s1").answer_version == 3


def test_classifier_detects_cosmetic_change():
    clf = TurnClassifier()
    session = SessionState(session_id="s1", last_answer="Some prior answer.", last_query="battery capacity")
    is_cosmetic, constraint = clf.classify("put that in bullet points", session)
    assert is_cosmetic is True and constraint is None


def test_classifier_first_turn_is_not_cosmetic():
    clf = TurnClassifier()
    session = SessionState(session_id="s1")
    is_cosmetic, constraint = clf.classify("what is the battery capacity?", session)
    assert is_cosmetic is False
    assert constraint == "what is the battery capacity?"


def test_classifier_detects_new_constraint_on_refinement():
    clf = TurnClassifier()
    session = SessionState(session_id="s1", last_query="travel reimbursement policy")
    is_cosmetic, constraint = clf.classify("international travel reimbursement policy", session)
    assert is_cosmetic is False
    assert "international" in constraint.lower()


def test_citation_validator_keeps_known_and_strips_unknown():
    session = SessionState(session_id="s1")
    session.cached_chunks["c1"] = CachedChunk("c1", "Doc_9", "text", "1")
    answer = "Fact A [Doc_9 §1]. Hallucinated fact [Doc_99 §5]."
    cleaned, kept, dropped = validate_and_strip_citations(answer, session)
    assert "[Doc_9 §1]" in cleaned
    assert "[Doc_99 §5]" not in cleaned
    assert kept == ["[Doc_9 §1]"]
    assert dropped == ["[Doc_99 §5]"]


def test_offline_synthesizer_cites_new_chunks():
    backend = OfflineSynthesizer()
    chunk = CachedChunk("c1", "Doc_1", "Battery is 4000mAh.", "2")
    out = backend.synthesize("battery capacity", [], [chunk])
    assert "Battery is 4000mAh." in out and "[Doc_1 §2]" in out


def test_offline_synthesizer_restyle_bullet_preserves_citation():
    backend = OfflineSynthesizer()
    prior = "Battery is 4000mAh. [Doc_1 §2]"
    out = backend.restyle("put that in bullet points", prior)
    assert out.count("- ") == 1
    assert "[Doc_1 §2]" in out


def test_full_turn_no_retrieval_when_cosmetic():
    async def run():
        synth = SessionAwareSynthesizer()
        retriever = _FakeRetriever({"battery": [_Hit("c1", "Doc_1", "Battery is 4000mAh.", {"section": "2"})]})
        t1 = await synth.process_turn("s1", "T1", "what is the battery capacity?", retriever)
        assert t1.retrieval_required is True
        t2 = await synth.process_turn("s1", "T2", "make that shorter", retriever)
        assert t2.retrieval_required is False and t2.is_cosmetic is True
        assert retriever.calls == ["what is the battery capacity?"]  # no second call
    asyncio.run(run())


def test_full_turn_delta_retrieval_on_new_constraint():
    async def run():
        synth = SessionAwareSynthesizer()
        retriever = _FakeRetriever({
            "travel": [_Hit("c1", "Doc_9", "Domestic travel is capped at $150/day.", {"section": "1"})],
            "international": [_Hit("c2", "Doc_9", "International travel needs pre-approval.", {"section": "3"})],
        })
        t1 = await synth.process_turn("s1", "T1", "travel reimbursement policy", retriever)
        assert "Doc_9" in t1.citations[0]
        t2 = await synth.process_turn("s1", "T2", "international travel reimbursement policy", retriever)
        assert t2.retrieval_required is True
        assert "pre-approval" in t2.answer
        assert "capped at $150" in t2.answer  # prior claim retained, not discarded
    asyncio.run(run())


def test_citations_survive_across_versions():
    async def run():
        synth = SessionAwareSynthesizer()
        retriever = _FakeRetriever({"x": [_Hit("c1", "Doc_5", "Fact X.", {"section": "1"})]})
        t1 = await synth.process_turn("s1", "T1", "tell me about x", retriever)
        assert t1.version == 1
        t2 = await synth.process_turn("s1", "T2", "reformat as bullet points", retriever)
        assert t2.version == 1  # cosmetic turns don't bump the version
        assert t2.citations == t1.citations
    asyncio.run(run())


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
