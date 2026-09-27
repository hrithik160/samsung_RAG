"""Tests for Phase 5 (Telemetry + Eval). Run: python component5/test_component5.py"""
import asyncio
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import run_eval as R                                   # noqa: E402

eval_data = R.eval_data
_CACHE = {}


def _records():
    if "r" not in _CACHE:
        recs = []
        for cls in R.VARIANTS:
            recs += asyncio.run(R.run_variant(cls, eval_data.SESSIONS))
        _CACHE["r"] = recs
    return _CACHE["r"]


def _df():
    import pandas as pd
    return pd.DataFrame(_records())


class _H:
    def __init__(self, doc_id):
        self.doc_id = doc_id


def test_dataset_integrity():
    for s in eval_data.SESSIONS:
        for t in s["turns"]:
            assert t["kind"] in {"fact", "refine", "cosmetic", "ack", "partial"}
            assert t["should_retrieve"] == (t["kind"] in {"fact", "refine"})
            assert bool(t["gold_docs"]) == t["should_retrieve"]
            for d in t["gold_docs"]:
                assert d in eval_data.CORPUS, d


def test_score_turn_recall_and_mrr():
    spec = {"kind": "fact", "gold_docs": ["A", "B"], "should_retrieve": True}
    out = {"retrieved": True, "hits": [_H("X"), _H("A"), _H("Y")], "answer": "t [A]"}
    sc = R.score_turn(spec, out, set(), ["A", "B", "X", "Y"])
    assert sc["recall_at_3"] == 0.5 and sc["mrr"] == 0.5
    assert not sc["task_correct"]                       # only half the gold docs found


def test_missed_retrieval_scores_zero():
    spec = {"kind": "fact", "gold_docs": ["A"], "should_retrieve": True}
    out = {"retrieved": False, "hits": [], "answer": ""}
    sc = R.score_turn(spec, out, set(), ["A"])
    assert sc["recall_at_3"] == 0.0 and sc["mrr"] == 0.0 and not sc["decision_correct"]


def test_cosmetic_needs_cache_reuse_and_preserved_citations():
    spec = {"kind": "cosmetic", "gold_docs": [], "should_retrieve": False}
    good = R.score_turn(spec, {"retrieved": False, "hits": [], "answer": "- x [A] y [B]"}, {"A", "B"}, ["A", "B"])
    empty = R.score_turn(spec, {"retrieved": False, "hits": [], "answer": ""}, {"A", "B"}, ["A", "B"])
    junk = R.score_turn(spec, {"retrieved": True, "hits": [_H("Z")], "answer": "z [Z]"}, {"A"}, ["A", "Z"])
    assert good["task_correct"] and good["cache_reused"]
    assert not empty["task_correct"] and not empty["cache_reused"]   # no memory -> nothing to reuse
    assert not junk["task_correct"]                                   # retrieved on a cosmetic ask


def test_all_variants_cover_all_turns():
    n = sum(len(s["turns"]) for s in eval_data.SESSIONS)
    df = _df()
    assert set(df.variant) == {"A_always_retrieve", "B_controller_only", "C_full_pipeline"}
    assert all(len(g) == n for _, g in df.groupby("variant"))


def test_baseline_retrieves_every_turn():
    df = _df()
    a = df[df.variant == "A_always_retrieve"]
    assert a.retrieved.all()


def test_controller_variants_retrieve_less_than_baseline():
    s = R.summarise(_df())
    assert s.loc["B_controller_only", "retrievals"] < s.loc["A_always_retrieve", "retrievals"]
    assert s.loc["C_full_pipeline", "retrievals"] <= s.loc["B_controller_only", "retrievals"]


def test_only_full_pipeline_reuses_cache():
    s = R.summarise(_df())
    assert s.loc["A_always_retrieve", "cache_reuse_rate"] == 0
    assert s.loc["B_controller_only", "cache_reuse_rate"] == 0
    assert s.loc["C_full_pipeline", "cache_reuse_rate"] > 0


def test_full_pipeline_wins_refinement_accuracy():
    s = R.summarise(_df())
    assert s.loc["C_full_pipeline", "refinement_accuracy"] > s.loc["B_controller_only", "refinement_accuracy"]
    assert s.loc["C_full_pipeline", "refinement_accuracy"] > s.loc["A_always_retrieve", "refinement_accuracy"]


def test_main_writes_log_csv_and_chart():
    R.main()
    for f in ("turn_log.jsonl", "summary.csv", "eval_summary.png"):
        p = os.path.join(R.OUT_DIR, f)
        assert os.path.getsize(p) > 0, f
    with open(os.path.join(R.OUT_DIR, "turn_log.jsonl")) as fh:
        row = json.loads(fh.readline())
    assert {"variant", "latency_ms", "decision", "recall_at_3", "cache_reused"} <= set(row)


if __name__ == "__main__":
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    for name, fn in tests:
        fn()
        print(f"PASS  {name}")
    print(f"\n{len(tests)} passed")
