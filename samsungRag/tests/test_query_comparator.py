from app.retrieval_controller.query_comparator import (
    compare_queries,
    jaccard,
    normalize_query,
)


def test_normalize():
    assert normalize_query("  Find Venues, in PUNE! ") == "find venues in pune"


def test_identical():
    r = compare_queries("find venues", "find venues", [], [], True)
    assert r.is_identical and not r.is_meaningful_change


def test_new_constraint_detection():
    r = compare_queries(
        "find venues in pune for 30 people",
        "find venues in pune",
        ["location:pune", "quantity:for 30 people"],
        ["location:pune"],
        True,
    )
    assert r.new_constraints == ["quantity:for 30 people"]
    assert r.is_meaningful_change


def test_cosmetic_vs_meaningful():
    assert jaccard("find venues in pune", "find venues in pune") == 1.0
    r = compare_queries("find venues in pune", "explain travel policy", [], [], True)
    assert r.is_meaningful_change


def test_first_query():
    r = compare_queries("find venues", "", ["location:pune"], [], False)
    assert r.is_first_query and r.is_meaningful_change
