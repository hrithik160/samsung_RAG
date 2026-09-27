"""Run with:  python test_component3.py   (or: pytest -q)
Uses offline stand-in models, so no downloads are needed."""
import asyncio
import tempfile

from fusion import rrf
from index import HybridIndex, chunk_document
from models import HashingEmbedder, TokenOverlapReranker
from retriever import CorpusRetriever, RetrieverConfig
from schema import SubQuery

DOCS = {
    "visa": "Tourist visa applications for Japan require a valid passport, a completed form and proof of funds. "
            "Processing usually takes five business days.",
    "baggage": "Checked baggage allowance on international flights is two bags of 23 kilograms each. "
               "Overweight baggage incurs a fee per kilogram.",
    "hotel": "Hotel cancellation policy: free cancellation up to 48 hours before check in. "
             "Late cancellation is charged one night.",
    "refund": "Refunds for cancelled flights are issued to the original payment method within seven days.",
    "insurance": "Travel insurance covers medical emergencies, trip cancellation and lost baggage.",
    "transit": "Airport transit passengers do not need a visa when staying under twenty four hours airside.",
    "loyalty": "Loyalty members earn double points on hotel stays booked through the partner portal.",
}


def make_retriever(**cfg):
    chunks = [c for d, t in DOCS.items() for c in chunk_document(d, t)]
    idx = HybridIndex(HashingEmbedder()).build(chunks)
    rr = TokenOverlapReranker()
    return CorpusRetriever(idx, rr, RetrieverConfig(final_k=4, **cfg)), rr


def test_rrf_math():
    out = rrf([["a", "b", "c"], ["b", "c", "d"]], k=60)
    assert out[0][0] == "b"
    assert abs(out[0][1] - (1 / 62 + 1 / 61)) < 1e-12
    assert [x for x, _ in rrf([["a"], ["b"]])] == ["a", "b"]  # deterministic tie-break


def test_single_query_finds_right_doc():
    r, _ = make_retriever()
    res = asyncio.run(r.retrieve("what is the hotel cancellation policy"))
    assert res.hits[0].doc_id == "hotel"


def test_multi_intent_coverage():
    r, _ = make_retriever(min_per_subquery=1)
    res = asyncio.run(r.retrieve([SubQuery(text="visa requirements passport Japan"),
                                  SubQuery(text="checked baggage weight allowance")]))
    docs = [h.doc_id for h in res.hits]
    assert "visa" in docs and "baggage" in docs
    assert len(set(h.chunk_id for h in res.hits)) == len(res.hits)  # no duplicates


def test_modes_and_hybrid_metadata():
    for mode in ("hybrid", "dense", "sparse"):
        r, _ = make_retriever(mode=mode)
        assert asyncio.run(r.retrieve("refund cancelled flight")).hits[0].doc_id == "refund"
    r, _ = make_retriever(mode="hybrid")
    h = asyncio.run(r.retrieve("refund cancelled flight")).hits[0]
    assert h.dense_rank is not None and h.sparse_rank is not None


def test_cache_avoids_rerank_calls():
    r, rr = make_retriever()
    asyncio.run(r.retrieve("visa passport"))
    calls = rr.calls
    res = asyncio.run(r.retrieve("visa passport"))
    assert rr.calls == calls and res.cache_hits == 1


def test_overlap_for_reflector():
    r, _ = make_retriever()
    early = asyncio.run(r.retrieve("hotel cancellation"))
    final = asyncio.run(r.retrieve("hotel cancellation policy 48 hours"))
    assert 0.0 <= early.overlap(final, k=3) <= 1.0
    assert early.overlap(early, k=3) == 1.0


def test_max_per_doc():
    long_doc = " ".join(f"baggage rule number {i} about weight" for i in range(60))
    chunks = chunk_document("big", long_doc, max_words=20, overlap=0) + \
        [c for d, t in DOCS.items() for c in chunk_document(d, t)]
    idx = HybridIndex(HashingEmbedder()).build(chunks)
    r = CorpusRetriever(idx, TokenOverlapReranker(), RetrieverConfig(final_k=5, max_per_doc=2))
    res = asyncio.run(r.retrieve("baggage weight rule"))
    assert sum(h.doc_id == "big" for h in res.hits) <= 2


def test_save_load_roundtrip():
    chunks = [c for d, t in DOCS.items() for c in chunk_document(d, t)]
    idx = HybridIndex(HashingEmbedder()).build(chunks)
    with tempfile.TemporaryDirectory() as d:
        idx.save(d)
        idx2 = HybridIndex.load(d, HashingEmbedder())
    assert [c.chunk_id for c in idx2.chunks] == [c.chunk_id for c in idx.chunks]
    assert idx2.dense_search("visa", 3) == idx.dense_search("visa", 3)
    assert idx2.sparse_search("visa", 3) == idx.sparse_search("visa", 3)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok  ", name)
