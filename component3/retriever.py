"""Component 3 - Corpus Retrieval & Fusion.

Pipeline for `CorpusRetriever.retrieve(sub_queries)`:

  per sub-query (cached by query text):
      dense top-N  ┐
                   ├─ RRF ─> top `candidate_k` ─> cross-encoder rerank ─> top `keep`
      BM25 top-N   ┘
  across sub-queries:
      weighted RRF over the per-sub-query lists
      + coverage guarantee (every intent keeps >= `min_per_subquery` chunks)
      + dedup by chunk id and `max_per_doc` cap  ->  final `final_k` hits

`mode` gives you the hybrid-vs-dense-only ablation with one flag.
"""
from __future__ import annotations

import asyncio
import math
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, replace
from typing import Optional, Sequence, Union

from fusion import rrf
from index import HybridIndex
from models import Reranker
from schema import Hit, RetrievalResult, SubQuery

QueryLike = Union[str, SubQuery, Sequence[Union[str, SubQuery]]]


@dataclass
class RetrieverConfig:
    mode: str = "hybrid"          # "hybrid" | "dense" | "sparse"   (ablation switch)
    dense_k: int = 50
    sparse_k: int = 50
    rrf_k: int = 60
    dense_weight: float = 1.0
    sparse_weight: float = 1.0
    candidate_k: int = 20         # candidates sent to the reranker per sub-query
    per_query_k: int = 5          # chunks each sub-query contributes to cross-intent fusion
    final_k: int = 5              # chunks returned to synthesis
    min_per_subquery: int = 2     # coverage guarantee across intents
    max_per_doc: int = 3          # anti-redundancy
    rerank: bool = True
    cache_size: int = 512
    uncertainty_threshold: float = 0.65  # if top rerank score is below this, bypass generation


class CorpusRetriever:
    def __init__(self, index: HybridIndex, reranker: Optional[Reranker] = None,
                 config: Optional[RetrieverConfig] = None):
        self.index = index
        self.reranker = reranker
        self.cfg = config or RetrieverConfig()
        if self.cfg.mode not in ("hybrid", "dense", "sparse"):
            raise ValueError(f"unknown mode {self.cfg.mode!r}")
        self._cache: "OrderedDict[tuple, list[Hit]]" = OrderedDict()
        self._lock = threading.Lock()  # controller may fire parallel threads

    # ------------------------------------------------------------------ public API
    async def retrieve(self, query: QueryLike) -> RetrievalResult:
        t_start = time.perf_counter()
        sqs = self._coerce(query)
        timings = {"dense_ms": 0.0, "sparse_ms": 0.0, "rerank_ms": 0.0}
        
        # Dispatch simultaneous queries
        tasks = [self._retrieve_one(sq.query_text(), timings) for sq in sqs]
        results = await asyncio.gather(*tasks)

        cache_hits = 0
        per_sq: dict[str, list[Hit]] = {}
        for sq, (hits, hit_cache) in zip(sqs, results):
            cache_hits += hit_cache
            per_sq[sq.qid] = hits

        t0 = time.perf_counter()
        final = self._fuse(sqs, per_sq)
        timings["fusion_ms"] = (time.perf_counter() - t0) * 1000
        timings["total_ms"] = (time.perf_counter() - t_start) * 1000

        n = max(len(sqs), 1)
        k_eff = self._per_query_k(n)
        
        # Uncertainty check
        uncertainty_bypass = not bool(final)
        if final and final[0].rerank_score is not None:
            if final[0].rerank_score < self.cfg.uncertainty_threshold:
                uncertainty_bypass = True
        
        return RetrievalResult(
            query_texts=[sq.query_text() for sq in sqs],
            hits=final,
            per_subquery={qid: hs[:k_eff] for qid, hs in per_sq.items()},
            timings_ms=timings,
            cache_hits=cache_hits,
            uncertainty_bypass=uncertainty_bypass,
        )

    def clear_cache(self) -> None:
        with self._lock:
            self._cache.clear()

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _coerce(query: QueryLike) -> list[SubQuery]:
        items = [query] if isinstance(query, (str, SubQuery)) else list(query)
        out = []
        for i, it in enumerate(items):
            sq = SubQuery(text=it) if isinstance(it, str) else replace(it)
            if not sq.qid:
                sq.qid = f"sq{i}"
            if sq.query_text():
                out.append(sq)
        if not out:
            raise ValueError("retrieve() needs at least one non-empty query")
        return out

    def _keep(self) -> int:
        return max(self.cfg.final_k, self.cfg.per_query_k)

    def _per_query_k(self, n_sub: int) -> int:
        return max(self.cfg.per_query_k, math.ceil(self.cfg.final_k / n_sub))

    async def _retrieve_one(self, text: str, timings: dict) -> tuple[list[Hit], int]:
        """Hybrid retrieve + rerank for one sub-query. Returns (hits, was_cache_hit)."""
        key = (self.cfg.mode, self.cfg.rerank, self.cfg.candidate_k, self._keep(), text)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
                return self._cache[key], 1

        def _sync_retrieve():
            cfg, idx = self.cfg, self.index
            dense: list[tuple[int, float]] = []
            sparse: list[tuple[int, float]] = []
            if cfg.mode in ("hybrid", "dense"):
                t0 = time.perf_counter()
                dense = idx.dense_search(text, cfg.dense_k)
                timings["dense_ms"] += (time.perf_counter() - t0) * 1000
            if cfg.mode in ("hybrid", "sparse"):
                t0 = time.perf_counter()
                sparse = idx.sparse_search(text, cfg.sparse_k)
                timings["sparse_ms"] += (time.perf_counter() - t0) * 1000
    
            dense_rank = {i: r for r, (i, _) in enumerate(dense, 1)}
            sparse_rank = {i: r for r, (i, _) in enumerate(sparse, 1)}
            fused = rrf([[i for i, _ in dense], [i for i, _ in sparse]], k=cfg.rrf_k,
                        weights=[cfg.dense_weight, cfg.sparse_weight])[: cfg.candidate_k]
    
            cand = [(i, s) for i, s in fused]
            rerank_scores: dict[int, float] = {}
            if cfg.rerank and self.reranker is not None and cand:
                t0 = time.perf_counter()
                scores = self.reranker.score(text, [idx.chunks[i].text for i, _ in cand])
                timings["rerank_ms"] += (time.perf_counter() - t0) * 1000
                rerank_scores = {i: sc for (i, _), sc in zip(cand, scores)}
                # Drop individually weak hits before multi-query fusion. A
                # strong result for one intent must not promote unrelated
                # low-score chunks from another intent into the answer.
                cand = [(i, score) for i, score in cand
                        if rerank_scores[i] >= cfg.uncertainty_threshold]
                # rerank score first; RRF score keeps ordering stable on ties
                cand.sort(key=lambda p: (-rerank_scores[p[0]], -p[1], p[0]))
    
            hits = []
            for i, rrf_score in cand[: self._keep()]:
                c = idx.chunks[i]
                hits.append(Hit(
                    chunk_id=c.chunk_id, doc_id=c.doc_id, text=c.text,
                    score=rerank_scores.get(i, rrf_score),
                    dense_rank=dense_rank.get(i), sparse_rank=sparse_rank.get(i),
                    rerank_score=rerank_scores.get(i), meta=dict(c.meta),
                ))
            return hits

        hits = await asyncio.to_thread(_sync_retrieve)

        with self._lock:
            self._cache[key] = hits
            while len(self._cache) > self.cfg.cache_size:
                self._cache.popitem(last=False)
        return hits, 0

    def _fuse(self, sqs: list[SubQuery], per_sq: dict[str, list[Hit]]) -> list[Hit]:
        cfg = self.cfg
        n = len(sqs)
        k_eff = self._per_query_k(n)
        lists = {sq.qid: per_sq[sq.qid][:k_eff] for sq in sqs}

        by_id: dict[str, Hit] = {}
        owners: dict[str, list[str]] = {}
        for qid, hs in lists.items():
            for h in hs:
                by_id.setdefault(h.chunk_id, h)
                owners.setdefault(h.chunk_id, []).append(qid)

        fused = rrf([[h.chunk_id for h in lists[sq.qid]] for sq in sqs],
                    k=cfg.rrf_k, weights=[sq.weight for sq in sqs])
        fused_score = dict(fused)

        selected: list[str] = []
        per_doc: dict[str, int] = {}

        def try_add(cid: str) -> bool:
            if cid in selected or len(selected) >= cfg.final_k:
                return False
            doc = by_id[cid].doc_id
            if per_doc.get(doc, 0) >= cfg.max_per_doc:
                return False
            selected.append(cid)
            per_doc[doc] = per_doc.get(doc, 0) + 1
            return True

        # 1) coverage: every intent keeps its best chunks (avoids one intent drowning the rest)
        floor = max(1, min(cfg.min_per_subquery, cfg.final_k // n))
        for sq in sqs:
            taken = 0
            for h in lists[sq.qid]:
                if taken >= floor:
                    break
                if h.chunk_id in selected or try_add(h.chunk_id):
                    taken += 1
        # 2) fill the remainder by fused rank
        for cid, _ in fused:
            try_add(cid)

        selected.sort(key=lambda c: (-fused_score[c], c))
        return [replace(by_id[c], score=fused_score[c], sub_query_ids=owners[c]) for c in selected]
