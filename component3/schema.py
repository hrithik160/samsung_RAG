"""Shared data types for Component 3 (Corpus Retrieval & Fusion)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    meta: dict = field(default_factory=dict)


@dataclass
class SubQuery:
    """One retrieval intent, as emitted by Component 2 (Multi-Intent Decomposer).

    Only `text` is required. `domain/entity/aspect` follow the {domain, entity,
    aspect} schema so fusion/dedup can reason about sub-queries later.
    """

    text: str = ""
    domain: Optional[str] = None
    entity: Optional[str] = None
    aspect: Optional[str] = None
    weight: float = 1.0
    qid: str = ""

    def query_text(self) -> str:
        if self.text.strip():
            return self.text.strip()
        return " ".join(p for p in (self.entity, self.aspect) if p).strip()


@dataclass
class Hit:
    chunk_id: str
    doc_id: str
    text: str
    score: float                      # final ordering score at the level it is reported
    dense_rank: Optional[int] = None  # 1-based rank in dense list (None = not retrieved)
    sparse_rank: Optional[int] = None
    rerank_score: Optional[float] = None
    sub_query_ids: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)


@dataclass
class RetrievalResult:
    query_texts: list
    hits: list                        # final fused, deduped hits (best first)
    per_subquery: dict                # qid -> list[Hit]
    timings_ms: dict
    cache_hits: int = 0
    uncertainty_bypass: bool = False  # Set to True if top result confidence is below threshold

    def top_ids(self, k: int = 5) -> list:
        return [h.chunk_id for h in self.hits[:k]]

    def overlap(self, other: "RetrievalResult", k: int = 5) -> float:
        """Top-k chunk-ID overlap in [0, 1]. Used by the reflector (Component 1/4):
        `provisional.overlap(final)` -> how much of the final top-k the early
        retrieval already had."""
        a, b = set(self.top_ids(k)), set(other.top_ids(k))
        denom = min(k, len(b))
        return len(a & b) / denom if denom else 1.0

    def to_event(self) -> dict[str, Any]:
        """JSON-serialisable form for the telemetry / retrieval_events log."""
        return {
            "sub_queries": self.query_texts,
            "citations": [
                {"chunk_id": h.chunk_id, "doc_id": h.doc_id,
                 "section": h.meta.get("section"), "score": round(h.score, 6),
                 "sub_query_ids": h.sub_query_ids}
                for h in self.hits
            ],
            "timings_ms": {k: round(v, 2) for k, v in self.timings_ms.items()},
            "cache_hits": self.cache_hits,
        }
