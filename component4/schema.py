"""Shared data types for Component 4 (Session-Aware Synthesis)."""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class CachedChunk:
    chunk_id: str
    doc_id: str
    text: str
    section: str = ""


@dataclass
class Claim:
    """One synthesized statement plus the citations that back it, versioned
    so a later cosmetic turn can be re-rendered without re-deriving facts."""
    claim_id: str
    text: str
    citations: list          # list[str], each like "Doc_12 §2"
    version: int


@dataclass
class SessionState:
    """Ephemeral, session-bound memory (design doc: 'In-memory Python
    dictionaries or Redis'). One instance per session_id."""
    session_id: str
    answer_version: int = 0
    cached_chunks: dict = field(default_factory=dict)     # chunk_id -> CachedChunk
    active_claims: list = field(default_factory=list)     # list[Claim]
    last_answer: str = ""
    last_query: str = ""

    def known_doc_ids(self) -> set:
        return {c.doc_id for c in self.cached_chunks.values()}

    def known_citations(self) -> set:
        return {(c.doc_id, c.section) for c in self.cached_chunks.values() if c.section}

    def add_chunks(self, hits) -> None:
        """Accepts Component 3 `Hit` objects (or any object with
        chunk_id/doc_id/text) and merges them into the cache."""
        for h in hits:
            section = h.meta.get("section", "") if hasattr(h, "meta") else ""
            self.cached_chunks[h.chunk_id] = CachedChunk(
                chunk_id=h.chunk_id, doc_id=h.doc_id, text=h.text, section=section,
            )


@dataclass
class SynthesisTurn:
    """Result of processing one user utterance."""
    session_id: str
    turn_id: str
    utterance: str
    retrieval_required: bool
    is_cosmetic: bool
    new_constraint: Optional[str]
    answer: str
    citations: list           # citation tags actually present in `answer`, post-validation
    dropped_citations: list   # tags stripped because they didn't resolve against the cache
    version: int
    uncertainty: bool = False
    timestamp: float = field(default_factory=time.time)

    def to_event(self) -> dict[str, Any]:
        return {
            "component": "session_aware_synthesis",
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "utterance": self.utterance,
            "retrieval_required": self.retrieval_required,
            "is_cosmetic": self.is_cosmetic,
            "new_constraint": self.new_constraint,
            "version": self.version,
            "citations": self.citations,
            "dropped_citations": self.dropped_citations,
            "uncertainty": self.uncertainty,
            "timestamp": round(self.timestamp, 3),
        }
