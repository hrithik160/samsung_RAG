"""Shared data types for Component 2 (Multi-Intent Decomposer).

Mirrors the style of Component 3's schema.py so the two modules compose
cleanly. A `SubQueryItem` here is the *authoring* format (what the
decomposition step produces / validates); `to_subquery_kwargs()` converts it
into the constructor kwargs for Component 3's `schema.SubQuery`, so callers
can do `SubQuery(**item.to_subquery_kwargs())` without Component 2 needing to
import Component 3 directly (keeps the two components decoupled).
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Optional

# Allowed intent categories (design doc: factual lookup, comparison,
# definition, numerical, follow-up/contextual).
INTENT_TYPES = {"factual", "comparison", "definition", "numerical", "followup"}


@dataclass
class SubQueryItem:
    """One atomic, independently-answerable sub-query."""

    sub_query: str
    intent_type: str = "factual"
    depends_on: Optional[int] = None   # index into the sibling sub_query list, or None
    priority: int = 1                  # lower = higher priority; ties broken by list order
    domain: Optional[str] = None
    entity: Optional[str] = None
    aspect: Optional[str] = None

    def __post_init__(self) -> None:
        if self.intent_type not in INTENT_TYPES:
            self.intent_type = "factual"
        if not self.sub_query or not self.sub_query.strip():
            raise ValueError("SubQueryItem.sub_query must be non-empty")
        self.sub_query = self.sub_query.strip()

    def to_subquery_kwargs(self, qid: str = "") -> dict:
        """Kwargs for Component 3's `schema.SubQuery(**kwargs)`."""
        return {
            "text": self.sub_query,
            "domain": self.domain,
            "entity": self.entity,
            "aspect": self.aspect,
            "weight": 1.0 if self.priority <= 1 else 1.0 / self.priority,
            "qid": qid,
        }


@dataclass
class DecompositionResult:
    """Full output of one decomposition pass, ready for telemetry logging."""

    original_query: str
    session_id: str
    turn_id: str
    sub_queries: list  # list[SubQueryItem]
    used_llm: bool = False
    gate_triggered: bool = False   # True if the complexity gate routed to the LLM path
    fallback_used: bool = False    # True if validation rejected the LLM output and we fell back
    timestamp: float = field(default_factory=time.time)

    def to_component3_subqueries(self):
        """Build Component 3 `SubQuery` objects directly (imports schema lazily)."""
        from schema import SubQuery  # Component 3's schema.py, if on sys.path
        return [
            SubQuery(**sq.to_subquery_kwargs(qid=f"{self.turn_id}.{i}"))
            for i, sq in enumerate(self.sub_queries)
        ]

    def to_event(self) -> dict[str, Any]:
        """JSON-serialisable telemetry record."""
        return {
            "component": "multi_intent_decomposer",
            "session_id": self.session_id,
            "turn_id": self.turn_id,
            "original_query": self.original_query,
            "timestamp": round(self.timestamp, 3),
            "used_llm": self.used_llm,
            "gate_triggered": self.gate_triggered,
            "fallback_used": self.fallback_used,
            "sub_queries": [
                {
                    "sub_query": sq.sub_query,
                    "intent_type": sq.intent_type,
                    "depends_on": sq.depends_on,
                    "priority": sq.priority,
                    "domain": sq.domain,
                    "entity": sq.entity,
                    "aspect": sq.aspect,
                }
                for sq in self.sub_queries
            ],
        }
