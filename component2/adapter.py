"""Implements Component 1's `DecomposerAdapter` Protocol
(`app/retrieval_controller/adapters/decomposer_adapter.py`) using the real
`MultiIntentDecomposer`, replacing `MockDecomposerAdapter`.

Component 1 only needs `.decompose(request) -> response` where `request`
exposes `.query/.session_id/.turn_id` and `response` exposes
`.subqueries: list[str]` / `.intents: list[str]` -- Python duck-typing
means we don't need to import Component 1's pydantic classes at all, so
this file has zero dependency on Component 1's package existing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

from decomposer import MultiIntentDecomposer, LLMDecomposer


@dataclass
class DecomposerResponse:
    """Structurally identical to Component 1's own DecomposerResponse
    (same field names/types), so `resp.subqueries` / `resp.intents` work
    whether the controller expects the pydantic class or this one."""
    subqueries: list = field(default_factory=list)
    intents: list = field(default_factory=list)


class RealDecomposerAdapter:
    """Drop-in replacement for `MockDecomposerAdapter`:

        from adapter import RealDecomposerAdapter
        controller = RetrievalController(decomposer=RealDecomposerAdapter())

    Pass `llm_call` to use the LLM-backed decomposer; omit it to use the
    offline rule-based backend (zero dependencies, good for tests/CI).
    """

    def __init__(self, llm_call=None):
        backend = LLMDecomposer(llm_call) if llm_call is not None else None
        self.decomposer = MultiIntentDecomposer(llm_backend=backend)

    def decompose(self, request) -> DecomposerResponse:
        query = getattr(request, "query", "") or ""
        session_id = getattr(request, "session_id", "default")
        turn_id = getattr(request, "turn_id", "T0")
        if not query.strip():
            return DecomposerResponse(subqueries=[], intents=[])
        result = self.decomposer.decompose(query, session_id=session_id, turn_id=turn_id)
        return DecomposerResponse(
            subqueries=[sq.sub_query for sq in result.sub_queries],
            intents=[sq.intent_type for sq in result.sub_queries],
        )

    def decompose_full(self, query: str, session_id: str = "default", turn_id: str = "T0"):
        """Escape hatch back to the rich `DecompositionResult` (with
        domain/entity/aspect, priorities, depends_on) for callers that want
        more than the Component 1 Protocol's flattened str lists -- e.g.
        Component 3 needs the full `SubQueryItem`s, not just text."""
        return self.decomposer.decompose(query, session_id=session_id, turn_id=turn_id)
