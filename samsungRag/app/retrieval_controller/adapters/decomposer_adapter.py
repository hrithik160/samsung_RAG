"""Decomposer adapter interface (Member 2 integration point).

The Retrieval Controller answers WHEN to retrieve. Member 2's
Multi-Intent Decomposer answers WHAT to retrieve. This module defines
the boundary so the controller works with or without Member 2.

How Member 2 connects later:
1. Implement the DecomposerAdapter Protocol (see below).
2. Pass the implementation into RetrievalController(decomposer=...).
3. The controller calls it opportunistically (non-blocking for MVP);
   full orchestration (controller -> decomposer -> retrieval) is wired
   by Member 4 / API layer using RetrievalDecision.query.
"""

from typing import List, Protocol
from pydantic import BaseModel, Field


class DecomposerRequest(BaseModel):
    """Input Member 2 receives when controller emits RETRIEVE."""

    query: str
    session_id: str
    turn_id: str

    model_config = {"extra": "ignore"}


class DecomposerResponse(BaseModel):
    """Output Member 2 returns. Controller does NOT depend on its internals."""

    subqueries: List[str] = Field(default_factory=list)
    intents: List[str] = Field(default_factory=list)

    model_config = {"extra": "ignore"}


class DecomposerAdapter(Protocol):
    """Interface Member 2 must implement. No import-time dependency."""

    def decompose(self, request: DecomposerRequest) -> DecomposerResponse:
        ...


class MockDecomposerAdapter:
    """Test double: echo query as a single subquery. No real decomposition."""

    def decompose(self, request: DecomposerRequest) -> DecomposerResponse:
        q = request.query.strip()
        if not q:
            return DecomposerResponse(subqueries=[], intents=[])
        return DecomposerResponse(subqueries=[q], intents=["general"])
