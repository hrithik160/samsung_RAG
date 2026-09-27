"""Shared API contracts for the streaming RAG pipeline.

Owned by Member 1 for the MVP, but these models are the integration
boundary for Members 2 (decomposer), 3 (retrieval), and 4 (synthesis/UI).
"""

from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class DecisionType(str, Enum):
    """WHEN to retrieve (Member 1). NOT what to retrieve (Member 2)."""

    WAIT = "WAIT"
    RETRIEVE = "RETRIEVE"
    NO_RETRIEVAL = "NO_RETRIEVAL"


class TriggerType(str, Enum):
    """Why the decision was made."""

    PROVISIONAL = "PROVISIONAL"
    FINAL = "FINAL"
    NEW_CONSTRAINT = "NEW_CONSTRAINT"
    QUERY_CHANGE = "QUERY_CHANGE"
    CONTEXTUAL = "CONTEXTUAL"
    NONE = "NONE"


class StreamingInput(BaseModel):
    """One partial or final transcript chunk from the client."""

    session_id: str = Field(..., min_length=1, description="Session identifier")
    turn_id: str = Field(..., min_length=1, description="Turn identifier within the session")
    transcript: str = Field(
        default="",
        description="Partial or complete user transcript. Empty allowed; controller returns WAIT.",
    )
    is_final: bool = Field(default=False, description="True when user finished speaking")
    timestamp: float = Field(..., description="Client event time, epoch seconds")
    language: Optional[str] = Field(default="en", description="BCP-47-ish language tag")
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "ignore"}


class RetrievalDecision(BaseModel):
    """Structured decision emitted by the Retrieval Controller."""

    schema_version: str = Field(default="1.0")
    session_id: str
    turn_id: str
    decision: DecisionType
    query: str = Field(description="Normalized query to retrieve on (empty if no retrieval)")
    confidence: float = Field(ge=0.0, le=1.0)
    reason_codes: List[str] = Field(default_factory=list)
    trigger: TriggerType
    is_final: bool
    timestamp: float = Field(description="Server decision time, epoch seconds")
    state_version: int = Field(ge=0)
    metadata: Dict[str, Any] = Field(default_factory=dict)

    model_config = {"extra": "ignore"}
