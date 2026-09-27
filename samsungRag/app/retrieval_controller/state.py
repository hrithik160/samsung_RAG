"""Session state abstraction.

In-memory for MVP. NOT suitable for multi-process production deployment
because each process would hold isolated state. Replace InMemorySessionStore
with a Redis/DB-backed implementation of SessionStore for production.
"""

from dataclasses import dataclass, field
from threading import Lock
from typing import Dict, List, Optional, Protocol


@dataclass
class SessionState:
    """Per-session retrieval state."""

    session_id: str
    previous_transcript: str = ""
    normalized_query: str = ""
    detected_intent: str = "UNKNOWN"
    extracted_entities: Dict[str, List[str]] = field(default_factory=dict)
    extracted_constraints: List[str] = field(default_factory=list)
    last_retrieval_query: str = ""
    last_retrieval_timestamp: float = 0.0
    last_decision: str = "NONE"
    state_version: int = 0


class SessionStore(Protocol):
    """Replaceable storage interface."""

    def get(self, session_id: str) -> Optional[SessionState]:
        ...

    def save(self, state: SessionState) -> SessionState:
        ...


class InMemorySessionStore:
    """Thread-safe in-memory store. Sessions are fully isolated by session_id."""

    def __init__(self) -> None:
        self._sessions: Dict[str, SessionState] = {}
        self._lock = Lock()

    def get(self, session_id: str) -> Optional[SessionState]:
        with self._lock:
            return self._sessions.get(session_id)

    def save(self, state: SessionState) -> SessionState:
        with self._lock:
            self._sessions[state.session_id] = state
            return state

    def clear(self) -> None:
        with self._lock:
            self._sessions.clear()
