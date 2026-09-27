"""Retrieval Controller orchestration (16-step pipeline).

Steps: validate -> normalize -> empty check -> ack -> presentation ->
incomplete -> info-seeking -> entities -> compare -> change/constraint
detection -> decide -> reason/confidence -> trigger -> telemetry -> persist.
"""

import time
from typing import Any, Dict, Optional

from app.retrieval_controller.config import ControllerConfig
from app.retrieval_controller.decision_engine import decide
from app.retrieval_controller.entity_extractor import extract_entities
from app.retrieval_controller.intent_detector import detect_intent
from app.retrieval_controller.query_comparator import (
    ComparisonResult,
    compare_queries,
    normalize_query,
)
from app.retrieval_controller.state import InMemorySessionStore, SessionState, SessionStore
from app.retrieval_controller.telemetry import emit_decision_telemetry
from app.shared.models import RetrievalDecision, StreamingInput


class RetrievalController:
    """Session-aware WHEN-to-retrieve controller. Decomposer-optional."""

    def __init__(
        self,
        store: Optional[SessionStore] = None,
        config: Optional[ControllerConfig] = None,
        decomposer: Any = None,
    ) -> None:
        self.store: SessionStore = store or InMemorySessionStore()
        self.config = config or ControllerConfig()
        self.decomposer = decomposer  # Member 2 plugs in later; unused for decisions.

    def decide(self, payload: StreamingInput) -> RetrievalDecision:
        """Run the full pipeline and persist session state safely."""
        start = time.perf_counter()
        transcript = (payload.transcript or "").strip()
        normalized = normalize_query(transcript)

        state = self.store.get(payload.session_id)
        is_first = state is None
        if is_first:
            state = SessionState(session_id=payload.session_id)

        assert state is not None
        intent = detect_intent(transcript if transcript else normalized)
        entities = extract_entities(transcript)
        constraints = entities.constraints()

        comparison: ComparisonResult = compare_queries(
            current_normalized=normalized,
            previous_normalized=state.normalized_query,
            current_constraints=constraints,
            previous_constraints=state.extracted_constraints,
            has_previous=not is_first and bool(state.normalized_query),
            meaningful_threshold=self.config.meaningful_change_threshold,
        )

        engine = decide(
            transcript=transcript,
            normalized=normalized,
            is_final=payload.is_final,
            intent=intent,
            comparison=comparison,
            config=self.config,
        )

        # Persist state: version increments every turn; retrieval fields
        # update only on RETRIEVE so repeats can be suppressed.
        state.previous_transcript = transcript
        state.normalized_query = normalized
        state.detected_intent = (
            "ACK" if intent.is_acknowledgement
            else "PRESENTATION" if intent.is_presentation_only
            else "INFO_SEEKING" if intent.is_information_seeking
            else "INCOMPLETE" if intent.is_incomplete
            else "UNKNOWN"
        )
        state.extracted_entities = entities.as_dict()
        state.extracted_constraints = constraints
        state.last_decision = engine.decision.value
        state.state_version += 1
        if engine.decision.value == "RETRIEVE":
            state.last_retrieval_query = normalized
            state.last_retrieval_timestamp = payload.timestamp
        self.store.save(state)

        now = time.time()
        decision = RetrievalDecision(
            schema_version=self.config.schema_version,
            session_id=payload.session_id,
            turn_id=payload.turn_id,
            decision=engine.decision,
            query=engine.query,
            confidence=round(min(1.0, max(0.0, engine.confidence)), 4),
            reason_codes=engine.reason_codes,
            trigger=engine.trigger,
            is_final=payload.is_final,
            timestamp=now,
            state_version=state.state_version,
            metadata=dict(payload.metadata or {}),
        )

        processing_ms = (time.perf_counter() - start) * 1000.0
        emit_decision_telemetry(
            {
                "session_id": payload.session_id,
                "turn_id": payload.turn_id,
                "decision": engine.decision.value,
                "confidence": decision.confidence,
                "reason_codes": engine.reason_codes,
                "trigger": engine.trigger.value,
                "processing_time_ms": round(processing_ms, 3),
                "state_version": state.state_version,
                "query_changed": comparison.query_changed,
                "new_constraints": comparison.new_constraints,
                "is_final": payload.is_final,
            }
        )
        return decision

    def preview_decomposition(self, decision: RetrievalDecision) -> Dict[str, Any]:
        """Optionally ask Member 2's decomposer for subqueries (non-required)."""
        if self.decomposer is None or decision.decision.value != "RETRIEVE":
            return {"subqueries": [], "intents": []}
        from app.retrieval_controller.adapters.decomposer_adapter import DecomposerRequest

        resp = self.decomposer.decompose(
            DecomposerRequest(
                query=decision.query,
                session_id=decision.session_id,
                turn_id=decision.turn_id,
            )
        )
        return {"subqueries": resp.subqueries, "intents": resp.intents}
