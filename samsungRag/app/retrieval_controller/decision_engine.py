"""Pure decision logic: WAIT / RETRIEVE / NO_RETRIEVAL.

Kept side-effect free for easy unit testing. Session I/O lives in controller.py.
"""

from dataclasses import dataclass, field
from typing import List

from app.retrieval_controller.config import ControllerConfig
from app.retrieval_controller.intent_detector import IntentSignal, is_provisionally_incomplete
from app.retrieval_controller.policies import select_trigger, should_suppress_duplicate
from app.retrieval_controller.query_comparator import ComparisonResult
from app.shared.models import DecisionType, TriggerType


def _looks_truncated(transcript: str) -> bool:
    """True when a chunk ends mid-utterance (ellipsis / dangling function word)."""
    raw = (transcript or "").strip()
    if not raw:
        return True
    if raw.endswith(("...", "…", "--", "—", "-", ",")):
        return True
    low = raw.lower().rstrip(".…!? ")
    dangling = (" to", " for", " with", " in", " on", " about", " of", " and", " or", " a", " an", " the")
    if any(low == d.strip() or low.endswith(d) for d in dangling):
        return True
    # Bare opener with no object, e.g. "I need to", "I want".
    if low in ("i need to", "i need", "i want", "i want to", "i am looking for", "um", "uh"):
        return True
    return False


@dataclass
class EngineResult:
    """Decision engine output (before state persistence)."""

    decision: DecisionType
    reason_codes: List[str] = field(default_factory=list)
    trigger: TriggerType = TriggerType.NONE
    confidence: float = 0.0
    query: str = ""


def decide(
    *,
    transcript: str,
    normalized: str,
    is_final: bool,
    intent: IntentSignal,
    comparison: ComparisonResult,
    config: ControllerConfig,
) -> EngineResult:
    """Apply the 16-step pipeline decision rules deterministically."""
    # 3. Empty input -> WAIT (streaming silence / missing chunk).
    if not normalized:
        return EngineResult(
            decision=DecisionType.WAIT,
            reason_codes=["EMPTY_INPUT"],
            trigger=TriggerType.NONE,
            confidence=config.confidence_wait,
            query="",
        )
    # 4. Acknowledgements / conversational replies -> NO_RETRIEVAL.
    if intent.is_acknowledgement:
        return EngineResult(
            decision=DecisionType.NO_RETRIEVAL,
            reason_codes=["CONVERSATIONAL_RESPONSE"],
            trigger=TriggerType.NONE,
            confidence=config.confidence_no_retrieval,
            query="",
        )
    # 5. Presentation-only -> NO_RETRIEVAL.
    if intent.is_presentation_only:
        return EngineResult(
            decision=DecisionType.NO_RETRIEVAL,
            reason_codes=["PRESENTATION_ONLY"],
            trigger=TriggerType.NONE,
            confidence=config.confidence_no_retrieval,
            query="",
        )
    # 6. Incomplete intent -> WAIT. Truncated streaming fragments
    # (trailing "...", dangling prepositions) WAIT even if they contain
    # info-seeking keywords like "need" — there is no actionable object yet.
    if intent.is_incomplete and (not intent.is_information_seeking or _looks_truncated(transcript)):
        return EngineResult(
            decision=DecisionType.WAIT,
            reason_codes=["INCOMPLETE_INTENT"],
            trigger=TriggerType.NONE,
            confidence=config.confidence_wait,
            query=transcript.strip(),
        )
    # Provisional completeness gate (Rule 1): when the stream is still
    # open (is_final=false) and the transcript is grammatically unfinished
    # or has no meaningful object, WAIT — before any INTENT_STABLE path.
    # Applies regardless of info-seeking keywords ("need"/"want") so that
    # "I need a" / "I want to know about" wait, while complete requests
    # ("... venues in Pune") pass through. Final transcripts with a real
    # object also pass (Rule 3); final fragments with no object still WAIT.
    if is_provisionally_incomplete(transcript):
        return EngineResult(
            decision=DecisionType.WAIT,
            reason_codes=["INCOMPLETE_INTENT"],
            trigger=TriggerType.NONE,
            confidence=config.confidence_wait,
            query=transcript.strip(),
        )
    # Short fragment that is neither info-seeking nor ack -> WAIT.
    if (
        len(normalized.split()) < config.min_tokens_for_retrieval
        or len(normalized) < config.min_chars_for_retrieval
    ) and not intent.is_information_seeking:
        return EngineResult(
            decision=DecisionType.WAIT,
            reason_codes=["INCOMPLETE_INTENT"],
            trigger=TriggerType.NONE,
            confidence=config.confidence_wait,
            query=transcript.strip(),
        )
    # A declarative continuation can add a factual constraint without being
    # phrased as a question ("actually, make it 30 people"). If the active
    # session already has a retrieval context and entity extraction found a
    # new constraint, retrieve its delta instead of suppressing it as chat.
    if not intent.is_information_seeking and not comparison.is_first_query and comparison.new_constraints:
        return EngineResult(
            decision=DecisionType.RETRIEVE,
            reason_codes=["NEW_CONSTRAINT"],
            trigger=select_trigger(
                is_final=is_final,
                has_new_constraints=True,
                has_meaningful_change=True,
                is_first_retrieval=False,
            ),
            confidence=config.confidence_new_constraint,
            query=transcript.strip(),
        )
    # 7. Non-informational remainder -> NO_RETRIEVAL.
    if not intent.is_information_seeking:
        return EngineResult(
            decision=DecisionType.NO_RETRIEVAL,
            reason_codes=["NON_INFORMATIONAL"],
            trigger=TriggerType.NONE,
            confidence=config.confidence_no_retrieval,
            query="",
        )
    # 10/11. Session-aware comparisons (skip on first query).
    if not comparison.is_first_query:
        # 7 (repeated). Identical -> suppress unless policy says otherwise.
        if should_suppress_duplicate(
            comparison.is_identical, config.suppress_identical_retrieval
        ):
            return EngineResult(
                decision=DecisionType.NO_RETRIEVAL,
                reason_codes=["QUERY_UNCHANGED"],
                trigger=TriggerType.NONE,
                confidence=0.88,
                query="",
            )
        # 11. New constraints -> RETRIEVE.
        if comparison.new_constraints:
            return EngineResult(
                decision=DecisionType.RETRIEVE,
                reason_codes=["NEW_CONSTRAINT"],
                trigger=select_trigger(
                    is_final=is_final,
                    has_new_constraints=True,
                    has_meaningful_change=True,
                    is_first_retrieval=False,
                ),
                confidence=config.confidence_new_constraint,
                query=transcript.strip(),
            )
        # 10. Meaningful change -> RETRIEVE.
        if comparison.is_meaningful_change:
            return EngineResult(
                decision=DecisionType.RETRIEVE,
                reason_codes=["QUERY_CHANGE"],
                trigger=select_trigger(
                    is_final=is_final,
                    has_new_constraints=False,
                    has_meaningful_change=True,
                    is_first_retrieval=False,
                ),
                confidence=config.confidence_query_change,
                query=transcript.strip(),
            )
        # Cosmetic rewording only -> no new retrieval.
        if comparison.is_cosmetic_change:
            return EngineResult(
                decision=DecisionType.NO_RETRIEVAL,
                reason_codes=["COSMETIC_CHANGE"],
                trigger=TriggerType.NONE,
                confidence=0.82,
                query="",
            )
    # 12. Stable information request -> RETRIEVE.
    trigger = select_trigger(
        is_final=is_final,
        has_new_constraints=False,
        has_meaningful_change=False,
        is_first_retrieval=comparison.is_first_query,
    )
    return EngineResult(
        decision=DecisionType.RETRIEVE,
        reason_codes=["INTENT_STABLE"],
        trigger=trigger,
        confidence=config.confidence_stable + (0.03 if is_final else 0.0),
        query=transcript.strip(),
    )
