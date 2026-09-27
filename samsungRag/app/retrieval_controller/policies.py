"""Trigger + dedup policies. Keeps decision_engine declarative."""

from app.shared.models import TriggerType


def select_trigger(
    *,
    is_final: bool,
    has_new_constraints: bool,
    has_meaningful_change: bool,
    is_first_retrieval: bool,
) -> TriggerType:
    """Precedence: NEW_CONSTRAINT > QUERY_CHANGE > FINAL/PROVISIONAL."""
    if has_new_constraints:
        return TriggerType.NEW_CONSTRAINT
    if has_meaningful_change and not is_first_retrieval:
        return TriggerType.QUERY_CHANGE
    if is_final:
        return TriggerType.FINAL
    return TriggerType.PROVISIONAL


def should_suppress_duplicate(is_identical: bool, suppress: bool) -> bool:
    """Avoid excessive retrieval calls when nothing meaningfully changed."""
    return bool(is_identical and suppress)
