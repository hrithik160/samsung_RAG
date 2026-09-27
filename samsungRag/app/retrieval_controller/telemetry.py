"""Structured telemetry via stdlib logging (no external infra for MVP)."""

import logging
import time
from typing import Any, Dict

logger = logging.getLogger("retrieval_controller")
if not logger.handlers:
    handler = logging.StreamHandler()
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
logger.setLevel(logging.INFO)


def emit_decision_telemetry(event: Dict[str, Any]) -> Dict[str, Any]:
    """Log a single decision event and return it (testable)."""
    event = dict(event)
    event.setdefault("component", "retrieval-controller")
    event.setdefault("emitted_at", time.time())
    logger.info("retrieval_decision %s", event)
    return event
