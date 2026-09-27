"""Controller configuration. Tunable without code changes."""

from dataclasses import dataclass


@dataclass
class ControllerConfig:
    """Thresholds and policy switches for the Retrieval Controller."""

    min_tokens_for_retrieval: int = 3
    min_chars_for_retrieval: int = 12
    # Suppress RETRIEVE when the normalized query exactly matches the
    # last query we already retrieved on in the same session.
    suppress_identical_retrieval: bool = True
    # Jaccard similarity below this => meaningful change.
    # Above it (but not identical) => cosmetic wording change.
    meaningful_change_threshold: float = 0.85
    # Confidence values (documented heuristics, not calibrated probabilities).
    confidence_stable: float = 0.87
    confidence_new_constraint: float = 0.89
    confidence_query_change: float = 0.84
    confidence_wait: float = 0.72
    confidence_no_retrieval: float = 0.91

    schema_version: str = "1.0"
