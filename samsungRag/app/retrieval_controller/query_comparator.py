"""Query normalization and comparison.

Distinguishes cosmetic wording changes from retrieval-relevant changes
(new entities/constraints, meaningful intent shifts).
"""

import re
from dataclasses import dataclass, field
from typing import List


def normalize_query(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    t = text.lower().strip()
    t = re.sub(r"[^a-z0-9\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def tokens(text: str) -> List[str]:
    """Tokenize normalized text."""
    return normalize_query(text).split() if text.strip() else []


def jaccard(a: str, b: str) -> float:
    """Token Jaccard similarity on normalized queries."""
    sa, sb = set(tokens(a)), set(tokens(b))
    if not sa and not sb:
        return 1.0
    if not sa or not sb:
        return 0.0
    return len(sa & sb) / len(sa | sb)


@dataclass
class ComparisonResult:
    """Structured diff between current query and session state."""

    normalized_current: str = ""
    normalized_previous: str = ""
    is_first_query: bool = True
    is_identical: bool = False
    jaccard_similarity: float = 0.0
    is_meaningful_change: bool = False
    is_cosmetic_change: bool = False
    new_constraints: List[str] = field(default_factory=list)
    query_changed: bool = False


def compare_queries(
    current_normalized: str,
    previous_normalized: str,
    current_constraints: List[str],
    previous_constraints: List[str],
    has_previous: bool,
    meaningful_threshold: float = 0.85,
) -> ComparisonResult:
    """Compare current vs previous normalized queries + constraint sets."""
    result = ComparisonResult(
        normalized_current=current_normalized,
        normalized_previous=previous_normalized,
        is_first_query=not has_previous,
    )
    if not has_previous or not previous_normalized:
        result.is_first_query = True
        result.query_changed = bool(current_normalized)
        result.is_meaningful_change = bool(current_normalized)
        result.new_constraints = sorted(set(current_constraints))
        return result
    result.is_identical = current_normalized == previous_normalized
    result.jaccard_similarity = jaccard(current_normalized, previous_normalized)
    result.query_changed = not result.is_identical
    new_c = sorted(set(current_constraints) - set(previous_constraints))
    result.new_constraints = new_c
    if result.is_identical:
        result.is_meaningful_change = False
        result.is_cosmetic_change = False
    elif new_c:
        result.is_meaningful_change = True
    elif result.jaccard_similarity < meaningful_threshold:
        result.is_meaningful_change = True
    else:
        result.is_cosmetic_change = True
    return result
