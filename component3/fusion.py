"""Reciprocal Rank Fusion (Cormack et al., 2009)."""
from __future__ import annotations

from typing import Hashable, Optional, Sequence


def rrf(
    rankings: Sequence[Sequence[Hashable]],
    k: int = 60,
    weights: Optional[Sequence[float]] = None,
) -> list[tuple[Hashable, float]]:
    """Fuse several ranked lists of ids.

    score(d) = sum_i  w_i / (k + rank_i(d))      (rank is 1-based)

    Returns [(id, score)] best first. Ties are broken by id so results are
    deterministic (important for the reflector's top-k overlap checks).
    """
    if weights is not None and len(weights) != len(rankings):
        raise ValueError("weights must match rankings in length")
    scores: dict[Hashable, float] = {}
    for i, ranking in enumerate(rankings):
        w = 1.0 if weights is None else weights[i]
        for rank, item in enumerate(ranking, start=1):
            scores[item] = scores.get(item, 0.0) + w / (k + rank)
    return sorted(scores.items(), key=lambda kv: (-kv[1], str(kv[0])))
