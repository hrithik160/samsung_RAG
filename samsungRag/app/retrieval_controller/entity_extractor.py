"""Lightweight entity/constraint extractor (regex heuristics, modular).

Identifies locations, quantities, times, topics, qualifiers.
Replaceable with a stronger NLP model later without changing callers.
"""

import re
from dataclasses import dataclass, field
from typing import Dict, List

_KNOWN_LOCATIONS = [
    "pune", "bengaluru", "bangalore", "delhi", "mumbai", "chennai",
    "hyderabad", "kolkata", "jaipur", "ahmedabad", "kochi", "goa",
]

_TIME_PATTERNS = [
    r"\btomorrow\b", r"\btoday\b", r"\bnext\s+week\b", r"\bnext\s+month\b",
    r"\bthis\s+weekend\b", r"\b20\d\d\b",
    r"\bmonday\b", r"\btuesday\b", r"\bwednesday\b", r"\bthursday\b",
    r"\bfriday\b", r"\bsaturday\b", r"\bsunday\b",
]

_QUANTITY_PATTERNS = [
    r"\bfor\s+(\d+)\s*(people|persons?|users?|members?|seats?|guests?)\b",
    r"\b(\d+)\s*(people|persons?|users?|members?|seats?|guests?)\b",
    r"\bcapacity\s*(of\s*)?(\d+)\b",
]

_TOPIC_KEYWORDS = [
    "travel", "reimbursement", "policy", "policies", "venue", "venues",
    "workshop", "crop", "disease", "diseases", "hotel", "flight",
    "office", "budget", "training", "conference",
]

_QUALIFIER_KEYWORDS = [
    "budget", "cheap", "premium", "luxury", "international", "domestic",
    "english", "hindi", "marathi", "small", "large", "outdoor", "indoor",
    "ac", "non-ac", "vegetarian", "beginner", "advanced",
]


@dataclass
class ExtractedEntities:
    """Structured extraction result."""

    locations: List[str] = field(default_factory=list)
    quantities: List[str] = field(default_factory=list)
    times: List[str] = field(default_factory=list)
    topics: List[str] = field(default_factory=list)
    qualifiers: List[str] = field(default_factory=list)

    def constraints(self) -> List[str]:
        """Flatten location/quantity/time/qualifier hits into constraint labels."""
        out: List[str] = []
        out += [f"location:{v}" for v in self.locations]
        out += [f"quantity:{v}" for v in self.quantities]
        out += [f"time:{v}" for v in self.times]
        out += [f"qualifier:{v}" for v in self.qualifiers]
        return sorted(set(out))

    def as_dict(self) -> Dict[str, List[str]]:
        return {
            "locations": self.locations,
            "quantities": self.quantities,
            "times": self.times,
            "topics": self.topics,
            "qualifiers": self.qualifiers,
        }


def extract_entities(text: str) -> ExtractedEntities:
    """Extract entities from raw transcript text."""
    low = text.lower()
    ent = ExtractedEntities()
    for loc in _KNOWN_LOCATIONS:
        if re.search(r"\b" + re.escape(loc) + r"\b", low):
            # Normalize bengaluru/bangalore variant
            ent.locations.append("bengaluru" if loc == "bangalore" else loc)
    ent.locations = sorted(set(ent.locations))
    for pat in _QUANTITY_PATTERNS:
        for m in re.finditer(pat, low):
            ent.quantities.append(re.sub(r"\s+", " ", m.group(0).strip()))
    ent.quantities = sorted(set(ent.quantities))
    for pat in _TIME_PATTERNS:
        for m in re.finditer(pat, low):
            ent.times.append(m.group(0).strip())
    ent.times = sorted(set(ent.times))
    for kw in _TOPIC_KEYWORDS:
        if re.search(r"\b" + re.escape(kw) + r"\b", low):
            ent.topics.append(kw)
    ent.topics = sorted(set(ent.topics))
    for kw in _QUALIFIER_KEYWORDS:
        if re.search(r"\b" + re.escape(kw) + r"\b", low):
            ent.qualifiers.append(kw)
    ent.qualifiers = sorted(set(ent.qualifiers))
    return ent
