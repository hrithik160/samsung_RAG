"""Heuristic intent detector. Explainable, deterministic, imperfect by design.

Limitations: regex/keyword heuristics only; no semantic understanding,
no multilingual support beyond keyword lists, sarcasm/ellipsis not handled.
Each signal returns reason codes so callers can explain decisions.
"""

import re
from dataclasses import dataclass, field
from typing import List

_ACK_PATTERNS = [
    r"^(ok|okay|okay\s*,?\s*thanks?|thanks?|thank\s*you|got\s*it|great|perfect|nice|cool|alright|awesome|yes|yeah|yep|sure)\b[\s!.]*$",
]

_ACK_KEYWORDS = {
    "thanks", "thank you", "got it", "okay", "ok", "great", "perfect",
    "nice", "cool", "alright", "awesome",
}

_PRESENTATION_PATTERNS = [
    r"\bbullet\s*points?\b",
    r"\breformat\b",
    r"\bprevious\s+(answer|response|result|output|reply)\b",
    r"\blast\s+(answer|response|result|output|reply)\b",
    r"\babove\s+(answer|response|result|output)\b",
    r"\bmake\s+it\s+(shorter|longer|concise|clearer|simpler)\b",
    r"\bput\b.{0,40}\bin\b.{0,40}\b(table|bullets|points|format)\b",
    r"\bsummar(y|ize|ise)\b.{0,30}\b(above|previous|last|that)\b",
    r"\brewrite\b",
    r"\btranslate\b",
    r"\b(repeat|restate)\b.{0,40}\b(answer|response|that|above|previous|last)\b",
    r"\brephrase\b",
    r"\bsimplify\b",
]

_QUESTION_WORDS = {
    "what", "where", "when", "how", "why", "which", "who", "whom", "whose",
    "find", "search", "explain", "summarize", "summarise", "show", "list", "tell", "give", "need",
    "want", "looking", "describe", "compare", "recommend", "suggest",
}

_IMPERATIVE_VERBS = {
    "find", "show", "list", "explain", "summarize", "summarise", "tell", "give", "search",
    "describe", "compare", "recommend", "suggest", "need",
}

_INCOMPLETE_TRAILING = ("...", "…", "--", "—", "-", ",", " and", " or", " to", " for", " with", " in", " on", " about", " of")
_INCOMPLETE_OPENERS = (
    "i need to", "i want to", "i am trying to", "i was going to",
    "so i", "um", "uh", "like i",
)

# General grammatical signals (not an exact-phrase blocklist).
# A transcript ending in one of these function words is mid-utterance,
# e.g. "I need a", "Tell me about the", "Can you help me with".
_DANGLING_END_WORDS = frozenset({
    "a", "an", "the",
    "to", "for", "with", "in", "on", "about", "of", "from", "into",
    "and", "or",
    "me", "my", "our", "your", "their", "this", "that", "these", "those",
    "is", "are", "be", "am",
})

# Request openers used ONLY as prefix-strips: the remainder after the
# opener must contain a meaningful object, otherwise the request is
# unfinished (e.g. "I want to know about" -> remainder "" -> incomplete,
# but "I want to know about venues in Pune" -> remainder has object).
_OPENER_PREFIXES = (
    "i want to know about",
    "can you help me with",
    "could you help me with",
    "i am looking for",
    "im looking for",
    "i m looking for",
    "looking for",
    "tell me about",
    "find me",
    "show me",
    "give me",
    "i need",
    "i want",
    "help me with",
)

_STOPWORDS = frozenset({
    "i", "me", "my", "you", "your", "he", "she", "it", "we", "they",
    "a", "an", "the",
    "to", "for", "with", "in", "on", "about", "of", "from", "into",
    "and", "or",
    "is", "are", "be", "am", "can", "could", "would", "should",
    "do", "does", "please", "know",
})


@dataclass
class IntentSignal:
    """Explainable intent classification for one transcript."""

    is_acknowledgement: bool = False
    is_presentation_only: bool = False
    is_incomplete: bool = False
    is_information_seeking: bool = False
    has_actionable_request: bool = False
    reason_codes: List[str] = field(default_factory=list)


def detect_acknowledgement(normalized: str) -> bool:
    """True for short conversational replies like 'Okay, thanks'."""
    text = normalized.strip().lower()
    if not text:
        return False
    for pat in _ACK_PATTERNS:
        if re.search(pat, text):
            return True
    # Short text dominated by ack keywords, e.g. "ok thanks!"
    if len(text.split()) <= 4 and any(k in text for k in _ACK_KEYWORDS):
        # Avoid misclassifying "thanks for the venues in Pune" (has entities/intent)
        if not any(w in text for w in ("find", "need", "want", "show", "list", "explain", "policy", "venue", "where", "what")):
            return True
    return False


def detect_presentation_only(normalized: str) -> bool:
    """True for formatting requests about an already-generated answer."""
    text = normalized.lower()
    return any(re.search(p, text) for p in _PRESENTATION_PATTERNS)


def detect_incomplete(normalized: str) -> bool:
    """True when the transcript looks like an unfinished streaming chunk."""
    text = normalized.strip()
    if not text:
        return True
    low = text.lower()
    # Generic dangling-ending check at ANY length: "I need a", "Find me a".
    # This runs before the length gate (the old code only checked tails
    # for transcripts >= 12 chars, which missed short fragments).
    if ends_with_dangling_word(low):
        return True
    if low in ("i", "i need to", "i need", "i want", "i want to", "um", "uh"):
        return True
    # Opener with no meaningful object remainder, e.g. "I want to know about".
    stripped_opener, matched = strip_opener_prefix(low)
    if matched and not has_meaningful_object(low):
        return True
    if len(text) < 12 or len(text.split()) < 3:
        # Very short fragments are incomplete unless they are a crisp
        # imperative ("List venues") — checked by caller via info-seeking path.
        # Keep heuristic: short + no question/imperative structure => incomplete.
        if low.endswith("...") or low.endswith("…"):
            return True
        # Fall through: let decision engine decide via token thresholds.
        return True if len(text.split()) < 2 else False
    for opener in _INCOMPLETE_OPENERS:
        if low == opener or low.startswith(opener + " ") and len(low.split()) <= 4:
            # e.g. "I need to..." with nothing after
            remainder = low[len(opener):].strip(" .…")
            if not remainder or remainder in ("...",):
                return True
    stripped = low.rstrip(".…!? ")
    for tail in _INCOMPLETE_TRAILING:
        if stripped.endswith(tail):
            return True
    # Ends mid-sentence on filler
    if re.search(r"\b(um+|uh+)\s*$", low):
        return True
    return False


def ends_with_dangling_word(text: str) -> bool:
    """True if the last token is a function word that demands continuation."""
    toks = re.findall(r"[a-z']+", text.lower())
    return bool(toks) and toks[-1] in _DANGLING_END_WORDS


def strip_opener_prefix(text: str) -> tuple:
    """Strip the longest known request-opener prefix; return (remainder, matched)."""
    low = text.lower().strip()
    best = ""
    for prefix in _OPENER_PREFIXES:
        if low == prefix:
            if len(prefix) > len(best):
                best = prefix
        elif low.startswith(prefix + " "):
            if len(prefix) > len(best):
                best = prefix
    if not best:
        return low, False
    return low[len(best):].strip(" .…!?,;:"), True


def has_meaningful_object(text: str) -> bool:
    """True when the request contains a content-bearing object.

    Strips a leading request opener, then requires at least one
    non-stopword token (length >= 3). This is deliberately generic:
    any opener + bare article/preposition ("I need a", "Tell me
    about the") has no object, while opener + subject ("I want to
    know about venues in Pune") does.
    """
    low = text.lower().strip()
    if not low:
        return False
    remainder, matched = strip_opener_prefix(low)
    scope = remainder if matched else low
    toks = re.findall(r"[a-z']+", scope)
    return any(t not in _STOPWORDS and len(t) >= 3 for t in toks)


def is_provisionally_incomplete(transcript: str) -> bool:
    """Conservative streaming-completeness gate for is_final=false.

    Returns True when a provisional transcript is grammatically
    unfinished or lacks a meaningful request object. Used by the
    decision engine BEFORE the generic INTENT_STABLE check.
    Complete requests (opener + real object) return False here.
    """
    raw = (transcript or "").strip()
    if not raw:
        return True
    low = raw.lower()
    if ends_with_dangling_word(low):
        return True
    if not has_meaningful_object(low):
        # No content object: bare openers ("I", "I need", "I want to"),
        # lone verbs, or pure function-word strings are incomplete.
        # Chit-chat without request verbs is handled by the caller's
        # non-informational path; this gate targets request-like input.
        toks = set(re.findall(r"[a-z']+", low))
        if toks & (_QUESTION_WORDS | {"i", "help", "know", "looking"}):
            return True
        if len(re.findall(r"[a-z']+", low)) <= 2:
            return True
    return False


def detect_information_seeking(normalized: str) -> bool:
    """True for questions, instructions, or information-seeking requests."""
    text = normalized.strip().lower()
    if not text:
        return False
    if text.endswith("?"):
        return True
    tokens = set(re.findall(r"[a-z']+", text))
    if tokens & _QUESTION_WORDS:
        # Require a subject/task, not just the word "need" alone.
        if len(tokens) >= 3:
            return True
        first = text.split()[0] if text.split() else ""
        if first in _IMPERATIVE_VERBS:
            return True
    first = text.split()[0] if text.split() else ""
    if first in _IMPERATIVE_VERBS and len(text.split()) >= 2:
        return True
    return False


def detect_intent(normalized: str) -> IntentSignal:
    """Run all detectors and return an explainable signal."""
    signal = IntentSignal()
    if detect_acknowledgement(normalized):
        signal.is_acknowledgement = True
        signal.reason_codes.append("CONVERSATIONAL_RESPONSE")
    if detect_presentation_only(normalized):
        signal.is_presentation_only = True
        if "PRESENTATION_ONLY" not in signal.reason_codes:
            signal.reason_codes.append("PRESENTATION_ONLY")
    if detect_incomplete(normalized):
        signal.is_incomplete = True
        signal.reason_codes.append("INCOMPLETE_INTENT")
    if detect_information_seeking(normalized):
        signal.is_information_seeking = True
        signal.has_actionable_request = True
        signal.reason_codes.append("INFO_SEEKING")
    return signal
