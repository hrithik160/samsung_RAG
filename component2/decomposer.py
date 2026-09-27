"""Component 2 - Multi-Intent Decomposer.

    Query --> [ComplexityGate] --gate says "simple"--> pass through as 1 sub-query
                     |
                gate says "complex"
                     v
            [Decomposer backend]  (LLMDecomposer in prod, RuleBasedDecomposer offline)
                     |
              [validate_subqueries]  -- reject/fallback on malformed or lossy output
                     v
            DecompositionResult

Run standalone:  python decomposer.py
"""
from __future__ import annotations

import json
import re
from typing import Optional, Protocol, Sequence

from schema import SubQueryItem, DecompositionResult

_WORD = re.compile(r"[a-zA-Z0-9']+")
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "of", "for", "to", "and", "or",
    "in", "on", "at", "it", "its", "this", "that", "what", "which", "who", "how",
    "do", "does", "did", "with", "me", "i", "my", "our", "we", "you", "your",
    "please", "can", "could", "would", "tell", "about",
}

# --------------------------------------------------------------------------- complexity gate
_CONJUNCTIONS = re.compile(r"\b(and|as well as|also)\b", re.I)
_COMPARATIVE = re.compile(r"\b(vs\.?|versus|compare(d)?|difference between|better than|which is)\b", re.I)
_MULTI_QMARK = re.compile(r"\?.*\?")


class ComplexityGate:
    """Cheap rule-based pre-filter (spaCy-style heuristics, kept regex-only so
    this module has zero hard dependencies). Trivially simple, single-intent
    queries short-circuit here and never reach the LLM."""

    def is_complex(self, query: str) -> bool:
        q = query.strip()
        if not q:
            return False
        if _MULTI_QMARK.search(q):
            return True
        if _COMPARATIVE.search(q):
            return True
        if _CONJUNCTIONS.search(q):
            # "and" alone is noisy ("mac and cheese") -- require it to be
            # joining two clause-like chunks (both sides have >= 2 tokens).
            for m in _CONJUNCTIONS.finditer(q):
                left, right = q[: m.start()].strip(), q[m.end():].strip()
                if len(_WORD.findall(left)) >= 2 and len(_WORD.findall(right)) >= 2:
                    return True
        if q.count(";") >= 1:
            return True
        return False


# --------------------------------------------------------------------------- intent classification (shared heuristic)
def classify_intent(text: str) -> str:
    t = text.lower()
    if _COMPARATIVE.search(t):
        return "comparison"
    if t.startswith(("what is", "what are", "define", "who is", "who was")):
        return "definition"
    if re.search(r"\b(how many|how much|percent|%|\d)\b", t):
        return "numerical"
    if t.startswith(("it", "that", "those", "what about", "and")) or len(_WORD.findall(t)) <= 3:
        return "followup"
    return "factual"


# --------------------------------------------------------------------------- pronoun / ellipsis resolution
def resolve_references(text: str, last_entity: Optional[str]) -> str:
    """Best-effort resolution of pronouns / implicit references back to the
    last known entity, so each sub-query stays self-contained. This is a
    heuristic stand-in for the LLM's own resolution in the production path."""
    if not last_entity:
        return text
    t = text.strip()
    pronoun_lead = re.match(r"^(it|that|this|those|they)\b(.*)", t, re.I)
    if pronoun_lead:
        return f"{last_entity}{pronoun_lead.group(2)}"
    if re.match(r"^(what about|and)\b", t, re.I):
        rest = re.sub(r"^(what about|and)\b", "", t, flags=re.I).strip()
        return f"{last_entity} {rest}".strip()
    return t


# --------------------------------------------------------------------------- backends
class DecomposerBackend(Protocol):
    def split(self, query: str, last_entity: Optional[str] = None) -> list[SubQueryItem]:
        """Return >=1 SubQueryItem. Raise on unrecoverable failure."""


class RuleBasedDecomposer:
    """Offline, deterministic backend -- no LLM calls, no downloads. Used for
    unit tests / CI and as the guaranteed fallback when the LLM backend
    returns malformed JSON."""

    def split(self, query: str, last_entity: Optional[str] = None) -> list[SubQueryItem]:
        q = resolve_references(query.strip(), last_entity)
        parts = self._split_clauses(q)
        entity = self._single_named_entity(q) or last_entity
        items = []
        for i, part in enumerate(parts):
            part = part.strip(" ?.,;")
            if not part:
                continue
            if entity and (entity.casefold() in q.casefold() or entity == last_entity) and entity.casefold() not in part.casefold():
                part = f"{entity} {part}"
            items.append(SubQueryItem(
                sub_query=part,
                intent_type=classify_intent(part),
                priority=i + 1,
            ))
        return items or [SubQueryItem(sub_query=q or query, intent_type=classify_intent(q or query))]

    @staticmethod
    def _single_named_entity(query: str) -> Optional[str]:
        spans = re.findall(r"\b[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*\b", query)
        spans = [s for s in spans if s.split()[0].lower() not in {"is", "are", "what", "how", "which", "does", "actually"}]
        # Multiple distinct proper-noun groups are ambiguous (for example,
        # a comparison), so only propagate an unambiguous entity.
        unique = list(dict.fromkeys(spans))
        return unique[0] if len(unique) == 1 else None

    @staticmethod
    def _split_clauses(q: str) -> list[str]:
        # comparative phrasing: "X vs Y", "difference between X and Y" -> two clauses
        m = re.search(r"difference between (.+?) and (.+)", q, re.I)
        if m:
            return [f"{m.group(1).strip()}", f"{m.group(2).strip()}"]
        m = re.search(r"(.+?)\s+(?:vs\.?|versus)\s+(.+)", q, re.I)
        if m:
            return [m.group(1).strip(), m.group(2).strip()]
        # generic conjunction / semicolon split, but only on clause-like joins
        pieces = re.split(r"\s*;\s*|\b(?:and|as well as|also)\b", q, flags=re.I)
        pieces = [p for p in pieces if len(_WORD.findall(p)) >= 2]
        return pieces if len(pieces) >= 2 else [q]


class LLMDecomposer:
    """Wraps a real LLM call. `llm_call(prompt: str) -> str` should return the
    model's raw text; this class handles prompting, JSON parsing, Pydantic-
    style validation and a single retry on malformed output before raising."""

    SYSTEM_PROMPT = (
        "You split a user query into atomic, independently-answerable "
        "sub-queries. Resolve pronouns and implicit references (e.g. 'it', "
        "'that year') into explicit entities. Return ONLY a JSON array, no "
        "prose, no markdown fences. Each element: "
        '{"sub_query": str, "intent_type": str, "depends_on": int|null, "priority": int}. '
        "intent_type must be one of factual, comparison, definition, numerical, followup."
    )

    FEWSHOT = (
        "Query: \"What's the difference between the S24 and S24 Ultra cameras, "
        "and does the base model support the S Pen?\"\n"
        "[\n"
        '  {"sub_query": "S24 camera specifications", "intent_type": "factual", "depends_on": null, "priority": 1},\n'
        '  {"sub_query": "S24 Ultra camera specifications", "intent_type": "factual", "depends_on": null, "priority": 1},\n'
        '  {"sub_query": "does the base Galaxy S24 support the S Pen", "intent_type": "factual", "depends_on": null, "priority": 2}\n'
        "]\n"
    )

    def __init__(self, llm_call):
        self.llm_call = llm_call

    def _prompt(self, query: str, last_entity: Optional[str]) -> str:
        ctx = f"\nPrior entity in this session (resolve pronouns against it): {last_entity}" if last_entity else ""
        return f"{self.SYSTEM_PROMPT}\n\n{self.FEWSHOT}\nQuery: \"{query}\"{ctx}\n"

    def split(self, query: str, last_entity: Optional[str] = None) -> list[SubQueryItem]:
        raw = self.llm_call(self._prompt(query, last_entity))
        items = self._parse(raw)
        if items is None:
            # one retry with a stricter reminder
            raw = self.llm_call(self._prompt(query, last_entity) + "\nReturn ONLY the JSON array.")
            items = self._parse(raw)
        if items is None:
            raise ValueError("LLM decomposer returned malformed JSON after retry")
        return items

    @staticmethod
    def _parse(raw: str) -> Optional[list[SubQueryItem]]:
        text = raw.strip()
        text = re.sub(r"^```(json)?|```$", "", text.strip(), flags=re.M).strip()
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            return None
        if not isinstance(data, list) or not data:
            return None
        out = []
        try:
            for i, obj in enumerate(data):
                out.append(SubQueryItem(
                    sub_query=obj["sub_query"],
                    intent_type=obj.get("intent_type", "factual"),
                    depends_on=obj.get("depends_on"),
                    priority=obj.get("priority", i + 1),
                ))
        except (KeyError, TypeError, ValueError):
            return None
        return out


# --------------------------------------------------------------------------- validation
def validate_subqueries(original_query: str, items: Sequence[SubQueryItem]) -> bool:
    """Coverage / sanity check: reject empty lists, and reject sets whose
    combined tokens don't roughly reconstruct the original question's scope
    (catches silent truncation / hallucinated splits)."""
    if not items:
        return False
    orig_tokens = {t for t in _WORD.findall(original_query.lower()) if t not in _STOPWORDS}
    if not orig_tokens:
        return True
    covered = {t for sq in items for t in _WORD.findall(sq.sub_query.lower()) if t not in _STOPWORDS}
    overlap = len(orig_tokens & covered) / len(orig_tokens)
    return overlap >= 0.5


# --------------------------------------------------------------------------- main entry point
class MultiIntentDecomposer:
    """Component 2 façade: gate -> backend -> validate -> DecompositionResult.

    `llm_backend` is optional; if omitted (or if it raises / fails
    validation), the offline `RuleBasedDecomposer` is used, so this class
    works with zero external dependencies out of the box.
    """

    def __init__(self, llm_backend: Optional[DecomposerBackend] = None):
        self.gate = ComplexityGate()
        self.llm_backend = llm_backend
        self.rule_backend = RuleBasedDecomposer()
        self._last_entity: dict[str, str] = {}   # session_id -> last resolved entity, for follow-ups

    def decompose(self, query: str, session_id: str = "default", turn_id: str = "T0") -> DecompositionResult:
        last_entity = self._last_entity.get(session_id)
        is_complex = self.gate.is_complex(query)

        if not is_complex:
            item = SubQueryItem(
                sub_query=resolve_references(query.strip(), last_entity) or query.strip(),
                intent_type=classify_intent(query),
            )
            result = DecompositionResult(
                original_query=query, session_id=session_id, turn_id=turn_id,
                sub_queries=[item], used_llm=False, gate_triggered=False,
            )
            self._remember_entity(session_id, result)
            return result

        used_llm, fallback = False, False
        items: list[SubQueryItem] = []
        if self.llm_backend is not None:
            try:
                candidate = self.llm_backend.split(query, last_entity)
                if validate_subqueries(query, candidate):
                    items, used_llm = candidate, True
                else:
                    fallback = True
            except Exception:
                fallback = True

        if not items:
            items = self.rule_backend.split(query, last_entity)
            if not validate_subqueries(query, items):
                items = [SubQueryItem(sub_query=query.strip(), intent_type=classify_intent(query))]

        result = DecompositionResult(
            original_query=query, session_id=session_id, turn_id=turn_id,
            sub_queries=items, used_llm=used_llm, gate_triggered=True, fallback_used=fallback,
        )
        self._remember_entity(session_id, result)
        return result

    def _remember_entity(self, session_id: str, result: DecompositionResult) -> None:
        # crude entity tracker: last capitalised token / longest noun-ish token
        # across this turn's sub-queries, used to resolve next turn's pronouns.
        text = " ".join(sq.sub_query for sq in result.sub_queries)
        caps = re.findall(r"\b[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*\b", text)
        caps = [c for c in caps if c.split()[0].lower() not in {"is", "are", "what", "how", "which", "does", "actually"}]
        if caps:
            self._last_entity[session_id] = max(caps, key=len)


if __name__ == "__main__":
    d = MultiIntentDecomposer()
    demo_queries = [
        "What is the battery capacity of the S24?",
        "What's the difference between the S24 and S24 Ultra cameras, and does the base model support the S Pen?",
        "Is it IP68 rated and how fast does it charge?",
        "What about in Pune?",
    ]
    for i, q in enumerate(demo_queries, 1):
        res = d.decompose(q, session_id="demo", turn_id=f"T{i}")
        print(f"\nQuery: {q}")
        print(f"  gate_triggered={res.gate_triggered} used_llm={res.used_llm} fallback={res.fallback_used}")
        for sq in res.sub_queries:
            print(f"    - [{sq.intent_type}] {sq.sub_query}")
