"""Component 4 - Session-Aware Synthesis.

    Utterance --> [TurnClassifier] --cosmetic--> restyle cached answer, reuse citations, NO retrieval
                        |
                   factual revision
                        v
              [delta or fresh retrieval]  (via an injected Component-3-shaped retriever)
                        v
              merge new chunks into session cache
                        v
              [Synthesizer]  -- LLM in prod, offline deterministic stand-in for tests
                        v
              [CitationValidator]  -- strip any [Doc_ID §Section] tag not in the cache
                        v
              SynthesisTurn (+ bumped answer_version)

Run standalone:  python session_synthesis.py
"""
from __future__ import annotations

import re
from typing import Optional, Protocol, Sequence

from schema import CachedChunk, Claim, SessionState, SynthesisTurn

_WORD = re.compile(r"[a-zA-Z0-9']+")
_STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "of", "for", "to", "and", "or",
    "in", "on", "at", "it", "its", "this", "that", "what", "which", "who", "how",
    "do", "does", "did", "with", "me", "i", "my", "our", "we", "you", "your",
    "please", "can", "could", "would", "tell", "about", "by",
}

CITATION_RE = re.compile(r"\[([\w\.\-]+)\s*§\s*([\w\.\-]+)\]")

# --------------------------------------------------------------------------- session store
class SessionStore:
    """In-memory session table (design doc: 'Python/Redis-based state store').
    Swap for a Redis-backed implementation later without changing callers --
    only `get`/`save` need to move over the wire."""

    def __init__(self):
        self._sessions: dict[str, SessionState] = {}

    def get(self, session_id: str) -> SessionState:
        if session_id not in self._sessions:
            self._sessions[session_id] = SessionState(session_id=session_id)
        return self._sessions[session_id]

    def save(self, state: SessionState) -> None:
        self._sessions[state.session_id] = state


# --------------------------------------------------------------------------- turn classification
_COSMETIC_PATTERNS = re.compile(
    r"\b(bullets?|bullet points?|repeat (the )?(last )?answer|reformat|rephrase|simplify|shorten|shorter|summar(y|ize)|"
    r"as a (list|table)|in a table|more concise|less formal|more formal|"
    r"restate|re-?write|make it (shorter|longer|simpler)|translate)\b",
    re.I,
)


class TurnClassifier:
    """Rule-based first pass (design doc: 'structured classifier prompt'); an
    LLM classifier can be dropped in later by swapping this class for one
    with the same `.classify()` signature."""

    def classify(self, utterance: str, session: SessionState) -> tuple[bool, Optional[str]]:
        """Returns (is_cosmetic, new_constraint_text_or_None)."""
        if _COSMETIC_PATTERNS.search(utterance) and session.last_answer:
            return True, None

        if not session.last_query:
            return False, utterance.strip()   # first turn: whole utterance is the "constraint"

        prev_tokens = {t for t in _WORD.findall(session.last_query.lower()) if t not in _STOPWORDS}
        cur_tokens = {t for t in _WORD.findall(utterance.lower()) if t not in _STOPWORDS}
        new_tokens = cur_tokens - prev_tokens
        overlap = len(cur_tokens & prev_tokens) / len(prev_tokens) if prev_tokens else 0.0

        if not new_tokens:
            # nothing new at all and not a cosmetic pattern -> treat as cosmetic/no-op
            return True, None
        if overlap >= 0.25:
            # Keep the utterance intact: removing "old" context tokens can
            # erase the entity and conjunctions needed to retrieve a delta.
            return False, utterance.strip()
        # looks like an unrelated fresh question
        return False, utterance.strip()


# --------------------------------------------------------------------------- retrieval adapter
class DeltaRetriever(Protocol):
    async def retrieve(self, query) -> "Any":
        """Shaped like Component 3's `CorpusRetriever.retrieve` -- accepts a
        str or list[SubQuery] and returns something with a `.hits` list of
        objects exposing chunk_id/doc_id/text/meta."""


# --------------------------------------------------------------------------- synthesis backends
class SynthesisBackend(Protocol):
    def synthesize(self, instruction: str, claims: Sequence[Claim], new_chunks: Sequence[CachedChunk]) -> str:
        ...

    def restyle(self, instruction: str, prior_answer: str) -> str:
        ...


class OfflineSynthesizer:
    """Deterministic, no-LLM stand-in for tests / offline demos."""

    def synthesize(self, instruction: str, claims: Sequence[Claim], new_chunks: Sequence[CachedChunk]) -> str:
        lines = []
        seen = set()

        def append_once(sentence: str) -> None:
            key = re.sub(r"\W+", " ", CITATION_RE.sub("", sentence).casefold()).strip()
            if key and key not in seen:
                seen.add(key)
                lines.append(sentence)

        for claim in claims:
            for sentence in re.split(r"(?<=[.!?])\s+(?!\[)", claim.text.strip()):
                if sentence.strip():
                    append_once(sentence.strip())
        for ch in new_chunks:
            citation = f"[{ch.doc_id} §{ch.section}]"
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", ch.text.strip()) if s.strip()]
            for sentence in sentences:
                append_once(f"{sentence.rstrip('.!?')} {citation}.")
        return " ".join(lines) if lines else "No relevant information found in the corpus."

    def restyle(self, instruction: str, prior_answer: str) -> str:
        # split on sentence boundaries, but never split right before a
        # trailing citation tag like "... $150/day. [Doc_9 §1]"
        split_re = r"(?<=[.!?])\s+(?!\[)"
        if re.search(r"bullet", instruction, re.I):
            sentences = [s.strip() for s in re.split(split_re, prior_answer) if s.strip()]
            return "\n".join(f"- {s}" for s in sentences)
        if re.search(r"short(en|er)", instruction, re.I):
            # The offline backend cannot safely paraphrase factual claims;
            # preserve the complete cited answer rather than dropping facts.
            return prior_answer
        return prior_answer


class LLMSynthesizer:
    """Wraps a real LLM call. `llm_call(prompt: str) -> str`."""

    def __init__(self, llm_call):
        self.llm_call = llm_call

    def synthesize(self, instruction: str, claims: Sequence[Claim], new_chunks: Sequence[CachedChunk]) -> str:
        evidence = "\n".join(f"[{c.text}] (cite as one of: {', '.join(c.citations)})" for c in claims)
        evidence += "\n" + "\n".join(
            f"[{ch.text}] (cite as: {ch.doc_id} §{ch.section})" for ch in new_chunks
        )
        prompt = (
            "Answer the user using ONLY the evidence below. Every factual sentence "
            "must end with an inline citation formatted strictly as [Doc_ID §Section]. "
            "Do not invent citations.\n\n"
            f"Evidence:\n{evidence}\n\nInstruction: {instruction}\n"
        )
        return self.llm_call(prompt)

    def restyle(self, instruction: str, prior_answer: str) -> str:
        prompt = (
            f"Reformat the following answer per this instruction: '{instruction}'. "
            "Keep every citation tag exactly as-is; do not add or remove facts.\n\n"
            f"Answer:\n{prior_answer}\n"
        )
        return self.llm_call(prompt)


# --------------------------------------------------------------------------- citation validation
def validate_and_strip_citations(answer: str, session: SessionState) -> tuple[str, list, list]:
    """Returns (cleaned_answer, kept_tags, dropped_tags). A tag is kept only
    if its Doc_ID exists among the session's cached chunk doc_ids."""
    known = session.known_citations()
    kept, dropped = [], []

    def _check(m: re.Match) -> str:
        doc_id, section = m.group(1), m.group(2)
        tag = f"[{doc_id} §{section}]" if section else f"[{doc_id} §]"
        if (doc_id, section) in known:
            kept.append(tag)
            return m.group(0)
        dropped.append(tag)
        return ""

    cleaned = CITATION_RE.sub(_check, answer)
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()
    return cleaned, kept, dropped


def has_uncited_sentences(answer: str) -> bool:
    """Fail closed when a factual output sentence has no inline citation."""
    without_tags = CITATION_RE.sub("", answer)
    cited_parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", answer) if p.strip()]
    bare_parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+", without_tags) if p.strip()]
    # Unmatched citation-like text is also an audit failure.
    if re.search(r"\[[^\]]*§[^\]]*\]", without_tags):
        return True
    return any(part and not CITATION_RE.search(cited_parts[i]) for i, part in enumerate(bare_parts))


def _session_entity_context(session: SessionState) -> Optional[str]:
    # When multiple retrieved document IDs share an unambiguous family prefix
    # (S24_Battery + S24_Stylus), use it to ground elliptical follow-ups.
    prefixes: dict[str, int] = {}
    for chunk in session.cached_chunks.values():
        prefix = chunk.doc_id.split("_", 1)[0]
        prefixes[prefix] = prefixes.get(prefix, 0) + 1
    frequent = sorted((p for p, n in prefixes.items() if n >= 2), key=lambda p: (-prefixes[p], p))
    if len(frequent) == 1:
        return frequent[0]
    # Otherwise recover a proper-name anchor from the latest factual query.
    candidates = re.findall(r"\b[A-Z][a-zA-Z0-9]*(?:\s+[A-Z][a-zA-Z0-9]*)*\b", session.last_query)
    candidates = [c for c in candidates if c.split()[0].lower() not in {"what", "is", "how", "does", "the", "and"}]
    return max(candidates, key=len) if candidates else None


# --------------------------------------------------------------------------- main façade
class SessionAwareSynthesizer:
    def __init__(self, store: Optional[SessionStore] = None, backend: Optional[SynthesisBackend] = None):
        self.store = store or SessionStore()
        self.backend = backend or OfflineSynthesizer()
        self.classifier = TurnClassifier()

    async def process_turn(
        self,
        session_id: str,
        turn_id: str,
        utterance: str,
        retriever: Optional[DeltaRetriever] = None,
    ) -> SynthesisTurn:
        session = self.store.get(session_id)
        is_cosmetic, new_constraint = self.classifier.classify(utterance, session)

        if is_cosmetic:
            raw_answer = self.backend.restyle(utterance, session.last_answer or "")
            cleaned, kept, dropped = validate_and_strip_citations(raw_answer, session)
            if dropped or has_uncited_sentences(cleaned):
                # A formatting request must not lose its source trail. If a
                # backend changes or drops citations, retain the validated
                # cached response unchanged.
                cleaned = session.last_answer
                cleaned, kept, _ = validate_and_strip_citations(cleaned, session)
            turn = SynthesisTurn(
                session_id=session_id, turn_id=turn_id, utterance=utterance,
                retrieval_required=False, is_cosmetic=True, new_constraint=None,
                answer=cleaned, citations=kept, dropped_citations=dropped,
                version=session.answer_version,
            )
            session.last_answer = cleaned
            self.store.save(session)
            return turn

        new_chunks: list[CachedChunk] = []
        uncertain = False
        if retriever is not None:
            query_text = utterance
            if session.last_query and new_constraint:
                anchor = _session_entity_context(session)
                if anchor and anchor.casefold() not in query_text.casefold():
                    query_text = f"{anchor} {query_text}"
            result = await retriever.retrieve(query_text)
            session.add_chunks(result.hits)
            new_chunks = [session.cached_chunks[h.chunk_id] for h in result.hits]
            uncertain = bool(getattr(result, "uncertainty_bypass", False) or not result.hits)

        if uncertain:
            raw_answer = (
                "I couldn't verify an answer to that request from the available corpus. "
                "Please provide another source or rephrase the question."
            )
        else:
            raw_answer = self.backend.synthesize(utterance, session.active_claims, new_chunks)
        cleaned, kept, dropped = validate_and_strip_citations(raw_answer, session)
        if not uncertain and has_uncited_sentences(cleaned):
            cleaned = (
                "I couldn't verify a fully cited answer to that request from the available corpus. "
                "Please provide another source or rephrase the question."
            )
            kept = []
            uncertain = True

        session.answer_version += 1
        claim = Claim(
            claim_id=f"{session_id}.{session.answer_version}",
            text=cleaned, citations=kept, version=session.answer_version,
        )
        # Retain the latest consolidated answer, rather than replaying every
        # historical version into subsequent refinement prompts.
        session.active_claims = [claim] if cleaned else []
        session.last_answer = cleaned
        session.last_query = utterance
        self.store.save(session)

        return SynthesisTurn(
            session_id=session_id, turn_id=turn_id, utterance=utterance,
                retrieval_required=True, is_cosmetic=False, new_constraint=new_constraint,
                answer=cleaned, citations=kept, dropped_citations=dropped,
                version=session.answer_version, uncertainty=uncertain,
        )


if __name__ == "__main__":
    import asyncio
    from dataclasses import dataclass, field

    @dataclass
    class _FakeHit:
        chunk_id: str
        doc_id: str
        text: str
        meta: dict = field(default_factory=dict)

    @dataclass
    class _FakeResult:
        hits: list

    class _FakeRetriever:
        """Stands in for Component 3's CorpusRetriever for this demo."""
        async def retrieve(self, query):
            if "international" in query.lower():
                return _FakeResult(hits=[_FakeHit("c2", "Doc_9", "International travel requires pre-approval from finance.", {"section": "3"})])
            return _FakeResult(hits=[_FakeHit("c1", "Doc_9", "Domestic travel reimbursement is capped at $150/day.", {"section": "1"})])

    async def demo():
        synth = SessionAwareSynthesizer()
        retriever = _FakeRetriever()

        t1 = await synth.process_turn("s1", "T1", "What is the travel reimbursement policy?", retriever)
        print("T1:", t1.answer, "| cosmetic:", t1.is_cosmetic)

        t2 = await synth.process_turn("s1", "T2", "put that in bullet points", retriever)
        print("T2:", t2.answer, "| cosmetic:", t2.is_cosmetic)

        t3 = await synth.process_turn("s1", "T3", "what about international travel reimbursement?", retriever)
        print("T3:", t3.answer, "| new_constraint:", t3.new_constraint)

    asyncio.run(demo())
