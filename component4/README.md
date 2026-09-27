# Component 4 – Session-Aware Synthesis

Manages conversational memory across turns. Newly introduced constraints
mutate existing statements (via targeted delta retrieval) instead of
discarding session history or replaying a full-corpus search; purely
cosmetic requests ("put that in bullet points") bypass retrieval entirely
and just restyle the cached answer, citations unchanged.

## Setup & Testing

```powershell
python test_component4.py     # offline unit tests, fake retriever, no downloads needed
python session_synthesis.py   # 3-turn demo: factual -> cosmetic -> factual refinement
```

Stdlib only — no hard dependencies.

## Architecture

```
utterance --> TurnClassifier --cosmetic--> restyle cached answer, reuse citations, NO retrieval
                   |
              factual revision
                   v
        delta or fresh retrieval (via any Component-3-shaped retriever)
                   v
        merge new chunks into SessionState.cached_chunks
                   v
        Synthesizer.synthesize(instruction, active_claims, new_chunks)
                   v
        validate_and_strip_citations()  -- drop any [Doc_ID §Section] not in the cache
                   v
        SynthesisTurn  (+ SessionState.answer_version bumped on factual turns only)
```

* **`SessionState`** — `session_id`, `answer_version`, `cached_chunks`
  (chunk_id → text/doc/section), `active_claims` (versioned, cited
  statements), `last_answer`, `last_query`. Swap `SessionStore` for a
  Redis-backed one later without touching callers.
* **`TurnClassifier`** — rule-based first pass matching the design doc's
  "structured classifier prompt" behaviour: cosmetic-pattern regex (bullet
  points, reformat, shorten, table, translate, ...) vs. token-diff against
  `last_query` to detect a genuinely new constraint. Swap in an LLM
  classifier later behind the same `.classify()` signature.
* **`OfflineSynthesizer` / `LLMSynthesizer`** — same pluggable-backend
  pattern as Component 2's decomposer. The offline one concatenates cited
  claim text + newly retrieved chunks deterministically and formats bullet
  lists. It retains the original answer for shortening/translation requests
  when it cannot safely transform the text. Configure an LLM backend for
  those transformations; it prompts for strict `[Doc_ID §Section]` citations.
* **`validate_and_strip_citations`** — regex-extracts every citation tag,
  keeps it only if its `Doc_ID` is present in the session's cached chunks,
  and reports `dropped_citations` for telemetry — this is the "strip
  hallucinated tags" grounding gate from the design doc.

## Plugging into Component 3 for delta retrieval

Anything with an `async def retrieve(query) -> result-with-.hits` works —
Component 3's `CorpusRetriever` fits directly:

```python
import sys; sys.path.append("../component3")
from retriever import CorpusRetriever  # already-built index/config

from session_synthesis import SessionAwareSynthesizer
synth = SessionAwareSynthesizer()

turn = await synth.process_turn("sess1", "T4", "what about international travel?", retriever=my_corpus_retriever)
print(turn.answer, turn.citations, turn.dropped_citations)
```

## Telemetry

`SynthesisTurn.to_event()` gives a JSON-safe record — `retrieval_required`,
`is_cosmetic`, `new_constraint`, `version`, kept/dropped citations — for the
Phase 5 telemetry log.
