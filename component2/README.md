# Component 2 – Multi-Intent Decomposer

Routing brain between raw user input and Component 3. Detects whether a query
carries one intent or several bundled sub-questions, and — if several —
splits it into atomic, independently-answerable sub-queries with pronouns
and implicit references resolved into explicit entities.

## Setup & Testing

```powershell
python test_component2.py     # offline unit tests, no downloads / API keys required
python decomposer.py          # small interactive-style demo, prints decomposition traces
```

There are no hard dependencies (stdlib only), so this runs anywhere Python 3.10+ runs.

## Architecture

```
query --> ComplexityGate --simple--> single SubQueryItem, passthrough (no LLM call)
              |
           complex
              v
   LLMDecomposer (if configured) --valid JSON--> SubQueryItem list
              |  malformed / no backend
              v
   RuleBasedDecomposer (offline, deterministic fallback)
              v
   validate_subqueries()  -- rejects empty / lossy splits, else keeps result
              v
   DecompositionResult
```

* **`ComplexityGate`** — cheap regex pre-filter (conjunctions joining two
  clause-like chunks, comparative language, multiple `?`, `;`). Trivial
  single-intent queries never reach an LLM.
* **`RuleBasedDecomposer`** — zero-dependency offline backend, used directly
  in tests/CI and as the automatic fallback when the LLM path fails.
* **`LLMDecomposer`** — wraps any `llm_call(prompt: str) -> str` (OpenAI,
  Anthropic, a local model, whatever your team already has). Enforces a
  strict JSON-array shape, retries once on malformed output, then raises so
  the façade can fall back.
* **`validate_subqueries`** — token-coverage check against the original
  query; catches silent truncation or off-topic splits before they reach
  Component 3.
* **`resolve_references` / session memory in `MultiIntentDecomposer`** — a
  light entity tracker so a bare follow-up like `"what about in Pune?"`
  resolves against the previous turn's entity.

## Plugging in a real LLM

```python
from decomposer import MultiIntentDecomposer, LLMDecomposer

def call_my_llm(prompt: str) -> str:
    # e.g. an Anthropic/OpenAI client call that returns response text
    ...

decomposer = MultiIntentDecomposer(llm_backend=LLMDecomposer(call_my_llm))
result = decomposer.decompose(
    "What's the difference between the S24 and S24 Ultra cameras, and does the base model support the S Pen?",
    session_id="sess1", turn_id="T3",
)
for sq in result.sub_queries:
    print(sq.intent_type, "->", sq.sub_query)
```

## Wiring into Component 3

`SubQueryItem.to_subquery_kwargs()` builds the exact kwargs Component 3's
`schema.SubQuery` expects, so no shared import is required at module load
time:

```python
import sys; sys.path.append("../component3")
from schema import SubQuery

subqueries = [SubQuery(**item.to_subquery_kwargs(qid=f"{turn_id}.{i}"))
              for i, item in enumerate(result.sub_queries)]
retrieval_result = await retriever.retrieve(subqueries)
```

See `../demo_full_pipeline.py` for a runnable Component 2 → 3 → 4 chain.

## Telemetry

`DecompositionResult.to_event()` returns a JSON-safe dict (`used_llm`,
`gate_triggered`, `fallback_used`, per-sub-query intent/priority/depends_on)
matching the structured-logging shape the other components use.
