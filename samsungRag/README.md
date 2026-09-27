# Streaming RAG — Retrieval Controller (Component 1)

## 1. Project overview
This folder contains the standalone Retrieval Controller service, which decides **WHEN**
retrieval should happen for progressively arriving user transcripts. The repository root
also includes the Component 2–5 prototypes and an offline integration pipeline.

## 2. Retrieval Controller responsibilities
- Accept partial/final transcripts (`StreamingInput`).
- Output `WAIT`, `RETRIEVE`, or `NO_RETRIEVAL` with reason codes, trigger, confidence.
- Detect intent stability, incompleteness, acknowledgements, presentation-only requests.
- Extract lightweight entities/constraints and compare against session state.
- Suppress duplicate retrieval; surface new constraints / query changes.
- Maintain isolated per-session state; emit structured telemetry.

## 3. WHEN vs WHAT to retrieve
- **Member 1 (this code) — WHEN:** `WAIT` (unstable/incomplete), `RETRIEVE` (stable/new info),
  `NO_RETRIEVAL` (chat/formatting only). Never decomposes queries or fetches documents.
- **Member 2 — WHAT:** splits a `RETRIEVE` query into subqueries/intents. Consumes
  `RetrievalDecision.query` via `DecomposerAdapter` (`app/retrieval_controller/adapters/decomposer_adapter.py`).
  A `MockDecomposerAdapter` is included so this service runs standalone.

## 4. Project structure
```
app/
  main.py                        # FastAPI factory, GET /health
  api/routes.py                  # POST /api/v1/retrieval-controller/decide (thin)
  shared/models.py               # StreamingInput, RetrievalDecision, DecisionType, TriggerType
  retrieval_controller/
    controller.py                # 16-step orchestration pipeline
    decision_engine.py           # pure WAIT/RETRIEVE/NO_RETRIEVAL rules
    intent_detector.py           # ack / presentation / incomplete / info-seeking heuristics
    entity_extractor.py          # location, quantity, time, topic, qualifier regexes
    query_comparator.py          # normalize, identical/meaningful/cosmetic diff
    state.py                     # SessionState + SessionStore + InMemorySessionStore
    policies.py                  # trigger precedence + dedup policy
    config.py / telemetry.py / exceptions.py / models.py
    adapters/decomposer_adapter.py  # Protocol + Mock (Member 2 boundary)
tests/  test_controller.py test_decision_engine.py test_state.py test_query_comparator.py test_api.py
```

## 5. Setup and installation
```powershell
python --version  # 3.11+
pip install -r requirements.txt
```

## 6. How to run the FastAPI server
```powershell
python -m uvicorn app.main:app --reload --port 8000
# GET http://localhost:8000/health
# POST http://localhost:8000/api/v1/retrieval-controller/decide
```

## 7. How to run tests
```powershell
python -m pytest -v
# Expected: 31 passed (health, controller examples 1-7, engine, state, comparator, API)
```

## 8. API request and response examples
Request:
```json
{"session_id":"S001","turn_id":"T001","transcript":"I need a workshop venue in Pune","is_final":false,"timestamp":1726800000.0,"language":"en","metadata":{}}
```
Response:
```json
{"schema_version":"1.0","session_id":"S001","turn_id":"T001","decision":"RETRIEVE","query":"I need a workshop venue in Pune","confidence":0.87,"reason_codes":["INTENT_STABLE"],"trigger":"PROVISIONAL","is_final":false,"timestamp":1726800000.0,"state_version":1,"metadata":{}}
```
Decisions: `WAIT` + `INCOMPLETE_INTENT` for `"I need to..."`; `NO_RETRIEVAL` + `PRESENTATION_ONLY`
for `"Put the previous answer in bullet points"`; `NO_RETRIEVAL` + `CONVERSATIONAL_RESPONSE`
for `"Okay, thanks"`; `RETRIEVE` + `NEW_CONSTRAINT` when `for 30 people` is added.

## 9. Decision logic explanation
Pipeline: validate → normalize → empty→WAIT → ack→NO_RETRIEVAL → presentation→NO_RETRIEVAL →
truncated/incomplete→WAIT → non-informational→NO_RETRIEVAL → session compare
(identical→suppress as `QUERY_UNCHANGED`; new constraints→`NEW_CONSTRAINT`;
meaningful change→`QUERY_CHANGE`; cosmetic→`COSMETIC_CHANGE`) → else stable→`RETRIEVE`
(`INTENT_STABLE`). Triggers: `NEW_CONSTRAINT` > `QUERY_CHANGE` > `FINAL`/`PROVISIONAL`;
`NONE` when no retrieval. Confidence values are fixed heuristic weights, not calibrated probabilities.

## 10. Integration instructions
- **Component 2 (Decomposer):** use `component2/adapter.py::RealDecomposerAdapter` to implement `DecomposerAdapter.decompose(DecomposerRequest) -> DecomposerResponse`
  (see `adapters/decomposer_adapter.py`), pass it to `RetrievalController(decomposer=...)` or call the
  HTTP endpoint and decompose `RetrievalDecision.query`. Test with `MockDecomposerAdapter`.
- **Component 3 (Retrieval):** trigger actual search only when `decision == RETRIEVE`, using `query`,
  `reason_codes`, `trigger`, `session_id/turn_id`.
- **Component 4 (Synthesis/UI/Telemetry):** render `WAIT` as listening state, `NO_RETRIEVAL` as chat-only,
  `RETRIEVE` with progress; log the telemetry event
  (`session_id, turn_id, decision, confidence, reason_codes, trigger, processing_time_ms, state_version, query_changed, new_constraints`).
- State: replace `InMemorySessionStore` with Redis/DB via the `SessionStore` Protocol for multi-process deploys.

## 11. Known limitations
- Regex/keyword heuristics only; no semantic understanding, negation, or robust multilingual support.
- Short imperatives (e.g. 2-token "List venues") retrieve provisionally; may over-retrieve on fragments.
- "please"-style additions can cross the Jaccard 0.85 threshold and count as `QUERY_CHANGE`.
- In-memory state is per-process; not suitable for multi-process production.
- Confidence scores are explainable constants, not calibrated.

## 12. Future improvements
- Embedding-based stability / semantic diff; stronger entity extraction (spaCy/LLM, Member 2 assist).
- Redis-backed `SessionStore`, debouncing/throttling for streaming bursts.
- Calibration of confidence, per-session policy tuning, `CONTEXTUAL` trigger enrichment.
- Contract tests with Members 2–4; OpenAPI examples; Docker + CI.
