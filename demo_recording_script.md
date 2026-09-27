# Demo recording script (target length: 4 minutes)

Record a single terminal window after starting with `./run.ps1`. Keep the terminal font large enough to read. The demo uses isolated workshop, travel, and unsupported-query sessions. Use a fresh `--events` path so the trace contains only this run.

## 0:00–0:30 — Explain the flow

Show the repository root and say: “This demo sends transcript turns through the controller, decomposer, local retriever, session-aware synthesizer, and JSONL telemetry. The default corpus and models are offline fixtures.”

## 0:30–1:10 — WAIT and early retrieval

Show the incomplete “I need to know” turn returning `WAIT`, followed by a stable but non-final S24 battery question triggering provisional retrieval. Point to the controller decision and retrieval trigger in the output.

## 1:10–2:00 — Compound request and citations

Use the workshop venue query from the local corpus, followed by the compound cancellation and catering request. Show the decomposed sub-queries, fused evidence, and citations in `[Doc_ID §Section]` form. Explain that corpus citations map to local source sections.

## 2:00–2:50 — Late refinement

Add one new constraint to the same session. Show the answer version increasing and the previous supported fact remaining available with its citation.

## 2:50–3:20 — Presentation-only request

Ask for the answer in bullets. Show that the answer is restyled without a new retrieval call and citations remain unchanged.

## 3:20–4:00 — Uncertainty and telemetry

Use a question unsupported by the local corpus and show the explicit uncertainty response. Open the generated JSONL file and point to decision, trigger, citations, answer version, processing time, token estimates, and inference cost.

## Close

State the limits: these are offline fixtures, not production-quality embeddings or a semantic entailment guarantee. Name the corpus/model swap needed for the deployment evaluation.
