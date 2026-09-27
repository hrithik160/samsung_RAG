# Streaming Live RAG — Architecture Brief

## Objective

The system accepts incrementally arriving transcript text and retrieves only when the intent is stable enough to search. It decomposes compound requests, retrieves evidence from a closed local corpus, maintains answer state for the active session, and emits cited answers or an explicit uncertainty response.

## Request path

1. **Controller (`samsungRag/`)** receives a `StreamingInput` with session and turn IDs, timestamp, transcript, and finality. It returns `WAIT`, `RETRIEVE`, or `NO_RETRIEVAL` with reason and trigger metadata.
2. **Decomposer (`component2/`)** resolves follow-ups and splits compound requests into independently searchable sub-queries.
3. **Retriever (`component3/`)** embeds local document chunks and queries with Gemini Embedding, runs dense and BM25 retrieval, fuses rankings, deduplicates results, and preserves `Doc_ID`, section, and chunk IDs. The persistent local vector index is refreshed when source files change.
4. **Session synthesizer (`component4/`)** uses Gemini 3.6 Flash to apply factual deltas or restyle cached answers. Each factual turn advances the answer version. Unknown citations and uncited generated sentences fail closed.
5. **Telemetry and evaluation (`component5/`)** record turn decisions, evidence references, versions, latency, token estimates, and model cost. The harness compares end-to-end variants and retrieval ablations against labelled fixtures.

## State and trust boundaries

Conversation state is keyed by `session_id` and held in memory by default. It is ephemeral and process-local; multi-process deployment needs a session-store implementation with the same interface. Corpus content is the only allowed evidence source. The offline synthesizer quotes retrieved sentences directly. The optional LLM adapters are extension points; citation allow-list checks establish source provenance, but production semantic support still requires reviewed entailment evaluation.

The index builder reads local files from `documents/` by default: text, Markdown, and text-based PDF. PDF citations use one-based page numbers; Markdown citations use one-based heading-section numbers; plain text uses section 1. Scanned PDFs need OCR before ingestion. Gemini API calls require `GEMINI_API_KEY`; source chunks are sent for embedding and retrieved evidence is sent to Gemini 3.6 Flash for answer generation.

## Failure behavior

- Incomplete or unstable input: wait for more text.
- Chat acknowledgement: do not retrieve.
- Formatting-only request: reuse the cached answer and citations.
- No hits, low reranker confidence, or missing citation coverage: return an explicit uncertainty response.
- Unknown document/section citation: strip it; if factual sentences remain uncited, fail closed.

## Run and evidence

From a clean Windows checkout with Python 3.10, add a Gemini API key to `.env`, put supported source files in `documents/`, and run `./run.ps1` for the interactive assistant. The first run embeds and caches the corpus; changed documents trigger an index rebuild. To use a different corpus, pass `-CorpusDir PATH` to `run.ps1` or `--corpus-dir PATH` to the Python entry point. `./run.ps1 -Offline -Demo` runs the deterministic fixture conversation. Run `./run.ps1 -Eval` for the labelled offline benchmark.

The offline report is fixture evidence only. Gemini-backed runs use live API calls and require valid credentials. Before production claims, evaluate on the intended corpus, replay setup on a clean machine, review observed edge failures, and record the five-minute demonstration.
