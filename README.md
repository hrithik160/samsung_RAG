# Samsung Streaming RAG

This repository implements a streaming, session-aware retrieval-augmented generation pipeline. The default assistant uses Gemini 3.6 Flash for decomposition and answer generation, Gemini embeddings for semantic retrieval, and BM25 rank fusion. Offline stand-ins remain available for evaluation and development.

## Live assistant (Windows)

1. Get a Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey).
2. Copy `.env.example` to `.env` and set `GEMINI_API_KEY` in `.env`.
3. Put `.txt`, `.md`, `.markdown`, or text-based `.pdf` documents in `documents/` (subfolders are supported).
4. Start the interactive assistant:

```powershell
.\run.ps1
```

The first launch embeds and indexes the documents using `gemini-embedding-001`; the local vector index is cached in `component3/index_store/`. Individual document embeddings are checkpointed in SQLite as they finish, so restarting after a quota interruption reuses completed work. The default limiter embeds at most 60 text chunks per minute in batches of 8 and sends at most 10 generation requests per minute. Adjust `GEMINI_EMBEDDING_ITEMS_PER_MINUTE`, `GEMINI_EMBEDDING_BATCH_SIZE`, and `GEMINI_GENERATION_REQUESTS_PER_MINUTE` in `.env` to stay below your account's quotas. Transient 429/5xx errors honor Gemini's retry delay and use bounded backoff. Later launches reuse the index unless a source file changes. The assistant asks Gemini 3.6 Flash to decompose complex requests and synthesize answers from retrieved evidence. Citations are checked against the retrieved document and section cache before display. The default model IDs can be changed with `GEMINI_MODEL` and `GEMINI_EMBEDDING_MODEL` in `.env`.

The setup script creates `.venv` with Python 3.10 and installs the pinned dependencies from `requirements-lock.txt`. To choose another corpus folder:

```powershell
.\.venv\Scripts\python.exe demo_full_pipeline.py --corpus-dir .\my_corpus --events .\component5\out\my_session.jsonl
```

PDF citations use page numbers; Markdown citations use heading section numbers. Scanned PDFs need OCR first. Source documents are read from your local folder, then document chunks and questions are sent to the configured Gemini API for embedding and generation. API usage can incur charges under your Google account.

## Offline demo and evaluation

```powershell
.\run.ps1 -Offline -Demo
.\run.ps1 -Eval
```

The offline path uses the hand-authored `component5/demo_corpus/` fixtures and makes no Gemini requests. Evaluation outputs are fixture evidence, not production-model benchmarks.

## Docker Quickstart

To run the project in a clean, reproducible container (satisfying the Reproducibility Gate), ensure Docker is installed and run:

```bash
# Run the evaluation suite (Clean Machine Test)
docker compose up eval --build

# Run the offline demo
docker compose up demo --build
```

This bypasses the Windows-specific `.ps1` runner and cleanly executes inside a lightweight Linux container.

## Components

| Component | Role | Location |
|---|---|---|
| 1. Retrieval controller | Decide when to wait, retrieve, or suppress retrieval | `samsungRag/` |
| 2. Multi-intent decomposer | Resolve follow-ups and split compound queries | `component2/` |
| 3. Corpus retrieval and fusion | Dense/BM25 retrieval, fusion, reranking, source metadata | `component3/` |
| 4. Session synthesis | Reuse, refine, version, and cite answers | `component4/` |
| 5. Telemetry and evaluation | JSONL traces, benchmark metrics, ablations | `component5/` |

The pipeline is `WAIT`/`NO_RETRIEVAL`/`RETRIEVE` → decomposition → hybrid retrieval and fusion → session synthesis. `[Doc_ID §Section]` citations are checked against retrieved session evidence; missing or uncited factual answers fail closed. Session memory is in-process and ephemeral.

## Evaluation

```powershell
.\.venv\Scripts\python.exe component5\run_eval.py
```

This generates `component5/out/turn_log.jsonl`, `summary.csv`, `retrieval_ablations.csv`, `acceptance_gates.csv`, `eval_summary.png`, and `benchmark_report.md`. The labelled evaluation corpus includes Samsung examples and guide-inspired venue/travel fixtures. The integration demo corpus is in `component5/demo_corpus/`. Results are smoke-test evidence, not production benchmarks; the report states that limitation and lists the measured edge cases.

The evaluation compares always-retrieve, controller/decomposer/retriever, and full session-aware variants. It also compares hybrid, dense-only, and sparse-only retrieval with reranking on/off. Six acceptance gates are reported, including explicit clean-machine replay status. The demo corpus in `component5/demo_corpus/` is hand-authored training data, not an approved source corpus.

## Deliverables and limitations

- `SYSTEM_ARCHITECTURE.md`: concise architecture brief.
- `demo_recording_script.md`: timed script for the requested five-minute video; the video itself must be recorded by the project team.
- `component5/telemetry_schema.json`: JSON Schema for integrated turn events.
- `component5/out/benchmark_report.md`: generated benchmark and gate results.

Session state uses an in-memory store, so it does not survive process restarts or coordinate multiple workers. Direct dependency versions and their resolved Windows/Python 3.10 dependencies are pinned in `requirements.txt` and `pylock.toml`. The Gemini API adapter uses `GEMINI_API_KEY` from `.env`; do not commit that file. Citation provenance is checked, but semantic entailment on a real corpus still needs evaluation and human review before production deployment.

## Existing component checks

Run the component-specific checks described in each `component*/README.md` and `samsungRag/README.md` when verification is requested. This change set has not run those tests.
