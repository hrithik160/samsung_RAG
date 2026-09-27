"""End-to-end demo, all four components wired together for real:

    Component 1 (Retrieval Controller) decides WAIT / RETRIEVE / NO_RETRIEVAL
                    v
    Component 2 (Multi-Intent Decomposer) splits a RETRIEVE query into sub-queries
                    v
    Component 3 (Corpus Retrieval & Fusion) searches the index for each sub-query
                    v
    Component 4 (Session-Aware Synthesis) merges evidence, cites, and remembers
                    across turns so a later cosmetic ask ("bullet points") or a
                    new constraint ("...for international travel") doesn't
                    re-run the whole pipeline from scratch.

Run:  python demo_full_pipeline.py
"""
from __future__ import annotations

import asyncio
import argparse
import importlib
import hashlib
import json
import os
import re
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
C1_DIR = os.path.join(BASE_DIR, "samsungRag")
C2_DIR = os.path.join(BASE_DIR, "component2")
C3_DIR = os.path.join(BASE_DIR, "component3")
C4_DIR = os.path.join(BASE_DIR, "component4")
sys.path.insert(0, BASE_DIR)
from observability import JsonlEventLogger  # noqa: E402


def _isolated_import(path: str, module_names: list[str]) -> dict:
    """Components 2/3/4 each ship their own bare-named `schema.py` (etc).
    Import one component's modules at a time, evicting stale same-named
    entries from `sys.modules` first, so they don't shadow one another.
    The returned module objects stay fully usable even after the cache is
    evicted for the *next* component's import."""
    for name in module_names:
        sys.modules.pop(name, None)
    old_path = sys.path[:]
    sys.path = [path] + [p for p in sys.path if p not in (C2_DIR, C3_DIR, C4_DIR)]
    try:
        return {name: importlib.import_module(name) for name in module_names}
    finally:
        sys.path = old_path


# Component 1 uses a proper "app.*" package (no bare-name clash with 2/3/4),
# so it can live on sys.path permanently, same as orchestrator.py does.
sys.path.append(C1_DIR)
from app.shared.models import StreamingInput                                    # noqa: E402
from app.retrieval_controller.controller import RetrievalController             # noqa: E402

c2 = _isolated_import(C2_DIR, ["schema", "decomposer", "adapter"])
c3 = _isolated_import(C3_DIR, ["schema", "models", "index", "retriever", "fusion"])
c4 = _isolated_import(C4_DIR, ["schema", "session_synthesis"])

RealDecomposerAdapter = c2["adapter"].RealDecomposerAdapter
SubQuery = c3["schema"].SubQuery
HybridIndex, chunk_document = c3["index"].HybridIndex, c3["index"].chunk_document
HashingEmbedder, TokenOverlapReranker = c3["models"].HashingEmbedder, c3["models"].TokenOverlapReranker
CorpusRetriever, RetrieverConfig = c3["retriever"].CorpusRetriever, c3["retriever"].RetrieverConfig
load_corpus = _isolated_import(C3_DIR, ["corpus"])["corpus"].load_corpus
SessionAwareSynthesizer = c4["session_synthesis"].SessionAwareSynthesizer
LLMSynthesizer = c4["session_synthesis"].LLMSynthesizer


def load_dotenv(path: str) -> None:
    """Load simple KEY=VALUE settings without replacing shell environment values."""
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as source:
        for raw in source:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key, value = key.strip(), value.strip().strip("\"'")
            if key:
                os.environ.setdefault(key, value)


def corpus_fingerprint(corpus_dir: str, embedding_model: str) -> str:
    root = os.path.realpath(corpus_dir)
    digest = hashlib.sha256(embedding_model.encode("utf-8"))
    paths = sorted(
        os.path.join(base, name)
        for base, _, names in os.walk(root)
        for name in names
        if os.path.splitext(name)[1].lower() in {".txt", ".md", ".markdown", ".pdf"}
    )
    for path in paths:
        digest.update(os.path.relpath(path, root).replace("\\", "/").encode("utf-8"))
        with open(path, "rb") as source:
            for block in iter(lambda: source.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()

class DecomposerToRetrieverAdapter:
    """Shaped like Component 3's own `CorpusRetriever` from Component 4's
    point of view (`async def retrieve(query) -> RetrievalResult`), but
    runs the query through Component 2's real decomposer first."""

    def __init__(self, decomposer_adapter, retriever, session_id: str):
        self.decomposer_adapter = decomposer_adapter
        self.retriever = retriever
        self.session_id = session_id
        self._turn = 0
        self.last_subqueries = []
        self.last_result = None

    async def retrieve(self, query: str):
        self._turn += 1
        decomp = self.decomposer_adapter.decompose_full(query, session_id=self.session_id, turn_id=f"T{self._turn}")
        subqueries = [
            SubQuery(**item.to_subquery_kwargs(qid=f"T{self._turn}.{i}"))
            for i, item in enumerate(decomp.sub_queries)
        ]
        self.last_subqueries = [sq.text for sq in subqueries]
        print(f"      [Component 2] {len(subqueries)} sub-query(ies): {[sq.text for sq in subqueries]}")
        self.last_result = await self.retriever.retrieve(subqueries)
        return self.last_result


async def main():
    load_dotenv(os.path.join(BASE_DIR, ".env"))
    parser = argparse.ArgumentParser(description="Run the Gemini-backed streaming RAG assistant.")
    parser.add_argument("--backend", choices=("gemini", "offline"),
                        default=os.environ.get("RAG_BACKEND", "gemini"),
                        help="Gemini live models by default; offline mode must be selected explicitly")
    parser.add_argument("--corpus-dir", default=os.environ.get("RAG_CORPUS_DIR", os.path.join(BASE_DIR, "documents")),
                        help="Folder of local .txt/.md/.pdf files (default: ./documents)")
    parser.add_argument("--events", default=os.environ.get(
                            "RAG_EVENTS_PATH", os.path.join(BASE_DIR, "component5", "out", "demo_events.jsonl")),
                        help="JSONL telemetry output path")
    parser.add_argument("--session-id", default="Demo_Session")
    parser.add_argument("--index-dir", default=os.environ.get(
                            "RAG_INDEX_DIR", os.path.join(BASE_DIR, "component3", "index_store")),
                        help="Persistent local vector index cache")
    parser.add_argument("--rebuild-index", action="store_true", help="Re-embed all documents and refresh the local index")
    parser.add_argument("--demo", action="store_true", help="Run the bundled scripted fixture conversation")
    args = parser.parse_args()
    corpus_dir = args.corpus_dir
    chunks = load_corpus(corpus_dir)
    llm_call = None
    backend = None
    if args.backend == "gemini":
        from gemini_backend import GeminiBackend, GeminiEmbedder, GeminiReranker

        backend = GeminiBackend()
        llm_call = backend.generate_text
        embedding_model = backend.embedding_model
        embedder = GeminiEmbedder(
            backend, cache_path=os.path.join(args.index_dir, "embeddings.sqlite3")
        )
        reranker = GeminiReranker(backend)
        rerank_threshold = 0.55
        print(f"Embedding {len(chunks)} chunks with {embedding_model}; this may take a moment on first run...")
    else:
        backend = None
        embedding_model = "offline-hashing-v1"
        embedder = HashingEmbedder()
        reranker = TokenOverlapReranker()
        rerank_threshold = 0.1
        print("Using explicit offline backend.")

    fingerprint = corpus_fingerprint(corpus_dir, embedding_model)
    manifest_path = os.path.join(args.index_dir, "manifest.json")
    index_path = os.path.join(args.index_dir, "chunks.jsonl")
    matrix_path = os.path.join(args.index_dir, "dense.npy")
    manifest = None
    if os.path.isfile(manifest_path):
        with open(manifest_path, encoding="utf-8") as source:
            manifest = json.load(source)
    if (not args.rebuild_index and manifest and manifest.get("fingerprint") == fingerprint
            and os.path.isfile(index_path) and os.path.isfile(matrix_path)):
        print(f"Loading cached document index from {args.index_dir}")
        idx = HybridIndex.load(args.index_dir, embedder)
    else:
        print(f"Building index for {len(chunks)} chunks from {corpus_dir}")
        idx = HybridIndex(embedder).build(chunks)
        idx.save(args.index_dir)
        os.makedirs(args.index_dir, exist_ok=True)
        with open(manifest_path, "w", encoding="utf-8") as target:
            json.dump({"fingerprint": fingerprint, "embedding_model": embedding_model}, target, indent=2)
    retriever = CorpusRetriever(
        idx, reranker,
        RetrieverConfig(final_k=5, per_query_k=5, rerank=True,
                        uncertainty_threshold=rerank_threshold),
    )

    decomposer_adapter = RealDecomposerAdapter(llm_call=llm_call)
    controller = RetrievalController(decomposer=decomposer_adapter)
    delta_retriever = DecomposerToRetrieverAdapter(decomposer_adapter, retriever, args.session_id)
    synthesizer_backend = LLMSynthesizer(llm_call) if llm_call else None
    synth = SessionAwareSynthesizer(backend=synthesizer_backend)
    telemetry_model = f"{backend.model}+{backend.embedding_model}" if backend else "offline-hashing"

    turns = [
        ("workshop", "I need to know", False),
        ("workshop", "I need a workshop venue in Pune", False),
        ("workshop", "Actually, the workshop is for 30 people and I need cancellation terms and catering options", True),
        ("workshop", "put that in bullet points", True),
        ("travel", "Summarize the travel reimbursement rule for an employee trip", True),
        ("travel", "The trip was international and the booking was made after travel", True),
        ("travel", "Please repeat the answer in two bullets", True),
        ("unsupported", "What is the campus parking policy?", True),
    ] if args.demo else None
    logger = JsonlEventLogger(args.events)

    if turns is None:
        print(f"Ready. Documents: {corpus_dir}. Type /quit to exit.")
        try:
            while True:
                utterance = input("\nYou: ").strip()
                if utterance.lower() in {"/quit", "/exit"}:
                    break
                if not utterance:
                    continue
                await handle_turn(
                    utterance=utterance, session_id=args.session_id,
                    turn_id=f"T{getattr(main, '_turn_number', 0) + 1}", is_final=True,
                    controller=controller, delta_retriever=delta_retriever, synth=synth,
                    logger=logger, telemetry_model=telemetry_model,
                )
                main._turn_number = getattr(main, "_turn_number", 0) + 1
        finally:
            if hasattr(embedder, "close"):
                embedder.close()
            if backend:
                backend.close()
        print(f"\nTurn telemetry: {args.events}")
        return

    try:
        for i, (session_label, utterance, is_final) in enumerate(turns, 1):
            session_id = f"{args.session_id}_{session_label}"
            await handle_turn(
                utterance=utterance, session_id=session_id, turn_id=f"T{i}", is_final=is_final,
                controller=controller, delta_retriever=delta_retriever, synth=synth,
                logger=logger, telemetry_model=telemetry_model,
            )
    finally:
        if hasattr(embedder, "close"):
            embedder.close()
        if backend:
            backend.close()
    print(f"\nTurn telemetry: {args.events}")


async def handle_turn(utterance, session_id, turn_id, is_final, controller,
                      delta_retriever, synth, logger, telemetry_model):
    delta_retriever.session_id = session_id
    started = time.perf_counter()
    payload = StreamingInput(
        session_id=session_id, turn_id=turn_id, transcript=utterance,
        is_final=is_final, timestamp=time.time(),
    )
    c1_decision = controller.decide(payload)
    print(f"\n[Turn {turn_id}] User: {utterance!r}")
    print(f"  [Component 1] {c1_decision.decision.value} "
          f"(reason={c1_decision.reason_codes}, trigger={c1_decision.trigger.value}, "
          f"confidence={c1_decision.confidence})")

    if c1_decision.decision.value == "WAIT":
        print("  -> waiting for more of the utterance; nothing downstream runs yet.")
        logger.write_turn(session_id=session_id, turn_id=turn_id, transcript=utterance,
                          decision="WAIT", reason_codes=c1_decision.reason_codes,
                          trigger=c1_decision.trigger.value,
                          processing_ms=(time.perf_counter() - started) * 1000,
                          model=telemetry_model, inference_cost_usd=None)
        return

    if c1_decision.decision.value == "NO_RETRIEVAL" and not set(c1_decision.reason_codes) & {"PRESENTATION_ONLY", "COSMETIC_CHANGE"}:
        print("  -> controller suppressed retrieval; no downstream retrieval or synthesis runs.")
        logger.write_turn(session_id=session_id, turn_id=turn_id, transcript=utterance,
                          decision="NO_RETRIEVAL", reason_codes=c1_decision.reason_codes,
                          trigger=c1_decision.trigger.value,
                          processing_ms=(time.perf_counter() - started) * 1000,
                          model=telemetry_model, inference_cost_usd=None)
        return

    text = c1_decision.query if c1_decision.decision.value == "RETRIEVE" else utterance
    delta_retriever.last_result = None
    delta_retriever.last_subqueries = []
    turn = await synth.process_turn(session_id, turn_id, text, retriever=delta_retriever)
    print(f"  [Component 4] retrieval_required={turn.retrieval_required} is_cosmetic={turn.is_cosmetic}")
    print(f"  Answer: {turn.answer}")
    if turn.dropped_citations:
        print(f"  (dropped unverifiable citations: {turn.dropped_citations})")
    citations = []
    state = synth.store.get(session_id)
    for tag in turn.citations:
        match = re.fullmatch(r"\[([\w.-]+)\s*§\s*([\w.-]+)\]", tag)
        if not match:
            continue
        doc_id, section = match.groups()
        chunk = next((c for c in state.cached_chunks.values()
                      if c.doc_id == doc_id and c.section == section), None)
        if chunk:
            citations.append({"doc_id": doc_id, "section": section, "chunk_id": chunk.chunk_id})
    logger.write_turn(
        session_id=session_id, turn_id=turn_id, transcript=utterance,
        decision=c1_decision.decision.value, reason_codes=c1_decision.reason_codes,
        trigger=c1_decision.trigger.value,
        sub_queries=delta_retriever.last_subqueries, citations=citations,
        answer_version=turn.version, uncertainty=turn.uncertainty,
        processing_ms=(time.perf_counter() - started) * 1000,
        input_tokens_estimate=len(utterance.split()),
        output_tokens_estimate=len(turn.answer.split()),
        model=telemetry_model, inference_cost_usd=None,
    )


if __name__ == "__main__":
    asyncio.run(main())
