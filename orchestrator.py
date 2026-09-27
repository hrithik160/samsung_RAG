import sys
import os
import asyncio
import importlib
import time

# --- Paths ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
C1_DIR = os.path.join(BASE_DIR, "samsungRag")
C2_DIR = os.path.join(BASE_DIR, "component2")
C3_DIR = os.path.join(BASE_DIR, "component3")

# --- Component 1 Imports (proper "app.*" package, safe on sys.path permanently) ---
sys.path.append(C1_DIR)
from app.shared.models import StreamingInput
from app.retrieval_controller.controller import RetrievalController


def _isolated_import(path, module_names):
    """Component 2 and Component 3 each ship a bare-named `schema.py`.
    Import one component's modules at a time, evicting stale same-named
    entries from sys.modules first, so one doesn't shadow the other."""
    for name in module_names:
        sys.modules.pop(name, None)
    old_path = sys.path[:]
    sys.path = [path] + [p for p in sys.path if p not in (C2_DIR, C3_DIR)]
    try:
        return {name: importlib.import_module(name) for name in module_names}
    finally:
        sys.path = old_path


# --- Component 2 Imports (real decomposer, replacing MockDecomposerAdapter) ---
c2 = _isolated_import(C2_DIR, ["schema", "decomposer", "adapter"])
RealDecomposerAdapter = c2["adapter"].RealDecomposerAdapter

# --- Component 3 Imports ---
c3 = _isolated_import(C3_DIR, ["schema", "models", "index", "retriever", "fusion"])
HybridIndex, chunk_document = c3["index"].HybridIndex, c3["index"].chunk_document
HashingEmbedder, TokenOverlapReranker = c3["models"].HashingEmbedder, c3["models"].TokenOverlapReranker
CorpusRetriever, RetrieverConfig = c3["retriever"].CorpusRetriever, c3["retriever"].RetrieverConfig
SubQuery = c3["schema"].SubQuery


# --- Mock Corpus ---
SAMSUNG_MANUAL = {
    "S24_Display": "The Galaxy S24 features a Dynamic AMOLED 2X display with a 120Hz refresh rate. To save battery, you can switch to 60Hz in the display settings.",
    "S24_Camera": "The camera system includes a 50MP main sensor, a 12MP ultrawide, and a 10MP telephoto with 3x optical zoom. Night mode automatically turns on in low light.",
    "S24_Battery": "The battery capacity is 4000mAh. It supports 25W fast charging and 15W wireless charging. Use the official Samsung charger for best results.",
    "S24_Water": "The device is IP68 rated, meaning it is water and dust resistant. It can survive in 1.5 meters of fresh water for up to 30 minutes.",
    "S24_Stylus": "Unlike the S24 Ultra, the base Galaxy S24 does not support the S Pen stylus."
}


async def main():
    print("Initializing Orchestrator (Connecting C1 -> C2 -> C3)...")

    # 1. Initialize Component 1 (Retrieval Controller) with the REAL Component 2 decomposer
    decomposer = RealDecomposerAdapter()
    controller = RetrievalController(decomposer=decomposer)

    # 2. Initialize Component 3 (Corpus Retriever)
    print("Building Component 3 search index...")
    chunks = [c for doc_id, text in SAMSUNG_MANUAL.items() for c in chunk_document(doc_id, text)]
    idx = HybridIndex(HashingEmbedder()).build(chunks)
    c3_config = RetrieverConfig(final_k=2, uncertainty_threshold=0.1)
    retriever = CorpusRetriever(idx, TokenOverlapReranker(), c3_config)

    session_id = "Session_001"
    turn_id = 1

    print("\n--- System Ready ---")
    print("Type a question (e.g. 'how fast does it charge?' or 'Okay thanks'). Type 'exit' to quit.\n")

    while True:
        user_input = input(f"[{session_id} T{turn_id}] User: ")
        if user_input.lower() in ['exit', 'quit']:
            break

        if not user_input.strip():
            continue

        # STEP 1: Component 1 decides IF we need to retrieve
        payload = StreamingInput(
            session_id=session_id,
            turn_id=f"T{turn_id}",
            transcript=user_input,
            is_final=True,
            timestamp=time.time(),
        )
        decision = controller.decide(payload)

        print(f"  [Component 1] Decision: {decision.decision.value} (Reason: {decision.reason_codes[0]})")

        if decision.decision.value == "RETRIEVE":
            # STEP 2: Component 2 (real) decomposes the query
            full = decomposer.decompose_full(decision.query, session_id=session_id, turn_id=f"T{turn_id}")
            subqueries = [
                SubQuery(**item.to_subquery_kwargs(qid=f"T{turn_id}.{i}"))
                for i, item in enumerate(full.sub_queries)
            ]

            print(f"  [Component 2] Decomposed into: {[sq.text for sq in subqueries]}")

            # STEP 3: Component 3 performs the search
            result = await retriever.retrieve(subqueries)
            print(f"  [Component 3] Found {len(result.hits)} results in {result.timings_ms['total_ms']:.2f} ms")

            if result.uncertainty_bypass:
                print("    \u26a0\ufe0f UNCERTAINTY GATE TRIGGERED (Low relevance)")

            for hit in result.hits:
                print(f"    -> [{hit.chunk_id}] (Score: {hit.score:.2f}) {hit.text}")
        else:
            print("  -> Bypassing search (handled by chat/formatting memory).")

        turn_id += 1
        print("-" * 50)

if __name__ == "__main__":
    asyncio.run(main())
