import sys
import os
import asyncio
import importlib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
C1_DIR = os.path.join(BASE_DIR, "samsungRag")
C2_DIR = os.path.join(BASE_DIR, "component2")
C3_DIR = os.path.join(BASE_DIR, "component3")

sys.path.append(C1_DIR)
from app.shared.models import StreamingInput
from app.retrieval_controller.controller import RetrievalController


def _isolated_import(path, module_names):
    """Component 2 and 3 each ship a bare-named `schema.py`; import one at a
    time so neither shadows the other."""
    for name in module_names:
        sys.modules.pop(name, None)
    old_path = sys.path[:]
    sys.path = [path] + [p for p in sys.path if p not in (C2_DIR, C3_DIR)]
    try:
        return {name: importlib.import_module(name) for name in module_names}
    finally:
        sys.path = old_path


c2 = _isolated_import(C2_DIR, ["schema", "decomposer", "adapter"])
RealDecomposerAdapter = c2["adapter"].RealDecomposerAdapter

c3 = _isolated_import(C3_DIR, ["schema", "models", "index", "retriever", "fusion"])
HybridIndex, chunk_document = c3["index"].HybridIndex, c3["index"].chunk_document
HashingEmbedder, TokenOverlapReranker = c3["models"].HashingEmbedder, c3["models"].TokenOverlapReranker
CorpusRetriever, RetrieverConfig = c3["retriever"].CorpusRetriever, c3["retriever"].RetrieverConfig
SubQuery = c3["schema"].SubQuery

SAMSUNG_MANUAL = {
    "S24_Display": "The Galaxy S24 features a Dynamic AMOLED 2X display with a 120Hz refresh rate.",
    "S24_Battery": "The battery capacity is 4000mAh. It supports 25W fast charging.",
}

async def main():
    print("--- DEMO START ---")
    decomposer = RealDecomposerAdapter()
    controller = RetrievalController(decomposer=decomposer)

    chunks = [c for doc_id, text in SAMSUNG_MANUAL.items() for c in chunk_document(doc_id, text)]
    idx = HybridIndex(HashingEmbedder()).build(chunks)
    c3_config = RetrieverConfig(final_k=2, uncertainty_threshold=0.1)
    retriever = CorpusRetriever(idx, TokenOverlapReranker(), c3_config)

    session_id = "Demo_Session"

    # Simulate user sending 3 messages in a row
    messages = [
        "How fast does the S24 charge?",
        "Okay thanks!",
        "What is the screen refresh rate?"
    ]

    for turn_id, msg in enumerate(messages, 1):
        print(f"\n[USER MESSAGE]: '{msg}'")

        payload = StreamingInput(session_id=session_id, turn_id=f"T{turn_id}", transcript=msg, is_final=True, timestamp=123456789.0)
        decision = controller.decide(payload)

        print(f"  [Component 1] Decision: {decision.decision.value} (Reason: {decision.reason_codes[0]})")

        if decision.decision.value == "RETRIEVE":
            full = decomposer.decompose_full(decision.query, session_id=session_id, turn_id=f"T{turn_id}")
            subqueries = [
                SubQuery(**item.to_subquery_kwargs(qid=f"T{turn_id}.{i}"))
                for i, item in enumerate(full.sub_queries)
            ]

            result = await retriever.retrieve(subqueries)
            for hit in result.hits:
                print(f"  [Component 3] Retrieved: [{hit.chunk_id}] {hit.text}")
        else:
            print("  [Component 3] (Skipped search!)")

if __name__ == "__main__":
    asyncio.run(main())
