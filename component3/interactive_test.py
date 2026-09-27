import asyncio
from index import HybridIndex, chunk_document
from models import HashingEmbedder, TokenOverlapReranker
from retriever import CorpusRetriever, RetrieverConfig
from schema import SubQuery

# A more realistic mock corpus for you to test with!
SAMSUNG_MANUAL = {
    "S24_Display": "The Galaxy S24 features a Dynamic AMOLED 2X display with a 120Hz refresh rate. To save battery, you can switch to 60Hz in the display settings.",
    "S24_Camera": "The camera system includes a 50MP main sensor, a 12MP ultrawide, and a 10MP telephoto with 3x optical zoom. Night mode automatically turns on in low light.",
    "S24_Battery": "The battery capacity is 4000mAh. It supports 25W fast charging and 15W wireless charging. Use the official Samsung charger for best results.",
    "S24_Water": "The device is IP68 rated, meaning it is water and dust resistant. It can survive in 1.5 meters of fresh water for up to 30 minutes.",
    "S24_Stylus": "Unlike the S24 Ultra, the base Galaxy S24 does not support the S Pen stylus."
}

async def run_interactive():
    print("Building search index from Samsung S24 Mock Manual...")
    
    # 1. Chunk and index the documents
    chunks = [c for doc_id, text in SAMSUNG_MANUAL.items() for c in chunk_document(doc_id, text)]
    idx = HybridIndex(HashingEmbedder()).build(chunks)
    
    # 2. Setup the Retriever
    config = RetrieverConfig(final_k=2, uncertainty_threshold=0.1)
    retriever = CorpusRetriever(idx, TokenOverlapReranker(), config)
    
    print("Ready! Type a question to search the manual (or type 'exit' to quit).")
    print("-" * 50)
    
    while True:
        user_input = input("\nYour Question: ")
        if user_input.lower() in ['exit', 'quit']:
            break
            
        if not user_input.strip():
            continue
            
        # 3. Retrieve
        query = SubQuery(text=user_input)
        result = await retriever.retrieve([query])
        
        # 4. Show results
        print(f"\n[Search took {result.timings_ms['total_ms']:.2f} ms]")
        if result.uncertainty_bypass:
            print("⚠️ UNCERTAINTY GATE TRIGGERED: I couldn't find a highly relevant match for that.")
        
        print("Top results:")
        for hit in result.hits:
            print(f" - [{hit.chunk_id}] (Relevance Score: {hit.score:.2f})")
            print(f"   Text: {hit.text}\n")

if __name__ == "__main__":
    try:
        asyncio.run(run_interactive())
    except KeyboardInterrupt:
        pass
