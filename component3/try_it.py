import asyncio
from index import HybridIndex, chunk_document
from models import HashingEmbedder, TokenOverlapReranker
from retriever import CorpusRetriever, RetrieverConfig
from schema import SubQuery

async def main():
    print("1. Creating a tiny mock corpus...")
    docs = {
        "Doc_12": "The cancellation policy requires 48 hours notice for a full refund.",
        "Doc_13": "Baggage allowance is strictly 23kg per passenger.",
        "Doc_14": "Tourist visas take 5 business days to process."
    }
    chunks = [c for d, t in docs.items() for c in chunk_document(d, t)]
    
    print("2. Building the Hybrid Index (using offline models)...")
    idx = HybridIndex(HashingEmbedder()).build(chunks)
    
    # We set a high uncertainty threshold here (0.9) just to demonstrate how the flag triggers
    # if the token overlap reranker score isn't perfect.
    config = RetrieverConfig(uncertainty_threshold=0.9)
    retriever = CorpusRetriever(idx, TokenOverlapReranker(), config)
    
    print("\n--- Running Query ---")
    query = [
        SubQuery(text="what is the baggage limit weight?"),
        SubQuery(text="how many days for visa?")
    ]
    
    print(f"SubQueries:\n - {query[0].text}\n - {query[1].text}\n")
    
    result = await retriever.retrieve(query)
    
    print(f"Uncertainty Bypass Triggered? {result.uncertainty_bypass}")
    print(f"Execution time: {result.timings_ms['total_ms']:.2f} ms\n")
    
    print("Top Fused Candidates Returned:")
    for hit in result.hits:
        print(f" => [{hit.chunk_id}] (Score: {hit.score:.2f}): {hit.text}")

if __name__ == "__main__":
    asyncio.run(main())
