# Component 3 – Corpus Retrieval & Fusion

This component serves as the high-throughput search engine and evidence filter. It executes searches across an isolated document index and compiles candidate chunks into a dense, deduplicated context window using Reciprocal Rank Fusion (RRF).

## Setup & Testing

```powershell
pip install numpy sentence-transformers      # sentence-transformers only needed for real BGE models
python test_component3.py                    # offline unit tests, no downloads required
python interactive_test.py                   # try it out yourself via terminal!
```

To see how Component 3 integrates with Component 1, run the orchestrator script located in the main project folder:
```powershell
cd ..
python demo_pipeline.py
```

## Architecture & Features
* **Asynchronous Execution:** Uses `asyncio.gather` for non-blocking concurrent querying.
* **Top-K Limits:** Fetches candidates per sub-query, fuses them via RRF, and sends the top 20 candidates to the cross-encoder to extract the final top 5 passages.
* **Uncertainty Gate:** Weak candidates below the configured relevance threshold are removed before fusion. If no usable evidence remains, `RetrievalResult.uncertainty_bypass` becomes `True` so synthesis can return an explicit uncertainty response.
* **Metadata Chunking:** Generates citation metadata exactly as required (`[Doc_ID §Section]`).

## Quickstart (real models)

```python
import asyncio
from models import BGEEmbedder, CrossEncoderReranker
from index import HybridIndex, chunk_document
from retriever import CorpusRetriever, RetrieverConfig
from schema import SubQuery

async def main():
    chunks = [c for doc_id, text in my_corpus.items() for c in chunk_document(doc_id, text)]
    index = HybridIndex(BGEEmbedder("BAAI/bge-large-en-v1.5")).build(chunks)
    index.save("index_dir")                       # later: HybridIndex.load("index_dir", embedder)
    
    config = RetrieverConfig(
        mode="hybrid", 
        candidate_k=20, 
        final_k=5, 
        uncertainty_threshold=0.65
    )
    retriever = CorpusRetriever(index, CrossEncoderReranker("BAAI/bge-reranker-large"), config)
    
    res = await retriever.retrieve([
        SubQuery(text="visa requirements for Japan"),
        SubQuery(text="checked baggage allowance")
    ])
    
    if res.uncertainty_bypass:
        print("Fact could not be verified in corpus.")
        
    res.top_ids(5)          # for the reflector's overlap check
    res.overlap(other, 5)   # provisional vs final top-k overlap
    res.to_event()          # JSON for retrieval_events telemetry

if __name__ == "__main__":
    asyncio.run(main())
```

## Ablation switch
`RetrieverConfig(mode="dense")` vs `mode="hybrid"` (or `"sparse"`) — same code path, one flag.
