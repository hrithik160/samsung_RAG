"""Hybrid (dense + BM25) index over a single closed corpus."""
from __future__ import annotations

import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Optional

import numpy as np

from models import Embedder
from schema import Chunk

_WORD = re.compile(r"\w+")
_STOP = frozenset(
    "a an the and or but of to in on at for from by with as is are was were be been it its this that "
    "these those i you he she we they my your our their do does did so if then than not no".split()
)


def tokenize(text: str) -> list[str]:
    return [t for t in _WORD.findall(text.lower()) if t not in _STOP]


# --------------------------------------------------------------------------- chunking
def chunk_document(doc_id: str, text: str, max_words: int = 180, overlap: int = 30,
                   meta: Optional[dict] = None) -> list[Chunk]:
    """Sliding word-window chunker. Swap for a structure-aware splitter if your corpus needs it."""
    words = text.split()
    if not words:
        return []
    step = max(1, max_words - overlap)
    chunks = []
    for n, start in enumerate(range(0, len(words), step)):
        piece = words[start:start + max_words]
        chunk_meta = dict(meta or {})
        # Chunk numbers are the stable section markers exposed in citations.
        # Preserve an explicitly supplied source section when present.
        chunk_meta.setdefault("section", str(n + 1))
        chunks.append(Chunk(f"{doc_id} §{chunk_meta['section']}.{n + 1}", doc_id, " ".join(piece), chunk_meta))
        if start + max_words >= len(words):
            break
    return chunks


# --------------------------------------------------------------------------- BM25
class BM25:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b
        self.postings: dict[str, list[tuple[int, int]]] = {}
        self.idf: dict[str, float] = {}
        self.doc_len: list[int] = []
        self.avg_len = 0.0

    def fit(self, docs_tokens: Iterable[list[str]]) -> "BM25":
        postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.doc_len = []
        for i, toks in enumerate(docs_tokens):
            self.doc_len.append(len(toks))
            for term, tf in Counter(toks).items():
                postings[term].append((i, tf))
        n = len(self.doc_len)
        self.avg_len = (sum(self.doc_len) / n) if n else 0.0
        self.postings = dict(postings)
        self.idf = {t: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for t, p in postings.items()}
        return self

    def search(self, q_tokens: list[str], k: int) -> list[tuple[int, float]]:
        scores: dict[int, float] = defaultdict(float)
        for term in set(q_tokens):
            for i, tf in self.postings.get(term, ()):
                norm = 1 - self.b + self.b * self.doc_len[i] / (self.avg_len or 1.0)
                scores[i] += self.idf[term] * tf * (self.k1 + 1) / (tf + self.k1 * norm)
        return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))[:k]


# --------------------------------------------------------------------------- hybrid index
class HybridIndex:
    def __init__(self, embedder: Embedder):
        self.embedder = embedder
        self.chunks: list[Chunk] = []
        self.matrix = np.zeros((0, 0), dtype=np.float32)
        self.bm25 = BM25()

    def __len__(self) -> int:
        return len(self.chunks)

    def build(self, chunks: list[Chunk], batch_size: int = 256) -> "HybridIndex":
        self.chunks = list(chunks)
        parts = [
            self.embedder.encode([c.text for c in self.chunks[i:i + batch_size]], is_query=False)
            for i in range(0, len(self.chunks), batch_size)
        ]
        self.matrix = np.vstack(parts).astype(np.float32) if parts else np.zeros((0, 0), np.float32)
        self._fit_sparse()
        return self

    def _fit_sparse(self) -> None:
        self.bm25 = BM25().fit(tokenize(c.text) for c in self.chunks)

    def dense_search(self, query: str, k: int) -> list[tuple[int, float]]:
        if not len(self.chunks):
            return []
        q = self.embedder.encode([query], is_query=True)[0]
        sims = self.matrix @ q
        k = min(k, len(sims))
        top = np.argpartition(-sims, k - 1)[:k]
        top = top[np.lexsort((top, -sims[top]))]  # score desc, index asc
        return [(int(i), float(sims[i])) for i in top]

    def sparse_search(self, query: str, k: int) -> list[tuple[int, float]]:
        return self.bm25.search(tokenize(query), k)

    # ---- persistence (BM25 is cheap to rebuild, so only chunks + embeddings are stored)
    def save(self, path: str | Path) -> None:
        p = Path(path)
        p.mkdir(parents=True, exist_ok=True)
        with open(p / "chunks.jsonl", "w", encoding="utf-8") as f:
            for c in self.chunks:
                f.write(json.dumps({"chunk_id": c.chunk_id, "doc_id": c.doc_id, "text": c.text,
                                    "meta": c.meta}) + "\n")
        np.save(p / "dense.npy", self.matrix)

    @classmethod
    def load(cls, path: str | Path, embedder: Embedder) -> "HybridIndex":
        p = Path(path)
        idx = cls(embedder)
        with open(p / "chunks.jsonl", encoding="utf-8") as f:
            idx.chunks = [Chunk(**json.loads(line)) for line in f if line.strip()]
        idx.matrix = np.load(p / "dense.npy")
        idx._fit_sparse()
        return idx
