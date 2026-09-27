"""Embedder / reranker interfaces plus concrete implementations.

Real models (need `pip install sentence-transformers`):
    BGEEmbedder, CrossEncoderReranker
Offline stand-ins (no downloads, deterministic) for unit tests / CI:
    HashingEmbedder, TokenOverlapReranker
"""
from __future__ import annotations

import hashlib
import re
from typing import Protocol, Sequence

import numpy as np

_WORD = re.compile(r"\w+")
_RERANK_STOPWORDS = frozenset("a an the is are was were be been do does did how what which and or for to in on at of it its that this i me my you your".split())


class Embedder(Protocol):
    def encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        """Return L2-normalised float32 array of shape (len(texts), dim)."""


class Reranker(Protocol):
    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        """Higher = more relevant. Same length/order as `texts`."""


# --------------------------------------------------------------------------- real models
class BGEEmbedder:
    QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

    def __init__(self, model_name: str = "BAAI/bge-small-en-v1.5", device: str | None = None,
                 batch_size: int = 64):
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size

    def encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        texts = [self.QUERY_PREFIX + t for t in texts] if is_query else list(texts)
        emb = self.model.encode(
            texts, normalize_embeddings=True, batch_size=self.batch_size, show_progress_bar=False
        )
        return np.asarray(emb, dtype=np.float32)


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-base", device: str | None = None,
                 batch_size: int = 32, max_length: int = 512):
        from sentence_transformers import CrossEncoder

        self.model = CrossEncoder(model_name, device=device, max_length=max_length)
        self.batch_size = batch_size

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        if not texts:
            return []
        preds = self.model.predict(
            [(query, t) for t in texts], batch_size=self.batch_size, show_progress_bar=False
        )
        # Cross-encoder checkpoints commonly return logits, while the
        # uncertainty gate expects a [0, 1] confidence-like score.
        vals = [float(s) for s in preds]
        if vals and (min(vals) < 0.0 or max(vals) > 1.0):
            return [1.0 / (1.0 + float(np.exp(-max(-60.0, min(60.0, s))))) for s in vals]
        return vals


# --------------------------------------------------------------------------- offline doubles
class HashingEmbedder:
    """Bag-of-words hashing embedder. Not semantic, but deterministic and fast."""

    def __init__(self, dim: int = 512):
        self.dim = dim

    def encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for tok in _WORD.findall(text.lower()):
                h = int(hashlib.md5(tok.encode()).hexdigest(), 16)
                out[row, h % self.dim] += 1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return out / norms


class TokenOverlapReranker:
    """Score = fraction of query tokens present in the passage. Counts calls for tests."""

    def __init__(self):
        self.calls = 0

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        self.calls += 1
        q = {t for t in _WORD.findall(query.lower()) if t not in _RERANK_STOPWORDS}
        if not q:
            return [0.0] * len(texts)
        return [len(q & {w for w in _WORD.findall(t.lower()) if w not in _RERANK_STOPWORDS}) / len(q)
                for t in texts]
