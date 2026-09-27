"""Small Gemini REST client for generation and retrieval embeddings.

Uses Google's public Gemini API with the API key sent in a header. The client
is synchronous because Component 3 runs retrieval work in worker threads.
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
import hashlib
from typing import Sequence

import httpx
import numpy as np


class GeminiBackend:
    API_ROOT = "https://generativelanguage.googleapis.com/v1beta"

    def __init__(self, api_key: str | None = None, model: str | None = None,
                 embedding_model: str | None = None, timeout: float = 120.0):
        self.api_key = api_key or os.environ.get("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY is missing. Set it in .env or the environment.")
        self.model = model or os.environ.get("GEMINI_MODEL", "gemini-3.6-flash")
        self.embedding_model = embedding_model or os.environ.get(
            "GEMINI_EMBEDDING_MODEL", "gemini-embedding-001"
        )
        self.embedding_items_per_minute = max(
            1, int(os.environ.get("GEMINI_EMBEDDING_ITEMS_PER_MINUTE", "60"))
        )
        self.embedding_batch_size = max(
            1, min(100, int(os.environ.get("GEMINI_EMBEDDING_BATCH_SIZE", "8")))
        )
        self.generation_requests_per_minute = max(
            1, int(os.environ.get("GEMINI_GENERATION_REQUESTS_PER_MINUTE", "10"))
        )
        self._embedding_rate_lock = threading.Lock()
        self._next_embedding_slot = 0.0
        self._generation_rate_lock = threading.Lock()
        self._next_generation_slot = 0.0
        self.http = httpx.Client(
            timeout=timeout,
            headers={"x-goog-api-key": self.api_key, "Content-Type": "application/json"},
        )

    def _post(self, endpoint: str, payload: dict) -> dict:
        for attempt in range(6):
            response = self.http.post(f"{self.API_ROOT}/{endpoint}", json=payload)
            if not response.is_error:
                return response.json()
            try:
                detail = response.json().get("error", {}).get("message", "")
            except ValueError:
                detail = ""
            detail = detail or response.reason_phrase
            retryable = response.status_code in (429, 500, 502, 503, 504)
            if not retryable or attempt == 5:
                extra = " Lower GEMINI_EMBEDDING_ITEMS_PER_MINUTE or check your Gemini quota/billing." if response.status_code == 429 else ""
                raise RuntimeError(f"Gemini API returned HTTP {response.status_code}: {detail}.{extra}")
            retry_after = response.headers.get("Retry-After")
            if retry_after:
                try:
                    delay = float(retry_after)
                except ValueError:
                    delay = 0.0
            else:
                match = re.search(r"retry in ([0-9.]+)s", detail, flags=re.I)
                delay = float(match.group(1)) if match else 0.0
            delay = max(delay, min(60.0, 2.0 ** (attempt + 1)))
            print(f"Gemini returned HTTP {response.status_code}; retrying in {delay:.1f}s (attempt {attempt + 1}/5).")
            time.sleep(delay)

    def _wait_for_embedding_quota(self, item_count: int) -> None:
        """Reserve embedding-item starts under the configured per-minute budget."""
        # Leave 10% headroom for burst/window accounting differences and other
        # callers that may share this Gemini project quota.
        interval = (60.0 / self.embedding_items_per_minute) * 1.1
        with self._embedding_rate_lock:
            now = time.monotonic()
            start_at = max(now, self._next_embedding_slot)
            self._next_embedding_slot = start_at + interval * item_count
        delay = start_at - now
        if delay > 0:
            print(f"Embedding quota pacing: waiting {delay:.1f}s before the next batch.")
            time.sleep(delay)

    def _wait_for_generation_quota(self) -> None:
        interval = 60.0 / self.generation_requests_per_minute
        with self._generation_rate_lock:
            now = time.monotonic()
            start_at = max(now, self._next_generation_slot)
            self._next_generation_slot = start_at + interval
        delay = start_at - now
        if delay > 0:
            print(f"Gemini generation quota pacing: waiting {delay:.1f}s before the next request.")
            time.sleep(delay)

    def generate_text(self, prompt: str, json_mode: bool = False) -> str:
        self._wait_for_generation_quota()
        generation_config = {"temperature": 0.2, "maxOutputTokens": 2048}
        if json_mode:
            generation_config["responseMimeType"] = "application/json"
        result = self._post(f"models/{self.model}:generateContent", {
            "contents": [{"role": "user", "parts": [{"text": prompt}]}],
            "generationConfig": generation_config,
        })
        candidates = result.get("candidates") or []
        parts = (candidates[0].get("content") or {}).get("parts") or [] if candidates else []
        text = "".join(part.get("text", "") for part in parts).strip()
        if not text:
            reason = candidates[0].get("finishReason", "no candidate") if candidates else "no candidate"
            raise RuntimeError(f"Gemini returned no text ({reason}).")
        return text

    def rerank(self, query: str, passages: Sequence[str]) -> list[float]:
        if not passages:
            return []
        entries = "\n".join(f"{i}: {text}" for i, text in enumerate(passages))
        prompt = (
            "You score passage relevance for retrieval. Use only the query and passages. "
            "Give each passage a relevance score from 0.0 to 1.0: 1.0 means it directly "
            "contains evidence that answers the query; 0.0 means unrelated. Return JSON only "
            "as {\"scores\":[number,...]} in the same order, with one score per passage.\n\n"
            f"Query: {query}\nPassages:\n{entries}"
        )
        raw = self.generate_text(prompt, json_mode=True)
        try:
            scores = json.loads(raw)["scores"]
            scores = [float(score) for score in scores]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("Gemini reranker returned malformed scores.") from exc
        if len(scores) != len(passages) or any(not 0.0 <= score <= 1.0 for score in scores):
            raise RuntimeError("Gemini reranker returned invalid score values or count.")
        return scores

    def embed(self, texts: Sequence[str], is_query: bool = False,
              batch_size: int = 16) -> np.ndarray:
        if not texts:
            return np.zeros((0, 768), dtype=np.float32)
        task_type = "RETRIEVAL_QUERY" if is_query else "RETRIEVAL_DOCUMENT"
        vectors: list[list[float]] = []
        for start in range(0, len(texts), batch_size):
            batch = texts[start:start + batch_size]
            self._wait_for_embedding_quota(len(batch))
            payload = {
                "requests": [
                    {
                        "model": f"models/{self.embedding_model}",
                        "content": {"parts": [{"text": text}]},
                        "embedContentConfig": {
                            "taskType": task_type,
                            "outputDimensionality": 768,
                        },
                    }
                    for text in batch
                ]
            }
            result = self._post("models/" + self.embedding_model + ":batchEmbedContents", payload)
            embeddings = result.get("embeddings", [])
            if len(embeddings) != len(batch):
                raise RuntimeError("Gemini returned a different number of embeddings than requested.")
            vectors.extend(item["values"] for item in embeddings)
        array = np.asarray(vectors, dtype=np.float32)
        norms = np.linalg.norm(array, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return array / norms

    def close(self) -> None:
        self.http.close()


class GeminiEmbedder:
    """Component 3 adapter for Gemini's task-specific embedding endpoint."""

    def __init__(self, backend: GeminiBackend, cache_path: str | None = None):
        self.backend = backend
        self.cache_path = cache_path
        self._cache_lock = threading.Lock()
        self._db = None
        if cache_path:
            os.makedirs(os.path.dirname(os.path.abspath(cache_path)), exist_ok=True)
            self._db = sqlite3.connect(cache_path, check_same_thread=False)
            self._db.execute(
                "CREATE TABLE IF NOT EXISTS document_embeddings "
                "(cache_key TEXT PRIMARY KEY, vector BLOB NOT NULL)"
            )
            self._db.commit()

    def encode(self, texts: Sequence[str], is_query: bool = False) -> np.ndarray:
        if is_query or self._db is None:
            return self.backend.embed(texts, is_query=is_query)
        if not texts:
            return np.zeros((0, 768), dtype=np.float32)

        def cache_key(text: str) -> str:
            return hashlib.sha256(
                (self.backend.embedding_model + "\0RETRIEVAL_DOCUMENT\0" + text).encode("utf-8")
            ).hexdigest()

        keys = [cache_key(text) for text in texts]
        vectors: list[np.ndarray | None] = [None] * len(texts)
        missing: list[int] = []
        with self._cache_lock:
            for i, key in enumerate(keys):
                row = self._db.execute(
                    "SELECT vector FROM document_embeddings WHERE cache_key = ?", (key,)
                ).fetchone()
                if row:
                    vectors[i] = np.frombuffer(row[0], dtype="<f4").copy()
                else:
                    missing.append(i)

        batch_size = self.backend.embedding_batch_size
        for start in range(0, len(missing), batch_size):
            positions = missing[start:start + batch_size]
            batch = [texts[i] for i in positions]
            embedded = self.backend.embed(batch, is_query=False, batch_size=len(batch))
            with self._cache_lock:
                self._db.executemany(
                    "INSERT OR REPLACE INTO document_embeddings(cache_key, vector) VALUES (?, ?)",
                    [(keys[i], np.asarray(vector, dtype="<f4").tobytes())
                     for i, vector in zip(positions, embedded)],
                )
                self._db.commit()
            for i, vector in zip(positions, embedded):
                vectors[i] = vector
            print(f"Document embedding cache: {min(start + len(positions), len(missing))}/{len(missing)} new chunks embedded.")

        return np.asarray(vectors, dtype=np.float32)

    def close(self) -> None:
        if self._db is not None:
            with self._cache_lock:
                self._db.close()
                self._db = None


class GeminiReranker:
    """Component 3 adapter using Gemini Flash to score fused candidates."""

    def __init__(self, backend: GeminiBackend):
        self.backend = backend

    def score(self, query: str, texts: Sequence[str]) -> list[float]:
        return self.backend.rerank(query, texts)
