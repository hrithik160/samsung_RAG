"""Append-only, session-scoped JSONL telemetry for the integrated demo."""
from __future__ import annotations

import json
import threading
import time
import uuid
from pathlib import Path


class JsonlEventLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self.run_id = uuid.uuid4().hex
        self._session_aliases: dict[str, str] = {}

    def _session_alias(self, session_id: str) -> str:
        if session_id not in self._session_aliases:
            self._session_aliases[session_id] = f"S{len(self._session_aliases) + 1:04d}"
        return self._session_aliases[session_id]

    def write_turn(self, *, session_id: str, turn_id: str, transcript: str,
                   decision: str, reason_codes: list[str], trigger: str,
                   processing_ms: float, sub_queries: list[str] | None = None,
                   citations: list[dict] | None = None, answer_version: int | None = None,
                   uncertainty: bool = False, model: str = "offline",
                   input_tokens_estimate: int = 0, output_tokens_estimate: int = 0,
                   inference_cost_usd: float | None = 0.0) -> dict:
        event = {
            "schema_version": "1.0",
            "run_id": self.run_id,
            "timestamp_unix": time.time(),
            # Aliases are unique only within this process run. Raw user or
            # client session identifiers never enter the persistent trace.
            "session_id": self._session_alias(session_id),
            "turn_id": turn_id,
            "transcript_length_chars": len(transcript),
            "decision": decision,
            "reason_codes": list(reason_codes),
            "retrieval_trigger": trigger,
            "sub_queries": list(sub_queries or []),
            "citations": list(citations or []),
            "answer_version": answer_version,
            "uncertainty": bool(uncertainty),
            "processing_ms": round(processing_ms, 3),
            "model": model,
            "input_tokens_estimate": input_tokens_estimate,
            "output_tokens_estimate": output_tokens_estimate,
            "inference_cost_usd": inference_cost_usd,
        }
        with self._lock, self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(event, ensure_ascii=False) + "\n")
            stream.flush()
        return event
