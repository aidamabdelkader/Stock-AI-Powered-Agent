from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


class AuditLogger:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS response_audit (
                    request_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    session_id TEXT,
                    question TEXT NOT NULL,
                    answer TEXT NOT NULL,
                    citations_json TEXT NOT NULL,
                    retrieved_json TEXT NOT NULL,
                    recommendation_intent INTEGER NOT NULL,
                    prompt_injection_signal INTEGER NOT NULL,
                    insufficient_evidence INTEGER NOT NULL,
                    confidence TEXT NOT NULL,
                    model TEXT NOT NULL,
                    prompt_version TEXT NOT NULL,
                    corpus_version TEXT,
                    input_tokens INTEGER NOT NULL,
                    output_tokens INTEGER NOT NULL,
                    estimated_cost_usd REAL NOT NULL,
                    latency_ms INTEGER NOT NULL,
                    validation_warnings_json TEXT NOT NULL,
                    error TEXT
                )
                """
            )

    def log(self, record: dict[str, Any]) -> None:
        payload = {
            "request_id": record["request_id"],
            "created_at": datetime.now(timezone.utc).isoformat(),
            "session_id": record.get("session_id"),
            "question": record.get("question", ""),
            "answer": record.get("answer", ""),
            "citations_json": json.dumps(record.get("citations", []), ensure_ascii=False),
            "retrieved_json": json.dumps(record.get("retrieved", []), ensure_ascii=False),
            "recommendation_intent": int(bool(record.get("recommendation_intent"))),
            "prompt_injection_signal": int(bool(record.get("prompt_injection_signal"))),
            "insufficient_evidence": int(bool(record.get("insufficient_evidence"))),
            "confidence": record.get("confidence", "low"),
            "model": record.get("model", "unknown"),
            "prompt_version": record.get("prompt_version", "unknown"),
            "corpus_version": record.get("corpus_version"),
            "input_tokens": int(record.get("input_tokens", 0)),
            "output_tokens": int(record.get("output_tokens", 0)),
            "estimated_cost_usd": float(record.get("estimated_cost_usd", 0.0)),
            "latency_ms": int(record.get("latency_ms", 0)),
            "validation_warnings_json": json.dumps(record.get("validation_warnings", [])),
            "error": record.get("error"),
        }
        columns = ", ".join(payload)
        placeholders = ", ".join(f":{name}" for name in payload)
        with self._connect() as connection:
            connection.execute(f"INSERT OR REPLACE INTO response_audit ({columns}) VALUES ({placeholders})", payload)

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM response_audit ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [dict(row) for row in rows]
