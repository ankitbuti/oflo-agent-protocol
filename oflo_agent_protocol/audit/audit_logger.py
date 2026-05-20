"""Structured, append-only JSONL audit logger — every LLM call is recorded."""
from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from oflo_agent_protocol.core.types import AuditRecord

logger = logging.getLogger(__name__)

_DEFAULT_DIR = os.getenv("OFLO_AUDIT_DIR", "./audit_logs")


class AuditLogger:
    """
    Writes one JSONL file per project under `audit_logs/<project_id>/agent_audit.jsonl`.

    Thread-safe via asyncio.Lock.
    """

    def __init__(self, project_id: str, log_dir: Optional[str] = None) -> None:
        self.project_id = project_id
        self._dir = Path(log_dir or _DEFAULT_DIR) / project_id
        self._dir.mkdir(parents=True, exist_ok=True)
        self._file = self._dir / "agent_audit.jsonl"
        self._lock = asyncio.Lock()
        self._buffer: List[Dict[str, Any]] = []
        self._flush_interval = 5  # seconds

    async def log(self, record: AuditRecord) -> None:
        record.project_id = self.project_id
        async with self._lock:
            try:
                with self._file.open("a", encoding="utf-8") as fh:
                    fh.write(json.dumps(record.to_dict()) + "\n")
            except OSError as e:
                logger.error("Audit write failed: %s", e)

    async def query(
        self,
        agent_id: Optional[str] = None,
        limit: int = 100,
        since_ts: Optional[str] = None,
    ) -> List[Dict[str, Any]]:
        records: List[Dict[str, Any]] = []
        if not self._file.exists():
            return records
        try:
            with self._file.open("r", encoding="utf-8") as fh:
                for line in fh:
                    r = json.loads(line)
                    if agent_id and r.get("agent_id") != agent_id:
                        continue
                    if since_ts and r.get("timestamp", "") < since_ts:
                        continue
                    records.append(r)
        except OSError:
            pass
        return records[-limit:]

    def get_summary(self) -> Dict[str, Any]:
        """Quick cost/token summary across all records."""
        total_cost = 0.0
        total_tokens = 0
        total_calls = 0
        errors = 0
        try:
            with self._file.open("r", encoding="utf-8") as fh:
                for line in fh:
                    r = json.loads(line)
                    total_cost += r.get("cost_usd", 0)
                    total_tokens += (r.get("token_usage") or {}).get("total_tokens", 0)
                    total_calls += 1
                    if not r.get("success", True):
                        errors += 1
        except OSError:
            pass
        return {
            "project_id": self.project_id,
            "total_calls": total_calls,
            "total_cost_usd": round(total_cost, 6),
            "total_tokens": total_tokens,
            "error_count": errors,
        }

    @staticmethod
    def hash_prompt(prompt: str) -> str:
        return hashlib.sha256(prompt.encode()).hexdigest()[:16]
