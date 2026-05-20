"""Real-time telemetry — token budgets, cost alerts, latency percentiles."""
from __future__ import annotations

import asyncio
import logging
import time
from collections import defaultdict, deque
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Callable, Deque, Dict, Optional

from oflo_agent_protocol.core.types import AuditRecord, ModelProvider, TokenUsage

logger = logging.getLogger(__name__)


@dataclass
class AgentMetrics:
    agent_id: str
    agent_name: str
    call_count: int = 0
    total_tokens: int = 0
    total_cost_usd: float = 0.0
    error_count: int = 0
    latencies_ms: Deque[float] = field(default_factory=lambda: deque(maxlen=100))

    def p50(self) -> float:
        if not self.latencies_ms:
            return 0.0
        s = sorted(self.latencies_ms)
        return s[len(s) // 2]

    def p95(self) -> float:
        if not self.latencies_ms:
            return 0.0
        s = sorted(self.latencies_ms)
        return s[int(len(s) * 0.95)]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent_id": self.agent_id,
            "agent_name": self.agent_name,
            "call_count": self.call_count,
            "total_tokens": self.total_tokens,
            "total_cost_usd": round(self.total_cost_usd, 6),
            "error_count": self.error_count,
            "latency_p50_ms": round(self.p50(), 1),
            "latency_p95_ms": round(self.p95(), 1),
        }


class Telemetry:
    """
    In-process telemetry collector for token/cost/latency.

    Subscribers can register callbacks via `on_alert()` to receive
    notifications when thresholds are breached.
    """

    def __init__(
        self,
        cost_budget_usd: Optional[float] = None,
        token_budget: Optional[int] = None,
    ) -> None:
        self._metrics: Dict[str, AgentMetrics] = {}
        self._project_cost = 0.0
        self._project_tokens = 0
        self._cost_budget = cost_budget_usd
        self._token_budget = token_budget
        self._alert_callbacks: list[Callable[[str, Dict[str, Any]], None]] = []
        self._lock = asyncio.Lock()

    def on_alert(self, callback: Callable[[str, Dict[str, Any]], None]) -> None:
        self._alert_callbacks.append(callback)

    async def record(self, record: AuditRecord) -> None:
        async with self._lock:
            m = self._metrics.setdefault(
                record.agent_id,
                AgentMetrics(agent_id=record.agent_id, agent_name=record.agent_name),
            )
            m.call_count += 1
            if record.token_usage:
                m.total_tokens += record.token_usage.total_tokens
                self._project_tokens += record.token_usage.total_tokens
            m.total_cost_usd += record.cost_usd
            self._project_cost += record.cost_usd
            if not record.success:
                m.error_count += 1
            if record.latency_ms:
                m.latencies_ms.append(record.latency_ms)

        await self._check_budgets()

    async def _check_budgets(self) -> None:
        if self._cost_budget and self._project_cost >= self._cost_budget:
            await self._fire_alert(
                "cost_budget_exceeded",
                {"budget_usd": self._cost_budget, "spent_usd": self._project_cost},
            )
        if self._token_budget and self._project_tokens >= self._token_budget:
            await self._fire_alert(
                "token_budget_exceeded",
                {"budget_tokens": self._token_budget, "used_tokens": self._project_tokens},
            )

    async def _fire_alert(self, alert_type: str, data: Dict[str, Any]) -> None:
        for cb in self._alert_callbacks:
            try:
                cb(alert_type, data)
            except Exception:
                pass

    def summary(self) -> Dict[str, Any]:
        return {
            "project_cost_usd": round(self._project_cost, 6),
            "project_tokens": self._project_tokens,
            "agents": {aid: m.to_dict() for aid, m in self._metrics.items()},
        }

    def agent_metrics(self, agent_id: str) -> Optional[AgentMetrics]:
        return self._metrics.get(agent_id)


@asynccontextmanager
async def timed_call(label: str = "") -> AsyncIterator[Dict[str, float]]:
    """Async context manager that records wall-clock ms into a dict."""
    data: Dict[str, float] = {}
    start = time.monotonic()
    try:
        yield data
    finally:
        data["latency_ms"] = (time.monotonic() - start) * 1000
        if label:
            logger.debug("%s took %.1f ms", label, data["latency_ms"])
