"""Agent registry — per-project, thread-safe agent catalogue."""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional

from oflo_agent_protocol.core.types import AgentStatus

logger = logging.getLogger(__name__)


class AgentRegistry:
    """
    In-memory, async-safe registry of agents for a single project.

    Agents register by ID; lookup by name is also supported.
    """

    def __init__(self, project_id: str) -> None:
        self.project_id = project_id
        self._agents: Dict[str, Any] = {}  # id → BaseAgentV2
        self._lock = asyncio.Lock()

    async def register(self, agent: Any) -> None:
        async with self._lock:
            self._agents[agent.id] = agent
            logger.info("[%s] Registered agent %s (%s)", self.project_id, agent.name, agent.id[:8])

    async def unregister(self, agent_id: str) -> bool:
        async with self._lock:
            if agent_id in self._agents:
                del self._agents[agent_id]
                return True
            return False

    def get(self, agent_id: str) -> Optional[Any]:
        return self._agents.get(agent_id)

    def get_by_name(self, name: str) -> Optional[Any]:
        for agent in self._agents.values():
            if agent.name == name:
                return agent
        return None

    def list_agents(self) -> List[Dict[str, Any]]:
        return [a.to_dict() for a in self._agents.values()]

    def active_agents(self) -> List[Any]:
        return [
            a for a in self._agents.values()
            if a.status in (AgentStatus.ACTIVE, AgentStatus.WORKING, AgentStatus.IDLE)
        ]

    def __len__(self) -> int:
        return len(self._agents)
