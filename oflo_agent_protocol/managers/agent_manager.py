"""AgentManager — per-project orchestrator of agent lifecycle, routing, and comms.

Each project (marketing, sales, trading, etc.) gets its own AgentManager instance.
The manager:
  - Creates and registers agents
  - Routes messages to the right agent (by capability, name, or ID)
  - Delegates tasks between agents
  - Exposes the project as both an MCP server and an A2A server
  - Maintains shared project-level telemetry and audit logging
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Dict, List, Optional, Type

from oflo_agent_protocol.audit.audit_logger import AuditLogger
from oflo_agent_protocol.audit.guardrails import GuardrailConfig
from oflo_agent_protocol.audit.telemetry import Telemetry
from oflo_agent_protocol.core.agent import BaseAgentV2
from oflo_agent_protocol.core.message import CanonicalMessage
from oflo_agent_protocol.core.registry import AgentRegistry
from oflo_agent_protocol.core.types import AgentStatus, ModelProvider, RoutingStrategy
from oflo_agent_protocol.runtimes.base_runtime import BaseRuntime

logger = logging.getLogger(__name__)


class AgentManager:
    """
    Per-project agent orchestrator.

    Typical usage::

        mgr = AgentManager(project_id="marketing", cost_budget_usd=5.0)

        analyst = mgr.create_agent("Analyst", system_prompt="You are a data analyst.")
        writer = mgr.create_agent("Writer", system_prompt="You write marketing copy.")

        @analyst.tool(description="Look up campaign metrics")
        async def get_metrics(campaign_id: str) -> dict:
            return {"clicks": 1200, "conversions": 48}

        reply = await mgr.route_message("Analyst", "Summarise last week's campaigns")

        # Delegate: Analyst → Writer
        await mgr.delegate("Analyst", "Writer", "Turn the analysis into a blog post")
    """

    def __init__(
        self,
        project_id: str,
        strategy: RoutingStrategy = RoutingStrategy.BALANCED,
        cost_budget_usd: Optional[float] = None,
        token_budget: Optional[int] = None,
        audit_dir: Optional[str] = None,
        guardrail_config: Optional[GuardrailConfig] = None,
    ) -> None:
        self.project_id = project_id
        self._strategy = strategy
        self._registry = AgentRegistry(project_id)
        self._audit = AuditLogger(project_id, log_dir=audit_dir)
        self._telemetry = Telemetry(
            cost_budget_usd=cost_budget_usd,
            token_budget=token_budget,
        )
        self._guardrail_config = guardrail_config or GuardrailConfig()
        self._logger = logging.getLogger(f"manager.{project_id}")

    # ------------------------------------------------------------------
    # Agent lifecycle
    # ------------------------------------------------------------------

    def create_agent(
        self,
        name: str,
        system_prompt: str = "You are a helpful assistant.",
        runtime: Optional[BaseRuntime] = None,
        strategy: Optional[RoutingStrategy] = None,
        max_tokens: int = 4096,
        temperature: float = 0.7,
        **kwargs: Any,
    ) -> BaseAgentV2:
        """Create, register, and return a new agent."""
        agent = BaseAgentV2(
            name=name,
            system_prompt=system_prompt,
            runtime=runtime,
            strategy=strategy or self._strategy,
            project_id=self.project_id,
            audit_logger=self._audit,
            telemetry=self._telemetry,
            guardrail_config=self._guardrail_config,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        asyncio.get_event_loop().run_until_complete(self._registry.register(agent))
        agent._status = AgentStatus.ACTIVE
        self._logger.info("Created agent '%s' (%s)", name, agent.id[:8])
        return agent

    async def register_agent(self, agent: BaseAgentV2) -> None:
        agent._project_id = self.project_id
        agent._audit = self._audit
        agent._telemetry = self._telemetry
        await self._registry.register(agent)

    async def remove_agent(self, agent_id: str) -> bool:
        return await self._registry.unregister(agent_id)

    # ------------------------------------------------------------------
    # Routing & delegation
    # ------------------------------------------------------------------

    async def route_message(self, agent_name: str, message: str) -> str:
        """Send a message to a named agent, return text reply."""
        agent = self._registry.get_by_name(agent_name)
        if agent is None:
            raise ValueError(f"No agent named '{agent_name}' in project '{self.project_id}'")
        return await agent.chat(message)

    async def route_to_capable(self, message: str, capability_hint: str = "") -> str:
        """Route to the first active agent (optionally filtered by capability)."""
        agents = self._registry.active_agents()
        if not agents:
            raise RuntimeError(f"No active agents in project '{self.project_id}'")
        target = agents[0]  # Round-robin / first-match; extend as needed
        return await target.chat(message)

    async def delegate(self, from_agent: str, to_agent: str, message: str) -> str:
        """Agent-to-agent delegation within the project."""
        source = self._registry.get_by_name(from_agent)
        target = self._registry.get_by_name(to_agent)
        if source is None:
            raise ValueError(f"Source agent '{from_agent}' not found")
        if target is None:
            raise ValueError(f"Target agent '{to_agent}' not found")

        # Get source agent's perspective first
        context = await source.chat(
            f"Prepare a handoff summary for this task: {message}"
        )
        # Deliver to target with context
        result = await target.chat(
            f"You are receiving a delegated task.\n\nContext from {from_agent}:\n{context}\n\nTask: {message}"
        )
        self._logger.info("Delegation %s → %s complete", from_agent, to_agent)
        return result

    async def broadcast(self, message: str) -> Dict[str, str]:
        """Broadcast a message to all active agents and collect replies."""
        agents = self._registry.active_agents()
        tasks = [agent.chat(message) for agent in agents]
        replies = await asyncio.gather(*tasks, return_exceptions=True)
        return {
            agent.name: (reply if isinstance(reply, str) else f"ERROR: {reply}")
            for agent, reply in zip(agents, replies)
        }

    async def chain(self, messages: List[Dict[str, str]]) -> List[str]:
        """
        Execute a pipeline: [{"agent": "A", "message": "..."}, {"agent": "B", "use_previous": True}].
        If `use_previous` is True, prepends the previous agent's reply to the message.
        """
        replies: List[str] = []
        prev = ""
        for step in messages:
            agent_name = step["agent"]
            msg = step.get("message", "")
            if step.get("use_previous") and prev:
                msg = f"Previous output:\n{prev}\n\nYour task: {msg}"
            reply = await self.route_message(agent_name, msg)
            replies.append(reply)
            prev = reply
        return replies

    # ------------------------------------------------------------------
    # Protocol exposure
    # ------------------------------------------------------------------

    def expose_as_mcp(self, port: int = 8080) -> Any:
        """Return a configured MCPServer exposing all project agents."""
        from oflo_agent_protocol.protocols.mcp.server import MCPServer
        srv = MCPServer(name=self.project_id, port=port)
        for agent in self._registry._agents.values():
            srv.register_agent(agent)
        return srv

    def expose_as_a2a(self, port: int = 9000, base_url: str = "") -> Any:
        """Return a configured A2AServer for the project's primary agent."""
        from oflo_agent_protocol.protocols.a2a.server import A2AServer
        from oflo_agent_protocol.protocols.a2a.types import AgentCard
        agents = self._registry.active_agents()
        if not agents:
            raise RuntimeError("No active agents to expose via A2A")
        primary = agents[0]
        url = base_url or f"http://localhost:{port}"
        card = AgentCard(
            name=self.project_id,
            description=f"Oflo project: {self.project_id}",
            url=url,
        )
        return A2AServer(card=card, agent=primary, port=port)

    # ------------------------------------------------------------------
    # Observability
    # ------------------------------------------------------------------

    def telemetry_summary(self) -> Dict[str, Any]:
        return self._telemetry.summary()

    async def audit_report(self, limit: int = 50) -> List[Dict[str, Any]]:
        return await self._audit.query(limit=limit)

    def agents(self) -> List[Dict[str, Any]]:
        return self._registry.list_agents()

    def on_cost_alert(self, callback: Any) -> None:
        self._telemetry.on_alert(callback)

    def __repr__(self) -> str:
        return f"AgentManager(project={self.project_id!r}, agents={len(self._registry)})"
